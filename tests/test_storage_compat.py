"""Regression tests for per-collection storage layout and legacy read fallback."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.storage.catalog import DocumentCatalog
from wenmai.storage.compat import (
    dir_has_entries,
    resolve_chroma_persist_path,
    resolve_read_path,
)
from wenmai.storage.fingerprints import FingerprintStore
from wenmai.storage.paths import collection_storage_bindings


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    return replace(
        test_settings,
        product=replace(test_settings.product, collection=other_id),
    )


def _chunk(document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:0000",
        document_id=document_id,
        text=text,
        metadata={
            "document_id": document_id,
            "title": document_id,
            "culture_domain": "妈祖",
            "审阅状态": "已通过",
        },
    )


def _commit(
    knowledge,
    document_id: str,
    text: str,
    *,
    source_path: str,
    sha256: str,
) -> None:
    knowledge.commit_document(
        source_path=source_path,
        sha256=sha256,
        document_id=document_id,
        status="ingested",
        chunks=[_chunk(document_id, text)],
    )


def test_resolve_read_path_prefers_new_layout(tmp_path: Path) -> None:
    new_path = tmp_path / "new" / "catalog.json"
    legacy_path = tmp_path / "legacy" / "catalog.json"
    new_path.parent.mkdir(parents=True)
    legacy_path.parent.mkdir(parents=True)
    new_path.write_text("{}", encoding="utf-8")
    legacy_path.write_text("{}", encoding="utf-8")

    assert (
        resolve_read_path(new_path, legacy_path, allow_legacy_fallback=True) == new_path
    )


def test_resolve_read_path_falls_back_to_legacy_for_default_collection(
    tmp_path: Path,
) -> None:
    new_path = tmp_path / "missing-new" / "catalog.json"
    legacy_path = tmp_path / "legacy-only" / "catalog.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("{}", encoding="utf-8")

    assert (
        resolve_read_path(new_path, legacy_path, allow_legacy_fallback=True) == legacy_path
    )


def test_resolve_read_path_skips_legacy_when_fallback_disabled(tmp_path: Path) -> None:
    new_path = tmp_path / "missing-new" / "catalog.json"
    legacy_path = tmp_path / "legacy-only" / "catalog.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("{}", encoding="utf-8")

    assert (
        resolve_read_path(new_path, legacy_path, allow_legacy_fallback=False) == new_path
    )


def test_resolve_chroma_persist_path_prefers_nonempty_new_dir(tmp_path: Path) -> None:
    new_path = tmp_path / "new-chroma"
    legacy_path = tmp_path / "legacy-chroma"
    new_path.mkdir()
    (new_path / "chroma.sqlite3").write_text("new", encoding="utf-8")
    legacy_path.mkdir()
    (legacy_path / "chroma.sqlite3").write_text("legacy", encoding="utf-8")

    assert (
        resolve_chroma_persist_path(new_path, legacy_path, allow_legacy_fallback=True)
        == new_path
    )


def test_resolve_chroma_persist_path_falls_back_to_nonempty_legacy_dir(
    tmp_path: Path,
) -> None:
    new_path = tmp_path / "empty-new"
    legacy_path = tmp_path / "legacy-chroma"
    new_path.mkdir()
    legacy_path.mkdir()
    (legacy_path / "chroma.sqlite3").write_text("legacy", encoding="utf-8")

    assert dir_has_entries(legacy_path)
    assert (
        resolve_chroma_persist_path(new_path, legacy_path, allow_legacy_fallback=True)
        == legacy_path
    )


def test_resolve_chroma_persist_path_skips_legacy_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    new_path = tmp_path / "empty-new"
    legacy_path = tmp_path / "legacy-chroma"
    new_path.mkdir()
    legacy_path.mkdir()
    (legacy_path / "chroma.sqlite3").write_text("legacy", encoding="utf-8")

    assert (
        resolve_chroma_persist_path(new_path, legacy_path, allow_legacy_fallback=False)
        == new_path
    )


def test_legacy_catalog_read_fallback_for_default_collection(test_settings: Settings) -> None:
    bindings = collection_storage_bindings(test_settings)
    bindings.legacy_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    bindings.legacy_catalog_path.write_text(
        json.dumps(
            {
                "documents": {
                    "legacy-only-doc": {
                        "culture_domain": "妈祖",
                        "title": "legacy-only-doc",
                        "chunks": [
                            {
                                "chunk_id": "legacy-only-doc:0000",
                                "document_id": "legacy-only-doc",
                                "preview": "legacy preview",
                                "审阅状态": "已通过",
                            }
                        ],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = DocumentCatalog.from_settings(test_settings)

    entry = catalog.get_document("legacy-only-doc")
    assert entry is not None
    assert entry.title == "legacy-only-doc"


def test_new_layout_catalog_preferred_over_legacy_for_default_collection(
    test_settings: Settings,
) -> None:
    bindings = collection_storage_bindings(test_settings)
    bindings.legacy_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    bindings.legacy_catalog_path.write_text(
        json.dumps(
            {
                "documents": {
                    "legacy-doc": {
                        "culture_domain": "妈祖",
                        "title": "legacy",
                        "chunks": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    bindings.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    bindings.catalog_path.write_text(
        json.dumps(
            {
                "documents": {
                    "new-doc": {
                        "culture_domain": "妈祖",
                        "title": "new-doc",
                        "chunks": [],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = DocumentCatalog.from_settings(test_settings)

    assert catalog.get_document("new-doc") is not None
    assert catalog.get_document("legacy-doc") is None


def test_legacy_fingerprint_read_fallback_for_default_collection(
    test_settings: Settings,
) -> None:
    bindings = collection_storage_bindings(test_settings)
    legacy_store = FingerprintStore(bindings.legacy_ingestion_history_path)
    legacy_store.upsert(
        source_path="/tmp/legacy-source.md",
        sha256="legacy-sha",
        document_id="legacy-doc",
        status="ingested",
    )
    legacy_store.close()

    store = FingerprintStore.from_settings(test_settings)
    record = store.get_by_source_path("/tmp/legacy-source.md")

    assert record is not None
    assert record.document_id == "legacy-doc"
    assert record.sha256 == "legacy-sha"
    store.close()


def test_non_default_collection_does_not_fallback_to_legacy_fingerprints(
    test_settings: Settings,
) -> None:
    bindings = collection_storage_bindings(test_settings)
    legacy_store = FingerprintStore(bindings.legacy_ingestion_history_path)
    legacy_store.upsert(
        source_path="/tmp/legacy-source.md",
        sha256="legacy-sha",
        document_id="legacy-doc",
        status="ingested",
    )
    legacy_store.close()

    other_settings = _other_collection_settings(test_settings)
    store = FingerprintStore.from_settings(other_settings)

    assert store.get_by_source_path("/tmp/legacy-source.md") is None
    store.close()


def test_legacy_chroma_path_fallback_for_default_collection(tmp_path: Path) -> None:
    new_path = tmp_path / "new-chroma"
    legacy_path = tmp_path / "legacy-chroma"
    new_path.mkdir()
    legacy_path.mkdir()
    (legacy_path / "chroma.sqlite3").write_text("legacy", encoding="utf-8")

    resolved = resolve_chroma_persist_path(
        new_path,
        legacy_path,
        allow_legacy_fallback=True,
    )

    assert resolved == legacy_path


def test_multiple_collections_coexist_in_new_layout(test_settings: Settings) -> None:
    default_knowledge = create_knowledge(test_settings)
    other_settings = _other_collection_settings(test_settings)
    other_knowledge = create_knowledge(other_settings)

    _commit(default_knowledge, "default-doc", "默认集合", source_path="/tmp/a.md", sha256="a")
    _commit(other_knowledge, "other-doc", "其他集合", source_path="/tmp/b.md", sha256="b")

    default_bindings = collection_storage_bindings(test_settings)
    other_bindings = collection_storage_bindings(other_settings)

    assert DocumentCatalog(default_bindings.catalog_path).get_document("default-doc")
    assert DocumentCatalog(other_bindings.catalog_path).get_document("other-doc")
    assert default_bindings.catalog_path != other_bindings.catalog_path
    assert default_bindings.chroma_path != other_bindings.chroma_path
    assert default_bindings.bm25_path != other_bindings.bm25_path


def test_same_source_path_in_different_collections_stays_independent(
    test_settings: Settings,
) -> None:
    shared_source = "/tmp/shared-source.md"
    default_knowledge = create_knowledge(test_settings)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings))

    _commit(
        default_knowledge,
        "default-shared",
        "默认集合共享源",
        source_path=shared_source,
        sha256="default-sha",
    )
    _commit(
        other_knowledge,
        "other-shared",
        "其他集合共享源",
        source_path=shared_source,
        sha256="other-sha",
    )

    assert default_knowledge.get_by_document_id("default-shared")
    assert other_knowledge.get_by_document_id("other-shared")

    default_plan = default_knowledge.plan_document(
        source_path=shared_source,
        sha256="default-sha",
        document_id="default-shared",
    )
    other_plan = other_knowledge.plan_document(
        source_path=shared_source,
        sha256="other-sha",
        document_id="other-shared",
    )
    assert default_plan.status == "skipped"
    assert other_plan.status == "skipped"

    default_knowledge.delete_document("default-shared")
    assert default_knowledge.get_by_document_id("default-shared") == []
    assert other_knowledge.get_by_document_id("other-shared")

    other_plan_after_delete = other_knowledge.plan_document(
        source_path=shared_source,
        sha256="other-sha",
        document_id="other-shared",
    )
    assert other_plan_after_delete.status == "skipped"

    reingest_plan = default_knowledge.plan_document(
        source_path=shared_source,
        sha256="default-sha",
        document_id="default-shared",
    )
    assert reingest_plan.status == "ingested"


def test_rebuild_in_one_collection_does_not_delete_other_collection_doc(
    test_settings: Settings,
) -> None:
    shared_source = "/tmp/shared-rebuild.md"
    default_knowledge = create_knowledge(test_settings)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings))

    _commit(
        default_knowledge,
        "default-v1",
        "默认 v1",
        source_path=shared_source,
        sha256="v1",
    )
    _commit(
        other_knowledge,
        "other-v1",
        "其他 v1",
        source_path=shared_source,
        sha256="v1",
    )

    rebuilt = default_knowledge.plan_document(
        source_path=shared_source,
        sha256="v2",
        document_id="default-v2",
    )
    assert rebuilt.status == "rebuilt"
    assert rebuilt.previous_document_id == "default-v1"

    default_knowledge.commit_document(
        source_path=shared_source,
        sha256="v2",
        document_id="default-v2",
        status="rebuilt",
        chunks=[_chunk("default-v2", "默认 v2")],
        previous_document_id=rebuilt.previous_document_id,
    )

    assert default_knowledge.get_by_document_id("default-v1") == []
    assert default_knowledge.get_by_document_id("default-v2")
    assert other_knowledge.get_by_document_id("other-v1")
