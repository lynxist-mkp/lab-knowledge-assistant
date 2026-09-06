"""Contract tests for collection scope, document facade, MCP envelope, and image refs."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_knowledge.config import Settings
from lab_knowledge.http.ops_service import OpsService
from lab_knowledge.knowledge import create_document_management, create_knowledge
from lab_knowledge.knowledge.collections import (
    CollectionReadModel,
    CollectionScope,
    UnknownCollectionError,
)
from lab_knowledge.knowledge.document_card import DocumentNotFoundError
from lab_knowledge.knowledge.domain import REVIEW_PENDING
from lab_knowledge.knowledge.image_refs import ImageNotFoundError, ImageReferenceService
from lab_knowledge.knowledge.store import Knowledge
from lab_knowledge.mcp.envelope import refs_from_citations, scope_for
from lab_knowledge.mcp.summary import GetDocumentSummaryError, get_document_summary
from lab_knowledge.mcp.tools.ask import ask_answer
from lab_knowledge.mcp.tools.collections import collections_get_stats, collections_list
from lab_knowledge.mcp.tools.documents import documents_delete, documents_get, documents_list
from lab_knowledge.mcp.tools.images import images_get_content, images_get_ref
from lab_knowledge.mcp.tools.reviews import (
    reviews_approve,
    reviews_list_pending,
    reviews_reject,
)
from lab_knowledge.models import Chunk, Citation
from lab_knowledge.storage.catalog import DocumentCatalog
from lab_knowledge.storage.images import ImageStore
from lab_knowledge.storage.paths import collection_storage_bindings
from tests.conftest import register_collection


def _commit(
    knowledge: Knowledge,
    document_id: str,
    text: str,
    *,
    culture_domain: str = "妈祖",
    review_status: str = "已通过",
    extra_metadata: dict[str, object] | None = None,
) -> None:
    knowledge.commit_document(
        source_path=f"/tmp/{document_id}.md",
        sha256=document_id,
        document_id=document_id,
        status="ingested",
        chunks=[
            Chunk(
                chunk_id=f"{document_id}:0000",
                document_id=document_id,
                text=text,
                metadata={
                    "document_id": document_id,
                    "title": document_id,
                    "culture_domain": culture_domain,
                    "审阅状态": review_status,
                    **(extra_metadata or {}),
                },
            )
        ],
    )


def _envelope_keys() -> set[str]:
    return {"data", "scope", "refs", "meta", "warnings"}


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def _settings_with_other_collection(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    return register_collection(test_settings, other_id)


def test_collection_stats_status_layering(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-approved", "已通过内容", review_status="已通过")
    _commit(knowledge, "doc-pending", "待审内容", review_status=REVIEW_PENDING)

    stats = CollectionReadModel(test_settings, knowledge).get_stats()

    assert stats.document_count == 2
    assert stats.chunk_count == 2
    assert stats.review.approved_documents == 1
    assert stats.review.pending_documents == 1
    assert stats.review.approved_chunks == 1
    assert stats.review.pending_chunks == 1
    assert len(stats.by_culture_domain) == 1
    assert stats.by_culture_domain[0].pending_chunks == 1


def test_collection_scope_defaults_to_configured_collection(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)

    scope = model.resolve_scope()

    assert isinstance(scope, CollectionScope)
    assert scope.collection_id == test_settings.product.collection
    assert scope.display_name == test_settings.product.name
    assert scope.settings is not test_settings
    assert scope.settings.product.collection == test_settings.product.collection
    assert scope.settings.product.name == test_settings.product.name


def test_collection_scope_explicit_default_matches_default_scope(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)

    implicit = model.resolve_scope()
    explicit = model.resolve_scope(test_settings.product.collection)

    assert explicit == implicit
    assert explicit.settings is not implicit.settings


def test_collection_storage_bindings_follow_scoped_settings(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    scope = CollectionReadModel(test_settings, knowledge).resolve_scope()

    bindings = collection_storage_bindings(scope.settings)
    collection_id = test_settings.product.collection

    assert bindings.collection_id == collection_id
    assert bindings.legacy_read_fallback is True
    assert bindings.images_root.name == collection_id
    assert bindings.chroma_path == bindings.legacy_chroma_path / collection_id
    assert bindings.bm25_path.name == collection_id
    assert bindings.catalog_path.name == "catalog.json"
    assert bindings.catalog_path.parent.name == collection_id
    assert bindings.ingestion_history_path.parent.name == collection_id
    assert bindings.image_index_path.parent.name == collection_id
    assert bindings.chroma_path.parent.exists()
    assert bindings.bm25_path.parent.exists()
    assert bindings.catalog_path.parent.exists()
    assert bindings.ingestion_history_path.parent.exists()
    assert bindings.image_index_path.parent.exists()


def test_collection_storage_bindings_isolate_physical_paths(test_settings: Settings) -> None:
    default_bindings = collection_storage_bindings(test_settings)
    other_settings = replace(
        test_settings,
        product=replace(test_settings.product, collection="other-collection"),
    )
    other_bindings = collection_storage_bindings(other_settings)

    assert other_bindings.legacy_read_fallback is False
    assert default_bindings.catalog_path != other_bindings.catalog_path
    assert default_bindings.chroma_path != other_bindings.chroma_path
    assert default_bindings.bm25_path != other_bindings.bm25_path
    assert default_bindings.image_index_path != other_bindings.image_index_path
    assert default_bindings.images_root != other_bindings.images_root
    assert default_bindings.ingestion_history_path != other_bindings.ingestion_history_path


def test_legacy_catalog_read_fallback_for_default_collection(test_settings: Settings) -> None:
    legacy_catalog = Path(test_settings.paths.catalog)
    if not legacy_catalog.is_absolute():
        legacy_catalog = test_settings.root / legacy_catalog
    legacy_catalog.parent.mkdir(parents=True, exist_ok=True)
    legacy_catalog.write_text(
        json.dumps(
            {
                "documents": {
                    "legacy-doc": {
                        "culture_domain": "妈祖",
                        "title": "legacy-doc",
                        "chunks": [
                            {
                                "chunk_id": "legacy-doc:0000",
                                "document_id": "legacy-doc",
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

    entry = catalog.get_document("legacy-doc")
    assert entry is not None
    assert entry.title == "legacy-doc"


def test_non_default_collection_does_not_fallback_to_legacy_catalog(
    test_settings: Settings,
) -> None:
    legacy_catalog = Path(test_settings.paths.catalog)
    if not legacy_catalog.is_absolute():
        legacy_catalog = test_settings.root / legacy_catalog
    legacy_catalog.parent.mkdir(parents=True, exist_ok=True)
    legacy_catalog.write_text(
        json.dumps(
            {
                "documents": {
                    "legacy-doc": {
                        "culture_domain": "妈祖",
                        "title": "legacy-doc",
                        "chunks": [],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    other_settings = replace(
        test_settings,
        product=replace(test_settings.product, collection="other-collection"),
    )

    catalog = DocumentCatalog.from_settings(other_settings)

    assert catalog.get_document("legacy-doc") is None


def test_commit_writes_to_per_collection_storage_layout(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "new-layout-doc", "新布局内容")

    bindings = collection_storage_bindings(test_settings)

    assert bindings.catalog_path.exists()
    assert bindings.ingestion_history_path.exists()
    catalog = DocumentCatalog(bindings.catalog_path)
    assert catalog.get_document("new-layout-doc") is not None


def test_create_knowledge_scoped_collections_do_not_share_catalog(
    test_settings: Settings,
) -> None:
    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "default-only-doc", "默认集合文档")

    other_settings = replace(
        test_settings,
        product=replace(test_settings.product, collection="other-collection"),
    )
    other_knowledge = create_knowledge(other_settings)
    _commit(other_knowledge, "other-only-doc", "其他集合文档")

    default_bindings = collection_storage_bindings(test_settings)
    other_bindings = collection_storage_bindings(other_settings)

    default_catalog = DocumentCatalog(default_bindings.catalog_path)
    other_catalog = DocumentCatalog(other_bindings.catalog_path)

    assert default_catalog.get_document("default-only-doc") is not None
    assert default_catalog.get_document("other-only-doc") is None
    assert other_catalog.get_document("other-only-doc") is not None
    assert other_catalog.get_document("default-only-doc") is None


def test_unknown_collection_id_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)
    with pytest.raises(UnknownCollectionError):
        model.get_stats("not-a-real-collection")

    with pytest.raises(UnknownCollectionError):
        model.resolve_scope("not-a-real-collection")


def test_document_management_list_and_delete(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "文档 A")
    _commit(knowledge, "doc-b", "文档 B", culture_domain="朱子")

    mgmt = create_document_management(test_settings, knowledge=knowledge)
    all_docs = mgmt.list_documents(collection_id=test_settings.product.collection)
    assert {doc.document_id for doc in all_docs} == {"doc-a", "doc-b"}

    matsu_only = mgmt.list_documents(
        collection_id=test_settings.product.collection,
        culture_domain="妈祖",
    )
    assert len(matsu_only) == 1
    assert matsu_only[0].document_id == "doc-a"

    mgmt.delete_document("doc-a", collection_id=test_settings.product.collection)
    remaining = mgmt.list_documents()
    assert len(remaining) == 1
    assert remaining[0].document_id == "doc-b"


def test_document_management_for_collection_returns_scoped_view(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-scoped", "作用域文档")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    scoped = mgmt.for_collection(test_settings.product.collection)

    assert scoped is not mgmt
    assert scoped.settings is not mgmt.settings
    assert scoped.settings.product.collection == test_settings.product.collection
    assert scoped.scope.collection_id == test_settings.product.collection
    docs = scoped.list_documents()
    assert {doc.document_id for doc in docs} == {"doc-scoped"}


def test_document_management_for_collection_is_idempotent(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))
    scoped = mgmt.for_collection(test_settings.product.collection)

    assert scoped.for_collection(test_settings.product.collection) is scoped
    assert scoped.for_collection(None) is scoped


def test_create_knowledge_with_collection_id_uses_collection_storage(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings, collection_id=test_settings.product.collection)
    _commit(knowledge, "doc-knowledge", "知识库作用域")
    scoped_mgmt = create_document_management(
        test_settings,
        collection_id=test_settings.product.collection,
    )
    assert scoped_mgmt.get_document("doc-knowledge").document_id == "doc-knowledge"


def test_create_document_management_with_collection_id_binds_scoped_view(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-factory", "工厂作用域文档")

    mgmt = create_document_management(
        test_settings,
        collection_id=test_settings.product.collection,
    )

    assert mgmt.settings is not test_settings
    assert mgmt.settings.product.collection == test_settings.product.collection
    assert mgmt.scope.collection_id == test_settings.product.collection
    assert {doc.document_id for doc in mgmt.list_documents()} == {"doc-factory"}


def test_document_management_for_collection_unknown_collection_raises(
    test_settings: Settings,
) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    with pytest.raises(UnknownCollectionError, match="unknown collection: missing"):
        mgmt.for_collection("missing")


def test_mcp_envelope_shape_for_collections(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-1", "统计用文档")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    listed = collections_list(mgmt)
    assert set(listed) == _envelope_keys()
    assert listed["scope"]["collection_id"] == test_settings.product.collection
    assert listed["data"][0]["collection_id"] == test_settings.product.collection

    stats = collections_get_stats(mgmt)
    assert set(stats) == _envelope_keys()
    assert stats["data"]["document_count"] == 1
    assert stats["meta"]["count"] == 1


def test_mcp_documents_list_uses_collection_bound_document_management(
    test_settings: Settings,
) -> None:
    captured: dict[str, object] = {}

    class RootDocumentManagement:
        settings = test_settings

        def list_documents(
            self,
            *,
            collection_id: str | None = None,
            culture_domain: str | None = None,
        ):
            captured["collection_id"] = collection_id
            captured["culture_domain"] = culture_domain
            return []

    result = documents_list(
        RootDocumentManagement(),  # type: ignore[arg-type]
        collection_id=test_settings.product.collection,
        culture_domain="妈祖",
    )

    assert captured["collection_id"] == test_settings.product.collection
    assert captured["culture_domain"] == "妈祖"
    assert result["scope"]["collection_id"] == test_settings.product.collection


def test_ops_service_uses_collection_bound_document_management(test_settings: Settings) -> None:
    captured: dict[str, object] = {}

    class RootDocumentManagement:
        def browse_groups(self, *, collection_id: str | None = None):
            captured["collection_id"] = collection_id
            return ["ok"]

    service = OpsService(
        test_settings,
        document_management=RootDocumentManagement(),  # type: ignore[arg-type]
    )

    assert service.browse_groups(collection_id=test_settings.product.collection) == ["ok"]
    assert captured["collection_id"] == test_settings.product.collection


def test_document_management_get_document_returns_domain_card(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-summary",
        "摘要正文",
        extra_metadata={"summary": "文档摘要", "tags": ["妈祖", "祖庙"]},
    )
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    card = mgmt.get_document(
        "doc-summary",
        collection_id=test_settings.product.collection,
    )

    assert card.document_id == "doc-summary"
    assert card.chunk_count == 1
    assert card.summary == "文档摘要"
    assert card.tags == ["妈祖", "祖庙"]


def test_document_management_get_document_unknown_collection_raises(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-summary", "摘要正文")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    with pytest.raises(UnknownCollectionError, match="unknown collection: missing"):
        mgmt.get_document("doc-summary", collection_id="missing")


def test_document_management_get_chunk_detail_returns_domain_read_model(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-chunk-detail",
        "片段详情正文",
        extra_metadata={"summary": "片段摘要", "tags": ["片段标签"], "chunk_title": "详情片段"},
    )
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    detail = mgmt.get_chunk_detail(
        "doc-chunk-detail:0000",
        collection_id=test_settings.product.collection,
    )

    assert detail is not None
    assert detail.document_id == "doc-chunk-detail"
    assert detail.text == "片段详情正文"
    assert detail.as_dict()["审阅状态"] == "已通过"
    assert detail.as_dict()["chunk_title"] == "详情片段"
    assert detail.as_dict()["summary"] == "片段摘要"
    assert detail.as_dict()["tags"] == ["片段标签"]


def test_mcp_get_document_summary_uses_document_management(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-mcp-summary", "MCP 摘要正文")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    summary = get_document_summary(
        "doc-mcp-summary",
        settings=test_settings,
        document_management=mgmt,
    )

    assert summary == mgmt.get_document(
        "doc-mcp-summary",
        collection_id=test_settings.product.collection,
    ).as_dict()


def test_mcp_get_document_summary_unknown_id_raises(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    with pytest.raises(GetDocumentSummaryError, match="document not found"):
        get_document_summary(
            "missing-doc",
            settings=test_settings,
            document_management=mgmt,
        )


def test_mcp_get_document_summary_routes_by_collection_id(
    test_settings: Settings,
) -> None:
    from dataclasses import replace

    from tests.conftest import register_collection

    other_id = "other-collection"
    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "default-summary-doc", "默认集合摘要")

    other_settings = replace(
        register_collection(test_settings, other_id),
        product=replace(test_settings.product, collection=other_id),
    )
    other_knowledge = create_knowledge(other_settings)
    _commit(other_knowledge, "other-summary-doc", "其他集合摘要")

    mgmt = create_document_management(
        register_collection(test_settings, other_id),
        knowledge=default_knowledge,
    )

    default_summary = get_document_summary(
        "default-summary-doc",
        settings=test_settings,
        document_management=mgmt,
        collection_id=test_settings.product.collection,
    )
    assert default_summary["document_id"] == "default-summary-doc"

    other_summary = get_document_summary(
        "other-summary-doc",
        settings=register_collection(test_settings, other_id),
        document_management=mgmt,
        collection_id=other_id,
    )
    assert other_summary["document_id"] == "other-summary-doc"


def test_mcp_get_document_summary_unknown_collection_raises(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    with pytest.raises(ValueError, match="unknown collection"):
        get_document_summary(
            "doc-1",
            settings=test_settings,
            document_management=mgmt,
            collection_id="missing",
        )


def test_mcp_documents_get_envelope(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-card",
        "卡片内容",
        extra_metadata={"summary": "卡片摘要", "tags": ["妈祖"], "chunk_title": "卡片片段"},
    )
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    result = documents_get(
        mgmt,
        "doc-card",
        collection_id=test_settings.product.collection,
    )
    assert set(result) == _envelope_keys()
    assert result["data"]["document_id"] == "doc-card"
    assert result["data"]["summary"] == "卡片摘要"
    assert result["data"]["tags"] == ["妈祖"]
    assert result["refs"]["document_ids"] == ["doc-card"]


def test_mcp_documents_list_envelope(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-list", "列表内容")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    result = documents_list(mgmt, collection_id=test_settings.product.collection)
    assert set(result) == _envelope_keys()
    assert result["meta"]["count"] == 1
    assert result["refs"]["document_ids"] == ["doc-list"]
    assert result["scope"]["collection_id"] == test_settings.product.collection


def test_mcp_documents_delete_envelope_and_effect(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-delete", "待删除内容")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    result = documents_delete(
        mgmt,
        "doc-delete",
        collection_id=test_settings.product.collection,
    )

    assert set(result) == _envelope_keys()
    assert result["data"] == {"document_id": "doc-delete", "deleted": True}
    assert result["refs"]["document_ids"] == ["doc-delete"]
    assert mgmt.list_documents(collection_id=test_settings.product.collection) == []


def test_mcp_documents_delete_unknown_id_raises(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    with pytest.raises(ValueError, match="document not found"):
        documents_delete(
            mgmt,
            "missing-doc",
            collection_id=test_settings.product.collection,
        )


def test_ask_answer_envelope_and_collection_scope(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-ask", "湄洲妈祖祖庙是妈祖信仰中心。")

    result = ask_answer(
        "妈祖信仰中心在哪里？",
        test_settings,
        collection_id=test_settings.product.collection,
        knowledge=knowledge,
    )
    assert set(result) == _envelope_keys()
    assert result["scope"]["collection_id"] == test_settings.product.collection
    assert "answer" in result["data"]
    assert result["meta"]["elapsed_ms"] is not None


def _assert_mcp_unknown_collection_raises(callable, *args, **kwargs) -> None:
    """MCP tools must wrap UnknownCollectionError in a plain ValueError."""
    with pytest.raises(ValueError, match="unknown collection") as exc_info:
        callable(*args, **kwargs)
    assert type(exc_info.value) is ValueError
    assert isinstance(exc_info.value.__cause__, UnknownCollectionError)
    assert exc_info.value.__cause__.collection_id == "missing"


def test_ask_answer_unknown_collection_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _assert_mcp_unknown_collection_raises(
        ask_answer,
        "问题",
        test_settings,
        collection_id="missing",
        knowledge=knowledge,
    )


def test_mcp_tools_unknown_collection_raise_value_error(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    _assert_mcp_unknown_collection_raises(collections_get_stats, mgmt, collection_id="missing")
    _assert_mcp_unknown_collection_raises(documents_list, mgmt, collection_id="missing")
    _assert_mcp_unknown_collection_raises(
        documents_get, mgmt, "doc-1", collection_id="missing"
    )
    _assert_mcp_unknown_collection_raises(
        documents_delete, mgmt, "doc-1", collection_id="missing"
    )
    _assert_mcp_unknown_collection_raises(reviews_list_pending, mgmt, collection_id="missing")
    _assert_mcp_unknown_collection_raises(
        reviews_approve, mgmt, "doc-1", collection_id="missing"
    )
    _assert_mcp_unknown_collection_raises(
        reviews_reject, mgmt, "doc-1", collection_id="missing"
    )
    _assert_mcp_unknown_collection_raises(images_get_ref, mgmt, "img-1", collection_id="missing")
    _assert_mcp_unknown_collection_raises(
        images_get_content, mgmt, "img-1", collection_id="missing"
    )


def test_mcp_reviews_pending_approve_reject_contract(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-review-a", "待审 A", review_status=REVIEW_PENDING)
    _commit(knowledge, "doc-review-b", "待审 B", review_status=REVIEW_PENDING)
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    pending = reviews_list_pending(
        mgmt,
        collection_id=test_settings.product.collection,
    )
    assert set(pending) == _envelope_keys()
    assert pending["meta"]["count"] == 2
    assert set(pending["refs"]["document_ids"]) == {"doc-review-a", "doc-review-b"}

    approved = reviews_approve(
        mgmt,
        "doc-review-a",
        collection_id=test_settings.product.collection,
    )
    assert approved["data"] == {"document_id": "doc-review-a", "审阅状态": "已通过"}

    rejected = reviews_reject(
        mgmt,
        "doc-review-b",
        collection_id=test_settings.product.collection,
    )
    assert rejected["data"] == {"document_id": "doc-review-b", "审阅状态": "已驳回"}

    remaining = reviews_list_pending(
        mgmt,
        collection_id=test_settings.product.collection,
    )
    assert remaining["data"] == []
    assert remaining["meta"]["count"] == 0


def test_mcp_reviews_unknown_id_raises(test_settings: Settings) -> None:
    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    with pytest.raises(ValueError, match="document not found"):
        reviews_approve(
            mgmt,
            "missing-doc",
            collection_id=test_settings.product.collection,
        )

    with pytest.raises(ValueError, match="document not found"):
        reviews_reject(
            mgmt,
            "missing-doc",
            collection_id=test_settings.product.collection,
        )


def test_refs_from_citations_extracts_image_ids() -> None:
    citations = [
        Citation(
            index=1,
            chunk_id="c1",
            document_id="d1",
            title="t",
            excerpt="正文 [IMAGE: abc123def] 说明",
        )
    ]
    refs = refs_from_citations(citations)
    assert refs.image_ids == ["abc123def"]
    assert refs.document_ids == ["d1"]
    assert refs.chunk_ids == ["c1"]


def test_image_ref_round_trip(test_settings: Settings) -> None:
    store = ImageStore(test_settings)
    image_bytes = b"\x89PNG\r\n\x1a\nfake"
    image_id = store.save(
        document_id="doc-img",
        source_path="/tmp/doc-img.md",
        page=1,
        image_bytes=image_bytes,
        mime_type="image/png",
    )

    service = ImageReferenceService(test_settings)
    ref = service.get_ref(image_id)
    assert ref is not None
    payload = ref.as_dict()
    assert payload["image_id"] == image_id
    assert payload["document_id"] == "doc-img"
    assert payload["mime_type"] == "image/png"
    assert "file_path" not in payload
    assert "source_path" not in payload

    listed = service.list_refs_for_document("doc-img")
    assert len(listed) == 1
    assert listed[0].image_id == image_id

    content = service.get_content(image_id)
    assert base64.b64decode(content.content_base64) == image_bytes

    mgmt = create_document_management(test_settings, knowledge=create_knowledge(test_settings))

    mcp_ref = images_get_ref(
        mgmt,
        image_id,
        collection_id=test_settings.product.collection,
    )
    assert set(mcp_ref) == _envelope_keys()
    assert mcp_ref["refs"]["image_ids"] == [image_id]

    mcp_content = images_get_content(
        mgmt,
        image_id,
        collection_id=test_settings.product.collection,
    )
    assert set(mcp_content) == _envelope_keys()
    assert mcp_content["data"]["mime_type"] == "image/png"
    assert "file_path" not in mcp_content["data"]
    assert "source_path" not in mcp_content["data"]


def test_same_image_bytes_in_different_documents_do_not_alias(test_settings: Settings) -> None:
    store = ImageStore(test_settings)
    image_bytes = b"\x89PNG\r\n\x1a\nshared"

    first_image_id = store.save(
        document_id="doc-a",
        source_path="/tmp/doc-a.md",
        page=1,
        image_bytes=image_bytes,
        mime_type="image/png",
    )
    second_image_id = store.save(
        document_id="doc-b",
        source_path="/tmp/doc-b.md",
        page=1,
        image_bytes=image_bytes,
        mime_type="image/png",
    )

    assert first_image_id != second_image_id
    assert store.get(first_image_id) is not None
    assert store.get(first_image_id).document_id == "doc-a"
    assert store.get(second_image_id) is not None
    assert store.get(second_image_id).document_id == "doc-b"


def test_ask_answer_keeps_image_refs_after_caption_replacement(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    image_id = ImageStore(test_settings).save(
        document_id="doc-captioned",
        source_path="/tmp/doc-captioned.md",
        page=1,
        image_bytes=b"\x89PNG\r\n\x1a\ncaptioned",
        mime_type="image/png",
    )
    _commit(
        knowledge,
        "doc-captioned",
        "湄洲妈祖祖庙有一幅仪式插图，并附文字说明。",
        extra_metadata={"image_ids": [image_id]},
    )

    result = ask_answer(
        "妈祖祖庙有什么插图说明？",
        test_settings,
        collection_id=test_settings.product.collection,
        knowledge=knowledge,
    )

    assert result["refs"]["image_ids"] == [image_id]


def test_scope_for_defaults_to_configured_collection(test_settings: Settings) -> None:
    scope = scope_for(test_settings)
    assert scope.collection_id == test_settings.product.collection


def test_document_management_routes_to_storage_backed_alternate_collection(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "default-only", "默认集合文档")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "other-only", "其他集合文档")

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    assert {doc.document_id for doc in mgmt.list_documents(collection_id=default_id)} == {
        "default-only"
    }
    assert {doc.document_id for doc in mgmt.list_documents(collection_id=other_id)} == {
        "other-only"
    }
    assert mgmt.get_document("other-only", collection_id=other_id).document_id == "other-only"

    with pytest.raises(DocumentNotFoundError):
        mgmt.get_document("other-only", collection_id=default_id)

    mgmt.delete_document("other-only", collection_id=other_id)
    assert {doc.document_id for doc in mgmt.list_documents(collection_id=default_id)} == {
        "default-only"
    }
    assert mgmt.list_documents(collection_id=other_id) == []


def test_document_management_for_collection_accepts_storage_backed_alternate(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "scoped-other", "作用域其他集合")

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))
    scoped = mgmt.for_collection(other_id)

    assert scoped.scope.collection_id == other_id
    assert scoped.settings.product.collection == other_id
    assert {doc.document_id for doc in scoped.list_documents()} == {"scoped-other"}


def test_document_management_review_routes_to_alternate_collection(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(
        default_knowledge,
        "pending-default",
        "默认待审",
        review_status=REVIEW_PENDING,
    )

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(
        other_knowledge,
        "pending-other",
        "其他待审",
        review_status=REVIEW_PENDING,
    )

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    assert {doc.document_id for doc in mgmt.list_pending_reviews(collection_id=other_id)} == {
        "pending-other"
    }
    assert {doc.document_id for doc in mgmt.list_pending_reviews(collection_id=default_id)} == {
        "pending-default"
    }

    mgmt.approve_review("pending-other", collection_id=other_id)
    mgmt.reject_review("pending-default", collection_id=default_id)

    assert mgmt.list_pending_reviews(collection_id=other_id) == []
    assert mgmt.list_pending_reviews(collection_id=default_id) == []


def test_document_management_images_route_to_alternate_collection(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    other_settings = _other_collection_settings(test_settings, other_id)

    other_knowledge = create_knowledge(other_settings)
    _commit(other_knowledge, "img-doc", "图片文档")

    default_image_id = ImageStore(test_settings).save(
        document_id="default-doc",
        source_path="/tmp/default-doc.md",
        page=1,
        image_bytes=b"\x89PNG\r\n\x1a\ndefault",
        mime_type="image/png",
    )
    other_image_id = ImageStore(other_settings).save(
        document_id="img-doc",
        source_path="/tmp/img-doc.md",
        page=1,
        image_bytes=b"\x89PNG\r\n\x1a\nother",
        mime_type="image/png",
    )

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    assert (
        mgmt.get_image_ref(
            default_image_id,
            collection_id=test_settings.product.collection,
        ).image_id
        == default_image_id
    )
    assert mgmt.get_image_ref(other_image_id, collection_id=other_id).image_id == other_image_id

    with pytest.raises(ImageNotFoundError):
        mgmt.get_image_ref(other_image_id, collection_id=test_settings.product.collection)

    default_refs = mgmt.image_refs_for_document(
        "default-doc",
        collection_id=test_settings.product.collection,
    )
    other_refs = mgmt.image_refs_for_document("img-doc", collection_id=other_id)
    assert [ref.image_id for ref in default_refs] == [default_image_id]
    assert [ref.image_id for ref in other_refs] == [other_image_id]

    default_content = mgmt.get_image_content(
        default_image_id,
        collection_id=test_settings.product.collection,
    )
    other_content = mgmt.get_image_content(other_image_id, collection_id=other_id)
    assert base64.b64decode(default_content.content_base64) == b"\x89PNG\r\n\x1a\ndefault"
    assert base64.b64decode(other_content.content_base64) == b"\x89PNG\r\n\x1a\nother"


def test_mcp_documents_routes_to_alternate_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "mcp-other", "MCP 其他集合")

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    listed = documents_list(mgmt, collection_id=other_id)
    assert listed["scope"]["collection_id"] == other_id
    assert listed["refs"]["document_ids"] == ["mcp-other"]

    fetched = documents_get(mgmt, "mcp-other", collection_id=other_id)
    assert fetched["data"]["document_id"] == "mcp-other"

    deleted = documents_delete(mgmt, "mcp-other", collection_id=other_id)
    assert deleted["data"]["deleted"] is True
    assert documents_list(mgmt, collection_id=other_id)["meta"]["count"] == 0


def test_mcp_reviews_route_to_alternate_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "review-other", "审阅其他", review_status=REVIEW_PENDING)

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    pending = reviews_list_pending(mgmt, collection_id=other_id)
    assert pending["scope"]["collection_id"] == other_id
    assert pending["refs"]["document_ids"] == ["review-other"]

    approved = reviews_approve(mgmt, "review-other", collection_id=other_id)
    assert approved["scope"]["collection_id"] == other_id
    assert reviews_list_pending(mgmt, collection_id=other_id)["meta"]["count"] == 0


def test_mcp_images_route_to_alternate_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    other_settings = _other_collection_settings(test_settings, other_id)
    other_knowledge = create_knowledge(other_settings)
    _commit(other_knowledge, "mcp-img-doc", "MCP 图片文档")

    other_image_id = ImageStore(other_settings).save(
        document_id="mcp-img-doc",
        source_path="/tmp/mcp-img-doc.md",
        page=1,
        image_bytes=b"\x89PNG\r\n\x1a\nmcp-other",
        mime_type="image/png",
    )

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    ref = images_get_ref(mgmt, other_image_id, collection_id=other_id)
    assert ref["scope"]["collection_id"] == other_id
    assert ref["refs"]["image_ids"] == [other_image_id]

    content = images_get_content(mgmt, other_image_id, collection_id=other_id)
    assert content["scope"]["collection_id"] == other_id
    assert base64.b64decode(content["data"]["content_base64"]) == b"\x89PNG\r\n\x1a\nmcp-other"


def test_ops_service_routes_browse_and_reviews_to_alternate_collection(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "ops-default", "运维默认")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "ops-other", "运维其他")
    _commit(
        other_knowledge,
        "ops-pending",
        "运维待审",
        review_status=REVIEW_PENDING,
    )

    service = OpsService(_settings_with_other_collection(test_settings, other_id))

    default_groups = service.browse_groups(collection_id=default_id)
    other_groups = service.browse_groups(collection_id=other_id)
    default_doc_ids = {
        doc.document_id for group in default_groups for doc in group.documents
    }
    other_doc_ids = {doc.document_id for group in other_groups for doc in group.documents}

    assert default_doc_ids == {"ops-default"}
    assert other_doc_ids == {"ops-other", "ops-pending"}

    pending = service.list_pending_reviews(collection_id=other_id)
    assert {item.document_id for item in pending} == {"ops-pending"}

    service.approve_review("ops-pending", collection_id=other_id)
    assert service.list_pending_reviews(collection_id=other_id) == []
    assert service.list_pending_reviews(collection_id=default_id) == []


def test_collection_read_model_stats_route_to_alternate_collection(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "stats-default", "默认统计")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "stats-other", "其他统计", review_status=REVIEW_PENDING)

    model = CollectionReadModel(
        _settings_with_other_collection(test_settings, other_id),
        default_knowledge,
    )

    default_stats = model.get_stats(default_id)
    other_stats = model.get_stats(other_id)

    assert default_stats.document_count == 1
    assert other_stats.document_count == 1
    assert default_stats.review.pending_documents == 0
    assert other_stats.review.pending_documents == 1


def test_collection_read_model_resolve_scope_accepts_storage_backed_alternate(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "scope-other", "作用域其他")

    scope = CollectionReadModel(
        _settings_with_other_collection(test_settings, other_id),
        create_knowledge(test_settings),
    ).resolve_scope(other_id)

    assert scope.collection_id == other_id
    assert scope.settings.product.collection == other_id


def test_mcp_collections_stats_route_to_alternate_collection(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "mcp-stats-other", "MCP 统计其他", review_status=REVIEW_PENDING)

    mgmt = create_document_management(_settings_with_other_collection(test_settings, other_id))

    stats = collections_get_stats(mgmt, collection_id=other_id)

    assert stats["scope"]["collection_id"] == other_id
    assert stats["data"]["document_count"] == 1
    assert stats["data"]["review"]["pending_documents"] == 1
    assert stats["meta"]["count"] == 1


def test_ops_service_overview_stats_route_to_alternate_collection(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "overview-default", "概览默认")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "overview-other", "概览其他")

    service = OpsService(_settings_with_other_collection(test_settings, other_id))

    default_stats = service.overview_stats(collection_id=default_id)
    other_stats = service.overview_stats(collection_id=other_id)

    assert default_stats.document_count == 1
    assert other_stats.document_count == 1
    assert default_stats.chunk_count == 1
    assert other_stats.chunk_count == 1
