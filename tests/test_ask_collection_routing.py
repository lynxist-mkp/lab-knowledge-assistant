"""Ask path collection routing: scoped knowledge binding and retrieval."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lab_knowledge.config import Settings
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.knowledge import create_knowledge
from lab_knowledge.knowledge.collections import (
    UnknownCollectionError,
    resolve_routable_collection_scope,
)
from lab_knowledge.mcp.tools.ask import ask_answer
from lab_knowledge.models import Chunk
from tests.conftest import register_collection


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def _commit(knowledge, document_id: str, text: str) -> None:
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
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )


def test_resolve_routable_collection_scope_accepts_registered_alternate(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "other-scope-doc", "其他集合作用域")

    scope = resolve_routable_collection_scope(
        register_collection(test_settings, other_id),
        other_id,
    )

    assert scope.collection_id == other_id
    assert scope.settings.product.collection == other_id


def test_resolve_routable_collection_scope_rejects_missing_storage(
    test_settings: Settings,
) -> None:
    with pytest.raises(UnknownCollectionError, match="unknown collection: missing"):
        resolve_routable_collection_scope(test_settings, "missing")


def test_run_ask_routes_retrieval_to_other_collection_storage(
    test_settings: Settings,
) -> None:
    default_id = test_settings.product.collection
    other_id = "other-collection"

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "default-doc", "DEFAULT_MARKER_abc 默认集合内容")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "other-doc", "OTHER_MARKER_xyz 其他集合内容")

    registered = register_collection(test_settings, other_id)
    other_result = run_ask(
        "OTHER_MARKER_xyz 在哪里",
        registered,
        collection_id=other_id,
    )
    default_result = run_ask(
        "DEFAULT_MARKER_abc 在哪里",
        registered,
        collection_id=default_id,
    )

    assert other_result.citations
    assert other_result.citations[0].document_id == "other-doc"
    assert default_result.citations
    assert default_result.citations[0].document_id == "default-doc"


def test_primary_ask_mcp_helper_explicit_collection_id_routes_to_scoped_storage(
    test_settings: Settings,
) -> None:
    from lab_knowledge.mcp.ask import ask_lab_knowledge

    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "surface-other-doc", "SURFACE_MARKER_qwe 表面路由")

    result = ask_lab_knowledge(
        "SURFACE_MARKER_qwe 在哪里",
        register_collection(test_settings, other_id),
        collection_id=other_id,
    )

    assert result["citations"]
    assert result["citations"][0]["document_id"] == "surface-other-doc"


def test_legacy_ask_mcp_helper_explicit_collection_id_routes_to_scoped_storage(
    test_settings: Settings,
) -> None:
    from lab_knowledge.mcp.ask import ask_wenmai

    other_id = "other-collection"
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "surface-other-doc", "SURFACE_MARKER_qwe 表面路由")

    result = ask_wenmai(
        "SURFACE_MARKER_qwe 在哪里",
        register_collection(test_settings, other_id),
        collection_id=other_id,
    )

    assert result["citations"]
    assert result["citations"][0]["document_id"] == "surface-other-doc"


def test_ask_answer_unknown_collection_still_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    with pytest.raises(ValueError, match="unknown collection"):
        ask_answer("问题", test_settings, collection_id="missing", knowledge=knowledge)
