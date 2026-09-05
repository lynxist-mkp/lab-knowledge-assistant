"""Contract tests for collection scope, document facade, MCP envelope, and image refs."""

from __future__ import annotations

import base64

import pytest

from wenmai.config import Settings
from wenmai.knowledge import create_document_management, create_knowledge
from wenmai.knowledge.collections import CollectionReadModel, UnknownCollectionError
from wenmai.knowledge.domain import REVIEW_PENDING
from wenmai.knowledge.image_refs import ImageReferenceService
from wenmai.knowledge.store import Knowledge
from wenmai.mcp.envelope import refs_from_citations, scope_for
from wenmai.mcp.tools.ask import ask_answer
from wenmai.mcp.tools.collections import collections_get_stats, collections_list
from wenmai.mcp.tools.documents import documents_delete, documents_get, documents_list
from wenmai.mcp.tools.images import images_get_content, images_get_ref
from wenmai.mcp.tools.reviews import (
    reviews_approve,
    reviews_list_pending,
    reviews_reject,
)
from wenmai.models import Chunk, Citation
from wenmai.storage.images import ImageStore


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


def test_unknown_collection_id_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)
    with pytest.raises(UnknownCollectionError):
        model.get_stats("not-a-real-collection")


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


def test_mcp_documents_get_envelope(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-card", "卡片内容")
    mgmt = create_document_management(test_settings, knowledge=knowledge)

    result = documents_get(
        mgmt,
        "doc-card",
        collection_id=test_settings.product.collection,
    )
    assert set(result) == _envelope_keys()
    assert result["data"]["document_id"] == "doc-card"
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


def test_ask_answer_unknown_collection_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    with pytest.raises(ValueError, match="unknown collection"):
        ask_answer("问题", test_settings, collection_id="missing", knowledge=knowledge)


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
