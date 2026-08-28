"""知识库 interface：写入后稠密稀疏可查，删则两边和配图一起没。"""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.storage.document_images import find_image_ids


def _chunk(chunk_id: str, document_id: str, text: str, culture_domain: str = "") -> Chunk:
    metadata: dict[str, str] = {
        "document_id": document_id,
        "title": document_id,
    }
    if culture_domain:
        metadata["culture_domain"] = culture_domain
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata=metadata,
    )


def test_upsert_makes_dense_and_sparse_searchable(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    result = knowledge.upsert(
        [
            _chunk("doc-a:0000", "doc-a", "船政学堂创办于马尾，是近代海军摇篮。"),
            _chunk("doc-b:0000", "doc-b", "湄洲祖庙是妈祖信仰的中心。"),
        ]
    )
    assert result.chunk_count == 2
    assert result.embed_dimension > 0

    dense = knowledge.dense_search("船政学堂在哪里", top_k=2)
    dense_ids = {item.chunk.chunk_id for item in dense}
    assert "doc-a:0000" in dense_ids
    assert all(item.chunk.text for item in dense)

    sparse = knowledge.sparse_search("船政学堂", top_k=2)
    assert sparse
    assert sparse[0].chunk.chunk_id == "doc-a:0000"
    assert sparse[0].chunk.text == "船政学堂创办于马尾，是近代海军摇篮。"
    assert sparse[0].score > 0


def test_sparse_ranks_more_relevant_chunk_higher(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    knowledge.upsert(
        [
            _chunk("doc-a:0000", "doc-a", "船政学堂船政学堂是近代海军摇篮。"),
            _chunk("doc-b:0000", "doc-b", "船政学堂曾在马尾设立分校，后来迁往别处。"),
        ]
    )
    hits = knowledge.sparse_search("船政学堂", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "doc-a:0000"
    assert hits[0].score > hits[1].score


def test_delete_document_removes_search_hits_and_images(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    knowledge.upsert(
        [
            _chunk("doc-a:0000", "doc-a", "船政学堂简介"),
            _chunk("doc-b:0000", "doc-b", "湄洲祖庙简介"),
        ]
    )
    images = knowledge.images
    placeholder = images.attach(
        document_id="doc-a",
        source_path="/tmp/a.md",
        page=1,
        image_bytes=b"\x89PNG\r\n\x1a\nnot-a-real-png",
        mime_type="image/png",
    )
    assert "[IMAGE:" in placeholder

    knowledge.delete_document("doc-a")

    assert knowledge.get_by_document_id("doc-a") == []
    remaining = knowledge.sparse_search("船政学堂", top_k=5)
    assert all(item.chunk.document_id != "doc-a" for item in remaining)
    temple = knowledge.sparse_search("湄洲祖庙", top_k=1)
    assert temple and temple[0].chunk.document_id == "doc-b"
    image_id = find_image_ids(placeholder)[0]
    assert not images.exists(image_id)


def test_plan_skips_unchanged_and_commit_rebuilds(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    source = "/tmp/doc.md"
    first = knowledge.plan_document(
        source_path=source, sha256="aaa", document_id="aaa"
    )
    assert first.status == "ingested"
    knowledge.commit_document(
        source_path=source,
        sha256="aaa",
        document_id="aaa",
        status="ingested",
        chunks=[_chunk("aaa:0000", "aaa", "旧版内容")],
    )

    skipped = knowledge.plan_document(
        source_path=source, sha256="aaa", document_id="aaa"
    )
    assert skipped.status == "skipped"
    assert knowledge.get_by_document_id("aaa")

    rebuilt = knowledge.plan_document(
        source_path=source, sha256="bbb", document_id="bbb"
    )
    assert rebuilt.status == "rebuilt"
    assert rebuilt.previous_document_id == "aaa"
    # plan is read-only: old doc still present until commit
    assert knowledge.get_by_document_id("aaa")
    knowledge.commit_document(
        source_path=source,
        sha256="bbb",
        document_id="bbb",
        status="rebuilt",
        chunks=[_chunk("bbb:0000", "bbb", "新版内容")],
        previous_document_id=rebuilt.previous_document_id,
    )
    assert knowledge.get_by_document_id("aaa") == []
    assert knowledge.get_by_document_id("bbb")


def test_delete_clears_fingerprint_so_reingest_is_not_skipped(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    source = "/tmp/clear-fp.md"
    knowledge.commit_document(
        source_path=source,
        sha256="doc1",
        document_id="doc1",
        status="ingested",
        chunks=[_chunk("doc1:0000", "doc1", "可删内容")],
    )
    knowledge.delete_document("doc1")
    assert knowledge.get_by_document_id("doc1") == []

    again = knowledge.plan_document(
        source_path=source, sha256="doc1", document_id="doc1"
    )
    assert again.status == "ingested"


def test_culture_domain_filter_hides_other_domain(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    knowledge.upsert(
        [
            _chunk("doc-haisi:0000", "doc-haisi", "通商口岸是海丝贸易节点。", "海丝"),
            _chunk("doc-ship:0000", "doc-ship", "通商口岸支撑了船政物资进口。", "船政"),
        ]
    )
    hits = knowledge.sparse_search("通商口岸", top_k=5, culture_domain="海丝")
    ids = {item.chunk.chunk_id for item in hits}
    assert "doc-haisi:0000" in ids
    assert "doc-ship:0000" not in ids
