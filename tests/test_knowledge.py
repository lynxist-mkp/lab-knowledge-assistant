"""知识库 interface：写入后稠密稀疏可查，删则两边和配图一起没。"""

from __future__ import annotations

import pytest

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.ops.observation import get_overview_stats
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


def _overview(knowledge) -> object:
    return get_overview_stats(knowledge._settings)


def _commit(
    knowledge,
    document_id: str,
    text: str,
    *,
    source_path: str | None = None,
    sha256: str | None = None,
    chunk_id: str | None = None,
    culture_domain: str = "",
    status: str = "ingested",
    previous_document_id: str | None = None,
):
    return knowledge.commit_document(
        source_path=source_path or f"/tmp/{document_id}.md",
        sha256=sha256 or document_id,
        document_id=document_id,
        status=status,
        chunks=[
            _chunk(
                chunk_id or f"{document_id}:0000",
                document_id,
                text,
                culture_domain,
            )
        ],
        previous_document_id=previous_document_id,
    )


def test_commit_makes_dense_and_sparse_searchable(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    result_a = _commit(knowledge, "doc-a", "船政学堂创办于马尾，是近代海军摇篮。")
    _commit(knowledge, "doc-b", "湄洲祖庙是妈祖信仰的中心。")
    assert result_a.chunk_count == 1
    assert result_a.embed_dimension > 0

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
    _commit(knowledge, "doc-a", "船政学堂船政学堂是近代海军摇篮。")
    _commit(knowledge, "doc-b", "船政学堂曾在马尾设立分校，后来迁往别处。")
    hits = knowledge.sparse_search("船政学堂", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "doc-a:0000"
    assert hits[0].score > hits[1].score


def test_delete_document_removes_search_hits_and_images(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂简介")
    _commit(knowledge, "doc-b", "湄洲祖庙简介")
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
    _commit(knowledge, "doc-haisi", "通商口岸是海丝贸易节点。", culture_domain="海丝")
    _commit(knowledge, "doc-ship", "通商口岸支撑了船政物资进口。", culture_domain="船政")
    hits = knowledge.sparse_search("通商口岸", top_k=5, culture_domain="海丝")
    ids = {item.chunk.chunk_id for item in hits}
    assert "doc-haisi:0000" in ids
    assert "doc-ship:0000" not in ids


def test_catalog_updates_on_commit_and_delete(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂简介", culture_domain="船政")
    _commit(knowledge, "doc-b", "湄洲祖庙简介", culture_domain="妈祖")

    overview = _overview(knowledge)
    assert overview.document_count == 2
    assert overview.chunk_count == 2

    groups = knowledge.browse_by_culture_domain()
    by_domain = {group.culture_domain: group for group in groups}
    assert by_domain["船政"].document_count == 1
    assert by_domain["妈祖"].chunk_count == 1

    knowledge.delete_document("doc-a")
    overview = _overview(knowledge)
    assert overview.document_count == 1
    assert overview.chunk_count == 1


def test_browse_uses_catalog_not_list_all(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂简介", culture_domain="船政")

    def fail_list_all() -> list[Chunk]:
        raise AssertionError("browse should not scan list_all()")

    monkeypatch.setattr(knowledge._store, "list_all", fail_list_all)

    assert _overview(knowledge).document_count == 1
    assert knowledge.browse_by_culture_domain()


def test_commit_rolls_back_dense_on_sparse_write_failure(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    doc_id = "rollback-doc"
    source = "/tmp/rollback.md"

    def fail_bm25_upsert(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated sparse write failure")

    monkeypatch.setattr(knowledge._bm25, "upsert", fail_bm25_upsert)

    with pytest.raises(RuntimeError, match="simulated sparse write failure"):
        knowledge.commit_document(
            source_path=source,
            sha256=doc_id,
            document_id=doc_id,
            status="ingested",
            chunks=[_chunk("rollback-doc:0000", doc_id, "会回滚的内容")],
        )

    assert knowledge.get_by_document_id(doc_id) == []
    assert _overview(knowledge).document_count == 0
    sparse = knowledge.sparse_search("回滚", top_k=5)
    assert all(item.chunk.document_id != doc_id for item in sparse)
    plan = knowledge.plan_document(
        source_path=source, sha256=doc_id, document_id=doc_id
    )
    assert plan.status == "ingested"


def test_commit_rolls_back_on_sparse_save_failure(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    doc_id = "save-fail-doc"
    source = "/tmp/save-fail.md"

    def fail_save() -> None:
        raise RuntimeError("simulated sparse save failure")

    monkeypatch.setattr(knowledge._bm25, "save", fail_save)

    with pytest.raises(RuntimeError, match="simulated sparse save failure"):
        knowledge.commit_document(
            source_path=source,
            sha256=doc_id,
            document_id=doc_id,
            status="ingested",
            chunks=[_chunk("save-fail-doc:0000", doc_id, "保存失败应回滚")],
        )

    assert knowledge.get_by_document_id(doc_id) == []
    assert _overview(knowledge).document_count == 0
    sparse = knowledge.sparse_search("回滚", top_k=5)
    assert all(item.chunk.document_id != doc_id for item in sparse)


def test_commit_rebuild_failure_preserves_previous_document(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    source = "/tmp/rebuild-fail.md"
    _commit(knowledge, "old-doc", "旧版保留", source_path=source, sha256="old-sha")

    def fail_after_upsert(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated fingerprint failure")

    monkeypatch.setattr(knowledge._fingerprints, "upsert", fail_after_upsert)

    with pytest.raises(RuntimeError, match="simulated fingerprint failure"):
        knowledge.commit_document(
            source_path=source,
            sha256="new-sha",
            document_id="new-doc",
            status="rebuilt",
            chunks=[_chunk("new-doc:0000", "new-doc", "新版未生效")],
            previous_document_id="old-doc",
        )

    assert knowledge.get_by_document_id("old-doc")
    assert knowledge.get_by_document_id("new-doc") == []
    plan = knowledge.plan_document(
        source_path=source, sha256="old-sha", document_id="old-doc"
    )
    assert plan.status == "skipped"


def test_commit_rebuild_delete_failure_restores_fingerprint(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    source = "/tmp/rebuild-delete-fail.md"
    _commit(knowledge, "old-doc", "旧版保留", source_path=source, sha256="old-sha")

    def fail_delete(_document_id: str) -> None:
        raise RuntimeError("simulated previous delete failure")

    monkeypatch.setattr(knowledge._write, "delete_document", fail_delete)

    with pytest.raises(RuntimeError, match="simulated previous delete failure"):
        knowledge.commit_document(
            source_path=source,
            sha256="new-sha",
            document_id="new-doc",
            status="rebuilt",
            chunks=[_chunk("new-doc:0000", "new-doc", "新版未生效")],
            previous_document_id="old-doc",
        )

    assert knowledge.get_by_document_id("old-doc")
    assert knowledge.get_by_document_id("new-doc") == []
    plan = knowledge.plan_document(
        source_path=source, sha256="old-sha", document_id="old-doc"
    )
    assert plan.status == "skipped"


def test_set_review_status_delegates_to_write_path(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-review", "审阅状态委托测试片段")

    called: dict[str, str] = {}

    def fake_update_review_status(document_id: str, status: str) -> None:
        called["document_id"] = document_id
        called["status"] = status

    monkeypatch.setattr(
        knowledge._write, "update_review_status", fake_update_review_status
    )

    knowledge.set_review_status("doc-review", "待审")

    assert called == {"document_id": "doc-review", "status": "待审"}


def test_pending_chunks_excluded_from_dense_and_sparse_search(
    test_settings: Settings,
) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-pending", "待审片段独有术语船政密档")
    _commit(knowledge, "doc-approved", "已通过片段船政学堂简介")

    knowledge.set_review_status("doc-pending", "待审")

    dense = knowledge.dense_search("船政密档", top_k=5)
    dense_ids = {item.chunk.chunk_id for item in dense}
    assert "doc-pending:0000" not in dense_ids

    sparse = knowledge.sparse_search("船政密档", top_k=5)
    sparse_ids = {item.chunk.chunk_id for item in sparse}
    assert "doc-pending:0000" not in sparse_ids


def test_approved_after_pending_becomes_searchable(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-flip", "翻转后可检索的船政密档内容")
    knowledge.set_review_status("doc-flip", "待审")
    assert not knowledge.sparse_search("船政密档", top_k=5)

    knowledge.set_review_status("doc-flip", "已通过")
    hits = knowledge.sparse_search("船政密档", top_k=5)
    assert hits and hits[0].chunk.document_id == "doc-flip"


def test_chunks_without_review_status_remain_searchable(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    knowledge.commit_document(
        source_path="/tmp/legacy.md",
        sha256="legacy",
        document_id="legacy",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="legacy:0000",
                document_id="legacy",
                text="旧数据无审阅状态字段",
                metadata={"document_id": "legacy", "title": "legacy"},
            )
        ],
    )
    hits = knowledge.sparse_search("旧数据", top_k=5)
    assert hits and hits[0].chunk.chunk_id == "legacy:0000"


def test_browse_lists_pending_chunks_with_review_status(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-pending", "待审库览片段", culture_domain="船政")
    knowledge.set_review_status("doc-pending", "待审")

    groups = knowledge.browse_by_culture_domain()
    by_domain = {group.culture_domain: group for group in groups}
    doc = by_domain["船政"].documents[0]
    assert doc.chunk_count == 1
    assert doc.chunks[0].review_status == "待审"
    assert doc.chunks[0].as_dict()["审阅状态"] == "待审"
