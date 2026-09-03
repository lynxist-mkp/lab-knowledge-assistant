"""入库质量门 seam: reject / approve / gray bands before load."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.ingestion.quality import evaluate_quality_gate
from wenmai.knowledge import create_knowledge
from wenmai.pipelines.ingestion import ingest_source
from wenmai.retrieval import resolve_retrieval_mode, run_fusion
from wenmai.tracing import get_ingestion_detail


def _run_retrieval(question, settings, *, knowledge=None, extra_queries=None, retrieval_mode=None):
    from wenmai.knowledge import create_knowledge
    from wenmai.retrieval import resolve_retrieval_mode, run_fusion
    knowledge = knowledge or create_knowledge(settings)
    mode, _ = resolve_retrieval_mode(settings, retrieval_mode, None)
    return run_fusion(
        knowledge,
        question,
        mode=mode,
        settings=settings,
        extra_queries=extra_queries or [],
    )



def _write_reject_markdown(path: Path) -> Path:
    body = "@#@$%^&*()!~`" * 80
    path.write_text(
        f"""---
title: 符号噪声材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_approve_markdown(path: Path) -> Path:
    body = "福建船政文化历史悠久，马尾船政学堂培养近代海军人才，见证闽派近代化历程。" * 8
    path.write_text(
        f"""---
title: 船政文化材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_gray_markdown(path: Path, test_settings: Settings) -> Path:
    prose = "福" * 70
    junk = "@" * 30
    body = prose + junk
    path.write_text(
        f"""---
title: 灰区船政材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    gate = evaluate_quality_gate(path.read_text(encoding="utf-8"), test_settings.quality_gate)
    assert 0.60 <= gate.ratio <= 0.80, gate.ratio
    assert gate.band == "gray"
    return path


def test_scanned_peek_low_ratio_routes_to_gray_not_approve(
    test_settings: Settings,
) -> None:
    gate = evaluate_quality_gate(
        "@#@$%^&*()!~`" * 80,
        test_settings.quality_gate,
        defer_reject=True,
    )
    assert gate.band == "gray"
    assert gate.ratio < test_settings.quality_gate.reject_below


def test_quality_gate_rejects_low_ratio_without_writing_knowledge(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_reject_markdown(tmp_path / "reject.md")
    gate = evaluate_quality_gate(source.read_text(encoding="utf-8"), test_settings.quality_gate)
    assert gate.band == "reject"

    result = ingest_source(source, test_settings)
    assert result.status == "rejected"
    assert result.chunk_count == 0

    knowledge = create_knowledge(test_settings)
    assert not knowledge.get_by_document_id(result.document_id)
    assert not knowledge.dense_search("船政", top_k=5)
    assert not knowledge.sparse_search("船政", top_k=5)

    detail = get_ingestion_detail(test_settings, result.trace_id)
    assert detail is not None
    labels = [step.label for step in detail.steps]
    assert labels[0] == "入库质量门"
    assert "读取" not in labels
    quality = detail.steps[0]
    assert quality.output.startswith("ratio=")
    assert "band=reject" in quality.output
    assert quality.error


def test_quality_gate_approves_normal_prose(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_approve_markdown(tmp_path / "approve.md")
    gate = evaluate_quality_gate(source.read_text(encoding="utf-8"), test_settings.quality_gate)
    assert gate.band == "approve"

    result = ingest_source(source, test_settings)
    assert result.status == "ingested"
    assert result.chunk_count >= 1

    chunks = create_knowledge(test_settings).get_by_document_id(result.document_id)
    assert chunks
    assert all(chunk.metadata.get("审阅状态") == "已通过" for chunk in chunks)

    hits = _run_retrieval("船政学堂", test_settings)
    assert hits.chunks
    assert "船政" in hits.chunks[0].chunk.text


def test_quality_gate_gray_marks_pending_and_filters_retrieval(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_gray_markdown(tmp_path / "gray.md", test_settings)
    result = ingest_source(source, test_settings)
    assert result.status == "ingested"
    assert result.chunk_count >= 1

    knowledge = create_knowledge(test_settings)
    chunks = knowledge.get_by_document_id(result.document_id)
    assert chunks
    assert all(chunk.metadata.get("审阅状态") == "待审" for chunk in chunks)

    pending = _run_retrieval("福", test_settings, knowledge=knowledge)
    assert not pending.chunks

    groups = knowledge.browse_by_culture_domain()
    all_docs = [doc for group in groups for doc in group.documents]
    doc = next(item for item in all_docs if item.document_id == result.document_id)
    assert doc.chunks[0].review_status == "待审"


def test_ingest_http_returns_rejected_status(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_reject_markdown(tmp_path / "reject_http.md")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["chunk_count"] == 0


def test_ingestion_trace_lists_quality_gate_before_load(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_approve_markdown(tmp_path / "trace.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    trace_id = ingest.json()["trace_id"]

    detail = client.get(f"/api/traces/{trace_id}")
    assert detail.status_code == 200
    payload = detail.json()
    labels = [step["label"] for step in payload["steps"]]
    assert labels[0] == "入库质量门"
    assert labels[1] == "读取"

    raw = get_ingestion_detail(test_settings, trace_id)
    assert raw is not None
    quality = raw.steps[0]
    assert "band=approve" in quality.output
