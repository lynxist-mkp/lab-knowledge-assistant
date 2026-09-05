"""灰区复判 seam: pass / fail / timeout on the gray 入库质量门 band."""

from __future__ import annotations

from pathlib import Path

from wenmai.config import Settings
from wenmai.ingestion.quality import evaluate_quality_gate
from wenmai.knowledge import create_knowledge
from wenmai.ops.observation import get_ingestion_detail
from wenmai.pipelines.ingestion import ingest_source
from wenmai.retrieval import retrieve


def _write_gray_markdown(path: Path, test_settings: Settings, extra: str = "") -> Path:
    prose = "福" * 70
    junk = "@" * 30
    body = prose + extra + junk
    path.write_text(
        f"""---
title: 灰区船政材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    gate = evaluate_quality_gate(
        path.read_text(encoding="utf-8"),
        test_settings.quality_gate,
    )
    assert gate.band == "gray", gate.ratio
    return path


def test_gray_review_pass_stamps_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    source = _write_gray_markdown(tmp_path / "gray-pass.md", test_settings)
    result = ingest_source(source, test_settings)
    assert result.status == "ingested"

    chunks = create_knowledge(test_settings).get_by_document_id(result.document_id)
    assert chunks
    assert all(chunk.metadata.get("审阅状态") == "已通过" for chunk in chunks)
    hits = retrieve("福", test_settings)
    assert hits.chunks

    detail = get_ingestion_detail(test_settings, result.trace_id)
    assert detail is not None
    labels = [step.label for step in detail.steps]
    assert "灰区复判" in labels
    assert "入库质量门" in labels


def test_gray_review_fail_stamps_pending(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    source = _write_gray_markdown(
        tmp_path / "gray-fail.md",
        test_settings,
        extra="不值得入库",
    )
    result = ingest_source(source, test_settings)
    assert result.status == "ingested"

    knowledge = create_knowledge(test_settings)
    chunks = knowledge.get_by_document_id(result.document_id)
    assert chunks
    assert all(chunk.metadata.get("审阅状态") == "待审" for chunk in chunks)
    assert not retrieve("福", test_settings, knowledge=knowledge).chunks


def test_gray_review_timeout_hard_rejects(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    test_settings.quality_gate.timeout_seconds = 1.0
    test_settings.fakes["multimodal"] = "timeout"
    source = _write_gray_markdown(tmp_path / "gray-timeout.md", test_settings)
    result = ingest_source(source, test_settings)
    assert result.status == "rejected"
    assert result.chunk_count == 0
    assert not create_knowledge(test_settings).get_by_document_id(result.document_id)

    detail = get_ingestion_detail(test_settings, result.trace_id)
    assert detail is not None
    gray = next(step for step in detail.steps if step.label == "灰区复判")
    assert gray.error


def test_gray_review_disabled_keeps_pending(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = False
    source = _write_gray_markdown(tmp_path / "gray-off.md", test_settings)
    result = ingest_source(source, test_settings)
    chunks = create_knowledge(test_settings).get_by_document_id(result.document_id)
    assert all(chunk.metadata.get("审阅状态") == "待审" for chunk in chunks)
    detail = get_ingestion_detail(test_settings, result.trace_id)
    assert detail is not None
    assert all(step.label != "灰区复判" for step in detail.steps)
