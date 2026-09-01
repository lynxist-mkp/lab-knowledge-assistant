"""Phase B batch: manifest ingest, ragas summary, bad-case stubs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from wenmai.config import Settings
from wenmai.eval import run_eval
from wenmai.eval.corpus_ingest import ingest_corpus_manifest, load_corpus_manifest
from wenmai.eval.phase_b import (
    DEFAULT_BAD_CASES_PATH,
    _resolve_bad_cases_path,
    append_bad_case_stubs,
    find_hit_at_5_misses,
    format_metrics_summary,
    run_phase_b_batch,
)
from wenmai.eval.golden import GoldItem
from wenmai.eval.ragas_metrics import compute_group_ragas_metrics, should_run_ragas
from wenmai.eval.views import parse_eval_run
from wenmai.generation import GenerationResult
from wenmai.models import Chunk, ScoredChunk


def _write_minpai_markdown(path: Path, title: str, body: str, domain: str = "妈祖") -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/{path.stem}
culture_domain: {domain}
space: minpai_culture
title: {title}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path, item_ids: list[str]) -> Path:
    payload = {
        "version": 1,
        "items": [{"id": item_id, "title": item_id} for item_id in item_ids],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_load_corpus_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.yaml", ["a", "b"])
    items = load_corpus_manifest(manifest)
    assert [item["id"] for item in items] == ["a", "b"]


def test_ingest_corpus_manifest_skips_missing_and_ingests_existing(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.yaml", ["present", "missing"])
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    _write_minpai_markdown(
        items_dir / "present.md",
        "在库文档",
        "闽派文化测试正文，用于批量入库脚本验证。",
    )

    result = ingest_corpus_manifest(test_settings, manifest, items_dir)

    assert result.total == 2
    assert result.attempted == 1
    assert result.missing == 1
    assert result.ingested == 1
    assert result.success is True


def test_ingest_corpus_manifest_idempotent_skip(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.yaml", ["doc-a"])
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    _write_minpai_markdown(items_dir / "doc-a.md", "文档 A", "第一段闽派文化内容。")

    first = ingest_corpus_manifest(test_settings, manifest, items_dir)
    second = ingest_corpus_manifest(test_settings, manifest, items_dir)

    assert first.ingested == 1
    assert second.skipped == 1
    assert second.ingested == 0


def test_ingest_corpus_manifest_dry_run_does_not_ingest(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.yaml", ["doc-a"])
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    _write_minpai_markdown(items_dir / "doc-a.md", "文档 A", "闽派文化 dry-run 测试。")

    result = ingest_corpus_manifest(
        test_settings,
        manifest,
        items_dir,
        dry_run=True,
    )

    assert result.attempted == 1
    assert result.ingested == 0
    assert result.skipped == 0


def test_ingest_corpus_manifest_exit_semantics_all_failed(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.yaml", ["bad"])
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    _write_minpai_markdown(items_dir / "bad.md", "坏文档", "x")

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ingest failed")

    monkeypatch.setattr("wenmai.eval.corpus_ingest.prepare_ingest_source", boom)

    result = ingest_corpus_manifest(test_settings, manifest, items_dir)

    assert result.attempted == 1
    assert result.failed == 1
    assert result.success is False


def test_should_run_ragas_without_key(
    test_settings: Settings,
    without_ragas_judge_key: None,
) -> None:
    assert should_run_ragas(test_settings, None) is False
    assert should_run_ragas(test_settings, False) is False
    assert should_run_ragas(test_settings, True) is True


def test_compute_group_ragas_metrics_unavailable_without_key(
    test_settings: Settings,
    without_ragas_judge_key: None,
) -> None:
    item = GoldItem(
        id="g1",
        question="问题？",
        evidence_doc_ids=["doc"],
        answerable=True,
        reference_answer="答案",
        category="单跳事实",
    )
    generation = GenerationResult(
        answer="回答",
        refused=False,
        citations=[],
        provider_name="fake",
        output_summary="",
        candidate_count=1,
    )
    chunks = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id="c1",
                document_id="d1",
                text="上下文",
                metadata={"source_path": "/tmp/doc.md"},
            ),
            score=1.0,
        )
    ]

    payload = compute_group_ragas_metrics(
        [item],
        {"g1": generation},
        {"g1": chunks},
        test_settings,
    )

    assert payload["status"] == "unavailable"
    assert "reason" in payload


def test_run_eval_attaches_ragas_with_fake_evaluator(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["matsu-intro"],'
        '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}\n',
        encoding="utf-8",
    )
    source = _write_minpai_markdown(
        tmp_path / "matsu-intro.md",
        "湄洲妈祖祖庙简介",
        "湄洲岛是妈祖信仰的发源地。",
    )
    test_settings.evaluation.golden_set = str(golden)
    test_settings.evaluation.runs = str(tmp_path / "runs")
    test_settings.providers.evaluator = "fake"
    test_settings.fakes = dict(test_settings.fakes or {})
    test_settings.fakes["evaluator"] = "ok"
    monkeypatch.setattr(
        "wenmai.eval.runner.should_run_ragas",
        lambda _settings, _ragas: True,
    )

    from fastapi.testclient import TestClient
    from wenmai.app import create_app

    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    run = run_eval(test_settings, ragas=True)
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    ragas = artifact["groups"]["rrf_rerank"]["metrics"]["ragas"]
    assert ragas["status"] == "ok"
    assert ragas["faithfulness"] == pytest.approx(0.9)
    assert ragas["context_precision"] == pytest.approx(0.8)
    assert parse_eval_run(artifact) is not None


def test_find_hit_at_5_misses_and_bad_case_stub(tmp_path: Path) -> None:
    from wenmai.eval.golden import GoldItem

    items = [
        GoldItem(
            id="hit",
            question="命中题",
            evidence_doc_ids=["doc-a"],
            answerable=True,
            reference_answer="",
            category="单跳事实",
        ),
        GoldItem(
            id="miss",
            question="未命中题",
            evidence_doc_ids=["doc-b"],
            answerable=True,
            reference_answer="",
            category="单跳事实",
        ),
    ]
    artifact = {
        "groups": {
            "rrf_rerank": {
                "items": {
                    "hit": {"retrieval": {"ranked_doc_ids": ["doc-a", "other"]}},
                    "miss": {"retrieval": {"ranked_doc_ids": ["other"]}},
                }
            }
        }
    }

    misses = find_hit_at_5_misses(artifact, items)
    assert len(misses) == 1
    assert misses[0]["item_id"] == "miss"

    bad_cases = tmp_path / "bad-cases.md"
    append_bad_case_stubs(bad_cases, misses, run_timestamp="20260101T000000Z")
    text = bad_cases.read_text(encoding="utf-8")
    assert "未命中题" in text
    assert "trace_id" in text


def test_format_metrics_summary_ragas_unavailable() -> None:
    summary = {
        "metrics": {
            "rrf_rerank": {
                "hit_at_5": 0.5,
                "mrr": 0.4,
                "refusal_accuracy": 0.9,
            }
        },
        "ragas": {"status": "unavailable", "reason": "no key"},
    }
    line = format_metrics_summary(summary)
    assert "Hit@5=0.500" in line
    assert "ragas=unavailable" in line


def test_run_phase_b_batch_skip_ingest(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["matsu-intro"],'
        '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}\n'
        '{"id":"g002","question":"不存在的问题？","evidence_doc_ids":[],'
        '"answerable":false,"reference_answer":"","category":"明确不可答"}\n',
        encoding="utf-8",
    )
    source = _write_minpai_markdown(
        tmp_path / "matsu-intro.md",
        "湄洲妈祖祖庙简介",
        "湄洲岛是妈祖信仰的发源地。",
    )
    test_settings.evaluation.golden_set = str(golden)
    test_settings.evaluation.runs = str(tmp_path / "runs")
    test_settings.providers.evaluator = "fake"
    test_settings.fakes = dict(test_settings.fakes or {})
    test_settings.fakes["evaluator"] = "ok"
    bad_cases = tmp_path / "bad-cases.md"

    from fastapi.testclient import TestClient
    from wenmai.app import create_app

    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    result = run_phase_b_batch(
        test_settings,
        skip_ingest=True,
        bad_cases_path=bad_cases,
        ragas=False,
    )

    assert result.summary_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "B"
    assert "rewrite_compare" in summary["artifacts"]
    assert summary["ragas"]["status"] == "unavailable"


def test_default_bad_cases_path_is_workspace_scratch(test_settings: Settings) -> None:
    resolved = _resolve_bad_cases_path(test_settings, DEFAULT_BAD_CASES_PATH)
    assert resolved == test_settings.root.parent / DEFAULT_BAD_CASES_PATH
    custom = Path("notes/bad.md")
    assert _resolve_bad_cases_path(test_settings, custom) == test_settings.root / custom
