"""Ablation runner writes run artifact with four group metrics."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval.runner import run_ablation_batch


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


def test_run_ablation_batch_writes_four_groups(test_settings: Settings, tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["matsu-intro"],'
            '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}',
            '{"id":"g002","question":"船政学堂是什么时候创办的？","evidence_doc_ids":[],'
            '"answerable":false,"reference_answer":"","category":"明确不可答"}',
        ],
    )
    source = _write_minpai_markdown(
        tmp_path / "matsu-intro.md",
        "湄洲妈祖祖庙简介",
        "湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。",
    )
    test_settings.evaluation.golden_set = str(golden)
    test_settings.evaluation.runs = str(tmp_path / "runs")
    test_settings.evaluation.ablations = ["dense_only", "sparse_only", "rrf", "rrf_rerank"]

    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    output_path = run_ablation_batch(test_settings)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["item_count"] == 2
    assert set(artifact["groups"].keys()) == {
        "dense_only",
        "sparse_only",
        "rrf",
        "rrf_rerank",
    }
    for group_name, group_data in artifact["groups"].items():
        metrics = group_data["metrics"]
        assert "hit_at_5" in metrics
        assert "mrr" in metrics
        assert "refusal_accuracy" in metrics
        assert "citation_coverage" in metrics
        assert metrics["answerable_count"] == 1
        assert metrics["unanswerable_count"] == 1
        config = group_data["config"]
        assert config["ablation_group"] == group_name
