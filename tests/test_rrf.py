"""RRF narrow seam + default hybrid query trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.components.retrieval.rrf import reciprocal_rank_fusion
from wenmai.config import Settings


def test_rrf_rank_starts_at_one() -> None:
    fused = reciprocal_rank_fusion([["only"]], k=60, top_k=1)
    assert fused == [("only", 1.0 / (60 + 1))]


def test_rrf_k_affects_score() -> None:
    at_60 = reciprocal_rank_fusion([["a"]], k=60, top_k=1)
    at_10 = reciprocal_rank_fusion([["a"]], k=10, top_k=1)
    assert at_60[0][1] == pytest.approx(1.0 / 61)
    assert at_10[0][1] == pytest.approx(1.0 / 11)
    assert at_60[0][1] != at_10[0][1]


def test_rrf_single_path_absence() -> None:
    fused = reciprocal_rank_fusion([["dense-only"], []], k=60, top_k=1)
    assert fused == [("dense-only", 1.0 / 61)]

    fused_sparse = reciprocal_rank_fusion([[], ["sparse-only"]], k=60, top_k=1)
    assert fused_sparse == [("sparse-only", 1.0 / 61)]


def test_rrf_accumulates_when_both_paths_hit() -> None:
    fused = reciprocal_rank_fusion(
        [["shared", "dense-only"], ["shared", "sparse-only"]],
        k=60,
        top_k=3,
    )
    scores = dict(fused)
    assert scores["shared"] == pytest.approx(2.0 / 61)
    assert scores["dense-only"] == pytest.approx(1.0 / 62)
    assert scores["sparse-only"] == pytest.approx(1.0 / 62)
    assert fused[0][0] == "shared"


def test_rrf_ask_records_dense_sparse_and_fusion_stages(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = tmp_path / "mazu.md"
    source.write_text(
        """---
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
""",
        encoding="utf-8",
    )
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])
    assert [stage["name"] for stage in query_trace["stages"]] == [
        "query_processing",
        "dense",
        "sparse",
        "fusion",
        "generation",
    ]

    fusion_stage = next(stage for stage in query_trace["stages"] if stage["name"] == "fusion")
    assert fusion_stage["method"] == "rrf"
    assert fusion_stage["dense_candidates"]
    assert fusion_stage["sparse_candidates"]
    assert fusion_stage["candidates"]
    assert len(fusion_stage["candidates"]) <= test_settings.retrieval.fused_k
