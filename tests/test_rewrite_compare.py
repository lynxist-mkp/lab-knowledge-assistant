"""Rewrite on/off compare eval: wiring and artifact shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wenmai.config import Settings
from wenmai.eval import list_eval_run_summaries, run_eval, run_rewrite_compare
from wenmai.eval.views import parse_eval_run_summary
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import Chunk
from wenmai.retrieval import retrieve


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata={"document_id": document_id, "source_path": f"/tmp/{document_id}.md"},
    )


def _commit(knowledge: Knowledge, document_id: str, text: str) -> None:
    knowledge.commit_document(
        source_path=f"/tmp/{document_id}.md",
        sha256=document_id,
        document_id=document_id,
        status="ingested",
        chunks=[_chunk(f"{document_id}:0000", document_id, text)],
    )


def _configure_rewrite_settings(
    test_settings: Settings,
    tmp_path: Path,
    *,
    rewriter: str = "lexicon",
    multi_query: bool = True,
) -> None:
    test_settings.query_processing.rewriter = rewriter
    test_settings.query_processing.multi_query = multi_query
    test_settings.query_processing.multi_query_n = 3
    test_settings.evaluation.runs = str(tmp_path / "runs")


def _prepare_rewrite_knowledge(test_settings: Settings) -> Knowledge:
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )
    decoys = [
        "武夷岩茶产于福建武夷山，是乌龙茶代表。",
        "土楼是客家民居建筑的重要形式。",
        "泉州开元寺东西塔是宋代石塔。",
        "片仔癀源于漳州，是名贵中成药。",
        "德化白瓷以温润如玉著称。",
        "寿山石雕是福州传统工艺。",
        "脱胎漆器是福州三宝之一。",
        "南音是闽南古老乐种。",
        "梨园戏是福建古老剧种。",
        "高甲戏流行于闽南地区。",
        "木偶戏在泉州传承悠久。",
        "闽剧是福建主要地方剧种。",
        "惠安女服饰独具特色。",
        "妈祖信俗列入世界非遗。",
        "三坊七巷是福州历史文化街区。",
    ]
    for index, text in enumerate(decoys):
        _commit(knowledge, f"decoy-{index:02d}", text)
    return knowledge


def _golden_path(tmp_path: Path) -> Path:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id":"g001","question":"这所院校始于哪一年？","evidence_doc_ids":["ship"],'
        '"answerable":true,"reference_answer":"1866年。","category":"单跳事实"}\n',
        encoding="utf-8",
    )
    return golden


def test_run_eval_default_skips_rewrite_even_when_configured(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _configure_rewrite_settings(test_settings, tmp_path)
    test_settings.evaluation.golden_set = str(_golden_path(tmp_path))
    test_settings.evaluation.ablations = ["sparse_only", "rrf"]
    knowledge = _prepare_rewrite_knowledge(test_settings)

    run = run_eval(test_settings, knowledge=knowledge)

    assert run.groups["sparse_only"].metrics.hit_at_5 == 0.0
    assert run.groups["sparse_only"].metrics.mrr == 0.0
    assert run.groups["rrf"].metrics.hit_at_5 == 0.0
    assert run.groups["rrf"].metrics.mrr == 0.0


def test_run_eval_passes_empty_extra_queries(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    without_ragas_judge_key: None,
) -> None:
    _configure_rewrite_settings(test_settings, tmp_path)
    test_settings.evaluation.golden_set = str(_golden_path(tmp_path))
    test_settings.evaluation.ablations = ["sparse_only"]
    knowledge = _prepare_rewrite_knowledge(test_settings)
    seen: list[list[str] | None] = []
    real_retrieve = retrieve

    def tracking_retrieve(
        question: str,
        settings: Settings,
        *,
        extra_queries: list[str] | None = None,
        **kwargs: object,
    ):
        seen.append(extra_queries)
        return real_retrieve(question, settings, extra_queries=extra_queries, **kwargs)

    monkeypatch.setattr("wenmai.pipelines.query_orchestration.retrieve", tracking_retrieve)
    run_eval(test_settings, knowledge=knowledge)
    assert seen == [[]]


def test_run_rewrite_compare_improves_hit_at_5(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _configure_rewrite_settings(test_settings, tmp_path)
    test_settings.evaluation.golden_set = str(_golden_path(tmp_path))
    knowledge = _prepare_rewrite_knowledge(test_settings)

    run = run_rewrite_compare(test_settings, knowledge=knowledge)

    assert set(run.groups) == {"rewrite_off", "rewrite_on"}
    assert run.groups["rewrite_off"].label == "无改写"
    assert run.groups["rewrite_on"].label == "术语归一+Multi-Query"
    assert run.groups["rewrite_off"].metrics.hit_at_5 == 0.0
    assert run.groups["rewrite_on"].metrics.hit_at_5 == 1.0
    assert run.groups["rewrite_off"].metrics.mrr == 0.0
    assert run.groups["rewrite_on"].metrics.mrr == 1.0


def test_rewrite_compare_artifact_shape(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _configure_rewrite_settings(test_settings, tmp_path)
    test_settings.evaluation.golden_set = str(_golden_path(tmp_path))
    knowledge = _prepare_rewrite_knowledge(test_settings)

    run = run_rewrite_compare(test_settings, knowledge=knowledge)
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["compare"] == "rewrite"
    assert "ablations" not in artifact
    assert set(artifact["groups"]) == {"rewrite_off", "rewrite_on"}
    for name in ("rewrite_off", "rewrite_on"):
        metrics = artifact["groups"][name]["metrics"]
        assert "hit_at_5" in metrics
        assert "mrr" in metrics
        config = artifact["groups"][name]["config"]
        assert config["query_rewrite"] == (name == "rewrite_on")
        assert "rewriter" in config["query_processing"]
        assert "multi_query" in config["query_processing"]
        assert "multi_query_n" in config["query_processing"]

    parsed = parse_eval_run_summary(artifact)
    assert parsed is not None
    listed = list_eval_run_summaries(test_settings)
    assert listed[0].timestamp == run.timestamp
    assert listed[0].groups["rewrite_on"].label == "术语归一+Multi-Query"
