"""Ablation config → retrieval mode mapping and artifact shape."""

from __future__ import annotations

from wenmai.eval.ablation import (
    AblationSpec,
    apply_ablation,
    config_snapshot,
    group_metrics_payload,
    resolve_ablation,
)


def test_resolve_ablation_maps_four_groups() -> None:
    assert resolve_ablation("dense_only") == AblationSpec(mode="dense_only", rerank_enabled=False)
    assert resolve_ablation("sparse_only") == AblationSpec(mode="sparse_only", rerank_enabled=False)
    assert resolve_ablation("rrf") == AblationSpec(mode="rrf", rerank_enabled=False)
    assert resolve_ablation("rrf_rerank") == AblationSpec(mode="rrf", rerank_enabled=True)


def test_apply_ablation_overrides_retrieval_fields(test_settings) -> None:
    overridden = apply_ablation(test_settings, "rrf")
    assert overridden.retrieval.mode == "rrf"
    assert overridden.retrieval.rerank_enabled is False

    rerank = apply_ablation(test_settings, "rrf_rerank")
    assert rerank.retrieval.mode == "rrf"
    assert rerank.retrieval.rerank_enabled is True


def test_config_snapshot_includes_group_and_retrieval(test_settings) -> None:
    snapshot = config_snapshot(test_settings, "dense_only")
    assert snapshot["ablation_group"] == "dense_only"
    assert snapshot["retrieval"]["mode"] == "dense_only"
    assert snapshot["retrieval"]["rerank_enabled"] is False


def test_group_metrics_payload_shape() -> None:
    payload = group_metrics_payload(0.8, 0.65, 1.0, 30, 12)
    assert payload == {
        "hit_at_5": 0.8,
        "mrr": 0.65,
        "refusal_accuracy": 1.0,
        "answerable_count": 30,
        "unanswerable_count": 12,
    }
