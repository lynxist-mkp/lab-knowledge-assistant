"""Ablation group names map to retrieval modes."""

from __future__ import annotations

from wenmai.eval.ablation import AblationSpec, resolve_ablation


def test_resolve_ablation_maps_four_groups() -> None:
    assert resolve_ablation("dense_only") == AblationSpec(mode="dense_only", rerank_enabled=False)
    assert resolve_ablation("sparse_only") == AblationSpec(mode="sparse_only", rerank_enabled=False)
    assert resolve_ablation("rrf") == AblationSpec(mode="rrf", rerank_enabled=False)
    assert resolve_ablation("rrf_rerank") == AblationSpec(mode="rrf", rerank_enabled=True)
