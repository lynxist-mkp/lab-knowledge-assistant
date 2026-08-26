"""Pure metrics seam: Hit@5, MRR, refusal accuracy vs GoldItem."""

from __future__ import annotations

from wenmai.eval.golden import GoldItem
from wenmai.eval.metrics import (
    hit_at_5,
    mean_reciprocal_rank,
    refusal_accuracy,
    unique_corpus_doc_ids,
)


def _gold(
    id: str,
    evidence: list[str],
    answerable: bool = True,
    category: str = "单跳事实",
) -> GoldItem:
    return GoldItem(
        id=id,
        question="q",
        evidence_doc_ids=evidence,
        answerable=answerable,
        reference_answer="a",
        category=category if answerable else "明确不可答",
    )


def test_hit_at_5_when_evidence_in_top_five() -> None:
    item = _gold("g1", ["doc-a"])
    ranked = ["doc-b", "doc-a", "doc-c", "doc-d", "doc-e"]

    assert hit_at_5(ranked, item) == 1.0


def test_hit_at_5_when_evidence_beyond_top_five() -> None:
    item = _gold("g1", ["doc-f"])
    ranked = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]

    assert hit_at_5(ranked, item) == 0.0


def test_hit_at_5_skips_unanswerable_items() -> None:
    item = _gold("g1", [], answerable=False)
    ranked = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]

    assert hit_at_5(ranked, item) is None


def test_mrr_at_first_rank() -> None:
    item = _gold("g1", ["doc-a"])
    ranked = ["doc-a", "doc-b", "doc-c"]

    assert mean_reciprocal_rank(ranked, item) == 1.0


def test_mrr_at_third_rank() -> None:
    item = _gold("g1", ["doc-c"])
    ranked = ["doc-a", "doc-b", "doc-c"]

    assert mean_reciprocal_rank(ranked, item) == 1 / 3


def test_mrr_zero_when_not_found() -> None:
    item = _gold("g1", ["doc-z"])
    ranked = ["doc-a", "doc-b", "doc-c"]

    assert mean_reciprocal_rank(ranked, item) == 0.0


def test_mrr_zero_when_evidence_beyond_top_five() -> None:
    item = _gold("g1", ["doc-f"])
    ranked = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e", "doc-f"]

    assert mean_reciprocal_rank(ranked, item) == 0.0


def test_mrr_uses_best_rank_when_multiple_evidence_ids() -> None:
    item = _gold("g1", ["doc-b", "doc-d"])
    ranked = ["doc-a", "doc-b", "doc-c", "doc-d"]

    assert mean_reciprocal_rank(ranked, item) == 0.5


def test_refusal_accuracy_counts_unanswerable_only() -> None:
    items = [
        _gold("g1", [], answerable=False),
        _gold("g2", [], answerable=False),
        _gold("g3", ["doc-a"], answerable=True),
    ]
    refused_flags = [True, False, False]

    assert refusal_accuracy(items, refused_flags) == 0.5


def test_unique_corpus_doc_ids_preserves_rank_and_dedupes() -> None:
    ranked = ["a", "a", "b", "c", "b", "d", "e", "f"]

    assert unique_corpus_doc_ids(ranked, top_k=5) == ["a", "b", "c", "d", "e"]
