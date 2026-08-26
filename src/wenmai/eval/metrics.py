from __future__ import annotations

from wenmai.eval.golden import GoldItem, corpus_id_from_source_path
from wenmai.models import ScoredChunk


def corpus_doc_ids_from_chunks(scored_chunks: list[ScoredChunk]) -> list[str]:
    """Map retrieved chunks to manifest corpus ids via source_path metadata."""
    ids: list[str] = []
    for item in scored_chunks:
        source_path = item.chunk.metadata.get("source_path")
        if source_path:
            ids.append(corpus_id_from_source_path(str(source_path)))
    return ids


def unique_corpus_doc_ids(ranked_doc_ids: list[str], top_k: int = 5) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in ranked_doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(doc_id)
        if len(result) >= top_k:
            break
    return result


def hit_at_5(ranked_doc_ids: list[str], item: GoldItem) -> float | None:
    """1.0 if any evidence doc appears in top-5 ranked ids; None if not answerable."""
    if not item.answerable:
        return None
    top = unique_corpus_doc_ids(ranked_doc_ids, top_k=5)
    if not item.evidence_doc_ids:
        return 0.0
    return 1.0 if any(doc_id in top for doc_id in item.evidence_doc_ids) else 0.0


def mean_reciprocal_rank(ranked_doc_ids: list[str], item: GoldItem) -> float | None:
    """Reciprocal rank of the first matching evidence doc within top-5; None if not answerable."""
    if not item.answerable:
        return None
    if not item.evidence_doc_ids:
        return 0.0
    top = unique_corpus_doc_ids(ranked_doc_ids, top_k=5)
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in item.evidence_doc_ids:
            return 1.0 / rank
    return 0.0


def refusal_accuracy(items: list[GoldItem], refused_flags: list[bool]) -> float:
    """Fraction of unanswerable items correctly refused."""
    if len(items) != len(refused_flags):
        raise ValueError("items and refused_flags length mismatch")
    unanswerable = [(item, refused) for item, refused in zip(items, refused_flags) if not item.answerable]
    if not unanswerable:
        return 1.0
    correct = sum(1 for item, refused in unanswerable if refused)
    return correct / len(unanswerable)


def aggregate_hit_at_5(items: list[GoldItem], ranked_per_item: list[list[str]]) -> float:
    scores = [
        score
        for item, ranked in zip(items, ranked_per_item)
        for score in [hit_at_5(ranked, item)]
        if score is not None
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_mrr(items: list[GoldItem], ranked_per_item: list[list[str]]) -> float:
    scores = [
        score
        for item, ranked in zip(items, ranked_per_item)
        for score in [mean_reciprocal_rank(ranked, item)]
        if score is not None
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
