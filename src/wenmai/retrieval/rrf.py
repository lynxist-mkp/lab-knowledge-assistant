"""Reciprocal rank fusion (Elastic formula).

Formula: score += 1.0 / (k + rank), rank starts at 1.
See: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_chunk_ids: list[list[str]],
    *,
    k: int,
    top_k: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_chunk_ids:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]
