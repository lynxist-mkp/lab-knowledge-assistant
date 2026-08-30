from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge.browse import OverviewStats
from wenmai.storage.catalog import DocumentCatalog


def build_overview_stats(
    settings: Settings,
    catalog: DocumentCatalog,
    *,
    avg_query_latency_ms: float | None = None,
    query_latency_p50_ms: float | None = None,
    query_latency_p95_ms: float | None = None,
    stage_latency: dict[str, dict[str, float | None]] | None = None,
) -> OverviewStats:
    _ = settings
    return OverviewStats(
        document_count=catalog.document_count,
        chunk_count=catalog.chunk_count,
        avg_query_latency_ms=avg_query_latency_ms,
        query_latency_p50_ms=query_latency_p50_ms,
        query_latency_p95_ms=query_latency_p95_ms,
        stage_latency=stage_latency,
    )
