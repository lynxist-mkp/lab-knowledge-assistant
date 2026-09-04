"""运维观测：概览统计的深 module。"""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge.browse import OverviewStats
from wenmai.storage.catalog import DocumentCatalog
from wenmai.tracing.latency import query_latency_percentiles
from wenmai.tracing.store import average_query_latency_ms


def load_overview_stats(settings: Settings) -> OverviewStats:
    """Settings → OverviewStats：目录计数 + 查询延迟分位（含 stage_latency）。"""
    catalog = DocumentCatalog.from_settings(settings)
    latency = query_latency_percentiles(
        settings,
        recent_n=settings.observability.query_latency_recent_n,
    )
    total = latency.get("total") or {}
    return OverviewStats(
        document_count=catalog.document_count,
        chunk_count=catalog.chunk_count,
        avg_query_latency_ms=average_query_latency_ms(settings),
        query_latency_p50_ms=total.get("p50"),
        query_latency_p95_ms=total.get("p95"),
        stage_latency=latency,
    )


__all__ = ["load_overview_stats"]
