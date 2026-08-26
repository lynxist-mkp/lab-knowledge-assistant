from __future__ import annotations

from dataclasses import asdict, dataclass

from wenmai.config import Settings
from wenmai.factories import vector_store as vector_store_factory
from wenmai.services.traces import average_query_latency_ms


@dataclass(frozen=True)
class OverviewStats:
    document_count: int
    chunk_count: int
    avg_query_latency_ms: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def get_overview_stats(settings: Settings) -> OverviewStats:
    store = vector_store_factory.create(settings)
    chunks = store.list_all()
    document_ids = {chunk.document_id for chunk in chunks if chunk.document_id}
    return OverviewStats(
        document_count=len(document_ids),
        chunk_count=len(chunks),
        avg_query_latency_ms=average_query_latency_ms(settings),
    )
