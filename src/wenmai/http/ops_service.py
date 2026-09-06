from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge.document_management import (
    DocumentManagement,
    create_document_management,
)
from wenmai.ops.observation import (
    ObservationHealthSnapshot,
    TaskInvestigationView,
    TaskProgressDetail,
    TaskProgressSummary,
    TraceDetail,
    TraceSummary,
    get_task_investigation,
    get_task_progress_detail,
    get_trace_detail,
    get_trace_summary,
    list_ingestion_summaries,
    list_query_summaries,
    list_task_progress_summaries,
    list_trace_degradations,
    load_health_snapshot,
    load_overview_stats,
)
from wenmai.tracing.ingestion_views import StageDegradation


class OpsService:
    """Service-layer seam for 运维看板 HTTP routes."""

    def __init__(
        self,
        settings: Settings,
        *,
        document_management: DocumentManagement | None = None,
    ) -> None:
        self._settings = settings
        self._document_management = document_management or create_document_management(
            settings
        )

    def overview_stats(self, *, collection_id: str | None = None):
        return load_overview_stats(self._settings, collection_id=collection_id)

    def health_snapshot(
        self,
        *,
        task_type: str | None = None,
        failure_kind: str | None = None,
        collection_id: str | None = None,
    ) -> ObservationHealthSnapshot:
        return load_health_snapshot(
            self._settings,
            task_type=task_type,
            failure_kind=failure_kind,
            collection_id=collection_id,
        )

    def browse_groups(self, *, collection_id: str | None = None):
        scoped = self._document_management.for_collection(collection_id)
        return scoped.browse_groups()

    def list_pending_reviews(self, *, collection_id: str | None = None):
        scoped = self._document_management.for_collection(collection_id)
        return scoped.list_pending_reviews()

    def approve_review(
        self, document_id: str, *, collection_id: str | None = None
    ) -> None:
        scoped = self._document_management.for_collection(collection_id)
        scoped.approve_review(document_id)

    def reject_review(self, document_id: str, *, collection_id: str | None = None) -> None:
        scoped = self._document_management.for_collection(collection_id)
        scoped.reject_review(document_id)

    def ingestion_traces(
        self,
        *,
        collection_id: str | None = None,
    ) -> list[TraceSummary]:
        return list_ingestion_summaries(self._settings, collection_id=collection_id)

    def query_traces(
        self,
        *,
        collection_id: str | None = None,
    ) -> list[TraceSummary]:
        return list_query_summaries(self._settings, collection_id=collection_id)

    def task_progress(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
        failure_kind: str | None = None,
        degraded: bool | None = None,
        has_trace: bool | None = None,
        config_fingerprint: str | None = None,
        needs_attention: bool | None = None,
        collection_id: str | None = None,
    ) -> list[TaskProgressSummary]:
        return list_task_progress_summaries(
            self._settings,
            task_type=task_type,
            status=status,
            failure_kind=failure_kind,
            degraded=degraded,
            has_trace=has_trace,
            config_fingerprint=config_fingerprint,
            needs_attention=needs_attention,
            collection_id=collection_id,
        )

    def task_progress_detail(
        self,
        task_id: str,
        *,
        collection_id: str | None = None,
    ) -> TaskProgressDetail | None:
        return get_task_progress_detail(
            self._settings,
            task_id,
            collection_id=collection_id,
        )

    def task_progress_investigation(
        self,
        task_id: str,
        *,
        collection_id: str | None = None,
    ) -> TaskInvestigationView | None:
        return get_task_investigation(
            self._settings,
            task_id,
            collection_id=collection_id,
        )

    def trace_detail(
        self,
        trace_id: str,
        *,
        collection_id: str | None = None,
    ) -> TraceDetail | None:
        return get_trace_detail(
            self._settings,
            trace_id,
            collection_id=collection_id,
        )

    def trace_summary(
        self,
        trace_id: str,
        *,
        collection_id: str | None = None,
    ) -> TraceSummary | None:
        return get_trace_summary(
            self._settings,
            trace_id,
            collection_id=collection_id,
        )

    def trace_degradations(
        self,
        trace_id: str,
        *,
        collection_id: str | None = None,
    ) -> list[StageDegradation] | None:
        return list_trace_degradations(
            self._settings,
            trace_id,
            collection_id=collection_id,
        )


def create_ops_service(
    settings: Settings,
    *,
    document_management: DocumentManagement | None = None,
) -> OpsService:
    return OpsService(settings, document_management=document_management)


__all__ = ["OpsService", "create_ops_service"]
