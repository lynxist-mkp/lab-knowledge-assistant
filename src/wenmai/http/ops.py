from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from wenmai.knowledge.document_card import DocumentNotFoundError
from wenmai.ops.observation import (
    get_task_progress_detail,
    get_trace_detail,
    get_trace_summary,
    list_ingestion_summaries,
    list_query_summaries,
    list_task_progress_summaries,
    list_trace_degradations,
    load_overview_stats,
)


def create_ops_router() -> APIRouter:
    router = APIRouter()

    @router.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request) -> HTMLResponse:
        templates = request.app.state.templates
        return templates.TemplateResponse(request, "ops.html", {})

    @router.get("/api/stats/overview")
    def api_overview_stats(request: Request) -> dict[str, object]:
        return load_overview_stats(request.app.state.settings).as_dict()

    @router.get("/api/browse")
    def api_browse(request: Request) -> list[dict[str, object]]:
        return [
            group.as_dict()
            for group in request.app.state.knowledge.browse_by_culture_domain()
        ]

    @router.get("/api/review/pending")
    def api_review_pending(request: Request) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in request.app.state.knowledge.list_pending_review_documents()
        ]

    @router.post("/api/review/{document_id}/approve")
    def api_review_approve(request: Request, document_id: str) -> dict[str, str]:
        try:
            request.app.state.knowledge.approve_review(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已通过"}

    @router.post("/api/review/{document_id}/reject")
    def api_review_reject(request: Request, document_id: str) -> dict[str, str]:
        try:
            request.app.state.knowledge.reject_review(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已驳回"}

    @router.get("/api/traces/ingestion")
    def api_ingestion_traces(request: Request) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in list_ingestion_summaries(request.app.state.settings)
        ]

    @router.get("/api/traces/query")
    def api_query_traces(request: Request) -> list[dict[str, object]]:
        return [
            item.as_dict() for item in list_query_summaries(request.app.state.settings)
        ]

    @router.get("/api/tasks/progress")
    def api_task_progress(
        request: Request,
        task_type: str | None = None,
        status: str | None = None,
        failure_kind: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in list_task_progress_summaries(
                request.app.state.settings,
                task_type=task_type,
                status=status,
                failure_kind=failure_kind,
            )
        ]

    @router.get("/api/tasks/progress/{task_id}")
    def api_task_progress_detail(request: Request, task_id: str) -> dict[str, object]:
        detail = get_task_progress_detail(request.app.state.settings, task_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="task progress not found")
        return detail.as_dict()

    @router.get("/api/traces/{trace_id}")
    def api_trace_detail(request: Request, trace_id: str) -> dict[str, object]:
        detail = get_trace_detail(request.app.state.settings, trace_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return detail.as_dict()

    @router.get("/api/traces/{trace_id}/summary")
    def api_trace_summary(request: Request, trace_id: str) -> dict[str, object]:
        summary = get_trace_summary(request.app.state.settings, trace_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return summary.as_dict()

    @router.get("/api/traces/{trace_id}/degradations")
    def api_trace_degradations(
        request: Request, trace_id: str
    ) -> list[dict[str, str]]:
        degradations = list_trace_degradations(request.app.state.settings, trace_id)
        if degradations is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return [item.as_dict() for item in degradations]

    return router
