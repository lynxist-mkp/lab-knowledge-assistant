from __future__ import annotations

import functools
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from wenmai.knowledge.collections import UnknownCollectionError
from wenmai.knowledge.document_card import DocumentNotFoundError


def _translate_unknown_collection(func: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except UnknownCollectionError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc

    return wrapper


def create_ops_router() -> APIRouter:
    router = APIRouter()

    @router.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request) -> HTMLResponse:
        templates = request.app.state.templates
        return templates.TemplateResponse(request, "ops.html", {})

    @router.get("/api/stats/overview")
    @_translate_unknown_collection
    def api_overview_stats(
        request: Request,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        return request.app.state.ops_service.overview_stats(
            collection_id=collection_id
        ).as_dict()

    @router.get("/api/stats/health")
    def api_health_snapshot(
        request: Request,
        task_type: str | None = None,
        failure_kind: str | None = None,
    ) -> dict[str, object]:
        return request.app.state.ops_service.health_snapshot(
            task_type=task_type,
            failure_kind=failure_kind,
        ).as_dict()

    @router.get("/api/browse")
    @_translate_unknown_collection
    def api_browse(
        request: Request,
        collection_id: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            group.as_dict()
            for group in request.app.state.ops_service.browse_groups(
                collection_id=collection_id
            )
        ]

    @router.get("/api/review/pending")
    @_translate_unknown_collection
    def api_review_pending(
        request: Request,
        collection_id: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in request.app.state.ops_service.list_pending_reviews(
                collection_id=collection_id
            )
        ]

    @router.post("/api/review/{document_id}/approve")
    @_translate_unknown_collection
    def api_review_approve(
        request: Request,
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, str]:
        try:
            request.app.state.ops_service.approve_review(
                document_id,
                collection_id=collection_id,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已通过"}

    @router.post("/api/review/{document_id}/reject")
    @_translate_unknown_collection
    def api_review_reject(
        request: Request,
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, str]:
        try:
            request.app.state.ops_service.reject_review(
                document_id,
                collection_id=collection_id,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已驳回"}

    @router.get("/api/traces/ingestion")
    @_translate_unknown_collection
    def api_ingestion_traces(
        request: Request,
        collection_id: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in request.app.state.ops_service.ingestion_traces(
                collection_id=collection_id
            )
        ]

    @router.get("/api/traces/query")
    @_translate_unknown_collection
    def api_query_traces(
        request: Request,
        collection_id: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in request.app.state.ops_service.query_traces(
                collection_id=collection_id
            )
        ]

    @router.get("/api/tasks/progress")
    def api_task_progress(
        request: Request,
        task_type: str | None = None,
        status: str | None = None,
        failure_kind: str | None = None,
        degraded: bool | None = None,
        has_trace: bool | None = None,
        config_fingerprint: str | None = None,
        needs_attention: bool | None = None,
    ) -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in request.app.state.ops_service.task_progress(
                task_type=task_type,
                status=status,
                failure_kind=failure_kind,
                degraded=degraded,
                has_trace=has_trace,
                config_fingerprint=config_fingerprint,
                needs_attention=needs_attention,
            )
        ]

    @router.get("/api/tasks/progress/{task_id}")
    def api_task_progress_detail(request: Request, task_id: str) -> dict[str, object]:
        detail = request.app.state.ops_service.task_progress_detail(task_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="task progress not found")
        return detail.as_dict()

    @router.get("/api/tasks/progress/{task_id}/investigation")
    @_translate_unknown_collection
    def api_task_progress_investigation(
        request: Request,
        task_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        detail = request.app.state.ops_service.task_progress_investigation(
            task_id,
            collection_id=collection_id,
        )
        if detail is None:
            raise HTTPException(status_code=404, detail="task progress not found")
        return detail.as_dict()

    @router.get("/api/traces/{trace_id}")
    @_translate_unknown_collection
    def api_trace_detail(
        request: Request,
        trace_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        detail = request.app.state.ops_service.trace_detail(
            trace_id,
            collection_id=collection_id,
        )
        if detail is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return detail.as_dict()

    @router.get("/api/traces/{trace_id}/summary")
    @_translate_unknown_collection
    def api_trace_summary(
        request: Request,
        trace_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        summary = request.app.state.ops_service.trace_summary(
            trace_id,
            collection_id=collection_id,
        )
        if summary is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return summary.as_dict()

    @router.get("/api/traces/{trace_id}/degradations")
    @_translate_unknown_collection
    def api_trace_degradations(
        request: Request,
        trace_id: str,
        collection_id: str | None = None,
    ) -> list[dict[str, str]]:
        degradations = request.app.state.ops_service.trace_degradations(
            trace_id,
            collection_id=collection_id,
        )
        if degradations is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return [item.as_dict() for item in degradations]

    return router
