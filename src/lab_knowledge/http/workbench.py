from __future__ import annotations

import functools
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lab_knowledge.generation import QueryGenerationError
from lab_knowledge.http.ask_governor import AskSaturationError
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.http.schemas import AskRequest
from lab_knowledge.knowledge.collections import UnknownCollectionError
from lab_knowledge.knowledge.document_card import DocumentNotFoundError


def _translate_unknown_collection(func: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except UnknownCollectionError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc

    return wrapper


def create_workbench_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def workbench_page(request: Request) -> HTMLResponse:
        templates = request.app.state.templates
        return templates.TemplateResponse(request, "workbench.html", {})

    @router.post("/ask")
    @_translate_unknown_collection
    def ask(
        request: Request,
        body: AskRequest,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        settings = request.app.state.settings
        knowledge = None if collection_id is not None else request.app.state.knowledge
        try:
            result = run_ask(
                body.question,
                settings,
                collection_id=collection_id,
                culture_domain=body.culture_domain,
                retrieval_mode=body.retrieval_mode,
                rerank_enabled=body.rerank_enabled,
                knowledge=knowledge,
                entrypoint="http",
            )
        except AskSaturationError as exc:
            raise HTTPException(status_code=503, detail=exc.as_dict()) from exc
        except UnknownCollectionError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except QueryGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "trace_id": exc.trace_id},
            ) from exc
        return result.as_dict()

    @router.get("/api/chunks/{chunk_id}")
    @_translate_unknown_collection
    def api_chunk_detail(
        request: Request,
        chunk_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        detail = request.app.state.document_management.get_chunk_detail(
            chunk_id,
            collection_id=collection_id,
        )
        if detail is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return detail.as_dict()

    @router.get("/api/documents/{document_id}")
    @_translate_unknown_collection
    def api_document_card(
        request: Request,
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return request.app.state.document_management.get_document(
                document_id,
                collection_id=collection_id,
            ).as_dict()
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
