from __future__ import annotations

import functools
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request

from lab_knowledge.eval import list_eval_run_summaries, run_eval
from lab_knowledge.knowledge.collections import (
    UnknownCollectionError,
    resolve_routable_collection_scope,
)
from lab_knowledge.knowledge.store import create_knowledge


def _translate_unknown_collection(func: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except UnknownCollectionError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc

    return wrapper


def create_eval_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/eval/runs")
    @_translate_unknown_collection
    def api_eval_runs(
        request: Request,
        collection_id: str | None = None,
    ) -> list[dict[str, object]]:
        settings = request.app.state.settings
        scope = resolve_routable_collection_scope(settings, collection_id)
        return [
            run.as_dict()
            for run in list_eval_run_summaries(
                settings,
                collection_id=scope.collection_id,
            )
        ]

    @router.post("/api/eval/runs")
    @_translate_unknown_collection
    def api_post_eval_runs(
        request: Request,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        settings = request.app.state.settings
        scope = resolve_routable_collection_scope(settings, collection_id)
        knowledge = request.app.state.knowledge
        if scope.settings.product.collection != settings.product.collection:
            knowledge = create_knowledge(scope.settings)
        return run_eval(scope.settings, knowledge=knowledge).as_dict()

    return router
