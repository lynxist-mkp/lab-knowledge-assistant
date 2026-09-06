from __future__ import annotations

import asyncio
import functools
import json
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from wenmai.config import Settings
from wenmai.http.schemas import IngestRequest
from wenmai.knowledge.collections import UnknownCollectionError, resolve_routable_collection_scope
from wenmai.knowledge.store import Knowledge, create_knowledge
from wenmai.pipelines.ingestion import ingest_source


def _translate_unknown_collection(func: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except UnknownCollectionError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc

    return wrapper


def _resolve_ingest_binding(
    settings: Settings,
    default_knowledge: Knowledge,
    collection_id: str | None,
) -> tuple[Settings, Knowledge]:
    scope = resolve_routable_collection_scope(settings, collection_id)
    if scope.settings.product.collection == settings.product.collection:
        return scope.settings, default_knowledge
    return scope.settings, create_knowledge(scope.settings)


def create_ingest_router() -> APIRouter:
    router = APIRouter()

    @router.post("/ingest")
    @_translate_unknown_collection
    def ingest(
        request: Request,
        body: IngestRequest,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        settings, knowledge = _resolve_ingest_binding(
            request.app.state.settings,
            request.app.state.knowledge,
            collection_id,
        )
        path = Path(body.source_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")
        try:
            result = ingest_source(
                path,
                settings,
                pdf_load_mode=body.pdf_load_mode,
                knowledge=knowledge,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.as_dict()

    @router.post("/api/ingestion/run")
    @_translate_unknown_collection
    async def api_ingestion_run(
        request: Request,
        body: IngestRequest,
        collection_id: str | None = None,
    ) -> StreamingResponse:
        settings, knowledge = _resolve_ingest_binding(
            request.app.state.settings,
            request.app.state.knowledge,
            collection_id,
        )
        path = Path(body.source_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")

        async def event_stream():
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_stage(stage: object) -> None:
                payload = {"event": "stage", "stage": stage.to_dict()}  # type: ignore[union-attr]
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            def run_ingest() -> None:
                try:
                    result = ingest_source(
                        path,
                        settings,
                        pdf_load_mode=body.pdf_load_mode,
                        on_stage=on_stage,
                        knowledge=knowledge,
                    )
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "done", "result": result.as_dict()},
                    )
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "error", "message": f"{type(exc).__name__}: {exc}"},
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=run_ingest, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
