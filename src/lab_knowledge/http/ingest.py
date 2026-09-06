from __future__ import annotations

import asyncio
import functools
import json
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from lab_knowledge.config import Settings
from lab_knowledge.http.schemas import IngestRequest
from lab_knowledge.ingestion.source_metadata import LiteratureMetadataOverrides
from lab_knowledge.knowledge.collections import (
    UnknownCollectionError,
    resolve_routable_collection_scope,
)
from lab_knowledge.knowledge.store import Knowledge, create_knowledge
from lab_knowledge.pipelines.ingest_directory import ingest_source_directory
from lab_knowledge.pipelines.ingestion import ingest_source


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


def _literature_overrides(body: IngestRequest) -> LiteratureMetadataOverrides | None:
    if body.literature is None:
        return None
    return LiteratureMetadataOverrides(
        title=body.literature.title,
        authors=body.literature.authors,
        year=body.literature.year,
    )


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
        if not path.exists():
            raise HTTPException(status_code=404, detail="source file not found")
        try:
            if path.is_dir():
                result = ingest_source_directory(
                    path,
                    settings,
                    pdf_load_mode=body.pdf_load_mode,
                    source_kind=body.source_kind,
                    literature=_literature_overrides(body),
                    knowledge=knowledge,
                )
                return result.as_dict()
            if not path.is_file():
                raise HTTPException(status_code=404, detail="source file not found")
            result = ingest_source(
                path,
                settings,
                pdf_load_mode=body.pdf_load_mode,
                source_kind=body.source_kind,
                literature=_literature_overrides(body),
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
        if not path.exists():
            raise HTTPException(status_code=404, detail="source file not found")

        async def event_stream():
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_stage(stage: object) -> None:
                payload = {"event": "stage", "stage": stage.to_dict()}  # type: ignore[union-attr]
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            def on_file_start(index: int, total: int, source_path: Path) -> None:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "event": "file_start",
                        "index": index,
                        "total": total,
                        "source_path": str(source_path),
                    },
                )

            def on_file_done(index: int, total: int, outcome: object) -> None:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "event": "file_done",
                        "index": index,
                        "total": total,
                        "outcome": outcome.as_dict(),  # type: ignore[union-attr]
                    },
                )

            def run_ingest() -> None:
                try:
                    if path.is_dir():
                        result = ingest_source_directory(
                            path,
                            settings,
                            pdf_load_mode=body.pdf_load_mode,
                            source_kind=body.source_kind,
                            literature=_literature_overrides(body),
                            knowledge=knowledge,
                            on_file_start=on_file_start,
                            on_file_done=on_file_done,
                        )
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"event": "done", "result": result.as_dict()},
                        )
                        return
                    if not path.is_file():
                        raise FileNotFoundError("source file not found")
                    result = ingest_source(
                        path,
                        settings,
                        pdf_load_mode=body.pdf_load_mode,
                        source_kind=body.source_kind,
                        literature=_literature_overrides(body),
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
