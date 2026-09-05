from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from wenmai.http.schemas import IngestRequest
from wenmai.pipelines.ingestion import ingest_source


def create_ingest_router() -> APIRouter:
    router = APIRouter()

    @router.post("/ingest")
    def ingest(request: Request, body: IngestRequest) -> dict[str, object]:
        settings = request.app.state.settings
        path = Path(body.source_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")
        try:
            result = ingest_source(
                path,
                settings,
                pdf_load_mode=body.pdf_load_mode,
                knowledge=request.app.state.knowledge,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.as_dict()

    @router.post("/api/ingestion/run")
    async def api_ingestion_run(
        request: Request, body: IngestRequest
    ) -> StreamingResponse:
        settings = request.app.state.settings
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
                        knowledge=request.app.state.knowledge,
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
