from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from wenmai.components.model_guard import in_batch, release_all_resources
from wenmai.config import Settings
from wenmai.eval import list_eval_runs, run_eval
from wenmai.knowledge.document_card import DocumentNotFoundError, build_document_card
from wenmai.ops.overview import build_overview_stats
from wenmai.pipelines.ingestion import ingest_source
from wenmai.pipelines.query import QueryGenerationError, ask_question
from wenmai.runtime import create_runtime
from wenmai.storage.catalog import DocumentCatalog
from wenmai.tracing import (
    get_trace_detail,
    get_trace_summary,
    list_ingestion_summaries,
    list_query_summaries,
    list_trace_degradations,
)
from wenmai.tracing.latency import query_latency_percentiles
from wenmai.tracing.store import average_query_latency_ms


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    culture_domain: str | None = None
    retrieval_mode: Literal["rrf", "dense_only", "sparse_only"] | None = None
    rerank_enabled: bool | None = None


def _templates_dir(settings: Settings) -> Path:
    templates = Path(settings.server.templates)
    if not templates.is_absolute():
        templates = settings.root / templates
    return templates


_idle_lock = threading.RLock()
_in_flight = 0
_idle_timer: threading.Timer | None = None


def _cancel_idle_timer() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _on_idle_timer_fire() -> None:
    global _idle_timer
    with _idle_lock:
        _idle_timer = None
        if _in_flight > 0:
            return
    if in_batch():
        return
    release_all_resources()


def _schedule_idle_timer(timeout_seconds: float) -> None:
    global _idle_timer
    with _idle_lock:
        _cancel_idle_timer()
        _idle_timer = threading.Timer(timeout_seconds, _on_idle_timer_fire)
        _idle_timer.daemon = True
        _idle_timer.start()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    with _idle_lock:
        _cancel_idle_timer()
    release_all_resources()


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime = create_runtime(settings)
    resolved = runtime.settings
    app = FastAPI(title=resolved.product.name, lifespan=_lifespan)
    app.state.settings = resolved
    app.state.knowledge = runtime.knowledge
    templates = Jinja2Templates(directory=str(_templates_dir(resolved)))

    if resolved.resources.process_idle_unload:

        @app.middleware("http")
        async def process_idle_unload_middleware(
            request: Request, call_next
        ):  # type: ignore[no-untyped-def]
            global _in_flight
            with _idle_lock:
                _cancel_idle_timer()
                _in_flight += 1
            try:
                return await call_next(request)
            finally:
                with _idle_lock:
                    _in_flight -= 1
                    if _in_flight == 0:
                        _schedule_idle_timer(
                            resolved.resources.process_idle_timeout_seconds
                        )

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, object]:
        path = Path(request.source_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")
        try:
            result = ingest_source(
                path,
                resolved,
                pdf_load_mode=request.pdf_load_mode,
                knowledge=app.state.knowledge,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.as_dict()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        try:
            result = ask_question(
                request.question,
                resolved,
                culture_domain=request.culture_domain,
                retrieval_mode=request.retrieval_mode,
                rerank_enabled=request.rerank_enabled,
                knowledge=app.state.knowledge,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except QueryGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "trace_id": exc.trace_id},
            ) from exc
        return result.as_dict()

    @app.get("/api/stats/overview")
    def api_overview_stats() -> dict[str, object]:
        catalog = DocumentCatalog.from_settings(resolved)
        latency = query_latency_percentiles(
            resolved,
            recent_n=resolved.observability.query_latency_recent_n,
        )
        total = latency.get("total") or {}
        return build_overview_stats(
            resolved,
            catalog,
            avg_query_latency_ms=average_query_latency_ms(resolved),
            query_latency_p50_ms=total.get("p50"),
            query_latency_p95_ms=total.get("p95"),
            stage_latency=latency,
        ).as_dict()

    @app.get("/api/browse")
    def api_browse() -> list[dict[str, object]]:
        return [
            group.as_dict()
            for group in app.state.knowledge.browse_by_culture_domain()
        ]

    @app.get("/api/review/pending")
    def api_review_pending() -> list[dict[str, object]]:
        return [
            item.as_dict()
            for item in app.state.knowledge.list_pending_review_documents()
        ]

    @app.post("/api/review/{document_id}/approve")
    def api_review_approve(document_id: str) -> dict[str, str]:
        try:
            app.state.knowledge.approve_review(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已通过"}

    @app.post("/api/review/{document_id}/reject")
    def api_review_reject(document_id: str) -> dict[str, str]:
        try:
            app.state.knowledge.reject_review(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return {"document_id": document_id, "审阅状态": "已驳回"}

    @app.get("/api/chunks/{chunk_id}")
    def api_chunk_detail(chunk_id: str) -> dict[str, object]:
        detail = app.state.knowledge.chunk_detail(chunk_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return detail

    @app.get("/api/documents/{document_id}")
    def api_document_card(document_id: str) -> dict[str, object]:
        try:
            return build_document_card(app.state.knowledge, document_id).as_dict()
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/traces/ingestion")
    def api_ingestion_traces() -> list[dict[str, object]]:
        return [item.as_dict() for item in list_ingestion_summaries(resolved)]

    @app.get("/api/traces/query")
    def api_query_traces() -> list[dict[str, object]]:
        return [item.as_dict() for item in list_query_summaries(resolved)]

    @app.get("/api/traces/{trace_id}")
    def api_trace_detail(trace_id: str) -> dict[str, object]:
        detail = get_trace_detail(resolved, trace_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return detail.as_dict()

    @app.get("/api/traces/{trace_id}/summary")
    def api_trace_summary(trace_id: str) -> dict[str, object]:
        summary = get_trace_summary(resolved, trace_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return summary.as_dict()

    @app.get("/api/traces/{trace_id}/degradations")
    def api_trace_degradations(trace_id: str) -> list[dict[str, str]]:
        degradations = list_trace_degradations(resolved, trace_id)
        if degradations is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return [item.as_dict() for item in degradations]

    @app.post("/api/ingestion/run")
    async def api_ingestion_run(request: IngestRequest) -> StreamingResponse:
        path = Path(request.source_path)
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
                        resolved,
                        pdf_load_mode=request.pdf_load_mode,
                        on_stage=on_stage,
                        knowledge=app.state.knowledge,
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

    @app.get("/", response_class=HTMLResponse)
    def workbench_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "workbench.html", {})

    @app.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "ops.html", {})

    @app.get("/api/eval/runs")
    def api_eval_runs() -> list[dict[str, object]]:
        return [run.as_dict() for run in list_eval_runs(resolved)]

    @app.post("/api/eval/runs")
    def api_post_eval_runs() -> dict[str, object]:
        return run_eval(resolved).as_dict()

    return app


def app() -> FastAPI:
    return create_app()
