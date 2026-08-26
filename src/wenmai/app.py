from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.pipelines.ingestion import ingest_markdown
from wenmai.pipelines.query import QueryGenerationError
from wenmai.services.ask import ask as ask_service
from wenmai.services.browse import browse_by_culture_domain, get_chunk_detail
from wenmai.services.ingestion_traces import (
    list_degradations,
    summarize_ingestion_trace,
)
from wenmai.services.query_traces import (
    get_dense_candidates,
    get_fusion_candidates,
    get_rank_changes,
    get_rerank_candidates,
    get_sparse_candidates,
    list_stage_latencies,
    summarize_query_trace,
)
from wenmai.services.eval_runs import eval_chart_data, get_eval_dashboard, list_eval_runs
from wenmai.services.stats import get_overview_stats
from wenmai.services.traces import get_trace_by_id, read_traces_by_type


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    culture_domain: str | None = None


def _templates_dir(settings: Settings) -> Path:
    templates = Path(settings.server.templates)
    if not templates.is_absolute():
        templates = settings.root / templates
    return templates


def create_app(settings: Settings | None = None) -> FastAPI:
    ensure_providers()
    resolved = settings or Settings.load()
    app = FastAPI(title=resolved.product.name)
    app.state.settings = resolved
    templates = Jinja2Templates(directory=str(_templates_dir(resolved)))

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, object]:
        path = Path(request.source_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")
        result = ingest_markdown(path, resolved, pdf_load_mode=request.pdf_load_mode)
        return result.as_dict()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        try:
            result = ask_service(
                request.question,
                resolved,
                culture_domain=request.culture_domain,
            )
        except QueryGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "trace_id": exc.trace_id},
            ) from exc
        return result.as_dict()

    @app.get("/api/stats/overview")
    def api_overview_stats() -> dict[str, object]:
        return get_overview_stats(resolved).as_dict()

    @app.get("/api/browse")
    def api_browse() -> list[dict[str, object]]:
        return [group.as_dict() for group in browse_by_culture_domain(resolved)]

    @app.get("/api/chunks/{chunk_id}")
    def api_chunk_detail(chunk_id: str) -> dict[str, object]:
        detail = get_chunk_detail(resolved, chunk_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return detail

    @app.get("/api/traces/ingestion")
    def api_ingestion_traces() -> list[dict[str, object]]:
        traces = read_traces_by_type(resolved, "ingestion")
        return [summarize_ingestion_trace(trace).as_dict() for trace in reversed(traces)]

    @app.get("/api/traces/query")
    def api_query_traces() -> list[dict[str, object]]:
        traces = read_traces_by_type(resolved, "query")
        return [summarize_query_trace(trace).as_dict() for trace in reversed(traces)]

    @app.get("/api/traces/{trace_id}")
    def api_trace_detail(trace_id: str) -> dict[str, object]:
        trace = get_trace_by_id(resolved, trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    @app.get("/api/traces/{trace_id}/summary")
    def api_trace_summary(trace_id: str) -> dict[str, object]:
        trace = get_trace_by_id(resolved, trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return summarize_ingestion_trace(trace).as_dict()

    @app.get("/api/traces/{trace_id}/degradations")
    def api_trace_degradations(trace_id: str) -> list[dict[str, str]]:
        trace = get_trace_by_id(resolved, trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return [item.as_dict() for item in list_degradations(trace)]

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
                    result = ingest_markdown(
                        path,
                        resolved,
                        pdf_load_mode=request.pdf_load_mode,
                        on_stage=on_stage,
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
    def overview_page(request: Request) -> HTMLResponse:
        stats = get_overview_stats(resolved)
        return templates.TemplateResponse(
            request,
            "overview.html",
            {
                "active_page": "overview",
                "stats": stats,
            },
        )

    @app.get("/browse", response_class=HTMLResponse)
    def browse_page(request: Request) -> HTMLResponse:
        groups = browse_by_culture_domain(resolved)
        return templates.TemplateResponse(
            request,
            "browse.html",
            {
                "active_page": "browse",
                "groups": groups,
            },
        )

    @app.get("/ingestion", response_class=HTMLResponse)
    def ingestion_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "ingestion.html",
            {"active_page": "ingestion"},
        )

    @app.get("/ingestion/traces", response_class=HTMLResponse)
    def ingestion_traces_page(request: Request) -> HTMLResponse:
        traces = read_traces_by_type(resolved, "ingestion")
        summaries = [summarize_ingestion_trace(trace) for trace in reversed(traces)]
        return templates.TemplateResponse(
            request,
            "ingestion_traces.html",
            {
                "active_page": "ingestion_traces",
                "traces": summaries,
            },
        )

    @app.get("/ingestion/traces/{trace_id}", response_class=HTMLResponse)
    def ingestion_trace_detail_page(request: Request, trace_id: str) -> HTMLResponse:
        trace = get_trace_by_id(resolved, trace_id)
        if trace is None or trace.get("trace_type") != "ingestion":
            raise HTTPException(status_code=404, detail="ingestion trace not found")
        stages = []
        for stage in trace.get("stages") or []:
            payload = dict(stage)
            payload["raw_json"] = json.dumps(stage, ensure_ascii=False, indent=2)
            stages.append(payload)
        trace_view = dict(trace)
        trace_view["stages"] = stages
        return templates.TemplateResponse(
            request,
            "ingestion_trace_detail.html",
            {
                "active_page": "ingestion_traces",
                "trace": trace_view,
                "summary": summarize_ingestion_trace(trace),
                "degradations": list_degradations(trace),
            },
        )

    @app.get("/query/traces", response_class=HTMLResponse)
    def query_traces_page(request: Request, trace_id: str | None = None) -> HTMLResponse:
        traces = read_traces_by_type(resolved, "query")
        summaries = [summarize_query_trace(trace) for trace in reversed(traces)]
        return templates.TemplateResponse(
            request,
            "query_traces.html",
            {
                "active_page": "query_traces",
                "traces": summaries,
                "highlight_trace_id": trace_id,
            },
        )

    @app.get("/query/traces/{trace_id}", response_class=HTMLResponse)
    def query_trace_detail_page(request: Request, trace_id: str) -> HTMLResponse:
        trace = get_trace_by_id(resolved, trace_id)
        if trace is None or trace.get("trace_type") != "query":
            raise HTTPException(status_code=404, detail="query trace not found")
        stages = []
        for stage in trace.get("stages") or []:
            payload = dict(stage)
            payload["raw_json"] = json.dumps(stage, ensure_ascii=False, indent=2)
            stages.append(payload)
        trace_view = dict(trace)
        trace_view["stages"] = stages
        summary = summarize_query_trace(trace)
        rerank_stage = next(
            (stage for stage in trace.get("stages") or [] if stage.get("name") == "rerank"),
            None,
        )
        rerank_fallback_reason = None
        if rerank_stage is not None:
            rerank_fallback_reason = rerank_stage.get("fallback_reason") or rerank_stage.get(
                "error"
            )
        return templates.TemplateResponse(
            request,
            "query_trace_detail.html",
            {
                "active_page": "query_traces",
                "trace": trace_view,
                "summary": summary,
                "stage_latencies": [item.as_dict() for item in list_stage_latencies(trace)],
                "dense_candidates": get_dense_candidates(trace),
                "sparse_candidates": get_sparse_candidates(trace),
                "fusion_candidates": get_fusion_candidates(trace),
                "rerank_candidates": get_rerank_candidates(trace),
                "rank_changes": get_rank_changes(trace),
                "rerank_fallback_reason": rerank_fallback_reason,
            },
        )

    @app.get("/api/eval/runs")
    def api_eval_runs() -> list[dict[str, object]]:
        return [run.as_dict() for run in list_eval_runs(resolved)]

    @app.get("/eval", response_class=HTMLResponse)
    def eval_page(request: Request) -> HTMLResponse:
        dashboard = get_eval_dashboard(resolved)
        return templates.TemplateResponse(
            request,
            "eval.html",
            {
                "active_page": "eval",
                "dashboard": dashboard,
                "chart_data": eval_chart_data(dashboard),
            },
        )

    return app


def app() -> FastAPI:
    return create_app()
