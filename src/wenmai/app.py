from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.pipelines.ingestion import ingest_markdown
from wenmai.pipelines.query import QueryGenerationError
from wenmai.services.ask import ask as ask_service
from wenmai.services.browse import browse_by_culture_domain, get_chunk_detail
from wenmai.services.stats import get_overview_stats


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


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
            result = ask_service(request.question, resolved)
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
            "placeholder.html",
            {"active_page": "ingestion", "page_title": "Ingestion 管理"},
        )

    @app.get("/ingestion/traces", response_class=HTMLResponse)
    def ingestion_traces_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "placeholder.html",
            {"active_page": "ingestion_traces", "page_title": "Ingestion 追踪"},
        )

    @app.get("/query/traces", response_class=HTMLResponse)
    def query_traces_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "placeholder.html",
            {"active_page": "query_traces", "page_title": "Query 追踪"},
        )

    @app.get("/eval", response_class=HTMLResponse)
    def eval_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "placeholder.html",
            {"active_page": "eval", "page_title": "评估面板"},
        )

    return app


def app() -> FastAPI:
    return create_app()
