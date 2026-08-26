from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.pipelines.ingestion import ingest_markdown
from wenmai.pipelines.query import QueryGenerationError, ask_question


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    ensure_providers()
    resolved = settings or Settings.load()
    app = FastAPI(title=resolved.product.name)
    app.state.settings = resolved

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
            result = ask_question(request.question, resolved)
        except QueryGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "trace_id": exc.trace_id},
            ) from exc
        return result.as_dict()

    return app


def app() -> FastAPI:
    return create_app()
