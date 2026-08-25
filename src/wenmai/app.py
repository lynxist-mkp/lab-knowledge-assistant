from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.pipelines.ingestion import ingest_markdown


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


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

    return app


def app() -> FastAPI:
    return create_app()
