from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from wenmai.components.model_guard import in_batch, release_all_resources
from wenmai.config import Settings
from wenmai.http import (
    create_eval_router,
    create_ingest_router,
    create_ops_router,
    create_workbench_router,
)
from wenmai.http.ops_service import create_ops_service
from wenmai.knowledge.document_management import create_document_management
from wenmai.runtime import create_runtime


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
    app.state.document_management = create_document_management(
        resolved,
        knowledge=runtime.knowledge,
    )
    app.state.ops_service = create_ops_service(
        resolved,
        document_management=app.state.document_management,
    )
    app.state.templates = Jinja2Templates(directory=str(_templates_dir(resolved)))

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

    app.include_router(create_workbench_router())
    app.include_router(create_ops_router())
    app.include_router(create_eval_router())
    app.include_router(create_ingest_router())

    return app


def app() -> FastAPI:
    return create_app()
