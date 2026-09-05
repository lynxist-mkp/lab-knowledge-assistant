from __future__ import annotations

import time
from dataclasses import dataclass

from wenmai.config import Settings
from wenmai.generation import QueryGenerationError
from wenmai.knowledge.store import Knowledge
from wenmai.models import AskResult


@dataclass(frozen=True)
class AskSurfaceResult:
    result: AskResult
    elapsed_ms: float


class AskSurfaceError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_surface(
    question: str,
    settings: Settings,
    *,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
    entrypoint: str = "surface",
) -> AskSurfaceResult:
    """Shared outward ask seam above run_ask for MCP and HTTP."""
    from wenmai.http.ask_service import run_ask

    started = time.perf_counter()
    try:
        result = run_ask(
            question,
            settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            entrypoint=entrypoint,
        )
    except QueryGenerationError as exc:
        raise AskSurfaceError(str(exc), exc.trace_id) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return AskSurfaceResult(result=result, elapsed_ms=elapsed_ms)


__all__ = ["AskSurfaceError", "AskSurfaceResult", "ask_surface"]
