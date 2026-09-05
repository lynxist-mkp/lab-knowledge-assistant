from __future__ import annotations

from fastapi import APIRouter, Request

from wenmai.eval import list_eval_run_summaries, run_eval


def create_eval_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/eval/runs")
    def api_eval_runs(request: Request) -> list[dict[str, object]]:
        return [run.as_dict() for run in list_eval_run_summaries(request.app.state.settings)]

    @router.post("/api/eval/runs")
    def api_post_eval_runs(request: Request) -> dict[str, object]:
        return run_eval(request.app.state.settings).as_dict()

    return router
