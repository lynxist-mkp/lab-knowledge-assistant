from wenmai.http.eval import create_eval_router
from wenmai.http.ingest import create_ingest_router
from wenmai.http.ops import create_ops_router
from wenmai.http.workbench import create_workbench_router

__all__ = [
    "create_eval_router",
    "create_ingest_router",
    "create_ops_router",
    "create_workbench_router",
]
