"""HTTP composition: surface routers mount with unchanged paths."""

from __future__ import annotations

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.http import create_ops_router, create_workbench_router


def test_surface_routes_remain_mounted(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))

    assert client.get("/").status_code == 200
    assert client.get("/ops").status_code == 200
    assert client.get("/api/stats/overview").status_code == 200
    assert client.get("/api/eval/runs").status_code == 200
    assert client.post("/ingest", json={"source_path": "/missing.md"}).status_code == 404


def test_ops_routes_are_not_mounted_on_workbench_router() -> None:
    workbench_paths = {route.path for route in create_workbench_router().routes}
    ops_paths = {route.path for route in create_ops_router().routes}

    assert "/api/browse" in ops_paths
    assert "/api/review/pending" in ops_paths
    assert "/api/review/{document_id}/approve" in ops_paths
    assert "/api/review/{document_id}/reject" in ops_paths

    assert "/api/browse" not in workbench_paths
    assert "/api/review/pending" not in workbench_paths
    assert "/api/review/{document_id}/approve" not in workbench_paths
    assert "/api/review/{document_id}/reject" not in workbench_paths
