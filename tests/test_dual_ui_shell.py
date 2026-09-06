"""HTTP seam: dual-surface shells and retired root HTML paths (#3)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from lab_knowledge.app import create_app
from lab_knowledge.config import Settings

_OLD_HTML_PATHS = (
    "/browse",
    "/ingestion",
    "/ingestion/traces",
    "/ingestion/traces/example-trace-id",
    "/query/traces",
    "/query/traces/example-trace-id",
    "/eval",
)


def test_workbench_and_ops_shells_render(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))

    workbench = client.get("/")
    assert workbench.status_code == 200
    assert "课题组知识助手" in workbench.text
    assert "检索工作台" in workbench.text
    assert "--color-primary" in workbench.text
    assert "#037AFF" in workbench.text
    assert "Ingestion 管理" not in workbench.text
    assert "评估面板" not in workbench.text
    assert "数据浏览" not in workbench.text

    ops = client.get("/ops")
    assert ops.status_code == 200
    assert "课题组知识助手" in ops.text
    assert "运维看板" in ops.text
    assert "--color-primary" in ops.text
    assert "#037AFF" in ops.text
    assert "Ingestion 管理" not in ops.text
    assert "评估面板" not in ops.text


def test_old_root_html_paths_are_gone(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    for path in _OLD_HTML_PATHS:
        response = client.get(path)
        assert response.status_code == 404, path
