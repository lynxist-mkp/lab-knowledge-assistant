"""HTTP seam: 编辑工作台可提问 (#4) and 出处页内只读抽屉 (#5)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings


def _write_markdown(path: Path, *, culture_domain: str, title: str, body: str) -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/{path.stem}
culture_domain: {culture_domain}
space: minpai_culture
title: {title}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _workbench_html(client: TestClient) -> str:
    response = client.get("/")
    assert response.status_code == 200
    return response.text


def test_workbench_landmarks_and_culture_domain_filter(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    # Brand and Surface
    assert "福云·文脉助手" in html
    assert "编辑工作台" in html
    assert "#037AFF" in html

    # Top bar 文化域 filter control and default options
    assert "文化域" in html
    assert "culture_domain" in html
    assert "culture-domain-select" in html
    assert "全部 / 不限" in html or "全部" in html
    assert "海丝" in html
    assert "朱子" in html
    assert "妈祖" in html
    assert "船政" in html

    # 提问 input affordance and action
    assert "提问" in html
    assert "question-input" in html
    assert "submit-btn" in html

    # Trace affordance (collapsed <details> pattern in card JS template)
    assert "Trace" in html
    assert "trace-details" in html
    assert "<details" in html
    assert "查看 Trace" in html

    # Footer compliance landmark
    assert "人工智能生成合成" in html
    assert "播出终审" in html

    # No ops primary nav chrome
    assert "Ingestion 管理" not in html
    assert "评估面板" not in html
    assert "数据浏览" not in html
    assert "库览" not in html
    assert "运维看板" not in html
    assert "/ops" not in html


def test_workbench_dual_shell_isolation(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))

    workbench = client.get("/")
    assert workbench.status_code == 200
    assert "编辑工作台" in workbench.text
    assert "运维看板" not in workbench.text

    ops = client.get("/ops")
    assert ops.status_code == 200
    assert "运维看板" in ops.text
    assert "编辑工作台" not in ops.text
    assert "人工智能生成合成" in ops.text
    assert "播出终审" in ops.text


def test_workbench_citation_drawer_shell_landmarks(test_settings: Settings) -> None:
    """#5: in-page read-only drawer shell present on SSR workbench page."""
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    # Drawer shell landmarks
    assert "id=\"citation-drawer\"" in html
    assert "role=\"dialog\"" in html
    assert "data-open" in html
    assert "id=\"citation-drawer-backdrop\"" in html
    assert "id=\"citation-drawer-close\"" in html
    assert "id=\"citation-drawer-body\"" in html
    assert "id=\"citation-drawer-text\"" in html
    assert "id=\"citation-drawer-loading\"" in html
    assert "id=\"citation-drawer-error\"" in html
    assert "id=\"citation-drawer-title\"" in html
    assert "只读出处" in html

    # Drawer data path wired in page JS (not ops navigation)
    assert "/api/chunks/" in html

    # Citation open hooks referenced in card JS (rendered after ask)
    assert "btn-view-chunk" in html
    assert "citation-item" in html
    assert "citation-ref" in html
    assert "data-chunk-id" in html

    # No full 库览 ops chrome / browse fetch on workbench
    assert "库览" not in html
    assert "/ops" not in html
    assert "fetch('/api/browse'" not in html
    assert 'fetch("/api/browse"' not in html


def test_workbench_ask_flow_uses_ask_endpoint(test_settings: Settings) -> None:
    """#4: ask submission posts to /ask from workbench page script."""
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    assert "fetch('/ask'" in html or "fetch(\"/ask\"" in html
    assert "ASK_TIMEOUT_MS" in html
    assert "AbortController" in html
    assert "ask-form" in html
    assert "拒答" in html or "已拒答" in html or "refused" in html


def test_workbench_citation_drawer_js_wiring(test_settings: Settings) -> None:
    """#5: workbench page JS wires in-page drawer — not ops / 库览 navigation."""
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    # Open / close drawer functions defined in page script
    assert "openCitationDrawer" in html
    assert "closeCitationDrawer" in html
    assert "async function openCitationDrawer" in html or "function openCitationDrawer" in html
    assert "function closeCitationDrawer" in html

    # Drawer fetch path: /api/chunks/ + encodeURIComponent (not ops or browse)
    assert "/api/chunks/" in html
    assert "encodeURIComponent" in html
    assert "fetch('/api/chunks/' + encodeURIComponent" in html or (
        "fetch('/api/chunks/'" in html and "encodeURIComponent(chunkId)" in html
    )
    assert "fetch('/ops" not in html
    assert 'fetch("/ops' not in html
    assert "fetch('/api/browse" not in html
    assert 'fetch("/api/browse' not in html

    # data-open toggled true on open, false on close
    assert "setAttribute('data-open', 'true')" in html
    assert "setAttribute('data-open', 'false')" in html

    # Click hooks for citation affordances (delegated on qa-list)
    assert "closest('.citation-ref')" in html
    assert "closest('.btn-view-chunk')" in html
    assert "closest('.citation-item')" in html
    assert "openCitationDrawer(chunkId" in html

    # Close on Escape, backdrop click, and #citation-drawer-close
    assert "e.key === 'Escape'" in html
    assert "closeCitationDrawer()" in html
    assert "drawerBackdrop.addEventListener('click', closeCitationDrawer)" in html
    assert "drawerCloseBtn.addEventListener('click', closeCitationDrawer)" in html
    assert "id=\"citation-drawer-close\"" in html
    assert "id=\"citation-drawer-backdrop\"" in html

    # 404 / not-found error path surfaces citation-drawer-error + 未找到
    assert "id=\"citation-drawer-error\"" in html
    assert "res.status === 404" in html
    assert "未找到" in html

    # Editor shell stays isolated — no full 库览 / ops chrome
    assert "库览" not in html
    assert "/ops" not in html


def test_workbench_chunk_api_returns_fragment_text(
    test_settings: Settings, tmp_path: Path
) -> None:
    """#5: drawer data path — GET /api/chunks/{chunk_id} returns fragment JSON contract."""
    source = _write_markdown(
        tmp_path / "mazu.md",
        culture_domain="妈祖",
        title="妈祖简介",
        body="妈祖信仰发源于湄洲岛。",
    )
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    # Discover chunk_id via public HTTP (ops browse API), not internal modules —
    # keeps this ticket's tests independent of WIP knowledge refactor.
    browse = client.get("/api/browse")
    assert browse.status_code == 200
    groups = browse.json()
    assert groups
    chunk_id = groups[0]["documents"][0]["chunks"][0]["chunk_id"]

    response = client.get(f"/api/chunks/{chunk_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_id"] == chunk_id
    assert body["culture_domain"] == "妈祖"
    assert body["title"] == "妈祖简介"
    assert "湄洲岛" in body["text"]

    missing = client.get("/api/chunks/nonexistent-chunk-id")
    assert missing.status_code == 404
