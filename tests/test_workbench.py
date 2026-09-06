"""HTTP seam: 检索工作台可提问 (#4) and 出处页内只读抽屉 (#5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import register_collection
from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk


def _write_markdown(path: Path, *, culture_domain: str, title: str, body: str) -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/{path.stem}
culture_domain: {culture_domain}
space: lab_knowledge
title: {title}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def _workbench_html(client: TestClient) -> str:
    response = client.get("/")
    assert response.status_code == 200
    return response.text


def test_workbench_landmarks_and_culture_domain_filter(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    # Brand and Surface
    assert "课题组知识助手" in html
    assert "检索工作台" in html
    assert "#037AFF" in html

    # Top bar 研究主题 filter control and default options
    assert "研究主题" in html
    assert "culture_domain" in html
    assert "culture-domain-select" in html
    assert "全部 / 不限" in html or "全部" in html
    assert "自然语言处理" in html
    assert "多模态" in html
    assert "检索增强" in html
    assert "强化学习" in html

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
    assert "实验资料核对" in html

    # No ops primary nav chrome (surface switch link is allowed)
    assert "Ingestion 管理" not in html
    assert "评估面板" not in html
    assert "数据浏览" not in html
    assert "库览" not in html
    assert "运维看板" not in html
    assert 'class="ops-nav-tab"' not in html


def test_workbench_dual_shell_isolation(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))

    workbench = client.get("/")
    assert workbench.status_code == 200
    assert '<p class="surface">检索工作台</p>' in workbench.text
    assert '<p class="surface">运维看板</p>' not in workbench.text

    ops = client.get("/ops")
    assert ops.status_code == 200
    assert '<p class="surface">运维看板</p>' in ops.text
    assert '<p class="surface">检索工作台</p>' not in ops.text
    assert "人工智能生成合成" in ops.text
    assert "实验资料核对" in ops.text

    # Cross-surface switch links use neutral labels, not the other surface name
    assert "知识库维护" in workbench.text
    assert "返回提问" in ops.text


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

    # Surface switch affordance for maintainers
    assert 'class="surface-switch"' in html
    assert "知识库维护" in html

    # No full 库览 ops chrome / browse fetch on workbench
    assert "库览" not in html
    assert "fetch('/api/browse'" not in html
    assert 'fetch("/api/browse"' not in html
    assert 'class="surface-switch"' in html


def test_workbench_ask_flow_uses_ask_endpoint(test_settings: Settings) -> None:
    """#4: ask submission posts to /ask from workbench page script."""
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    assert "buildWorkbenchApiUrl('/ask')" in html
    assert "fetch('/ask'" not in html
    assert 'fetch("/ask"' not in html
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

    # Drawer fetch path inherits page collection scope via helper.
    assert "function getCollectionIdFromUrl" in html
    assert "function buildWorkbenchApiUrl" in html
    assert "/api/chunks/" in html
    assert "encodeURIComponent" in html
    assert "fetch(" in html
    assert "buildWorkbenchApiUrl('/api/chunks/' + encodeURIComponent(chunkId))" in html
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
    assert 'class="ops-nav-tab"' not in html


def test_workbench_chunk_api_returns_fragment_text(
    test_settings: Settings, tmp_path: Path
) -> None:
    """#5: drawer data path — GET /api/chunks/{chunk_id} returns fragment JSON contract."""
    source = _write_markdown(
        tmp_path / "mazu.md",
        culture_domain="检索增强",
        title="RAG 简介",
        body="检索增强生成结合外部检索与语言模型。",
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
    assert body["culture_domain"] == "检索增强"
    assert body["title"] == "RAG 简介"
    assert "检索增强" in body["text"]

    missing = client.get("/api/chunks/nonexistent-chunk-id")
    assert missing.status_code == 404


def test_workbench_page_collection_scope_url_wiring(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    html = _workbench_html(client)

    assert "params.get('collection_id')" in html
    assert "function buildWorkbenchApiUrl" in html
    assert "buildWorkbenchApiUrl('/api/chunks/' + encodeURIComponent(chunkId))" in html
    assert "buildWorkbenchApiUrl('/api/traces/' + encodeURIComponent(traceId))" in html
    assert (
        "buildWorkbenchApiUrl('/api/traces/' + encodeURIComponent(traceId) + '/summary')"
        in html
    )
    assert "fetch('/api/traces/' + encodeURIComponent(traceId))" not in html
    assert "fetch('/api/traces/' + encodeURIComponent(traceId) + '/summary')" not in html
    assert "buildWorkbenchApiUrl('/ask')" in html
    assert "fetch('/ask'" not in html
    assert 'fetch("/ask"' not in html


def test_workbench_chunk_api_routes_by_collection_query_param(
    test_settings: Settings, tmp_path: Path
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    client = TestClient(create_app(settings))
    other_settings = _other_collection_settings(test_settings, other_id)
    other_client = TestClient(create_app(other_settings))

    source = _write_markdown(
        tmp_path / "other-collection.md",
        culture_domain="妈祖",
        title="其他集合文档",
        body="其他集合里的出处片段正文。",
    )
    ingest = other_client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    browse = other_client.get("/api/browse")
    assert browse.status_code == 200
    chunk_id = browse.json()[0]["documents"][0]["chunks"][0]["chunk_id"]

    default_detail = client.get(f"/api/chunks/{chunk_id}")
    assert default_detail.status_code == 404

    scoped_detail = client.get(
        f"/api/chunks/{chunk_id}",
        params={"collection_id": other_id},
    )
    assert scoped_detail.status_code == 200
    assert "其他集合里的出处片段正文" in scoped_detail.json()["text"]


def test_api_document_card_routes_by_collection_query_param(
    test_settings: Settings, tmp_path: Path
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    client = TestClient(create_app(settings))
    other_settings = _other_collection_settings(test_settings, other_id)
    other_client = TestClient(create_app(other_settings))

    source = _write_markdown(
        tmp_path / "other-document.md",
        culture_domain="海丝",
        title="其他集合文档卡片",
        body="其他集合里的文档详情正文。",
    )
    ingest = other_client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    default_detail = client.get(f"/api/documents/{document_id}")
    assert default_detail.status_code == 404

    scoped_detail = client.get(
        f"/api/documents/{document_id}",
        params={"collection_id": other_id},
    )
    assert scoped_detail.status_code == 200
    assert scoped_detail.json()["title"] == "其他集合文档卡片"

    unknown = client.get(
        f"/api/documents/{document_id}",
        params={"collection_id": "missing-collection"},
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "collection not found"


def _commit(knowledge, document_id: str, text: str) -> None:
    knowledge.commit_document(
        source_path=f"/tmp/{document_id}.md",
        sha256=document_id,
        document_id=document_id,
        status="ingested",
        chunks=[
            Chunk(
                chunk_id=f"{document_id}:0000",
                document_id=document_id,
                text=text,
                metadata={
                    "document_id": document_id,
                    "title": document_id,
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )


def test_workbench_ask_routes_by_collection_query_param(test_settings: Settings) -> None:
    other_id = "other-collection"
    default_id = test_settings.product.collection

    default_knowledge = create_knowledge(test_settings)
    _commit(default_knowledge, "default-doc", "DEFAULT_MARKER_abc 默认集合内容")

    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    _commit(other_knowledge, "other-doc", "OTHER_MARKER_xyz 其他集合内容")

    registered = register_collection(test_settings, other_id)
    client = TestClient(create_app(registered))

    other_response = client.post(
        "/ask",
        params={"collection_id": other_id},
        json={"question": "OTHER_MARKER_xyz 在哪里"},
    )
    assert other_response.status_code == 200
    other_body = other_response.json()
    assert other_body["citations"]
    assert other_body["citations"][0]["document_id"] == "other-doc"

    default_response = client.post(
        "/ask",
        params={"collection_id": default_id},
        json={"question": "DEFAULT_MARKER_abc 在哪里"},
    )
    assert default_response.status_code == 200
    default_body = default_response.json()
    assert default_body["citations"]
    assert default_body["citations"][0]["document_id"] == "default-doc"


def test_workbench_ask_unknown_collection_returns_404(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    response = client.post(
        "/ask",
        params={"collection_id": "missing-collection"},
        json={"question": "问题"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "collection not found"
