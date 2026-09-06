"""HTTP seam and contract tests: 运维看板 (#6 概览·库览·入库, #7 追踪·评测)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from tests.conftest import register_collection

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


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


def _ops_html(client: TestClient) -> str:
    response = client.get("/ops")
    assert response.status_code == 200
    return response.text


def test_ops_landmarks_and_overview_panel(test_settings: Settings) -> None:
    """Landmarks, brand, tokens, navigation, overview panel and stats fetch wiring."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # Brand and Surface
    assert "福云·文脉助手" in html
    assert "运维看板" in html
    assert "#037AFF" in html

    # Navigation tabs and panels
    assert 'id="ops-nav"' in html
    assert 'id="tab-overview"' in html
    assert 'id="tab-browse"' in html
    assert 'id="tab-ingest"' in html
    assert 'id="ops-panel-overview"' in html
    assert 'id="ops-panel-browse"' in html
    assert 'id="ops-panel-ingest"' in html

    # Overview panel landmarks & fetch path
    assert "概览" in html
    assert "知识库概览" in html
    assert "fetch('/api/stats/overview')" in html or 'fetch("/api/stats/overview")' in html
    assert 'id="stat-doc-count"' in html
    assert 'id="stat-chunk-count"' in html
    assert 'id="stat-avg-latency"' in html
    assert 'id="stat-latency-p50"' in html
    assert 'id="stat-latency-p95"' in html
    assert "document_count" in html
    assert "chunk_count" in html
    assert "avg_query_latency_ms" in html
    assert "query_latency_p50_ms" in html
    assert "query_latency_p95_ms" in html

    # Compliance footer inherited from shell
    assert "人工智能生成合成" in html
    assert "播出终审" in html


def test_ops_browse_panel_wiring(test_settings: Settings) -> None:
    """Browse panel: /api/browse fetch, 库览 landmark, and browse list container."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # Browse panel landmarks
    assert "库览" in html
    assert "文化域" in html
    assert 'id="ops-browse-list"' in html or 'id="browse-list"' in html
    assert 'id="btn-refresh-browse"' in html

    # Fetch wiring
    assert "fetch('/api/browse')" in html or 'fetch("/api/browse")' in html


def test_ops_ingestion_panel_wiring(test_settings: Settings) -> None:
    """Ingestion panel: /api/ingestion/run SSE stream fetch, 入库 landmark, and form inputs."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # Ingestion panel landmarks
    assert "入库" in html
    assert 'id="ingest-form"' in html
    assert 'id="ingest-source-path"' in html
    assert 'name="source_path"' in html
    assert 'id="ingest-submit-btn"' in html
    assert 'id="ingest-stages-list"' in html
    assert 'id="ingest-result-card"' in html

    # Ingestion SSE fetch wiring
    assert "/api/ingestion/run" in html
    assert "fetch('/api/ingestion/run'" in html or 'fetch("/api/ingestion/run"' in html


def test_ops_nav_isolation_from_editor(test_settings: Settings) -> None:
    """Ops surface must not contain editor controls or ask widgets."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # No editor controls
    assert 'id="ask-form"' not in html
    assert 'id="question-input"' not in html
    assert 'id="submit-btn"' not in html
    assert 'id="culture-domain-select"' not in html
    assert "openCitationDrawer" not in html
    assert 'id="citation-drawer"' not in html

    # Ops surface features are present
    assert "概览" in html
    assert "库览" in html
    assert "入库" in html
    assert "运维看板" in html


def test_ops_dual_shell_isolation(test_settings: Settings) -> None:
    """Dual shell isolation: /ops has 运维看板 not 编辑工作台; / has 编辑工作台 not 运维看板."""
    client = TestClient(create_app(test_settings))

    ops = client.get("/ops")
    assert ops.status_code == 200
    assert "运维看板" in ops.text
    assert "编辑工作台" not in ops.text

    workbench = client.get("/")
    assert workbench.status_code == 200
    assert "编辑工作台" in workbench.text
    assert "运维看板" not in workbench.text
    assert "库览" not in workbench.text
    assert 'class="ops-nav-tab"' not in workbench.text

    assert "返回提问" in ops.text
    assert "知识库维护" in workbench.text


def test_ops_overview_api_contract(test_settings: Settings, tmp_path: Path) -> None:
    """Overview API contract returns document_count, chunk_count, avg_query_latency_ms."""
    client = TestClient(create_app(test_settings))

    # Empty stats initially
    resp = client.get("/api/stats/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_count"] == 0
    assert data["chunk_count"] == 0
    assert "avg_query_latency_ms" in data
    assert "query_latency_p50_ms" in data
    assert "query_latency_p95_ms" in data
    assert data["query_latency_p50_ms"] is None
    assert data["query_latency_p95_ms"] is None

    # After ingesting a file
    source = _write_markdown(
        tmp_path / "zhuzi.md",
        culture_domain="朱子",
        title="朱子理学",
        body="朱熹是理学的集大成者，主张格物致知。",
    )
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    resp = client.get("/api/stats/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_count"] == 1
    assert data["chunk_count"] >= 1


def test_ops_browse_api_contract(test_settings: Settings) -> None:
    """Browse API contract returns culture domain groups with documents and chunk previews."""
    from wenmai.models import Chunk

    app = create_app(test_settings)
    app.state.knowledge.commit_document(
        source_path="/tmp/doc-haisi.md",
        sha256="doc-haisi",
        document_id="doc-haisi",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="doc-haisi:0000",
                document_id="doc-haisi",
                text="泉州是联合国教科文组织认定的海上丝绸之路起点之一。",
                metadata={
                    "document_id": "doc-haisi",
                    "title": "海上丝绸之路",
                    "culture_domain": "海丝",
                },
            ),
        ],
    )
    app.state.knowledge.commit_document(
        source_path="/tmp/doc-mazu.md",
        sha256="doc-mazu",
        document_id="doc-mazu",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="doc-mazu:0000",
                document_id="doc-mazu",
                text="妈祖是流传于中国沿海地区的民间信仰，发源于福建莆田湄洲岛。",
                metadata={
                    "document_id": "doc-mazu",
                    "title": "妈祖文化",
                    "culture_domain": "妈祖",
                },
            ),
        ],
    )

    client = TestClient(app)
    resp = client.get("/api/browse")
    assert resp.status_code == 200
    groups = resp.json()
    assert len(groups) == 2

    domains = {g["culture_domain"] for g in groups}
    assert "海丝" in domains
    assert "妈祖" in domains

    for group in groups:
        assert group["document_count"] >= 1
        assert group["chunk_count"] >= 1
        assert len(group["documents"]) >= 1
        doc = group["documents"][0]
        assert "document_id" in doc
        assert "title" in doc
        assert len(doc["chunks"]) >= 1
        chunk = doc["chunks"][0]
        assert "chunk_id" in chunk
        assert "document_id" in chunk
        assert "preview" in chunk
        assert "审阅状态" in chunk


def test_ops_browse_shows_pending_review_status(test_settings: Settings) -> None:
    from wenmai.models import Chunk

    app = create_app(test_settings)
    app.state.knowledge.commit_document(
        source_path="/tmp/doc-pending.md",
        sha256="doc-pending",
        document_id="doc-pending",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="doc-pending:0000",
                document_id="doc-pending",
                text="待审片段仅供库览展示。",
                metadata={
                    "document_id": "doc-pending",
                    "title": "待审材料",
                    "culture_domain": "船政",
                    "审阅状态": "待审",
                },
            ),
        ],
    )

    client = TestClient(app)
    resp = client.get("/api/browse")
    assert resp.status_code == 200
    groups = resp.json()
    ship_group = next(g for g in groups if g["culture_domain"] == "船政")
    chunk = ship_group["documents"][0]["chunks"][0]
    assert chunk["审阅状态"] == "待审"


def test_ops_collection_query_params_route_overview_and_browse(
    test_settings: Settings,
) -> None:
    from wenmai.models import Chunk

    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    other_knowledge.commit_document(
        source_path="/tmp/doc-other-ops.md",
        sha256="doc-other-ops",
        document_id="doc-other-ops",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="doc-other-ops:0000",
                document_id="doc-other-ops",
                text="其他集合运维路由",
                metadata={
                    "document_id": "doc-other-ops",
                    "title": "其他集合运维路由",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            ),
        ],
    )

    client = TestClient(create_app(settings))

    default_overview = client.get("/api/stats/overview")
    other_overview = client.get("/api/stats/overview", params={"collection_id": other_id})
    assert default_overview.status_code == 200
    assert other_overview.status_code == 200
    assert default_overview.json()["document_count"] == 0
    assert other_overview.json()["document_count"] == 1

    default_browse = client.get("/api/browse")
    other_browse = client.get("/api/browse", params={"collection_id": other_id})
    assert default_browse.status_code == 200
    assert other_browse.status_code == 200
    assert default_browse.json() == []
    other_groups = other_browse.json()
    assert len(other_groups) == 1
    assert other_groups[0]["documents"][0]["document_id"] == "doc-other-ops"


def test_ops_ingestion_run_sse_stream_contract(
    test_settings: Settings, tmp_path: Path
) -> None:
    """Ingestion SSE endpoint /api/ingestion/run streams stage and done events."""
    source = _write_markdown(
        tmp_path / "chuanzheng.md",
        culture_domain="船政",
        title="福建船政",
        body="福建船政创办于1866年，培养了大批中国近代海军和工程科技人才。",
    )

    client = TestClient(create_app(test_settings))
    response = client.post("/api/ingestion/run", json={"source_path": str(source)})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events: list[dict[str, object]] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            events.append(payload)

    event_types = [e.get("event") for e in events]
    assert "stage" in event_types
    assert "done" in event_types

    done_event = next(e for e in events if e.get("event") == "done")
    result = done_event["result"]
    assert result["status"] in ("ingested", "rebuilt", "skipped")
    assert result["chunk_count"] >= 1
    assert "trace_id" in result


def test_ops_trace_panel_wiring(test_settings: Settings) -> None:
    """Trace panel: 追踪 tab, subtabs, list containers, detail panel and fetch wiring."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # Tab and Panel landmarks
    assert "追踪" in html
    assert 'id="tab-trace"' in html
    assert 'id="ops-panel-trace"' in html
    assert 'id="trace-subtab-ingestion"' in html
    assert 'id="trace-subtab-query"' in html

    # List & Detail landmarks
    assert 'id="trace-ingestion-list"' in html
    assert 'id="trace-query-list"' in html
    assert 'id="btn-refresh-trace-ingestion"' in html
    assert 'id="btn-refresh-trace-query"' in html
    assert 'id="trace-detail-panel"' in html
    assert 'id="trace-detail-title"' in html
    assert 'id="trace-detail-body"' in html
    assert 'id="trace-detail-close"' in html

    # Fetch wiring
    assert "fetch('/api/traces/ingestion')" in html or 'fetch("/api/traces/ingestion")' in html
    assert "fetch('/api/traces/query')" in html or 'fetch("/api/traces/query")' in html
    assert "fetch('/api/traces/' + encodeURIComponent(" in html or (
        'fetch("/api/traces/" + encodeURIComponent(' in html
    )
    assert "white-space: nowrap" in html
    assert "btn-table-action" in html
    assert "function formatChunkPreview" in html
    assert 'id="trace-detail-backdrop"' in html


def test_ops_eval_panel_wiring(test_settings: Settings) -> None:
    """Eval panel: 评测 tab, eval runs list, run button, detail view, fetch/post wiring."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # Strictly use 评测 and avoid 评估面板
    assert "评测" in html
    assert "评估面板" not in html
    assert 'id="tab-eval"' in html
    assert 'id="ops-panel-eval"' in html
    assert 'id="eval-runs-list"' in html
    assert 'id="btn-refresh-eval"' in html
    assert 'id="btn-run-eval"' in html
    assert 'id="eval-run-detail"' in html

    # Group labels
    assert "Dense 单路" in html
    assert "Sparse 单路" in html
    assert "RRF 融合" in html
    assert "RRF + Rerank" in html

    # Fetch and run wiring — collection_id from page URL via buildEvalRunsUrl()
    assert "function buildEvalRunsUrl" in html
    assert "function getCollectionIdFromUrl" in html
    assert "URLSearchParams" in html
    assert "fetch(buildEvalRunsUrl())" in html
    assert "fetch(buildEvalRunsUrl()," in html
    assert "'/api/eval/runs?collection_id=' + encodeURIComponent(collectionId)" in html


def test_ops_eval_panel_collection_id_url_wiring(test_settings: Settings) -> None:
    """Eval panel: /ops?collection_id=... passes query through to eval API fetches."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    assert "params.get('collection_id')" in html
    assert "buildEvalRunsUrl()" in html
    assert "encodeURIComponent(collectionId)" in html


def test_ops_trace_detail_api_contract(test_settings: Settings, tmp_path: Path) -> None:
    """Integration contract: trace details for ingestion and query via /api/traces/{trace_id}."""
    source = _write_markdown(
        tmp_path / "mazu_trace.md",
        culture_domain="妈祖",
        title="妈祖信俗",
        body="妈祖是流传于中国沿海地区的民间信仰，发源于福建莆田湄洲岛。",
    )
    client = TestClient(create_app(test_settings))

    # 1. Ingest file and check ingestion trace detail
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    ingest_trace_id = ingest.json()["trace_id"]

    ingest_list_resp = client.get("/api/traces/ingestion")
    assert ingest_list_resp.status_code == 200
    ingest_traces = ingest_list_resp.json()
    assert any(t["trace_id"] == ingest_trace_id for t in ingest_traces)

    ingest_detail_resp = client.get(f"/api/traces/{ingest_trace_id}")
    assert ingest_detail_resp.status_code == 200
    ingest_detail = ingest_detail_resp.json()
    assert ingest_detail["trace_type"] == "ingestion"
    assert "steps" in ingest_detail
    assert len(ingest_detail["steps"]) > 0

    # 2. Ask question and check query trace detail
    ask = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert ask.status_code == 200
    query_trace_id = ask.json()["trace_id"]

    query_list_resp = client.get("/api/traces/query")
    assert query_list_resp.status_code == 200
    query_traces = query_list_resp.json()
    assert any(t["trace_id"] == query_trace_id for t in query_traces)

    query_detail_resp = client.get(f"/api/traces/{query_trace_id}")
    assert query_detail_resp.status_code == 200
    query_detail = query_detail_resp.json()
    assert query_detail["trace_type"] == "query"
    assert "stage_latencies" in query_detail
    assert len(query_detail["stage_latencies"]) > 0


def test_ops_five_capabilities_complete(test_settings: Settings) -> None:
    """All six ops nav tabs are present and isolated from editor controls."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # 6 nav tabs present
    assert 'id="tab-overview"' in html
    assert 'id="tab-browse"' in html
    assert 'id="tab-review"' in html
    assert 'id="tab-ingest"' in html
    assert 'id="tab-trace"' in html
    assert 'id="tab-eval"' in html

    assert "概览" in html
    assert "库览" in html
    assert "待审" in html
    assert "入库" in html
    assert "追踪" in html
    assert "评测" in html

    # Still no editor controls
    assert 'id="ask-form"' not in html
    assert 'id="citation-drawer"' not in html
    assert 'id="question-input"' not in html


def test_ops_review_panel_wiring(test_settings: Settings) -> None:
    """Review panel: 待审 tab, list container, approve/reject fetch wiring."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    assert 'id="tab-review"' in html
    assert 'id="ops-panel-review"' in html
    assert 'id="review-list"' in html
    assert 'id="btn-refresh-review"' in html
    assert "fetch('/api/review/pending')" in html or 'fetch("/api/review/pending")' in html
    assert "/api/review/" in html
    assert "通过" in html
    assert "驳回" in html
    assert "chunk-preview-row--pending" in html
    assert "review-chunk-list" in html
    assert "chunks-grid--collapsed" in html
    assert "data-expand-chunks" in html
    assert "data-view-chunk-full" in html
    assert 'id="ops-chunk-drawer"' in html
    assert "eval-compare-table" in html


def test_workbench_has_no_review_actions(test_settings: Settings) -> None:
    """编辑工作台 must not expose review APIs or buttons."""
    client = TestClient(create_app(test_settings))
    workbench = client.get("/")
    assert workbench.status_code == 200
    html = workbench.text

    assert "待审" not in html
    assert "/api/review/" not in html
    assert "btn-review-approve" not in html
    assert "btn-review-reject" not in html


def _plant_pending_doc(
    app,
    *,
    document_id: str,
    title: str,
    culture_domain: str,
    text: str,
) -> None:
    from wenmai.models import Chunk

    app.state.knowledge.commit_document(
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
                    "title": title,
                    "culture_domain": culture_domain,
                },
            ),
        ],
    )
    app.state.knowledge.set_review_status(document_id, "待审")


def test_review_pending_approve_reject_api_contract(
    test_settings: Settings, tmp_path: Path
) -> None:
    """List pending, approve makes ask hit, reject removes from browse."""
    app = create_app(test_settings)
    _plant_pending_doc(
        app,
        document_id="doc-approve-me",
        title="待审放行材料",
        culture_domain="妈祖",
        text="妈祖信仰发源于福建莆田湄洲岛，是沿海民间信俗。",
    )
    _plant_pending_doc(
        app,
        document_id="doc-reject-me",
        title="待审驳回材料",
        culture_domain="船政",
        text="福建船政创办于1866年，培养近代海军人才。",
    )

    client = TestClient(app)

    pending = client.get("/api/review/pending")
    assert pending.status_code == 200
    items = pending.json()
    assert len(items) == 2
    ids = {item["document_id"] for item in items}
    assert ids == {"doc-approve-me", "doc-reject-me"}
    for item in items:
        assert "title" in item
        assert "culture_domain" in item
        assert item["chunk_count"] >= 1
        assert "chunks" in item
        assert len(item["chunks"]) >= 1
        assert "preview" in item["chunks"][0]
        assert item["chunks"][0]["审阅状态"] == "待审"

    missing = client.post("/api/review/unknown-doc/approve")
    assert missing.status_code == 404

    refused = client.post(
        "/ask",
        json={"question": "妈祖信仰发源于哪里？"},
    )
    assert refused.status_code == 200
    assert refused.json()["refused"] is True

    approve = client.post("/api/review/doc-approve-me/approve")
    assert approve.status_code == 200
    assert approve.json()["审阅状态"] == "已通过"

    pending_after_approve = client.get("/api/review/pending")
    assert pending_after_approve.status_code == 200
    pending_ids = {item["document_id"] for item in pending_after_approve.json()}
    assert "doc-approve-me" not in pending_ids
    assert "doc-reject-me" in pending_ids

    answered = client.post(
        "/ask",
        json={"question": "妈祖信仰发源于哪里？"},
    )
    assert answered.status_code == 200
    body = answered.json()
    assert body["refused"] is False
    assert body["citations"]
    assert body["citations"][0]["document_id"] == "doc-approve-me"

    reject = client.post("/api/review/doc-reject-me/reject")
    assert reject.status_code == 200
    assert reject.json()["审阅状态"] == "已驳回"

    pending_final = client.get("/api/review/pending")
    assert pending_final.status_code == 200
    assert pending_final.json() == []

    browse = client.get("/api/browse")
    assert browse.status_code == 200
    all_doc_ids = {
        doc["document_id"]
        for group in browse.json()
        for doc in group["documents"]
    }
    assert "doc-reject-me" not in all_doc_ids
    assert "doc-approve-me" in all_doc_ids


def test_review_api_query_params_route_to_alternate_collection(
    test_settings: Settings,
) -> None:
    from wenmai.models import Chunk

    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    other_knowledge.commit_document(
        source_path="/tmp/doc-other-review.md",
        sha256="doc-other-review",
        document_id="doc-other-review",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="doc-other-review:0000",
                document_id="doc-other-review",
                text="其他集合待审材料。",
                metadata={
                    "document_id": "doc-other-review",
                    "title": "其他集合待审材料",
                    "culture_domain": "船政",
                },
            ),
        ],
    )
    other_knowledge.set_review_status("doc-other-review", "待审")

    client = TestClient(create_app(settings))

    pending = client.get("/api/review/pending", params={"collection_id": other_id})
    assert pending.status_code == 200
    items = pending.json()
    assert [item["document_id"] for item in items] == ["doc-other-review"]

    approve = client.post(
        "/api/review/doc-other-review/approve",
        params={"collection_id": other_id},
    )
    assert approve.status_code == 200
    assert approve.json()["审阅状态"] == "已通过"

    pending_after = client.get("/api/review/pending", params={"collection_id": other_id})
    assert pending_after.status_code == 200
    assert pending_after.json() == []


def test_api_traces_collection_query_params_route_list_and_detail(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    create_knowledge(_other_collection_settings(test_settings, other_id)).commit_document(
        source_path="/tmp/trace-api-other.md",
        sha256="trace-api-other",
        document_id="trace-api-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="trace-api-other:0000",
                document_id="trace-api-other",
                text="其他集合 trace API 材料。",
                metadata={
                    "document_id": "trace-api-other",
                    "title": "trace-api-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            ),
        ],
    )

    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_trace = {
        "trace_id": "legacy-api-query",
        "trace_type": "query",
        "started_at": "2026-06-01T12:00:00+00:00",
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": 10.0,
        "stages": [],
        "error": None,
        "metadata": {"question": "legacy"},
    }
    scoped_trace = {
        **legacy_trace,
        "trace_id": "scoped-api-query",
        "collection_id": other_id,
        "metadata": {"question": "scoped"},
    }
    trace_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (legacy_trace, scoped_trace)
        )
        + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(settings))

    default_list = client.get("/api/traces/query")
    assert default_list.status_code == 200
    assert [item["trace_id"] for item in default_list.json()] == ["legacy-api-query"]

    other_list = client.get("/api/traces/query", params={"collection_id": other_id})
    assert other_list.status_code == 200
    assert [item["trace_id"] for item in other_list.json()] == ["scoped-api-query"]

    default_detail = client.get("/api/traces/legacy-api-query")
    assert default_detail.status_code == 200
    assert default_detail.json()["trace_type"] == "query"

    missing_on_other = client.get(
        "/api/traces/legacy-api-query",
        params={"collection_id": other_id},
    )
    assert missing_on_other.status_code == 404

    scoped_detail = client.get(
        "/api/traces/scoped-api-query",
        params={"collection_id": other_id},
    )
    assert scoped_detail.status_code == 200
    assert scoped_detail.json()["question"] == "scoped"

    missing_on_default = client.get("/api/traces/scoped-api-query")
    assert missing_on_default.status_code == 404


def test_ops_health_snapshot_supports_collection_scope(test_settings: Settings) -> None:
    from wenmai.ops.ask_evidence import write_ask_evidence
    from wenmai.task_progress import TaskCounters, persist_task_progress

    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_settings = _other_collection_settings(settings, other_id)
    client = TestClient(create_app(settings))

    write_ask_evidence(
        settings,
        {
            "event": "saturation",
            "code": "busy",
            "entrypoint": "default",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 0.0,
        },
    )
    write_ask_evidence(
        other_settings,
        {
            "event": "saturation",
            "code": "timeout",
            "entrypoint": "other",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 100.0,
        },
    )
    persist_task_progress(
        other_settings,
        task_id="evaluation:other-failed",
        task_type="evaluation",
        status="failed",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, failed=1),
        failure_kind="runtime",
    )

    global_health = client.get("/api/stats/health")
    assert global_health.status_code == 200
    global_signals = {
        item["name"]: item["count"] for item in global_health.json()["signals"]
    }
    assert global_signals["ask_busy"] == 1
    assert global_signals["ask_timeout"] == 1

    scoped_health = client.get("/api/stats/health", params={"collection_id": other_id})
    assert scoped_health.status_code == 200
    scoped_signals = {
        item["name"]: item["count"] for item in scoped_health.json()["signals"]
    }
    assert scoped_signals["ask_busy"] == 0
    assert scoped_signals["ask_timeout"] == 1
    assert scoped_signals["failed"] == 1


def test_ops_unknown_collection_returns_404(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    unknown = {"collection_id": "missing"}

    collection_scoped_gets = (
        "/api/stats/overview",
        "/api/stats/health",
        "/api/browse",
        "/api/review/pending",
        "/api/traces/ingestion",
        "/api/traces/query",
        "/api/traces/any-trace",
        "/api/traces/any-trace/summary",
        "/api/traces/any-trace/degradations",
        "/api/tasks/progress/ingestion:demo/investigation",
    )
    for path in collection_scoped_gets:
        response = client.get(path, params=unknown)
        assert response.status_code == 404
        assert response.json()["detail"] == "collection not found"

    for path in (
        "/api/review/doc-1/approve",
        "/api/review/doc-1/reject",
    ):
        response = client.post(path, params=unknown)
        assert response.status_code == 404
        assert response.json()["detail"] == "collection not found"

