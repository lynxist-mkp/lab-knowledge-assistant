"""HTTP seam and contract tests: 运维看板 (#6 概览·库览·入库, #7 追踪·评测)."""

from __future__ import annotations

import json
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
    assert "document_count" in html
    assert "chunk_count" in html
    assert "avg_query_latency_ms" in html

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
    assert "/ops" not in workbench.text


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
    assert "fetch('/api/traces/' + encodeURIComponent(" in html or 'fetch("/api/traces/" + encodeURIComponent(' in html


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

    # Fetch and run wiring
    assert "fetch('/api/eval/runs')" in html or 'fetch("/api/eval/runs")' in html
    assert "fetch('/api/eval/runs', {" in html or 'fetch("/api/eval/runs", {' in html


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
    """All five ops nav tabs are present and isolated from editor controls."""
    client = TestClient(create_app(test_settings))
    html = _ops_html(client)

    # 5 nav tabs present
    assert 'id="tab-overview"' in html
    assert 'id="tab-browse"' in html
    assert 'id="tab-ingest"' in html
    assert 'id="tab-trace"' in html
    assert 'id="tab-eval"' in html

    assert "概览" in html
    assert "库览" in html
    assert "入库" in html
    assert "追踪" in html
    assert "评测" in html

    # Still no editor controls
    assert 'id="ask-form"' not in html
    assert 'id="citation-drawer"' not in html
    assert 'id="question-input"' not in html

