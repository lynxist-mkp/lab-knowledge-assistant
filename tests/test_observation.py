"""运维观测 seam：概览与 Trace 读。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from wenmai.config import Settings
from wenmai.ops.observation import (
    get_query_detail,
    list_query_summaries,
    load_overview_stats,
)


def test_load_overview_stats_empty_catalog(test_settings: Settings) -> None:
    stats = load_overview_stats(test_settings)
    assert stats.document_count == 0
    assert stats.chunk_count == 0


def test_list_query_summaries_via_observation_seam(
    test_settings: Settings, tmp_path: Path
) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    trace = {
        "trace_id": "obs-read-1",
        "trace_type": "query",
        "started_at": started.isoformat(),
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": 42.0,
        "stages": [],
        "error": None,
        "metadata": {"question": "闽派文化是什么？"},
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")

    summaries = list_query_summaries(test_settings)
    assert len(summaries) == 1
    assert summaries[0].trace_id == "obs-read-1"

    detail = get_query_detail(test_settings, "obs-read-1")
    assert detail is not None
    assert detail.summary.question == "闽派文化是什么？"
