from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wenmai.config import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Load settings with fake providers and isolated storage under tmp_path."""
    repo_settings = Path(__file__).resolve().parents[1] / "settings.yaml"
    raw = yaml.safe_load(repo_settings.read_text(encoding="utf-8"))
    raw["paths"] = {
        "chroma": str(tmp_path / "chroma"),
        "bm25": str(tmp_path / "bm25"),
        "ingestion_history": str(tmp_path / "ingestion_history.db"),
        "images": str(tmp_path / "images"),
        "image_index": str(tmp_path / "image_index.db"),
        "traces": str(tmp_path / "traces.jsonl"),
        "corpus": str(tmp_path / "corpus"),
        "prompts": raw["paths"]["prompts"],
    }
    raw["observability"] = {"trace_file": str(tmp_path / "traces.jsonl")}
    raw["providers"]["llm"] = "fake"
    raw["providers"]["vision"] = "fake"
    raw["providers"]["embedding"] = "fake"
    raw["providers"]["reranker"] = "fake"
    raw["providers"]["splitter"] = "recursive_zh"
    raw["providers"]["vector_store"] = "chroma"
    raw["providers"]["evaluator"] = "fake"
    raw["fakes"] = {
        "llm": "ok",
        "vision": "ok",
        "embedding": "ok",
        "reranker": "ok",
        "evaluator": "ok",
        "splitter": "ok",
        "vector_store": "ok",
    }
    return Settings.from_dict(raw, root=repo_settings.parent)
