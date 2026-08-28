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
    raw["providers"]["multimodal"] = "fake"
    raw["providers"]["embedding"] = "fake"
    raw["providers"]["reranker"] = "fake"
    raw["providers"]["splitter"] = "recursive_zh"
    raw["providers"]["vector_store"] = "chroma"
    raw["transform"]["stages"] = ["refiner", "enricher"]
    raw["fakes"] = {
        "multimodal": "ok",
        "embedding": "ok",
        "reranker": "ok",
        "splitter": "ok",
        "vector_store": "ok",
    }
    if "gemma" not in raw:
        raw["gemma"] = {
            "mlx_python": "python3",
            "resolve_script": str(repo_settings.parent / "scripts/resolve_modelscope_model.py"),
            "model": "mlx-community/gemma-4-e2b-it-mxfp4",
            "server_port": 8120,
            "server_url": "http://127.0.0.1:8120/",
            "idle_timeout_seconds": 60,
        }
    return Settings.from_dict(raw, root=repo_settings.parent)
