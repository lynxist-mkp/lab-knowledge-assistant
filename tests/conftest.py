from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lab_knowledge.config import CollectionRegistration, Settings


def register_collection(
    settings: Settings,
    collection_id: str,
    display_name: str = "测试集合",
) -> Settings:
    """Return settings with an additional explicitly registered collection."""
    if collection_id == settings.default_collection_id:
        return settings
    if any(reg.collection_id == collection_id for reg in settings.collections):
        return settings
    return replace(
        settings,
        collections=[
            *settings.collections,
            CollectionRegistration(collection_id=collection_id, display_name=display_name),
        ],
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Load settings with fake providers and isolated storage under tmp_path."""
    repo_settings = Path(__file__).resolve().parents[1] / "settings.yaml"
    raw = yaml.safe_load(repo_settings.read_text(encoding="utf-8"))
    raw["paths"] = {
        "chroma": str(tmp_path / "chroma"),
        "bm25": str(tmp_path / "bm25"),
        "ingestion_history": str(tmp_path / "ingestion_history.db"),
        "catalog": str(tmp_path / "catalog.json"),
        "images": str(tmp_path / "images"),
        "image_index": str(tmp_path / "image_index.db"),
        "traces": str(tmp_path / "traces.jsonl"),
        "corpus": str(tmp_path / "corpus"),
        "prompts": raw["paths"]["prompts"],
    }
    raw["observability"] = {
        "trace_file": str(tmp_path / "traces.jsonl"),
        "task_progress_file": str(tmp_path / "task_progress.jsonl"),
        "ask_evidence_file": str(tmp_path / "ask_evidence.jsonl"),
    }
    raw["providers"]["multimodal"] = "fake"
    raw["providers"]["embedding"] = "fake"
    raw["providers"]["reranker"] = "fake"
    raw["providers"]["evaluator"] = "fake"
    raw["providers"]["splitter"] = "recursive_zh"
    raw["providers"]["vector_store"] = "chroma"
    raw["transform"]["stages"] = ["refiner", "enricher"]
    raw["fakes"] = {
        "multimodal": "ok",
        "embedding": "ok",
        "reranker": "ok",
        "evaluator": "ok",
        "splitter": "ok",
        "vector_store": "ok",
    }
    raw["quality_gate"] = dict(raw.get("quality_gate") or {})
    raw["quality_gate"]["gray_review"] = False
    raw["generation"] = {"min_question_overlap": 0.0}
    raw["query_processing"] = {
        "rewriter": "none",
        "lexicon": str(repo_settings.parent / "data/lexicon/synonyms.yaml"),
        "multi_query": False,
    }
    raw["resources"] = dict(raw.get("resources") or {})
    raw["resources"]["query_window_batch"] = False
    raw["resources"]["ingest_window_batch"] = False
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


@pytest.fixture
def without_ragas_judge_key(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    """Configure zhipu Ragas judge and ensure its API key is unset."""
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    test_settings.evaluation.ragas_judge.model = "glm-5.3-flash"
    test_settings.evaluation.ragas_judge.base_url = "https://open.bigmodel.cn/api/paas/v4"
    test_settings.evaluation.ragas_judge.api_key_env = "ZHIPUAI_API_KEY"
