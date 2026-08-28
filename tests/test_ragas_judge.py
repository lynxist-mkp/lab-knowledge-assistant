"""Ragas judge backends: DeepSeek and 智谱."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wenmai.config import Settings, resolve_ragas_judge
from wenmai.eval.read import get_ragas_status


def test_empty_provider_defaults_to_zhipu() -> None:
    judge = resolve_ragas_judge({})
    assert judge.provider == "zhipu"
    assert judge.model == "glm-5.3-flash"


def test_resolve_ds_alias_to_deepseek() -> None:
    judge = resolve_ragas_judge({"provider": "ds"})
    assert judge.provider == "deepseek"
    assert judge.provider_label == "DeepSeek"
    assert judge.model == "deepseek-chat"
    assert judge.base_url == "https://api.deepseek.com"
    assert judge.api_key_env == "DEEPSEEK_API_KEY"


def test_resolve_zhipu_uses_glm_flash() -> None:
    judge = resolve_ragas_judge({"provider": "zhipu"})
    assert judge.provider == "zhipu"
    assert judge.provider_label == "智谱"
    assert judge.model == "glm-5.3-flash"
    assert judge.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert judge.api_key_env == "ZHIPUAI_API_KEY"


def test_resolve_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown ragas_judge provider"):
        resolve_ragas_judge({"provider": "openai"})


def test_settings_zhipu_provider(tmp_path: Path) -> None:
    repo_settings = Path(__file__).resolve().parents[1] / "settings.yaml"
    raw = yaml.safe_load(repo_settings.read_text(encoding="utf-8"))
    raw["evaluation"] = dict(raw["evaluation"])
    raw["evaluation"]["ragas_judge"] = {"provider": "zhipu"}
    settings = Settings.from_dict(raw, root=repo_settings.parent)
    assert settings.evaluation.ragas_judge.model == "glm-5.3-flash"
    assert settings.evaluation.ragas_judge.api_key_env == "ZHIPUAI_API_KEY"


def test_settings_accepts_legacy_top_level_ragas_judge(tmp_path: Path) -> None:
    repo_settings = Path(__file__).resolve().parents[1] / "settings.yaml"
    raw = yaml.safe_load(repo_settings.read_text(encoding="utf-8"))
    raw["evaluation"] = {
        "golden_set": raw["evaluation"]["golden_set"],
        "runs": raw["evaluation"]["runs"],
        "ablations": raw["evaluation"]["ablations"],
    }
    raw["ragas_judge"] = {"provider": "ds"}
    settings = Settings.from_dict(raw, root=repo_settings.parent)
    assert settings.evaluation.ragas_judge.provider == "deepseek"


def test_ragas_status_exposes_configured_judge(test_settings: Settings) -> None:
    status = get_ragas_status(test_settings)
    assert status.provider == "zhipu"
    assert status.provider_label == "智谱"
    assert status.model == "glm-5.3-flash"
    assert status.faithfulness.status in {"ready", "unavailable"}
