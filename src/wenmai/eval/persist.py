"""评测 run 持久化：runs 目录与 artifact 写入的单一 seam。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.eval.views import EvalRunView, parse_eval_run


def runs_dir(settings: Settings, *, mkdir: bool = False) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_json(settings: Settings, filename: str, payload: dict[str, Any]) -> Path:
    """Write any JSON under the eval runs directory (EvalRun or Phase B summary)."""
    output_path = runs_dir(settings, mkdir=True) / filename
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def persist_eval_artifact(settings: Settings, artifact: dict[str, Any]) -> EvalRunView:
    """Write EvalRun artifact and return the typed view."""
    timestamp = artifact.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("eval artifact missing timestamp")
    write_run_json(settings, f"{timestamp}.json", artifact)
    parsed = parse_eval_run(artifact)
    if parsed is None:
        raise RuntimeError("eval run artifact could not be parsed")
    return parsed


__all__ = ["persist_eval_artifact", "runs_dir", "write_run_json"]
