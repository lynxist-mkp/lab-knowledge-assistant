from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from lab_knowledge.config import Dolphin

SubprocessRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

_STDERR_EXCERPT_CHARS = 500


def extract_transcript(payload: dict) -> str:
    """Join segment text from Dolphin JSON, using only text_nospecial."""
    segments = payload.get("segments") or []
    return "".join(segment["text_nospecial"] for segment in segments)


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _build_command(audio_path: Path, output_path: Path, config: Dolphin) -> list[str]:
    return [
        "env",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONPATH",
        config.python,
        config.script,
        str(audio_path),
        "-o",
        str(output_path),
        "--model",
        config.model,
        "--lang",
        config.lang,
        "--region",
        config.region,
    ]


def _stderr_excerpt(stderr: str) -> str:
    text = stderr.strip()
    if len(text) <= _STDERR_EXCERPT_CHARS:
        return text
    return text[:_STDERR_EXCERPT_CHARS] + "…"


def transcribe_audio(
    audio_path: str | Path,
    *,
    config: Dolphin,
    runner: SubprocessRunner | None = None,
) -> str:
    """Run Dolphin via subprocess and return concatenated segment text."""
    audio = Path(audio_path)
    run = runner or _default_runner

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = run(_build_command(audio, output_path, config))
        if result.returncode != 0:
            excerpt = _stderr_excerpt(result.stderr or "")
            detail = f" (stderr: {excerpt})" if excerpt else ""
            raise RuntimeError(
                f"Dolphin transcription failed with exit code {result.returncode}{detail}"
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return extract_transcript(payload)
    finally:
        output_path.unlink(missing_ok=True)
