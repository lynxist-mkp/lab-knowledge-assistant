"""Smoke checks for scripts/stop_server.sh."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stop_server.sh"


def test_stop_server_script_exists() -> None:
    assert SCRIPT.is_file()


def test_stop_server_script_shebang_and_safety() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in text


def test_stop_server_script_contains_kill_patterns() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "uvicorn (wenmai|lab_knowledge)\\.app:create_app" in text
    assert "uvicorn (wenmai|lab_knowledge)\\.app:app" in text
    assert "mlx_vlm.server" in text
    assert "8120" in text
    assert "8111" in text
    assert "lsof -ti" in text
