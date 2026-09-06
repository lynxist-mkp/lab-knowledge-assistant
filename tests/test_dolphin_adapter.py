"""Dolphin adapter narrow seam: audio path in, transcript text out."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lab_knowledge.components.dolphin import transcribe_audio
from lab_knowledge.config import Dolphin

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dolphin"


@pytest.fixture
def dolphin_config() -> Dolphin:
    return Dolphin(
        python="/fake/dolphin-env/bin/python",
        script="/fake/scripts/dolphin_transcribe.py",
        model="small.cn",
        lang="zh",
        region="MINNAN",
    )


def _fixture_runner(fixture_name: str):
    fixture_path = FIXTURES / fixture_name

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        output_idx = cmd.index("-o")
        output_path = Path(cmd[output_idx + 1])
        output_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return runner


def test_transcribe_audio_concatenates_text_nospecial_from_segments(
    dolphin_config: Dolphin, tmp_path: Path
) -> None:
    audio = tmp_path / "demo.mp3"
    audio.write_bytes(b"fake-audio")

    text = transcribe_audio(
        audio,
        config=dolphin_config,
        runner=_fixture_runner("sample_output.json"),
    )

    assert text == "你好世界闽南语测试"


def test_transcribe_audio_never_includes_zh_minnan_markers(
    dolphin_config: Dolphin, tmp_path: Path
) -> None:
    audio = tmp_path / "demo.mp3"
    audio.write_bytes(b"fake-audio")

    text = transcribe_audio(
        audio,
        config=dolphin_config,
        runner=_fixture_runner("sample_output.json"),
    )

    assert "<zh>" not in text
    assert "<MINNAN>" not in text
    assert "notimestamp" not in text


def test_transcribe_audio_raises_clear_error_on_nonzero_exit(
    dolphin_config: Dolphin, tmp_path: Path
) -> None:
    audio = tmp_path / "demo.mp3"
    audio.write_bytes(b"fake-audio")

    def failing_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr="模型加载失败: file not found\nTraceback: ...",
        )

    with pytest.raises(RuntimeError, match="Dolphin transcription failed") as exc_info:
        transcribe_audio(audio, config=dolphin_config, runner=failing_runner)

    message = str(exc_info.value)
    assert "模型加载失败" in message
    assert "exit code 1" in message


def test_transcribe_audio_invokes_subprocess_with_env_strip_and_cli_args(
    dolphin_config: Dolphin, tmp_path: Path
) -> None:
    audio = tmp_path / "long-demo.mp3"
    audio.write_bytes(b"fake-audio")
    captured: list[list[str]] = []

    def capture_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        output_idx = cmd.index("-o")
        output_path = Path(cmd[output_idx + 1])
        output_path.write_text(
            (FIXTURES / "sample_output.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    transcribe_audio(audio, config=dolphin_config, runner=capture_runner)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:5] == ["env", "-u", "PYTHONHOME", "-u", "PYTHONPATH"]
    assert cmd[5] == dolphin_config.python
    assert cmd[6] == dolphin_config.script
    assert cmd[7] == str(audio)
    assert cmd[cmd.index("-o") + 1].endswith(".json")
    assert cmd[cmd.index("--model") + 1] == "small.cn"
    assert cmd[cmd.index("--lang") + 1] == "zh"
    assert cmd[cmd.index("--region") + 1] == "MINNAN"
