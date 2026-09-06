"""mlx_vlm lifecycle: reuse a healthy server; adapters keep separate ports."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lab_knowledge.components.mlx.server import MlxVlmProcessConfig, MlxVlmServerManager
from lab_knowledge.config import Settings


def _config(**overrides: object) -> MlxVlmProcessConfig:
    payload: dict[str, object] = {
        "mlx_python": "python3",
        "server_port": 8111,
        "server_url": "http://127.0.0.1:8111/",
        "model": "primary-model",
        "resolve_script": "/tmp/resolve.py",
        "idle_timeout_seconds": 30,
        "fallback_model": "fallback-model",
        "reuse_healthy": True,
        "ready_timeout_seconds": 5,
    }
    payload.update(overrides)
    return MlxVlmProcessConfig(**payload)  # type: ignore[arg-type]


def test_ensure_running_reuses_healthy_server_without_start() -> None:
    manager = MlxVlmServerManager(config=_config())
    with (
        patch.object(manager, "_health_ok", return_value=True),
        patch.object(manager, "_start_server") as start,
    ):
        manager.ensure_running()
    start.assert_not_called()
    assert manager._process is None


def test_ensure_running_starts_when_unhealthy() -> None:
    manager = MlxVlmServerManager(config=_config())
    with (
        patch.object(manager, "_health_ok", return_value=False),
        patch.object(manager, "_start_server") as start,
    ):
        manager.ensure_running()
    start.assert_called_once()


def test_start_server_tries_fallback_model_when_primary_resolve_fails() -> None:
    manager = MlxVlmServerManager(config=_config())

    def resolve(model_id: str) -> str:
        if model_id == "primary-model":
            raise RuntimeError("missing primary")
        return "/tmp/fallback"

    with (
        patch.object(manager, "_resolve_model_path", side_effect=resolve),
        patch("lab_knowledge.components.mlx.server.subprocess.Popen") as popen,
        patch.object(manager, "_wait_until_ready"),
    ):
        popen.return_value.poll.return_value = None
        manager._start_server()

    assert manager.active_mlx_model == "/tmp/fallback"
    cmd = popen.call_args.args[0]
    assert cmd[-1] == "/tmp/fallback"


def test_force_shutdown_kills_port_listeners_when_process_untracked() -> None:
    manager = MlxVlmServerManager(config=_config())
    with patch.object(manager, "_kill_process") as kill_process:
        with patch.object(manager, "_kill_port_listeners") as kill_port:
            manager.force_shutdown()
    kill_process.assert_called_once()
    kill_port.assert_called_once()


def test_paddle_and_gemma_configs_keep_separate_ports(test_settings: Settings) -> None:
    repo = Path(__file__).resolve().parents[1]
    settings = Settings.load(repo / "settings.yaml")
    gemma = MlxVlmProcessConfig(
        mlx_python=settings.gemma.mlx_python,
        server_port=settings.gemma.server_port,
        server_url=settings.gemma.server_url,
        model=settings.gemma.model,
        resolve_script=settings.gemma.resolve_script,
        idle_timeout_seconds=settings.gemma.idle_timeout_seconds,
        reuse_healthy=True,
        ready_timeout_seconds=180.0,
    )
    paddle = MlxVlmProcessConfig(
        mlx_python=settings.paddleocr.mlx_python,
        server_port=settings.paddleocr.server_port,
        server_url=settings.paddleocr.server_url,
        model=settings.paddleocr.mlx_model,
        resolve_script=str(
            Path(settings.paddleocr.script).resolve().parent / "resolve_modelscope_model.py"
        ),
        idle_timeout_seconds=settings.paddleocr.idle_timeout_seconds,
        fallback_model=settings.paddleocr.mlx_fallback_model,
        reuse_healthy=True,
        ready_timeout_seconds=120.0,
    )
    assert gemma.server_port != paddle.server_port
    assert gemma.reuse_healthy is True
    assert paddle.reuse_healthy is True
    assert paddle.fallback_model
