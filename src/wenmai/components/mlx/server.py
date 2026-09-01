from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_REGISTRY: dict[tuple[str, int, str], MlxVlmServerManager] = {}
_REGISTRY_LOCK = threading.Lock()
_SERVER_READY_PATH = "/health"


@dataclass(frozen=True)
class MlxVlmProcessConfig:
    mlx_python: str
    server_port: int
    server_url: str
    model: str
    resolve_script: str
    idle_timeout_seconds: float
    fallback_model: str | None = None
    reuse_healthy: bool = True
    ready_timeout_seconds: float = 180.0

    @property
    def model_candidates(self) -> list[str]:
        candidates = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            candidates.append(self.fallback_model)
        return candidates


@dataclass
class MlxVlmServerManager:
    """Lazy-start mlx_vlm.server and shut it down after idle timeout."""

    config: MlxVlmProcessConfig
    _process: subprocess.Popen[str] | None = None
    _idle_timer: threading.Timer | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_touch: float = 0.0
    _active_model_path: str | None = None

    @property
    def active_mlx_model(self) -> str | None:
        return self._active_model_path

    def ensure_running(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            if self._process is not None and self._process.poll() is None:
                self._last_touch = time.monotonic()
                return
            if self.config.reuse_healthy and self._health_ok():
                self._last_touch = time.monotonic()
                return
            self._start_server()
            self._last_touch = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._last_touch = time.monotonic()
            self._schedule_idle_shutdown()

    def _resolve_model_path(self, model_id: str) -> str:
        if Path(model_id).exists():
            return str(Path(model_id).resolve())
        script = Path(self.config.resolve_script)
        cmd = [self.config.mlx_python, str(script), model_id]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            excerpt = (result.stderr or result.stdout or "").strip()[:500]
            raise RuntimeError(
                f"ModelScope download failed for {model_id} (exit {result.returncode}): {excerpt}"
            )
        path = result.stdout.strip().splitlines()[-1]
        if not path or not Path(path).is_dir():
            raise RuntimeError(f"ModelScope resolve returned no directory for {model_id}")
        return path

    def _start_server(self) -> None:
        last_error: RuntimeError | None = None
        for model_id in self.config.model_candidates:
            self._active_model_path = None
            try:
                model_path = self._resolve_model_path(model_id)
            except RuntimeError as exc:
                last_error = exc
                continue
            cmd = [
                self.config.mlx_python,
                "-m",
                "mlx_vlm.server",
                "--port",
                str(self.config.server_port),
                "--model",
                model_path,
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                self._wait_until_ready()
                self._active_model_path = model_path
                return
            except RuntimeError as exc:
                last_error = exc
                self._kill_process()

        raise RuntimeError(
            "mlx_vlm.server failed to start with configured models: "
            f"{', '.join(self.config.model_candidates)}"
        ) from last_error

    def _health_ok(self) -> bool:
        url = self.config.server_url.rstrip("/") + _SERVER_READY_PATH
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status < 500
        except (urllib.error.URLError, TimeoutError):
            return False

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.ready_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("mlx_vlm.server exited before becoming ready")
            if self._health_ok():
                return
            time.sleep(0.5)
        raise RuntimeError("mlx_vlm.server did not become ready within timeout")

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_shutdown(self) -> None:
        self._cancel_idle_timer()
        timeout = self.config.idle_timeout_seconds
        self._idle_timer = threading.Timer(timeout, self._shutdown_if_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def force_shutdown(self) -> None:
        """Terminate owned subprocess and any listener still bound to our port."""
        with self._lock:
            self._cancel_idle_timer()
            self._kill_process()
            self._kill_port_listeners()
            self._active_model_path = None
            self._idle_timer = None

    def _kill_port_listeners(self) -> None:
        port = self.config.server_port
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return
        for token in result.stdout.strip().split():
            if not token.isdigit():
                continue
            pid = int(token)
            try:
                subprocess.run(["kill", "-TERM", str(pid)], check=False)
            except OSError:
                continue

    def _kill_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._active_model_path = None

    def _shutdown_if_idle(self) -> None:
        with self._lock:
            if time.monotonic() - self._last_touch < self.config.idle_timeout_seconds:
                self._schedule_idle_shutdown()
                return
            self._kill_process()
            self._idle_timer = None


def get_mlx_vlm_manager(config: MlxVlmProcessConfig) -> MlxVlmServerManager:
    key = (config.mlx_python, config.server_port, config.model)
    with _REGISTRY_LOCK:
        manager = _REGISTRY.get(key)
        if manager is None:
            manager = MlxVlmServerManager(config=config)
            _REGISTRY[key] = manager
        return manager


def shutdown_all_mlx_vlm_managers() -> None:
    with _REGISTRY_LOCK:
        managers = list(_REGISTRY.values())
    for manager in managers:
        manager.force_shutdown()


def _register_model_guard_unload() -> None:
    from wenmai.components.model_guard import ModelResource, register_unload

    register_unload(ModelResource.MLX_VLM, shutdown_all_mlx_vlm_managers)


_register_model_guard_unload()
