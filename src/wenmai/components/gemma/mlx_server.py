from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from wenmai.config import Gemma

_REGISTRY: dict[tuple[str, int, str], GemmaMlxServerManager] = {}
_REGISTRY_LOCK = threading.Lock()
_SERVER_READY_PATH = "/health"


@dataclass
class GemmaMlxServerManager:
    """Lazy-start mlx_vlm.server for local Gemma and shut down after idle timeout."""

    config: Gemma
    _process: subprocess.Popen[str] | None = None
    _idle_timer: threading.Timer | None = None
    _lock: threading.Lock = threading.Lock()
    _last_touch: float = 0.0
    _active_model_path: str | None = None

    def ensure_running(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            if self._process is None or self._process.poll() is not None:
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
        model_path = self._resolve_model_path(self.config.model)
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
        except RuntimeError:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
            self._active_model_path = None
            raise

    def _wait_until_ready(self, timeout_seconds: float = 180.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = self.config.server_url.rstrip("/") + _SERVER_READY_PATH
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("mlx_vlm.server exited before becoming ready")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status < 500:
                        return
            except (urllib.error.URLError, TimeoutError):
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

    def _shutdown_if_idle(self) -> None:
        with self._lock:
            if time.monotonic() - self._last_touch < self.config.idle_timeout_seconds:
                self._schedule_idle_shutdown()
                return
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
            self._active_model_path = None
            self._idle_timer = None


def get_gemma_server_manager(config: Gemma) -> GemmaMlxServerManager:
    key = (config.mlx_python, config.server_port, config.model)
    with _REGISTRY_LOCK:
        manager = _REGISTRY.get(key)
        if manager is None:
            manager = GemmaMlxServerManager(config=config)
            _REGISTRY[key] = manager
        return manager
