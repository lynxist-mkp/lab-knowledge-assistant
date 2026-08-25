from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from wenmai.config import PaddleOCR

_REGISTRY: dict[tuple[str, int, str], MlxVlmServerManager] = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class MlxVlmServerManager:
    """Lazy-start mlx_vlm.server and shut it down after idle timeout."""

    config: PaddleOCR
    _process: subprocess.Popen[str] | None = None
    _idle_timer: threading.Timer | None = None
    _lock: threading.Lock = threading.Lock()
    _last_touch: float = 0.0

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

    def _start_server(self) -> None:
        cmd = [
            self.config.mlx_python,
            "-m",
            "mlx_vlm.server",
            "--port",
            str(self.config.server_port),
            "--model",
            self.config.mlx_model,
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout_seconds: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = self.config.server_url.rstrip("/") + "/"
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
            self._idle_timer = None


def get_mlx_server_manager(config: PaddleOCR) -> MlxVlmServerManager:
    key = (config.mlx_python, config.server_port, config.mlx_model)
    with _REGISTRY_LOCK:
        manager = _REGISTRY.get(key)
        if manager is None:
            manager = MlxVlmServerManager(config=config)
            _REGISTRY[key] = manager
        return manager
