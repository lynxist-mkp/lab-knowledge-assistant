"""Bounded-concurrency governor for the 提问服务 seam."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from wenmai.ops.observation import has_running_long_tasks

if TYPE_CHECKING:
    from wenmai.config import AskConcurrencyConfig, Settings

AskSaturationCode = Literal["busy", "timeout", "long_task_active"]


@dataclass(frozen=True)
class AskGovernorSnapshot:
    in_flight: int
    max_in_flight: int
    total_acquired: int
    total_rejected_busy: int
    total_rejected_timeout: int
    total_rejected_long_task: int
    long_task_active: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "in_flight": self.in_flight,
            "max_in_flight": self.max_in_flight,
            "total_acquired": self.total_acquired,
            "total_rejected_busy": self.total_rejected_busy,
            "total_rejected_timeout": self.total_rejected_timeout,
            "total_rejected_long_task": self.total_rejected_long_task,
            "long_task_active": self.long_task_active,
        }


class AskSaturationError(Exception):
    """Raised when the ask governor cannot grant a slot within policy."""

    def __init__(
        self,
        code: AskSaturationCode,
        message: str,
        *,
        in_flight: int = 0,
        max_in_flight: int = 0,
        wait_ms: float = 0.0,
        entrypoint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.in_flight = in_flight
        self.max_in_flight = max_in_flight
        self.wait_ms = wait_ms
        self.entrypoint = entrypoint

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "in_flight": self.in_flight,
            "max_in_flight": self.max_in_flight,
            "wait_ms": self.wait_ms,
            "entrypoint": self.entrypoint,
        }


class AskConcurrencyGovernor:
    def __init__(self, config: AskConcurrencyConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._in_flight = 0
        self._total_acquired = 0
        self._total_rejected_busy = 0
        self._total_rejected_timeout = 0
        self._total_rejected_long_task = 0

    def snapshot(
        self,
        settings: Settings,
        *,
        collection_id: str | None = None,
    ) -> AskGovernorSnapshot:
        max_in_flight = self._effective_max_in_flight(
            settings,
            collection_id=collection_id,
        )
        long_task_active = has_running_long_tasks(
            settings,
            collection_id=collection_id,
        )
        with self._lock:
            return AskGovernorSnapshot(
                in_flight=self._in_flight,
                max_in_flight=max_in_flight,
                total_acquired=self._total_acquired,
                total_rejected_busy=self._total_rejected_busy,
                total_rejected_timeout=self._total_rejected_timeout,
                total_rejected_long_task=self._total_rejected_long_task,
                long_task_active=long_task_active,
            )

    @contextmanager
    def acquire(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        collection_id: str | None = None,
    ) -> Iterator[None]:
        config = settings.resources.ask
        if not config.enabled:
            yield
            return

        max_in_flight = self._effective_max_in_flight(
            settings,
            collection_id=collection_id,
        )
        if max_in_flight <= 0:
            self._reject_long_task(
                settings,
                entrypoint=entrypoint,
                max_in_flight=max_in_flight,
            )

        policy = config.saturation_policy.strip().lower()
        if policy == "busy":
            self._acquire_immediate(
                settings,
                entrypoint=entrypoint,
                max_in_flight=max_in_flight,
            )
            try:
                yield
            finally:
                self._release(settings)
            return

        self._acquire_with_wait(
            settings,
            entrypoint=entrypoint,
            max_in_flight=max_in_flight,
        )
        try:
            yield
        finally:
            self._release(settings)

    def _effective_max_in_flight(
        self,
        settings: Settings,
        *,
        collection_id: str | None = None,
    ) -> int:
        config = settings.resources.ask
        if config.long_task_guard and has_running_long_tasks(
            settings,
            collection_id=collection_id,
        ):
            return min(config.max_in_flight, config.long_task_max_in_flight)
        return config.max_in_flight

    def _acquire_immediate(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        max_in_flight: int,
    ) -> None:
        with self._lock:
            if self._in_flight >= max_in_flight:
                self._reject_busy(
                    settings,
                    entrypoint=entrypoint,
                    max_in_flight=max_in_flight,
                    wait_ms=0.0,
                )
            self._in_flight += 1
            self._total_acquired += 1

    def _acquire_with_wait(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        max_in_flight: int,
    ) -> None:
        config = settings.resources.ask
        wait_started = time.monotonic()
        deadline = wait_started + config.max_wait_seconds
        while True:
            with self._lock:
                if self._in_flight < max_in_flight:
                    self._in_flight += 1
                    self._total_acquired += 1
                    return
            if time.monotonic() >= deadline:
                wait_ms = (time.monotonic() - wait_started) * 1000
                self._reject_timeout(
                    settings,
                    entrypoint=entrypoint,
                    max_in_flight=max_in_flight,
                    wait_ms=wait_ms,
                )
            time.sleep(0.05)

    def _release(self, settings: Settings) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def _reject_busy(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        max_in_flight: int,
        wait_ms: float,
    ) -> None:
        with self._lock:
            self._total_rejected_busy += 1
            in_flight = self._in_flight
        _record_saturation(
            settings,
            code="busy",
            entrypoint=entrypoint,
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=wait_ms,
        )
        raise AskSaturationError(
            "busy",
            "提问服务繁忙：当前并发已满，请稍后重试。",
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=wait_ms,
            entrypoint=entrypoint,
        )

    def _reject_timeout(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        max_in_flight: int,
        wait_ms: float,
    ) -> None:
        with self._lock:
            self._total_rejected_timeout += 1
            in_flight = self._in_flight
        _record_saturation(
            settings,
            code="timeout",
            entrypoint=entrypoint,
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=wait_ms,
        )
        raise AskSaturationError(
            "timeout",
            "提问服务等待超时：队列等待超过上限。",
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=wait_ms,
            entrypoint=entrypoint,
        )

    def _reject_long_task(
        self,
        settings: Settings,
        *,
        entrypoint: str,
        max_in_flight: int,
    ) -> None:
        with self._lock:
            self._total_rejected_long_task += 1
            in_flight = self._in_flight
        _record_saturation(
            settings,
            code="long_task_active",
            entrypoint=entrypoint,
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=0.0,
        )
        raise AskSaturationError(
            "long_task_active",
            "长任务运行中：提问路径暂以保护模式限流。",
            in_flight=in_flight,
            max_in_flight=max_in_flight,
            wait_ms=0.0,
            entrypoint=entrypoint,
        )


_governor: AskConcurrencyGovernor | None = None


def get_ask_governor(settings: Settings) -> AskConcurrencyGovernor:
    global _governor
    if _governor is None:
        _governor = AskConcurrencyGovernor(settings.resources.ask)
    return _governor


def reset_ask_governor() -> None:
    global _governor
    _governor = None
def _record_saturation(
    settings: Settings,
    *,
    code: AskSaturationCode,
    entrypoint: str,
    in_flight: int,
    max_in_flight: int,
    wait_ms: float,
) -> None:
    from wenmai.ops.ask_evidence import safe_record_ask_evidence

    safe_record_ask_evidence(
        settings,
        {
            "event": "saturation",
            "code": code,
            "entrypoint": entrypoint,
            "in_flight": in_flight,
            "max_in_flight": max_in_flight,
            "wait_ms": wait_ms,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )

__all__ = [
    "AskConcurrencyGovernor",
    "AskGovernorSnapshot",
    "AskSaturationCode",
    "AskSaturationError",
    "get_ask_governor",
    "reset_ask_governor",
]
