from __future__ import annotations

import json
import threading
from pathlib import Path

from wenmai.tracing.context import TraceContext


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, trace: TraceContext) -> None:
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
