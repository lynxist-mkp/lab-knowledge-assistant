from __future__ import annotations

from pathlib import Path

from wenmai.config import Settings

_DIR_STORES = frozenset({"chroma", "bm25", "images", "corpus"})


def store_path(settings: Settings, name: str) -> Path:
    """Resolve one of the five stores (or traces) without touching the others."""
    raw = getattr(settings.paths, name)
    path = Path(raw)
    if not path.is_absolute():
        path = settings.root / path
    if name in _DIR_STORES:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
