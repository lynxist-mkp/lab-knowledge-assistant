"""Read-path resolution for per-collection storage with legacy fallback.

Rules are specified in ADR 0011 (multi-collection storage layout). Writes always
target the new per-collection paths from ``collection_storage_bindings()``;
this module only decides which on-disk location to read.
"""

from __future__ import annotations

from pathlib import Path


def path_exists(path: Path) -> bool:
    return path.exists()


def dir_has_entries(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def resolve_read_path(
    new_path: Path,
    legacy_path: Path,
    *,
    allow_legacy_fallback: bool,
) -> Path:
    """Prefer new layout; fall back to legacy only for the deployment default collection."""
    if path_exists(new_path):
        return new_path
    if allow_legacy_fallback and path_exists(legacy_path):
        return legacy_path
    return new_path


def resolve_chroma_persist_path(
    new_path: Path,
    legacy_path: Path,
    *,
    allow_legacy_fallback: bool,
) -> Path:
    """Chroma uses a directory; treat non-empty dirs as present storage."""
    if dir_has_entries(new_path):
        return new_path
    if allow_legacy_fallback and dir_has_entries(legacy_path):
        return legacy_path
    return new_path


__all__ = [
    "dir_has_entries",
    "path_exists",
    "resolve_chroma_persist_path",
    "resolve_read_path",
]
