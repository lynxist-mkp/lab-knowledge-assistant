from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CollectionStorageBindings:
    collection_id: str
    chroma_path: Path
    bm25_path: Path
    shared_catalog_path: Path
    shared_ingestion_history_path: Path
    images_root: Path
    shared_image_index_path: Path


def collection_storage_bindings(settings: Settings) -> CollectionStorageBindings:
    """Resolve storage paths for the selected collection without changing shared stores."""
    collection_id = settings.product.collection
    return CollectionStorageBindings(
        collection_id=collection_id,
        chroma_path=store_path(settings, "chroma"),
        bm25_path=store_path(settings, "bm25"),
        shared_catalog_path=store_path(settings, "catalog"),
        shared_ingestion_history_path=store_path(settings, "ingestion_history"),
        images_root=store_path(settings, "images") / collection_id,
        shared_image_index_path=store_path(settings, "image_index"),
    )
