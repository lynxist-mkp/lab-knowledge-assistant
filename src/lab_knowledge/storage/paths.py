from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lab_knowledge.config import Settings
from lab_knowledge.storage.compat import resolve_chroma_persist_path, resolve_read_path

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


def _collection_store_path(base: Path, collection_id: str, leaf: str) -> Path:
    path = base / collection_id / leaf
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class CollectionStorageBindings:
    collection_id: str
    chroma_path: Path
    bm25_path: Path
    catalog_path: Path
    ingestion_history_path: Path
    images_root: Path
    image_index_path: Path
    legacy_chroma_path: Path
    legacy_catalog_path: Path
    legacy_ingestion_history_path: Path
    legacy_image_index_path: Path
    legacy_read_fallback: bool

    def catalog_read_path(self) -> Path:
        return resolve_read_path(
            self.catalog_path,
            self.legacy_catalog_path,
            allow_legacy_fallback=self.legacy_read_fallback,
        )

    def ingestion_history_read_path(self) -> Path:
        return resolve_read_path(
            self.ingestion_history_path,
            self.legacy_ingestion_history_path,
            allow_legacy_fallback=self.legacy_read_fallback,
        )

    def image_index_read_path(self) -> Path:
        return resolve_read_path(
            self.image_index_path,
            self.legacy_image_index_path,
            allow_legacy_fallback=self.legacy_read_fallback,
        )

    def chroma_persist_path(self) -> Path:
        return resolve_chroma_persist_path(
            self.chroma_path,
            self.legacy_chroma_path,
            allow_legacy_fallback=self.legacy_read_fallback,
        )


def collection_storage_bindings(settings: Settings) -> CollectionStorageBindings:
    """Resolve per-collection storage paths with legacy read fallback for the default collection."""
    collection_id = settings.product.collection
    legacy_read_fallback = collection_id == settings.default_collection_id

    legacy_chroma_path = store_path(settings, "chroma")
    legacy_bm25_root = store_path(settings, "bm25")
    legacy_catalog_path = store_path(settings, "catalog")
    legacy_ingestion_history_path = store_path(settings, "ingestion_history")
    images_root = store_path(settings, "images")
    legacy_image_index_path = store_path(settings, "image_index")

    catalog_base = legacy_catalog_path.parent
    history_base = legacy_ingestion_history_path.parent
    image_index_base = legacy_image_index_path.parent

    return CollectionStorageBindings(
        collection_id=collection_id,
        chroma_path=legacy_chroma_path / collection_id,
        bm25_path=legacy_bm25_root / collection_id,
        catalog_path=_collection_store_path(catalog_base, collection_id, "catalog.json"),
        ingestion_history_path=_collection_store_path(
            history_base, collection_id, "ingestion_history.db"
        ),
        images_root=images_root / collection_id,
        image_index_path=_collection_store_path(image_index_base, collection_id, "image_index.db"),
        legacy_chroma_path=legacy_chroma_path,
        legacy_catalog_path=legacy_catalog_path,
        legacy_ingestion_history_path=legacy_ingestion_history_path,
        legacy_image_index_path=legacy_image_index_path,
        legacy_read_fallback=legacy_read_fallback,
    )
