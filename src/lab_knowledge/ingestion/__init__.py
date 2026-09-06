"""文档解析：path → LoadedDocument。入库只经 load_source。"""

from lab_knowledge.ingestion.loaders import LoadedDocument, SourceLoadError, load_source

__all__ = [
    "LoadedDocument",
    "SourceLoadError",
    "load_source",
]
