from __future__ import annotations

from lab_knowledge.components.bm25.index import Bm25Index
from lab_knowledge.components.bm25.tokenizer import ChineseTokenizer
from lab_knowledge.config import Settings
from lab_knowledge.storage.paths import collection_storage_bindings


def create(settings: Settings) -> Bm25Index:
    bindings = collection_storage_bindings(settings)
    tokenizer = ChineseTokenizer.from_settings(settings)
    return Bm25Index(
        collection=bindings.collection_id,
        persist_path=bindings.bm25_path,
        tokenizer=tokenizer,
        k1=settings.bm25.k1,
        b=settings.bm25.b,
    )
