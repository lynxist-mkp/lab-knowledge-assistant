from __future__ import annotations

from wenmai.components.bm25.index import Bm25Index
from wenmai.components.bm25.tokenizer import ChineseTokenizer
from wenmai.config import Settings
from wenmai.storage.paths import store_path


def create(settings: Settings) -> Bm25Index:
    tokenizer = ChineseTokenizer.from_settings(settings)
    return Bm25Index(
        collection=settings.product.collection,
        persist_path=store_path(settings, "bm25"),
        tokenizer=tokenizer,
        k1=settings.bm25.k1,
        b=settings.bm25.b,
    )
