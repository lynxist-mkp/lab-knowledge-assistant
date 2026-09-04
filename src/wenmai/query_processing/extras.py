"""提问预处理：术语归一 + Multi-Query 的深 module。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from wenmai.config import Settings
from wenmai.factories import query_rewrite as query_rewrite_factory
from wenmai.query_processing import multi_query


@dataclass(frozen=True)
class QueryExtras:
    term_extras: list[str]
    multi_query_extras: list[str]
    rewriter_provider_name: str
    extras_elapsed_ms: float

    @property
    def combined(self) -> list[str]:
        return self.term_extras + self.multi_query_extras


def prepare_query_extras(question: str, settings: Settings) -> QueryExtras:
    """Collect lexicon rewrites and Multi-Query paraphrases for retrieval + Trace."""
    started = time.perf_counter()
    rewriter = query_rewrite_factory.create(settings)
    term_extras = rewriter.extra_queries(question)
    mq_extras = multi_query.expand(question, settings)
    return QueryExtras(
        term_extras=term_extras,
        multi_query_extras=mq_extras,
        rewriter_provider_name=rewriter.provider_name,
        extras_elapsed_ms=(time.perf_counter() - started) * 1000,
    )


__all__ = ["QueryExtras", "prepare_query_extras"]
