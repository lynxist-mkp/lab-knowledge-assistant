from __future__ import annotations

from dataclasses import dataclass

from wenmai.config import Settings
from wenmai.factories import query_rewrite as query_rewrite_factory
from wenmai.query_processing import multi_query


@dataclass(frozen=True)
class CollectedExtras:
    term_extras: list[str]
    multi_query_extras: list[str]

    @property
    def combined(self) -> list[str]:
        return self.term_extras + self.multi_query_extras


def collect_extra_queries(question: str, settings: Settings) -> CollectedExtras:
    """Collect lexicon rewrites and Multi-Query paraphrases for retrieval."""
    rewriter = query_rewrite_factory.create(settings)
    term_extras = rewriter.extra_queries(question)
    mq_extras = multi_query.expand(question, settings)
    return CollectedExtras(term_extras=term_extras, multi_query_extras=mq_extras)


__all__ = ["CollectedExtras", "collect_extra_queries"]
