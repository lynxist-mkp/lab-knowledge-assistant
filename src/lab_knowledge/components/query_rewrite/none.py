from __future__ import annotations

from lab_knowledge.components.query_rewrite.base import BaseQueryRewriter
from lab_knowledge.factories.query_rewrite import registry


@registry.register("none")
class NoOpQueryRewriter(BaseQueryRewriter):
    provider_name = "none"

    def extra_queries(self, query: str) -> list[str]:
        return []
