from __future__ import annotations

from wenmai.components.query_rewrite.base import BaseQueryRewriter
from wenmai.factories.query_rewrite import registry


@registry.register("none")
class NoOpQueryRewriter(BaseQueryRewriter):
    provider_name = "none"

    def extra_queries(self, query: str) -> list[str]:
        return []
