from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQueryRewriter(ABC):
    """Returns extra query strings for retrieval; never includes the original."""

    provider_name: str = "local"

    @abstractmethod
    def extra_queries(self, query: str) -> list[str]:
        """Zero or more additional query strings to search alongside the original."""
