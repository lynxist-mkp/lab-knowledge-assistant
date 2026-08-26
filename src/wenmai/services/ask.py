from __future__ import annotations

from wenmai.config import Settings
from wenmai.models import AskResult
from wenmai.pipelines.query import ask_question


def ask(
    question: str,
    settings: Settings,
    culture_domain: str | None = None,
) -> AskResult:
    """Ask the knowledge base; shared entry point for HTTP and MCP."""
    return ask_question(question, settings, culture_domain=culture_domain)
