from __future__ import annotations

from dataclasses import dataclass

from lab_knowledge.components.model_guard import configure
from lab_knowledge.config import Settings
from lab_knowledge.factories.loader import ensure_providers
from lab_knowledge.knowledge import Knowledge, create_knowledge


@dataclass
class Runtime:
    """Shared composition root for HTTP and MCP entrypoints."""

    settings: Settings
    knowledge: Knowledge


def create_runtime(settings: Settings | None = None) -> Runtime:
    ensure_providers()
    resolved = settings or Settings.load()
    configure(exclusive=resolved.resources.single_model_exclusive)
    return Runtime(settings=resolved, knowledge=create_knowledge(resolved))
