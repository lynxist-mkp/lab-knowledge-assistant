from __future__ import annotations

from pathlib import Path

from lab_knowledge.components.query_rewrite.base import BaseQueryRewriter
from lab_knowledge.config import Settings
from lab_knowledge.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseQueryRewriter] = ProviderRegistry()

_loaded = False


def _ensure_implementations() -> None:
    global _loaded
    if _loaded:
        return
    import lab_knowledge.components.query_rewrite.lexicon  # noqa: F401
    import lab_knowledge.components.query_rewrite.none  # noqa: F401

    _loaded = True


def create(settings: Settings) -> BaseQueryRewriter:
    _ensure_implementations()
    rewriter_name = settings.query_processing.rewriter
    if rewriter_name == "lexicon":
        lexicon_path = Path(settings.query_processing.lexicon)
        if not lexicon_path.is_absolute():
            lexicon_path = settings.root / lexicon_path
        return registry.create("lexicon", lexicon_path=lexicon_path)
    return registry.create(rewriter_name)
