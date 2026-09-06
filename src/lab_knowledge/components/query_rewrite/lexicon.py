from __future__ import annotations

import logging
from pathlib import Path

import yaml

from lab_knowledge.components.query_rewrite.base import BaseQueryRewriter
from lab_knowledge.factories.query_rewrite import registry

logger = logging.getLogger(__name__)


def _apply_longest_match(text: str, surface_to_canonical: dict[str, str]) -> str:
    if not surface_to_canonical:
        return text
    surfaces = sorted(surface_to_canonical.keys(), key=len, reverse=True)
    parts: list[str] = []
    index = 0
    while index < len(text):
        matched = False
        for surface in surfaces:
            if text.startswith(surface, index):
                parts.append(surface_to_canonical[surface])
                index += len(surface)
                matched = True
                break
        if not matched:
            parts.append(text[index])
            index += 1
    return "".join(parts)


def _load_lexicon(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"lexicon must be a mapping, got {type(raw).__name__}")
    mapping: dict[str, str] = {}
    for surface, canonical in raw.items():
        surface_text = str(surface).strip()
        canonical_text = str(canonical).strip()
        if surface_text and canonical_text:
            mapping[surface_text] = canonical_text
    return mapping


@registry.register("lexicon")
class LexiconQueryRewriter(BaseQueryRewriter):
    provider_name = "lexicon"

    def __init__(self, *, lexicon_path: Path) -> None:
        self._lexicon_path = lexicon_path
        self._mapping: dict[str, str] | None = None
        self._load_error: str | None = None

    def _ensure_mapping(self) -> dict[str, str]:
        if self._mapping is not None or self._load_error is not None:
            return self._mapping or {}
        try:
            self._mapping = _load_lexicon(self._lexicon_path)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "failed to load query lexicon %s: %s",
                self._lexicon_path,
                self._load_error,
            )
            self._mapping = {}
        return self._mapping

    def extra_queries(self, query: str) -> list[str]:
        mapping = self._ensure_mapping()
        if not mapping:
            return []
        rewritten = _apply_longest_match(query, mapping)
        if rewritten == query:
            return []
        return [rewritten]
