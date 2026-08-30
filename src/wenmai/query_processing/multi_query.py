from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory

logger = logging.getLogger(__name__)

_MULTI_QUERY_MARKER = "多路改写"
_LINE_NUMBER_PREFIX = re.compile(r"^\d+[\.、\)]\s*")


def _load_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.query_processing.multi_query_prompt
    return prompt_path.read_text(encoding="utf-8")


def _build_prompt(template: str, question: str, n: int) -> str:
    return template.format(question=question, n=n)


def _parse_lines(raw: str, limit: int) -> list[str]:
    lines: list[str] = []
    for line in raw.strip().splitlines():
        cleaned = _LINE_NUMBER_PREFIX.sub("", line.strip())
        if not cleaned:
            continue
        lines.append(cleaned)
        if len(lines) >= limit:
            break
    return lines


def _dedupe_extras(question: str, extras: list[str], limit: int) -> list[str]:
    seen = {question}
    unique: list[str] = []
    for item in extras:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def expand(question: str, settings: Settings) -> list[str]:
    """Return up to multi_query_n LLM paraphrases of question; [] on disable/error."""
    if not settings.query_processing.multi_query:
        return []

    limit = settings.query_processing.multi_query_n
    if limit <= 0:
        return []

    template = _load_prompt(settings)
    prompt = _build_prompt(template, question, limit)
    llm = multimodal_factory.create(settings)
    timeout = settings.query_processing.multi_query_timeout_seconds

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.generate, prompt)
            raw = future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning("multi-query timed out after %ss", timeout)
        return []
    except Exception as exc:
        logger.warning("multi-query failed: %s: %s", type(exc).__name__, exc)
        return []

    paraphrases = _parse_lines(raw, limit)
    return _dedupe_extras(question, paraphrases, limit)


__all__ = ["expand", "_MULTI_QUERY_MARKER"]
