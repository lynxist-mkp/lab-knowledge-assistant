from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab_knowledge.config import Settings

CATEGORIES = ("单跳事实", "跨文档", "图内信息", "明确不可答")

_REQUIRED = ("id", "question", "evidence_doc_ids", "answerable", "reference_answer", "category")


@dataclass(frozen=True)
class GoldItem:
    id: str
    question: str
    evidence_doc_ids: list[str]
    answerable: bool
    reference_answer: str
    category: str
    status: str = "ready"
    evidence_quote: str = ""


def corpus_id_from_source_path(source_path: str) -> str:
    """Map an ingested chunk's source_path back to a manifest item id."""
    return Path(source_path).stem


def load_golden_set_from_settings(
    settings: Settings,
    *,
    include_placeholders: bool = False,
) -> list[GoldItem]:
    raw = Path(settings.evaluation.golden_set)
    path = raw if raw.is_absolute() else settings.root / raw
    return load_golden_set(path, include_placeholders=include_placeholders)


def load_golden_set(
    path: Path,
    *,
    include_placeholders: bool = False,
) -> list[GoldItem]:
    items: list[GoldItem] = []
    text = path.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSONL at line {line_no}: expected object")
        item = _parse_item(payload, line_no)
        if item.status == "placeholder" and not include_placeholders:
            continue
        items.append(item)
    return items


def _parse_item(payload: dict[str, Any], line_no: int) -> GoldItem:
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"invalid JSONL at line {line_no}: missing {', '.join(missing)}")

    category = str(payload["category"])
    if category not in CATEGORIES:
        raise ValueError(f"invalid JSONL at line {line_no}: unknown category {category!r}")

    evidence = payload["evidence_doc_ids"]
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise ValueError(
            f"invalid JSONL at line {line_no}: evidence_doc_ids must be a list of strings"
        )

    answerable = payload["answerable"]
    if not isinstance(answerable, bool):
        raise ValueError(f"invalid JSONL at line {line_no}: answerable must be a boolean")

    status = str(payload.get("status") or "ready")
    item = GoldItem(
        id=str(payload["id"]),
        question=str(payload["question"]),
        evidence_doc_ids=list(evidence),
        answerable=answerable,
        reference_answer=str(payload["reference_answer"]),
        category=category,
        status=status,
        evidence_quote=str(payload.get("evidence_quote") or ""),
    )
    _check_consistency(item, line_no)
    return item


def _check_consistency(item: GoldItem, line_no: int) -> None:
    if item.status == "placeholder":
        return
    if item.answerable and not item.evidence_doc_ids:
        raise ValueError(f"invalid JSONL at line {line_no}: answerable item needs evidence_doc_ids")
    if not item.answerable:
        if item.category != "明确不可答":
            raise ValueError(
                f"invalid JSONL at line {line_no}: unanswerable item must use category 明确不可答"
            )
        if item.evidence_doc_ids:
            raise ValueError(
                f"invalid JSONL at line {line_no}: unanswerable item must have empty evidence"
            )
