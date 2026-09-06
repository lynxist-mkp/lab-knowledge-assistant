"""Golden-set loader seam: path in, typed items out for the eval pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lab_knowledge.config import Settings
from lab_knowledge.eval.golden import (
    CATEGORIES,
    GoldItem,
    corpus_id_from_source_path,
    load_golden_set,
    load_golden_set_from_settings,
)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loader_reads_question_evidence_answerable_and_reference(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"朱熹出生于哪一年？","evidence_doc_ids":["zhuzi-wiki-zhuxi"],'
            '"answerable":true,"reference_answer":"1130年。","category":"单跳事实"}',
        ],
    )

    items = load_golden_set(path)

    assert items == [
        GoldItem(
            id="g001",
            question="朱熹出生于哪一年？",
            evidence_doc_ids=["zhuzi-wiki-zhuxi"],
            answerable=True,
            reference_answer="1130年。",
            category="单跳事实",
        )
    ]


def test_unanswerable_item_has_empty_evidence_ids(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g040","question":"林默娘出生于哪一年？","evidence_doc_ids":[],'
            '"answerable":false,"reference_answer":"","category":"明确不可答"}',
        ],
    )

    items = load_golden_set(path)

    assert len(items) == 1
    assert items[0].answerable is False
    assert items[0].evidence_doc_ids == []
    assert items[0].category == "明确不可答"


def test_image_placeholders_are_skipped_by_default(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"泉州别称是什么？","evidence_doc_ids":["haishi-wiki-citong"],'
            '"answerable":true,"reference_answer":"刺桐城。","category":"单跳事实"}',
            '{"id":"g043","question":"","evidence_doc_ids":[],"answerable":true,'
            '"reference_answer":"","category":"图内信息","status":"placeholder"}',
        ],
    )

    skipped = load_golden_set(path)
    included = load_golden_set(path, include_placeholders=True)

    assert [item.id for item in skipped] == ["g001"]
    assert [item.id for item in included] == ["g001", "g043"]
    assert included[1].status == "placeholder"


def test_loader_reads_settings_golden_set_path(test_settings: Settings, tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"马尾区隶属哪个市？","evidence_doc_ids":["chuanzheng-wiki-mawei"],'
            '"answerable":true,"reference_answer":"福州市。","category":"单跳事实"}',
        ],
    )
    test_settings.evaluation.golden_set = str(golden)

    items = load_golden_set_from_settings(test_settings)

    assert len(items) == 1
    assert items[0].evidence_doc_ids == ["chuanzheng-wiki-mawei"]


def test_malformed_line_raises_with_line_number(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", ["{not json"])

    with pytest.raises(ValueError, match="line 1"):
        load_golden_set(path)


def test_corpus_id_matches_manifest_item_id_from_source_path() -> None:
    assert (
        corpus_id_from_source_path("/tmp/items/zhuzi-wiki-zhuxi.md") == "zhuzi-wiki-zhuxi"
    )


def _is_short_or_colloquial(question: str) -> bool:
    cjk = sum(1 for ch in question if "\u4e00" <= ch <= "\u9fff")
    if cjk <= 12:
        return True
    colloquial_markers = ("啥", "哪年", "是谁", "咋", "叫啥", "几时", "多大", "在哪")
    return any(marker in question for marker in colloquial_markers)


def test_committed_golden_set_has_hundred_items_and_four_categories() -> None:
    repo = Path(__file__).resolve().parents[1]
    items = load_golden_set(repo / "data" / "eval" / "golden.jsonl", include_placeholders=True)

    assert len(items) == 100
    present = {item.category for item in items}
    assert present == set(CATEGORIES)

    unanswerable = [item for item in items if item.category == "明确不可答"]
    assert len(unanswerable) >= 8
    assert all(not item.answerable and item.evidence_doc_ids == [] for item in unanswerable)

    ready = [item for item in items if item.status != "placeholder"]
    assert all(item.question for item in ready)
    answerable = [item for item in ready if item.answerable]
    assert all(item.evidence_doc_ids and item.reference_answer for item in answerable)

    images = [item for item in items if item.category == "图内信息"]
    assert len(images) == 8
    assert all(item.status == "ready" for item in images)

    new_batch = [item for item in ready if item.id >= "g051"]
    short_colloquial = [item for item in new_batch if _is_short_or_colloquial(item.question)]
    assert len(short_colloquial) >= 10

    manifest_path = repo / "data" / "corpus" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    known = {entry["id"] for entry in manifest["items"]}
    for item in items:
        for doc_id in item.evidence_doc_ids:
            assert doc_id in known, f"{item.id} cites unknown corpus id {doc_id}"
