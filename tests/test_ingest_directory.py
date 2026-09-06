"""Bulk directory ingest as personal literature."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.pipelines.ingest_directory import ingest_source_directory, list_ingestable_files


def _write_literature_markdown(path: Path, *, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
culture_domain: 检索增强
---

{body}
""",
        encoding="utf-8",
    )
    return path


def test_list_ingestable_files_collects_nested_md_and_pdf(tmp_path: Path) -> None:
    root = tmp_path / "library"
    _write_literature_markdown(root / "paper-a.md", body="内容 A。")
    _write_literature_markdown(root / "nested" / "paper-b.md", body="内容 B。")
    (root / "notes.txt").write_text("skip me", encoding="utf-8")
    (root / "scan.pdf").write_bytes(b"%PDF-1.4\n")

    files = list_ingestable_files(root)

    assert {path.name for path in files} == {"paper-a.md", "paper-b.md", "scan.pdf"}


def test_directory_ingest_returns_summary_and_personal_literature_metadata(
    test_settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "papers"
    first = _write_literature_markdown(
        root / "Vaswani et al - Attention Is All You Need (2017).md",
        body="Transformer 架构通过自注意力机制建模序列依赖。",
    )
    second = _write_literature_markdown(
        root / "nested" / "Bengio et al - Deep Learning (2015).md",
        body="深度学习通过多层表示学习特征。",
    )

    client = TestClient(create_app(test_settings))
    response = client.post(
        "/ingest",
        json={"source_path": str(root), "source_kind": "personal_literature"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_kind"] == "personal_literature"
    assert body["total"] == 2
    assert body["attempted"] == 2
    assert body["ingested"] == 2
    assert body["skipped"] == 0
    assert body["failed"] == 0
    assert len(body["files"]) == 2
    assert {item["source_path"] for item in body["files"]} == {str(first), str(second)}
    assert all(item["status"] == "ingested" for item in body["files"])
    assert all(item["chunk_count"] >= 1 for item in body["files"])

    knowledge = create_knowledge(test_settings)
    for item in body["files"]:
        chunks = knowledge.get_by_document_id(item["document_id"])
        assert chunks
        assert chunks[0].metadata["source_kind"] == "personal_literature"
        assert chunks[0].metadata["source_label"] == "个人文献库"


def test_directory_ingest_reuses_idempotency_semantics(
    test_settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "papers"
    _write_literature_markdown(root / "paper-one.md", body="第一篇文献内容。")
    _write_literature_markdown(root / "paper-two.md", body="第二篇文献内容。")

    first = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )
    assert first.ingested == 2
    assert first.rebuilt == 0
    assert first.skipped == 0

    second = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )
    assert second.ingested == 0
    assert second.rebuilt == 0
    assert second.skipped == 2
    assert second.failed == 0
    assert all(item.status == "skipped" for item in second.files)


def test_empty_directory_ingest_returns_zero_totals(
    test_settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    result = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )

    assert result.total == 0
    assert result.attempted == 0
    assert result.ingested == 0
    assert result.rebuilt == 0
    assert result.skipped == 0
    assert result.failed == 0
    assert result.files == []


def test_directory_ingest_reports_failed_files_without_stopping_batch(
    test_settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "mixed"
    good = _write_literature_markdown(root / "good.md", body="可用文献内容。")
    bad = root / "broken.pdf"
    bad.write_bytes(b"not-a-pdf")

    result = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )

    assert result.total == 2
    assert result.attempted == 2
    assert result.ingested == 1
    assert result.failed == 1
    outcomes = {item.source_path: item for item in result.files}
    assert outcomes[str(good)].status == "ingested"
    assert outcomes[str(bad)].status == "failed"
    assert outcomes[str(bad)].error


def test_directory_ingest_counts_rebuilt_files_separately(
    test_settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "rebuilt"
    paper = _write_literature_markdown(root / "paper.md", body="第一版文献内容。")

    first = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )
    assert first.ingested == 1
    assert first.rebuilt == 0

    paper.write_text(
        """---
culture_domain: 检索增强
---

第二版文献内容，已经发生变化。
""",
        encoding="utf-8",
    )
    second = ingest_source_directory(
        root,
        test_settings,
        source_kind="personal_literature",
    )
    assert second.ingested == 0
    assert second.rebuilt == 1
    assert second.skipped == 0
    assert second.failed == 0
    assert second.files[0].status == "rebuilt"
