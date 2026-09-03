"""入库准入 — admit(path) three-state decisions before load."""

from __future__ import annotations

from pathlib import Path

from wenmai.config import Settings
from wenmai.ingestion.admission import AdmissionGate


def _write_reject_markdown(path: Path) -> Path:
    body = "@#@$%^&*()!~`" * 80
    path.write_text(
        f"""---
title: 符号噪声材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_approve_markdown(path: Path) -> Path:
    body = "福建船政文化历史悠久，马尾船政学堂培养近代海军人才，见证闽派近代化历程。" * 8
    path.write_text(
        f"""---
title: 船政文化材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_gray_markdown(path: Path, extra: str = "") -> Path:
    prose = "福" * 70
    junk = "@" * 30
    body = prose + extra + junk
    path.write_text(
        f"""---
title: 灰区船政材料
culture_domain: 船政
---

{body}
""",
        encoding="utf-8",
    )
    return path


def test_admission_rejects_low_ratio(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_reject_markdown(tmp_path / "reject.md")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "rejected"
    assert outcome.gray_outcome is None
    assert not outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate"]
    assert outcome.stages[0].error is not None


def test_admission_approves_high_ratio(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_approve_markdown(tmp_path / "approve.md")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "approved"
    assert outcome.gray_outcome is None
    assert not outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate"]


def test_admission_gray_without_review_is_pending_review(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = False
    source = _write_gray_markdown(tmp_path / "gray-off.md")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "pending_review"
    assert outcome.gray_outcome is None
    assert outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate"]


def test_admission_gray_pass_is_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    source = _write_gray_markdown(tmp_path / "gray-pass.md")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "approved"
    assert outcome.gray_outcome is not None
    assert outcome.gray_outcome.passed
    assert not outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate", "gray_review"]


def test_admission_gray_fail_is_pending_review(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    source = _write_gray_markdown(tmp_path / "gray-fail.md", extra="不值得入库")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "pending_review"
    assert outcome.gray_outcome is not None
    assert not outcome.gray_outcome.passed
    assert outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate", "gray_review"]


def test_admission_gray_timeout_is_rejected(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.quality_gate.gray_review = True
    test_settings.quality_gate.timeout_seconds = 1.0
    test_settings.fakes["multimodal"] = "timeout"
    source = _write_gray_markdown(tmp_path / "gray-timeout.md")
    outcome = AdmissionGate(test_settings).admit(source)
    assert outcome.decision == "rejected"
    assert outcome.gray_outcome is not None
    assert outcome.gray_outcome.hard_reject
    assert not outcome.stamp_pending_chunks
    assert [stage.name for stage in outcome.stages] == ["quality_gate", "gray_review"]
