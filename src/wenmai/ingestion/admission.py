"""入库准入 — orchestrates quality gate + gray review into a three-state decision."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wenmai.config import Settings
from wenmai.ingestion.gray_review import GrayReviewOutcome, run_gray_review
from wenmai.ingestion.quality import (
    QualityGateResult,
    SourcePeek,
    evaluate_quality_gate,
)

AdmissionDecision = Literal["rejected", "approved", "pending_review"]


@dataclass(frozen=True)
class AdmissionOutcome:
    decision: AdmissionDecision
    source_peek: SourcePeek
    gate_result: QualityGateResult
    gray_outcome: GrayReviewOutcome | None = None
    gray_elapsed_ms: float | None = None

    @property
    def stamp_pending_chunks(self) -> bool:
        return self.decision == "pending_review"


class AdmissionGate:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def decide(self, path: Path, peek: SourcePeek) -> AdmissionOutcome:
        gate_result = evaluate_quality_gate(
            peek.text,
            self._settings.quality_gate,
            defer_reject=peek.defer_reject,
        )
        if gate_result.band == "reject":
            return AdmissionOutcome(
                decision="rejected",
                source_peek=peek,
                gate_result=gate_result,
            )
        if gate_result.band == "approve":
            return AdmissionOutcome(
                decision="approved",
                source_peek=peek,
                gate_result=gate_result,
            )

        if not self._settings.quality_gate.gray_review:
            return AdmissionOutcome(
                decision="pending_review",
                source_peek=peek,
                gate_result=gate_result,
            )

        gray_started = time.perf_counter()
        gray_outcome = run_gray_review(path, peek, self._settings)
        gray_elapsed_ms = (time.perf_counter() - gray_started) * 1000
        if gray_outcome.hard_reject:
            return AdmissionOutcome(
                decision="rejected",
                source_peek=peek,
                gate_result=gate_result,
                gray_outcome=gray_outcome,
                gray_elapsed_ms=gray_elapsed_ms,
            )
        if gray_outcome.passed:
            return AdmissionOutcome(
                decision="approved",
                source_peek=peek,
                gate_result=gate_result,
                gray_outcome=gray_outcome,
                gray_elapsed_ms=gray_elapsed_ms,
            )
        return AdmissionOutcome(
            decision="pending_review",
            source_peek=peek,
            gate_result=gate_result,
            gray_outcome=gray_outcome,
            gray_elapsed_ms=gray_elapsed_ms,
        )
