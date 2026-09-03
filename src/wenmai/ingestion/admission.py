"""入库准入 — peek + 入库质量门 + 灰区复判 → 三态决策与 Trace stage 载荷."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from wenmai.config import Settings
from wenmai.ingestion.gray_review import GrayReviewOutcome, run_gray_review
from wenmai.ingestion.quality import (
    QualityGateResult,
    SourcePeek,
    evaluate_quality_gate,
    peek_source,
)
from wenmai.tracing.context import StageRecord
from wenmai.tracing.stages.ingestion import IngestionStage

AdmissionDecision = Literal["rejected", "approved", "pending_review"]


@dataclass(frozen=True)
class AdmissionOutcome:
    decision: AdmissionDecision
    source_peek: SourcePeek
    gate_result: QualityGateResult
    gray_outcome: GrayReviewOutcome | None = None
    gray_elapsed_ms: float | None = None
    stages: tuple[StageRecord, ...] = field(default_factory=tuple)

    @property
    def stamp_pending_chunks(self) -> bool:
        return self.decision == "pending_review"


class AdmissionGate:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def admit(self, path: Path) -> AdmissionOutcome:
        """入库准入 interface：path → 决策 + Trace stage 载荷."""
        peek = peek_source(path, self._settings)
        gate_started = time.perf_counter()
        gate_result = evaluate_quality_gate(
            peek.text,
            self._settings.quality_gate,
            defer_reject=peek.defer_reject,
        )
        gate_elapsed_ms = (time.perf_counter() - gate_started) * 1000

        if gate_result.band == "reject":
            return self._outcome(
                decision="rejected",
                peek=peek,
                gate_result=gate_result,
                gate_elapsed_ms=gate_elapsed_ms,
                path=path,
            )
        if gate_result.band == "approve":
            return self._outcome(
                decision="approved",
                peek=peek,
                gate_result=gate_result,
                gate_elapsed_ms=gate_elapsed_ms,
                path=path,
            )

        if not self._settings.quality_gate.gray_review:
            return self._outcome(
                decision="pending_review",
                peek=peek,
                gate_result=gate_result,
                gate_elapsed_ms=gate_elapsed_ms,
                path=path,
            )

        gray_started = time.perf_counter()
        gray_outcome = run_gray_review(path, peek, self._settings)
        gray_elapsed_ms = (time.perf_counter() - gray_started) * 1000
        if gray_outcome.hard_reject:
            decision: AdmissionDecision = "rejected"
        elif gray_outcome.passed:
            decision = "approved"
        else:
            decision = "pending_review"
        return self._outcome(
            decision=decision,
            peek=peek,
            gate_result=gate_result,
            gate_elapsed_ms=gate_elapsed_ms,
            path=path,
            gray_outcome=gray_outcome,
            gray_elapsed_ms=gray_elapsed_ms,
        )

    def _outcome(
        self,
        *,
        decision: AdmissionDecision,
        peek: SourcePeek,
        gate_result: QualityGateResult,
        gate_elapsed_ms: float,
        path: Path,
        gray_outcome: GrayReviewOutcome | None = None,
        gray_elapsed_ms: float | None = None,
    ) -> AdmissionOutcome:
        stages: list[StageRecord] = [
            IngestionStage.quality_gate(
                elapsed_ms=gate_elapsed_ms,
                path=path,
                gate_result=gate_result,
                peek=peek,
                decision=decision,
                gray_ran=gray_outcome is not None,
                reject_below=self._settings.quality_gate.reject_below,
            )
        ]
        if gray_outcome is not None:
            stages.append(
                IngestionStage.gray_review(
                    provider=gray_outcome.provider,
                    method=gray_outcome.method,
                    elapsed_ms=gray_elapsed_ms or 0.0,
                    output_summary=gray_outcome.output_summary,
                    error=gray_outcome.error,
                )
            )
        return AdmissionOutcome(
            decision=decision,
            source_peek=peek,
            gate_result=gate_result,
            gray_outcome=gray_outcome,
            gray_elapsed_ms=gray_elapsed_ms,
            stages=tuple(stages),
        )
