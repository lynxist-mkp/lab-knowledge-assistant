from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUERY_STAGE_ORDER = (
    "query_processing",
    "dense",
    "sparse",
    "fusion",
    "rerank",
    "generation",
)

QUERY_LABELS: dict[str, str] = {
    "query_processing": "查询处理",
    "dense": "嵌入检索",
    "sparse": "稀疏检索",
    "fusion": "融合",
    "rerank": "精排",
    "generation": "生成",
}

INGESTION_LABELS: dict[str, str] = {
    "quality_gate": "入库质量门",
    "load": "读取",
    "integrity": "跳过判定",
    "split": "切分",
    "transform": "清洗",
    "enricher": "补元数据",
    "captioner": "图转文",
    "embed": "嵌入",
    "upsert": "写入",
}

DEGRADATION_LABELS: dict[str, str] = {
    "enricher": "补元数据",
    "captioner": "图转文",
    "refiner": "清洗",
    "load": "读取",
    "pipeline": "入库",
}


def stage_label(name: str, labels: dict[str, str]) -> str:
    return labels.get(name, "步骤")


@dataclass
class StepRow:
    label: str
    method: str
    provider: str
    elapsed_ms: float
    output: str
    error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "method": self.method,
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
            "output": self.output,
            "error": self.error,
        }


def steps_from_record(record: dict[str, Any], labels: dict[str, str]) -> list[StepRow]:
    rows: list[StepRow] = []
    for stage in record.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "")
        rows.append(
            StepRow(
                label=stage_label(name, labels),
                method=str(stage.get("method") or ""),
                provider=str(stage.get("provider") or ""),
                elapsed_ms=float(stage.get("elapsed_ms") or 0.0),
                output=str(stage.get("output_summary") or ""),
                error=stage.get("error") if isinstance(stage.get("error"), str) else None,
            )
        )
    return rows
