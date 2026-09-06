"""Transform registry seam: config providers drive registered stages."""

from __future__ import annotations

from contextlib import contextmanager

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories.transform import registry, run_registered
from wenmai.ingestion.prepare import TransformTraceRecorder, prepare_chunks
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext


class _RecorderStub:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}
        self.stages: list[dict[str, object]] = []

    @contextmanager
    def stage(
        self,
        name: str,
        method: str,
        provider: str,
        input_summary: str = "",
    ):
        stage_info: dict[str, object] = {
            "name": name,
            "method": method,
            "provider": provider,
            "input_summary": input_summary,
        }
        try:
            yield stage_info
        finally:
            self.stages.append(stage_info)


def test_transform_config_restores_provider_keys(test_settings: Settings) -> None:
    assert test_settings.transform.refiner == "rule"
    assert test_settings.transform.enricher == "llm"
    assert test_settings.transform.captioner == "vision"


def test_prepare_chunks_delegates_to_registry(test_settings: Settings) -> None:
    recorder = _RecorderStub()
    chunks = [
        Chunk(
            chunk_id="doc:0001",
            document_id="doc",
            text="湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所。" * 5,
            metadata={},
        )
    ]
    result = prepare_chunks(chunks, test_settings, recorder)
    assert result[0].metadata.get("culture_domain") in test_settings.transform.domains
    assert any(stage.get("name") == "enricher" for stage in recorder.stages)


def test_changing_transform_provider_changes_behavior(test_settings: Settings) -> None:
    @registry.register("refiner.marker")
    class MarkerRefiner(BaseTransform):
        name = "refiner"

        def __init__(self, settings: Settings, **kwargs: object) -> None:
            pass

        def apply(self, chunks: list[Chunk], recorder: TransformTraceRecorder) -> list[Chunk]:
            for chunk in chunks:
                chunk.text = f"MARKED:{chunk.text}"
            return chunks

    test_settings.transform.stages = ["refiner"]
    test_settings.transform.refiner = "marker"
    trace = TraceContext(trace_type="ingestion")
    chunks = [
        Chunk(
            chunk_id="doc:0001",
            document_id="doc",
            text="妈祖信仰发源地。",
            metadata={},
        )
    ]
    result = run_registered(chunks, test_settings, trace)
    assert result[0].text.startswith("MARKED:")
