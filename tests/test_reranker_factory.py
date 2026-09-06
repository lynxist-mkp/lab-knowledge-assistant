"""Reranker factory caches provider instances for phase batching."""

from __future__ import annotations

import pytest

from lab_knowledge.config import Settings
from lab_knowledge.factories import reranker as reranker_factory
from lab_knowledge.models import Chunk, ScoredChunk
from lab_knowledge.retrieval.retrieve import rerank_chunks


def _scored_chunk(index: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"chunk-{index}",
            document_id="doc-1",
            text=f"片段{index}关于妈祖信仰的内容。",
            metadata={"title": f"标题{index}"},
        ),
        score=1.0 - index * 0.01,
    )


@pytest.fixture(autouse=True)
def _clear_reranker_cache() -> None:
    reranker_factory.clear_cache()
    yield
    reranker_factory.clear_cache()


def test_reranker_factory_reuses_instance(test_settings: Settings) -> None:
    first = reranker_factory.create(test_settings)
    second = reranker_factory.create(test_settings)
    assert first is second


def test_rerank_chunks_reuses_cached_reranker(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instantiate_calls = 0
    real_registry_create = reranker_factory.registry.create

    def counting_registry_create(name: str, **kwargs: object):
        nonlocal instantiate_calls
        instantiate_calls += 1
        return real_registry_create(name, **kwargs)

    monkeypatch.setattr(
        reranker_factory.registry, "create", counting_registry_create
    )
    chunks = [_scored_chunk(i) for i in range(3)]

    rerank_chunks(test_settings, "妈祖信仰的发源地在哪里？", chunks)
    rerank_chunks(test_settings, "船政学堂是什么时候创办的？", chunks)

    assert instantiate_calls == 1
