from __future__ import annotations

from lab_knowledge.knowledge import Knowledge
from lab_knowledge.models import Chunk, ScoredChunk


def _chunk_index(chunk: Chunk) -> int:
    index = chunk.metadata.get("chunk_index")
    if isinstance(index, int):
        return index
    suffix = chunk.chunk_id.rsplit(":", 1)[-1]
    return int(suffix)


def _document_chunks(knowledge: Knowledge, document_id: str) -> list[Chunk]:
    chunks = knowledge.get_by_document_id(document_id)
    chunks.sort(key=_chunk_index)
    return chunks


def expand_for_generation(
    hits: list[ScoredChunk],
    knowledge: Knowledge,
    n: int,
) -> tuple[list[ScoredChunk], list[str], list[str]]:
    """Expand hit chunks with ±n same-document neighbors for generation input."""
    original_ids = [item.chunk.chunk_id for item in hits]
    if n <= 0 or not hits:
        return hits, original_ids, list(original_ids)

    doc_cache: dict[str, list[Chunk]] = {}
    expanded: list[ScoredChunk] = []
    included_ids: list[str] = []
    seen_included: set[str] = set()

    for hit in hits:
        doc_id = hit.chunk.document_id
        if doc_id not in doc_cache:
            doc_cache[doc_id] = _document_chunks(knowledge, doc_id)
        doc_chunks = doc_cache[doc_id]
        if not doc_chunks:
            expanded.append(hit)
            if hit.chunk.chunk_id not in seen_included:
                included_ids.append(hit.chunk.chunk_id)
                seen_included.add(hit.chunk.chunk_id)
            continue

        hit_index = _chunk_index(hit.chunk)
        start = max(0, hit_index - n)
        end = min(len(doc_chunks), hit_index + n + 1)
        window = doc_chunks[start:end]

        combined_text = "\n".join(chunk.text.strip() for chunk in window if chunk.text.strip())
        expanded_chunk = Chunk(
            chunk_id=hit.chunk.chunk_id,
            document_id=hit.chunk.document_id,
            text=combined_text,
            embedding=hit.chunk.embedding,
            metadata=dict(hit.chunk.metadata),
        )
        expanded.append(ScoredChunk(chunk=expanded_chunk, score=hit.score))

        for chunk in window:
            if chunk.chunk_id not in seen_included:
                included_ids.append(chunk.chunk_id)
                seen_included.add(chunk.chunk_id)

    return expanded, original_ids, included_ids
