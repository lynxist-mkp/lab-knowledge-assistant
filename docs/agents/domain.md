# Domain docs

Use this file before exploring code that depends on repo vocabulary, ADRs, or deep-module seams.

## Exploration path

1. Read `CONTEXT.md` at the workspace root.
2. Read the ADRs for the feature area from both ADR directories.
3. If the task crosses a deep-module seam, read the matching glossary term and module-map row before opening implementation files.

If a file is missing, proceed. New glossary terms and ADRs can be created later by `/domain-modeling`.

Completion check: before coding, you should know the canonical term, the governing ADRs, and the public seam you are supposed to call through.

## Repo layout

- `CONTEXT.md`: canonical glossary for agents
- `docs/adr/`: product and infra ADRs
- `docs/agents/`: agent-facing pointers
- `wenmai-assistant/CONTEXT.md`: git-tracked copy of the glossary
- `wenmai-assistant/docs/adr/`: app ADRs
- `wenmai-assistant/src/wenmai/`: application code

The Cursor workspace root is not the git clone; git operations use `wenmai-assistant/`.

## ADR lookup

Search both directories by topic, not by number:

- `docs/adr/`: includes 扫描件 OCR, Gemma MLX, 双面 UI, 入库质量门
- `wenmai-assistant/docs/adr/`: includes 双面 UI copy, 评测不写 Trace, 评测共用生成前扩展, 深 module 收口

Two directories reuse some ADR numbers. When citing one, name it by title as well as number.

## Glossary discipline

When you name a domain concept in output, tests, refactors, or issue titles, use the term defined in `CONTEXT.md`, including that file's Avoid list.

If the concept is missing from the glossary, treat that as a signal: either the wording is invented or the domain model needs to be extended.

## Deep-module seams

Call through the seam's public interface. Do not reach into adapters unless the task is the seam itself.

| Term (`CONTEXT.md`) | Seam | Entry |
| --- | --- | --- |
| **提问编排** | ask + eval 共用四阶段编排 | `pipelines/query_orchestration.py` — `run_ask_works` / `run_eval_works`；builders: `ask_work_from_job` / `eval_work_from_item` / `gen_retry_work` |
| **提问预处理** | 术语归一 + Multi-Query | `query_processing/extras.py` — `prepare_query_extras` |
| **入库编排** | prepare → commit 两阶段入库 | `ingestion/orchestrator.py` + `pipelines/ingestion.py` |
| **知识库** | read/write 门面 | `knowledge/store.py` — `Knowledge` 委托 `ReadPath` / `WritePath` |
| **ReadPath** | 检索与审阅读取 | `knowledge/read.py` |
| **WritePath** | 入库写入与审阅变更 | `knowledge/write.py` |
| **运维观测** | Trace 读 + 概览 | `ops/observation.py` — `load_overview_stats` |
| **评测** | run / 看板 | `eval/runner.py` · `eval/read.py` · `eval/persist.py` |
| **QueryTrace** | 提问 Trace 读写 | `tracing/query_trace.py` |
| **PrepareTraceRecorder** | 入库 Trace 适配 | `tracing/prepare_recorder.py` |
| **检索** | 融合检索（不含精排） | `retrieval/retrieve.py` |

Notes:

- `retrieve()` only returns fused chunks.
- `rerank_chunks` is only called in 提问编排 Phase 3.
- Eval and `/ask` share the same orchestration, but eval does not write Trace.

## ADR conflicts

If your plan or output contradicts an ADR, say so explicitly before proceeding.

Example: `Contradicts ADR-0002 (local Gemma MLX via ModelScope), but worth reopening because ...`
