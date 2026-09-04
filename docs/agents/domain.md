# Domain Docs

How engineering skills consume this repo's domain documentation when exploring the codebase.

## Before exploring

1. Read **`CONTEXT.md`** (workspace root; `wenmai-assistant/CONTEXT.md` is the git-tracked copy — keep them in sync).
2. Skim ADRs that touch the area you're about to work in (see [ADR locations](#adr-locations)).
3. If the task crosses a **deep module** seam, read its glossary entry and the [module map](#deep-modules) row before opening implementation files.

If a file is missing, proceed. `/domain-modeling` creates glossary entries and ADRs lazily when terms or decisions resolve.

## File structure

```
/
├── CONTEXT.md              ← glossary (canonical for agents)
├── docs/adr/               ← product / infra ADRs (0001–0004)
├── docs/agents/            ← agent pointers (this file)
└── wenmai-assistant/
    ├── CONTEXT.md          ← same glossary, committed with app
    ├── docs/adr/           ← app ADRs (0003–0006; 0004 ≠ root 0004)
    └── src/wenmai/
```

Application code lives under `wenmai-assistant/src/wenmai/`. The Cursor workspace root is not the git clone; git operations use `wenmai-assistant/`.

## ADR locations

Two directories share numbering but not every number — read by topic, not by assuming one sequence:

| Directory | Numbers | Topics |
| --- | --- | --- |
| `docs/adr/` | 0001–0004 | 扫描件 OCR、Gemma MLX、双面 UI、**入库质量门** |
| `wenmai-assistant/docs/adr/` | 0003–0007 | 双面 UI（副本）、**评测不写 Trace**、评测共用生成前扩展、**深 module 收口**（0006/0007） |

When a skill says "check ADRs", search both directories for the feature area.

## Glossary terms

When output names a domain concept (issue title, refactor, hypothesis, test name), use the term as defined in `CONTEXT.md`, including that file's Avoid list.

A concept missing from the glossary is a signal — invented language, or a gap for `/domain-modeling`.

## Deep modules

Domain terms for the main seams. Call through the module's public interface; don't reach past it into adapters unless the task is the seam itself.

| Term (`CONTEXT.md`) | Seam | Entry |
| --- | --- | --- |
| **提问编排** | ask + eval 共用四阶段编排 | `pipelines/query_orchestration.py` — `run_ask_works` / `run_eval_works`（别名 `run_ask` / `run_eval`） |
| **提问预处理** | 术语归一 + Multi-Query | `query_processing/extras.py` — `prepare_query_extras` |
| **入库编排** | prepare → commit 两阶段入库 | `ingestion/orchestrator.py`（`prepare_ingest`）+ `pipelines/ingestion.py`（`run_prepare_commit` / `run_prepare_commit_batch`） |
| **知识库** | read/write 门面 | `knowledge/store.py` — `Knowledge` 委托 `ReadPath` / `WritePath` |
| **ReadPath** | 检索与审阅读取 | `knowledge/read.py` |
| **WritePath** | 入库写入与审阅变更 | `knowledge/write.py` |
| **运维观测** | Trace 读 + 概览 | `ops/observation.py` — `load_overview_stats` |
| **QueryTrace** | 提问 Trace 读写 | `tracing/query_trace.py` |
| **PrepareTraceRecorder** | 入库 Trace 适配 | `tracing/prepare_recorder.py` |
| **检索** | 融合检索（不含精排） | `retrieval/retrieve.py` — 精排仅在**提问编排** Phase 3 |

`retrieve()` 只返回融合 chunks；`rerank_chunks` 只在编排层调用。评测与 `/ask` 共用编排，评测不写 Trace（ADR 0004）。

## ADR conflicts

If output contradicts an existing ADR, surface it:

> _Contradicts ADR-0002 (local Gemma MLX via ModelScope) — but worth reopening because…_

Use the ADR file's title to identify which `0004` you mean when both directories have one.
