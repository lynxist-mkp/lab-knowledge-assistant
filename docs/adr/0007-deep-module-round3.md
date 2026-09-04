# ADR 0007: 深 module 收口（Round 3）

**Status:** accepted  
**Date:** 2026-09-04  
**Parent:** Round-3 architecture deepening  
**Related:** [ADR 0006 深 module 收口（Round 2）](0006-deep-module-consolidation.md)

## Context

Round 2 收口了 ReadPath/WritePath、入库 prepare/commit、QueryTrace 命名与 retrieve 纯化，但仍有浅 module：运维概览配方散在 app/tests、提问预处理与 Trace 二次建 rewriter、评测 artifact 配方重复、OrchestrationWork 宽袋泄漏到 Trace finalize。

## Decision

1. **运维观测**：`ops/observation.py` 的 `load_overview_stats(settings) -> OverviewStats` 为概览唯一 interface；删除浅 `ops/overview.py`。
2. **提问预处理**：`prepare_query_extras` → `QueryExtras`（extras + rewriter provider + elapsed）；Trace finalize 读 `rewriter_provider_name`，不再二次 `query_rewrite_factory.create`。
3. **评测 run 持久化**：`run_eval` / `run_rewrite_compare` 共用 artifact builder 与单一 runs-dir；`pipeline` 不再 re-export retrieve/rerank。（随后续 PR 落地）
4. **提问 Trace 写 + 提问编排入口**：意图 builders 隐藏 `OrchestrationWork`；`AskTracePayload` 为 QueryTrace 写 seam 输入，禁止 `work: object`。（随后续 PR 落地）

## Consequences

- 运维看板概览测试只打 `load_overview_stats`，不镜像 HTTP 配方。
- 提问预处理与 Trace 共享同一份 rewriter 元数据；ExtrasPhase 不再自行计时。
- 后续 PR 按 评测 →（Trace 写 + builders）补全本 ADR 条目 3–4。
