# ADR 0007: Round-3 深 module 收口

**Status:** accepted  
**Date:** 2026-09-03  
**Related:** [ADR 0006 深 module 收口](0006-deep-module-consolidation.md), [ADR 0005 评测共用生成前扩展](0005-eval-shared-generation-prep.md)

## Context

ADR-0006 之后仍有浅 wrapper 与 Trace/政策泄漏：**入库准入**的 peek/Trace 载荷散在编排；`retrieve()` 几乎等于 `run_fusion`；精排错位在检索包；邻块扩展与生成拆开；评测对外导出一袋 metrics；运维读侧与概览浅透传；语料批处理二次编排。

## Decision

1. **入库准入** `admit(path) → AdmissionOutcome`（含 Trace stage 载荷）；编排只 append 并按决策分支。
2. **检索** interface 为 `run_fusion`；删除浅 `retrieve()`；`rerank_chunks` 迁至 `pipelines/rerank.py`（提问编排 Phase 3）。
3. **生成** `generate_with_context`：扩展 + LLM + 拒答/出处；删除 `prepare_generation_context`；ask/评测共用（延续 ADR-0005）。
4. **评测** 对外只留 run / rewrite_compare / list / dashboard（+黄金集）；metrics 为 package-private。
5. **运维观测** 合一 Trace 列表/详情/降级与概览；删除 `build_overview_stats`。
6. 语料批处理委托 `run_prepare_commit_batch`，只保留 manifest 解析与记账。

## Consequences

- 测试与导航以 glossary 术语 + 上表 interface 为准。
- 精排不得重回 fusion（ADR-0005/0006 延续）。
