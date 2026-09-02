# ADR 0005: 评测与生产共用生成前扩展

**Status:** accepted  
**Date:** 2026-09-01  
**Parent:** [提问编排 #47](https://github.com/lynxist-mkp/wenmai-assistant/issues/47)  
**Related:** [ADR 0004 评测不写 Query Trace](0004-eval-no-trace.md)

## Context

编辑工作台 `/ask` 在生成前会通过 `expand_for_generation` 将检索命中的 chunk 与同一文档的相邻 chunk 合并，再送入 LLM。黄金集评测（`eval/pipeline.py`）原先在 `run_eval_group_batched` 中跳过该扩展，直接用精排后的 chunks 调用 `generate()`。

这导致评测与生产的生成输入不一致：评测可能低估出处覆盖率、高估拒答准确率，且无法在 artifact 中复现编辑提问时 LLM 实际看到的上下文。

## Decision

1. **统一提问编排**：`pipelines/query_orchestration.py` 提供 `run_ask_works` / `run_eval_works` 双入口（别名 `run_ask` / `run_eval`），共享四阶段批处理（多查询扩展 → 检索 → 精排 → `prepare_generation_context` + 生成）。
2. **生成前扩展统一**：评测与 `/ask` 均经 `prepare_generation_context`（封装 `expand_for_generation`）后再调用 `generate()`。
3. **检索不再内联精排**：`retrieve()` 只返回融合后的 chunks；精排仅在编排 Phase 3 通过 `rerank_chunks` 执行。
4. **评测仍不写 trace**：与 ADR 0004 一致，编排路径不调用 `save_trace`。
5. **删除 `eval_item()` 死代码**：评测只经 `run_eval_group_batched` → `run_eval_works` 批处理入口（ADR 0004 不写 trace 约束不变）。

## Consequences

- 评测生成指标（拒答准确率、出处覆盖率）与生产行为对齐，更有参考价值。
- **历史评测 run 与本次变更后的 run 不可直接对比**：扩展后的 prompt 更宽，指标可能偏移；对比时需重跑基线或标注变更版本。
- Hit@5 / MRR 等检索指标不受扩展影响（仍基于精排后的 ranked chunks 计算）。
- 运维 trace 仍只反映真实编辑提问，评测 artifact 是评测侧调试权威来源。
