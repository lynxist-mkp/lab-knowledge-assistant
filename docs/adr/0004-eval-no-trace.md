# ADR 0004: 评测不写 Query Trace

**Status:** accepted  
**Date:** 2026-08-28  
**Parent:** [黄金集评测 #18](https://github.com/lynxist-mkp/wenmai-assistant/issues/18)  
**Issue:** [#24 评测分路径且不写 Trace](https://github.com/lynxist-mkp/wenmai-assistant/issues/24)  
**Updated:** [#29 评测单次检索](https://github.com/lynxist-mkp/wenmai-assistant/issues/29), [ADR 0005 评测与生产共用生成前扩展](0005-eval-shared-generation-prep.md)

## Context

黄金集评测会对每条标注题、每个消融组反复跑完整提问链路。原先评测经 `ask_question()`，在 `finally` 里无条件 `save_trace`，导致运维追踪里混入大量评测产生的 query trace，污染真实编辑提问的观测数据，也让 trace 文件体积无意义膨胀。

评测需要的是可复现的指标与调试用的检索快照（排序后的 corpus doc id、chunk 摘要），而不是把每次评测当成一次生产提问去落盘。

## Decision

1. **评测经编排、不写 trace**：`eval/pipeline.py` 的 `run_eval_group_batched` 调用 `run_eval_works`（**提问编排**），与 `/ask` 共享四阶段批处理；全程不调用 `QueryTrace.save`。
   - **Hit@5 / MRR**：由检索结果计算。
   - **拒答准确率 / 出处覆盖率**：由生成结果得到 `refused` 与 `citations`。
   - **零 chunks**：仍调用 `generate()`，其内部短路为 `insufficient_evidence` 拒答，不调用 LLM。
2. **编辑工作台与 `/ask` 默认写 trace**：`ask_question` → `run_ask_works` 在 `record_trace=True` 时经 `QueryTrace` 落盘。
3. **结果只进评测 artifact**：每组除聚合指标外，附带 per-item 的 `retrieval` 快照（`ranked_doc_ids`、`ranked_chunks`）及生成侧 `refused` / `citation_count`，供离线排查。
4. **消融配置解析共用**：检索与生成均通过 `resolve_ablation()` 取得 `retrieval_mode` 与 `rerank_enabled`。
5. **重试不重复检索**：生成失败重试时复用已成功的检索结果，每条题每组至多一次 `retrieve()`。

## Consequences

- 运维追踪只反映真实用户/编辑提问，评测不再制造噪音。
- Hit@5/MRR 与生成解耦：生成失败时检索指标仍可计算（artifact 中保留检索快照）。
- 每条 golden × 每组只执行一次检索，评测耗时与向量库负载减半；生成失败重试不再触发二次检索。
- 需接受：评测 artifact 是检索调试的唯一权威来源，不能指望在 trace 文件里回放评测 run。
- ADR 0005 删除 `eval_item()` 后，本 ADR 的「不写 trace」约束由 `run_eval_works` 路径继承，决策不变。
