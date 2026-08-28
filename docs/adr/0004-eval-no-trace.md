# ADR 0004: 评测不写 Query Trace

**Status:** accepted  
**Date:** 2026-08-28  
**Parent:** [黄金集评测 #18](https://github.com/lynxist-mkp/wenmai-assistant/issues/18)  
**Issue:** [#24 评测分路径且不写 Trace](https://github.com/lynxist-mkp/wenmai-assistant/issues/24)

## Context

黄金集评测会对每条标注题、每个消融组反复跑完整提问链路。原先 `run_eval` 对每条题调用 `ask_question()`，在 `finally` 里无条件 `save_trace`，导致运维追踪里混入大量评测产生的 query trace，污染真实编辑提问的观测数据，也让 trace 文件体积无意义膨胀。

评测需要的是可复现的指标与调试用的检索快照（排序后的 corpus doc id、chunk 摘要），而不是把每次评测当成一次生产提问去落盘。

## Decision

1. **评测分两条路径**：
   - **Hit@5 / MRR**：直接调用 `retrieve()`，不经过生成 LLM。
   - **拒答准确率 / 出处覆盖率**：调用 `ask_question(record_trace=False)`，仍走生成以得到 `refused` 与 `citations`。
2. **评测全程不写 query trace**：`record_trace=False` 时跳过 `save_trace`；编辑工作台与 `/ask` 默认行为不变（仍写 trace）。
3. **结果只进评测 artifact**：每组除聚合指标外，附带 per-item 的 `retrieval` 快照（`ranked_doc_ids`、`ranked_chunks`）及生成侧 `refused` / `citation_count`，供离线排查。
4. **消融配置解析共用**：两条路径均通过 `resolve_ablation()` 取得 `retrieval_mode` 与 `rerank_enabled`。

## Consequences

- 运维追踪只反映真实用户/编辑提问，评测不再制造噪音。
- Hit@5/MRR 与生成解耦：生成失败时检索指标仍可计算（artifact 中保留检索快照）。
- 每条 golden × 每组会执行两次检索（直连 `retrieve` + `ask_question` 内检索），换取路径清晰与 trace 隔离；若后续成为瓶颈可再优化为单次检索复用。
- 需接受：评测 artifact 是检索调试的唯一权威来源，不能指望在 trace 文件里回放评测 run。
