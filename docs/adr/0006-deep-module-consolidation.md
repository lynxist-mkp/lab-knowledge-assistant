# ADR 0006: 深 module 收口（Round 2）

**Status:** accepted  
**Date:** 2026-09-02  
**Parent:** [架构加深 #51](https://github.com/lynxist-mkp/wenmai-assistant/pull/51)  
**Related:** [ADR 0005 评测与生产共用生成前扩展](0005-eval-shared-generation-prep.md)

## Context

Round 1 已抽出**提问编排**与**入库质量门**，但若干 seam 仍泄漏：`Knowledge` 读写未分、入库 trace 与 `TraceContext` 交织、批处理入库与单条路径分叉、query trace 读写散落、`retrieve()` 内联精排。

## Decision

1. **知识库 read/write 分离**：`ReadPath`（检索、按文档读取、列出待审）与 `WritePath`（plan/commit、删文档、审阅通过/驳回）对称；`Knowledge` 门面委托两者，`WritePath` 经 `ReadPath` 校验文档存在。
2. **入库编排两阶段统一**：`prepare_ingest` 产出 `PrepareBody` + `PrepareTraceRecorder`；`run_prepare_commit`（单条）与 `run_prepare_commit_batch`（批处理）共用 prepare → commit 序列；`ingest_batch` 委托批处理入口。
3. **入库 Trace 适配器**：`PrepareTraceRecorder` 位于 pipeline seam；底层 `prepare_chunks` 经 protocol 读 `trace_context`，不直接暴露 `TraceContext` 给编排层以外。
4. **提问 Trace 统一入口**：`QueryTrace` 为对外读写 module；删除 `summarize_query_trace` / `query_trace_detail` shim。
5. **检索纯化**：`retrieve()` 只返回融合结果；`rerank_chunks` 仅在**提问编排** Phase 3 执行（ADR 0005 延续）。
6. **编排 Phase 单例**：`ExtrasPhase` / `RetrievalPhase` / `RerankPhase` / `GenerationPhase` 模块级单例，避免重复实例化。

## Consequences

- 测试与 AI 导航以 glossary 术语 + 上表入口为准，不必从 HTTP handler 倒推全链路。
- 批处理与单条入库行为对齐，批处理回归由 `run_prepare_commit_batch` 单点保障。
- `IngestTraceRecorder`、`IngestPrepareBody`、`run_ingest_phases` 等旧名废弃；新代码与文档使用 `PrepareTraceRecorder`、`PrepareBody`、`run_prepare_commit`。
- `TraceRecorder` 仍为 `QueryTrace` 内部实现细节，不作为对外入口。
