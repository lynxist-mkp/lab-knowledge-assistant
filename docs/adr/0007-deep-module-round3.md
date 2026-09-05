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
3. **评测 run 持久化**：`persist_eval_artifact` / `runs_dir` / `write_run_json` 为唯一 runs seam；`run_eval` / `run_rewrite_compare` / Phase B 共用；`pipeline` 不再 re-export retrieve/rerank。
4. **提问 Trace 写 + 提问编排入口**：`AskTracePayload` 为 QueryTrace 写 seam；`ask_work_from_job` / `eval_work_from_item` / `gen_retry_work` 隐藏 `OrchestrationWork`（不再对外导出）。
5. **提问服务收口**：HTTP 与 MCP 的对外提问调用经 `http/ask_service.py::run_ask()` 统一收口，再委托到**提问编排**；服务层调用方不直接 import 编排细节。
6. **运维看板服务收口**：`http/ops.py` 的对外读写调用经 `http/ops_service.py::OpsService` 统一收口，再分别委托到 **文档管理** 与 **运维观测**；路由层不直接 import 观测读函数袋。

## Consequences

- 运维看板概览测试只打 `load_overview_stats`，不镜像 HTTP 配方。
- 提问预处理与 Trace 共享同一份 rewriter 元数据；ExtrasPhase 不再自行计时。
- 评测 artifact 配方与 runs 路径只维护一处；看板读与 Phase B 摘要走同一 runs-dir。
- Trace finalize 与编排 work bag 解耦；eval/ask 经意图 builders 进入编排。
- HTTP / MCP 契约调整与编排内部演化分层隔离；`run_ask()` 成为对外提问入口而非临时 helper。
- 运维看板页面契约与更深的观测/文档实现解耦；后续 `ops` 路由调整只改服务层一处。
