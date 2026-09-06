# ADR 0016: Ingest/Eval 外部入口与 eval runs 集合契约

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0009（集合作用域语义）、ADR 0012（Trace 集合过滤）、ADR 0014（Task progress 集合过滤）

## Context

入库与评测的外部 HTTP 入口仍绑定默认 `Settings`/`Knowledge`；eval runs 共用单一 runs 目录且历史记录缺少 `collection_id`。多集合部署时会出现「settings 已 scoped 但写入仍落默认集合」或评测列表跨集合串读。

## Decision

### 外部 HTTP 入口

同步 `POST /ingest`、流式 `POST /api/ingestion/run`、`GET/POST /api/eval/runs` 接受可选 **query** 参数 `collection_id`（不放 body、不放 header）。

1. 经 `resolve_routable_collection_scope` 解析到 scoped `Settings`。
2. 非默认集合显式路由时创建 scoped `Knowledge`；默认集合复用 `app.state.knowledge`。
3. 未知集合返回 `404` / `collection not found`；源文件不存在仍 `source file not found`。

### Eval runs 持久化

1. 不按集合拆分 runs 物理目录；继续共享 `settings.evaluation.runs`。
2. `persist_eval_artifact` 写入时盖章 `collection_id`（`settings.product.collection`）。
3. 读侧经 `eval_run_belongs_to_collection` / `list_eval_run_summaries(..., collection_id=...)` 过滤；规则与 ADR 0012 Trace 一致：legacy 无字段记录仅在默认集合可见。

### 明确不做

1. 不回填 legacy eval run 的 `collection_id`。
2. 不为非默认集合对 legacy 记录做可见性 fallback。
3. 本轮不扩展 eval dashboard 内部 UI 的集合选择（仅收口 public HTTP seam）。

## Consequences

- 外部自动化可按 `collection_id` 对目标集合入库与触发评测，写入与列表归属一致。
- 默认集合升级后 legacy eval runs 仍可见；多集合列表互不串读。
- 过滤与盖章集中在 `eval/persist.py` 与 `eval/read.py`，HTTP 层只做 scope 解析与 Knowledge 绑定。
