# ADR 0012: Trace 按集合过滤与 legacy 兼容

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0009（集合作用域语义）、ADR 0011（多集合存储布局与 legacy 兼容读取）

## Context

运维观测的目录计数已按 `collection_id` 路由，但 Trace 仍共用单一 JSONL 文件，且历史记录缺少 `collection_id` 字段。若不统一读侧过滤规则，概览延迟、Trace 列表/详情与调查视图会在多集合场景下串读。

Trace 与 catalog/Chroma 等不同：**不按集合拆物理文件**；集合边界由记录级 `collection_id` 与读侧 filter seam 表达。

## Decision

### 写入

1. 新写入 Trace（`save_trace`）必须持久化 `collection_id`，取值为写入时 `settings.product.collection`（scoped settings 视图下的当前集合）。
2. 调用方通过 scoped `Settings` 或显式 `collection_id` 路由到目标集合；Trace 文件本身仍落在 `settings.paths.traces`（共享 JSONL）。

### 读取过滤（`tracing/store.py`）

统一经 `read_trace_records` / `get_trace_record` / `filter_trace_records`：

| 记录 `collection_id` | 默认集合视图 | 非默认集合视图 |
| --- | --- | --- |
| 缺失（legacy） | 可见 | 不可见 |
| 等于目标 `collection_id` | 可见 | 可见 |
| 不等于目标 `collection_id` | 不可见 | 不可见 |

`collection_id=None` 传给读 API 时不做过滤（内部/测试用途）；运维观测与 HTTP 入口经 `resolve_routable_collection_scope` 解析后始终传入具体 `collection_id`。

### 运维观测与 HTTP

- `load_overview_stats`、`list_query_summaries`、`list_ingestion_summaries`、`get_trace_summary`、`get_trace_detail`、`get_task_investigation` 及 `/api/traces/*`、`/api/tasks/progress/*/investigation` 均接受可选 `collection_id`，并走同一 tracing filter seam。
- `query_latency_percentiles` / `average_query_latency_ms` 同样按 `collection_id` 过滤，避免概览延迟跨集合聚合。

### 明确不做

1. **不**按集合拆分 Trace JSONL 文件或迁移历史行到新路径。
2. **不**在启动时批量回填 legacy 记录的 `collection_id`。
3. **不**为非默认集合对无 `collection_id` 的 legacy Trace 做可见性 fallback（与 ADR 0011 侧存储规则一致）。
4. **不**引入 Trace 文件的磁盘自动发现或多租户隐式集合推断；集合仍只认显式配置/调用方传入。
5. **不**改变 ADR 0004：评测仍不写生产 `QueryTrace`。

## Consequences

- 默认集合升级后行为与单集合时代一致：旧 Trace 仍出现在概览与列表中。
- 多集合并存时，各集合运维视图只看到自己的 Trace 与新写入记录，延迟分位不再被其他集合拉高或拉低。
- 过滤逻辑集中在 `tracing/store.py` 与 `tracing/latency.py`，避免各 HTTP/观测入口重复实现兼容分支。
- 将来若需物理分文件，可在不改对外契约的前提下替换 store 实现；当前以最小 diff 交付集合读隔离。
