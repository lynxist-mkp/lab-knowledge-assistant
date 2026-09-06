# ADR 0014: Task Progress 按集合过滤与 legacy 兼容

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0008（task progress 证据模型）、ADR 0012（Trace 集合过滤）、ADR 0011（多集合存储布局）

## Context

运维观测的 Trace 已按 `collection_id` 读侧过滤，但 `task_progress` 仍共用单一 JSONL 且历史记录缺少 `collection_id`。列表、详情与 investigation 会在多集合场景下串读。

## Decision

Task progress 与 Trace 对齐：**不按集合拆物理文件**；集合边界由记录级 `collection_id` 与读侧 filter seam 表达。

### 写入

1. 新写入（`persist_task_progress` 及衍生 seam）必须持久化 `collection_id`，取值为写入时 `settings.product.collection`。
2. 调用方通过 scoped `Settings` 路由到目标集合；不增加并行 `collection_id` 参数。

### 读取过滤（`task_progress.py`）

统一经 `read_task_progress_records` / `get_task_progress` / `filter_task_progress_records`，规则与 ADR 0012 Trace 一致：

| 记录 `collection_id` | 默认集合视图 | 非默认集合视图 |
| --- | --- | --- |
| 缺失（legacy） | 可见 | 不可见 |
| 等于目标 `collection_id` | 可见 | 可见 |
| 不等于目标 `collection_id` | 不可见 | 不可见 |

### 运维观测与 HTTP

- `list_task_progress_summaries`、`get_task_progress_detail`、`get_task_investigation` 及 `/api/tasks/progress*` 接受可选 `collection_id`，经 `resolve_routable_collection_scope` 解析后走同一 filter seam。
- `/api/stats/health` 保持全局，不引入 `collection_id`。

### 明确不做

1. 不按集合拆分 task progress JSONL 或回填 legacy 记录。
2. 不靠 trace/document 反推 task progress 集合归属。
3. 不为非默认集合对无 `collection_id` 的 legacy 记录做可见性 fallback。

## Consequences

- 默认集合升级后行为与单集合时代一致。
- 多集合并存时各集合运维视图只看到自己的 task progress。
- 过滤逻辑集中在 `task_progress.py`，避免 HTTP/观测入口重复实现。
