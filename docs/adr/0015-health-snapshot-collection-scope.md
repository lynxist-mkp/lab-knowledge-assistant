# ADR 0015: Health Snapshot 与 Ask Evidence 集合口径

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0012（Trace 集合过滤）、ADR 0014（Task Progress 集合过滤）

## Context

Task progress 与 Trace 已支持按 `collection_id` 读侧过滤，但 `/api/stats/health` 仍为混合口径：task 信号经默认集合视图聚合，ask 饱和证据则全局计数。`ask_evidence` 缺少记录级 `collection_id`，`ask_governor` 的 long-task guard 也始终看全局 running long tasks。多集合运维视图下 health 与提问限流口径不一致。

## Decision

### Ask evidence（`ops/ask_evidence.py`）

1. 新写入必须持久化 `collection_id`，取值为 scoped `settings.product.collection`。
2. 读侧 filter 规则与 ADR 0012 Trace 一致：legacy 无字段记录仅在默认集合视图可见。
3. 共享 JSONL，不按集合拆物理文件。

### Health snapshot（`ops/observation.py`）

1. `/api/stats/health` 双模：不传 `collection_id` 时返回**全局**视图（task progress 与 ask evidence 均不过滤）；传入时返回该集合 scoped 视图。
2. scoped 模式下 task 信号与 ask 信号（`ask_busy`、`ask_timeout`、`ask_long_task_guard`）使用同一 `collection_id` 过滤，避免半全局半 scoped。

### Ask governor long-task guard（`http/ask_governor.py`）

1. 未带 `collection_id` 的提问路径：long-task guard 仍看**全局** running long tasks。
2. 带 `collection_id` 的提问路径：guard 只看该集合内 running long tasks。
3. **不**拆分 per-collection governor 实例；内存并发槽位仍为全局单实例。

## Consequences

- 默认集合升级后 legacy ask evidence 行为与单集合时代一致。
- 多集合 health 看板可按集合查看异常与 ask 饱和信号，不再串读。
- 提问限流在长任务仅影响其他集合时，scoped 提问仍可正常进入（全局槽位模型不变）。
- ADR 0014 中「health 保持全局」的表述由本 ADR 取代。
