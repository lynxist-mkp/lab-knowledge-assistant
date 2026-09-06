# ADR 0013: 显式集合注册发现

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0009（集合作用域语义）、ADR 0011（多集合存储布局）

## Context

多集合存储布局（ADR 0011）后，非默认集合曾通过「磁盘上已有存储即允许路由」被隐式发现。这导致空集合不可见、运维/MCP 无法提前声明集合，且 `list_collections()` 只能返回默认集合。

## Decision

1. 在 `settings.collections` 维护轻量注册表，每项至少包含 `collection_id` 与 `display_name`。
2. 默认集合真源仍为 `settings.product.collection`；未配置 `collections` 时行为与单集合部署一致。
3. `list_collections()` 返回默认集合 + 注册表中额外集合（去重，默认优先）。
4. `resolve_routable_collection_scope()` 与 `resolve_collection_id()` 仅承认默认集合与注册集合；**不做磁盘自动发现**。
5. 已注册但尚无存储的集合仍可见、可路由；统计为 0，不抛 `UnknownCollectionError`。
6. 未注册集合继续抛 `UnknownCollectionError`。

## Consequences

- 新建集合需先在配置中注册，方可被 MCP、文档管理与运维观测路由。
- 已有存储但未注册的集合不再被隐式承认；迁移时需补注册项。
- 展示名可在注册表中覆盖，默认集合亦可用注册项覆盖 `display_name`。
