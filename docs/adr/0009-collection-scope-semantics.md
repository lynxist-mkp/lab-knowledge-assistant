# ADR 0009: 集合作用域语义

**Status:** accepted  
**Date:** 2026-09-05  

## Context

对外契约需要明确三个相关但不同的概念：**知识库**（系统能力）、**集合**（材料作用域容器）、**研究主题**（集合内的内容切分维度；元数据字段名仍为 `culture_domain`）。此前 `settings.product.collection` 已隐式界定索引与配图边界，但缺少一等读模型与稳定对外命名，导致 HTTP、MCP 与运维观测各自拼装 scope。

`Knowledge` 作为深 module 已委托 `ReadPath` / `WritePath`，名称应保留；不应把「集合」与「知识库」混称为同一层对象。

## Decision

1. **知识库** 继续指可检索片段集合这一系统能力；`Knowledge` 类名与 `knowledge/store.py` seam 不变。
2. **集合** 作为一等 scope 容器对外暴露，至少包含 `collection_id`、展示名与按审阅状态分层的统计；当前单租户部署下默认集合来自 `settings.product.collection`。
3. **研究主题** 是集合内的内容维度，用于浏览、过滤与提问可选范围；不等同于集合，也不作为集合 ID。
4. 新增 `CollectionReadModel` 作为集合读侧 seam，从配置与 `DocumentCatalog` / `Knowledge` 汇总统计，不写入索引。
5. 调用方可选传入 `collection_id`；未传则使用默认集合；传入未知 ID 时返回明确错误，而不是静默落到错误索引。
6. 本轮先收口**集合**的对外契约与错误语义，不要求同一周内完成真多集合路由；单集合实现可先把 `collection_id` 作为显式 scope 验证与回传字段。

## Consequences

- MCP 与文档管理 facade 可共享同一 scope 对象，减少各入口自拼字段。
- 将来多集合并存时，可在不改动 `ReadPath` / `WritePath` 的前提下扩展 `CollectionReadModel.list_collections()`。
- 术语表（`CONTEXT.md`）中的 **集合** 与 **研究主题** 边界在代码中有对应类型，而非仅配置注释。
- 对外先暴露 `collection_id` 契约，不等于立即承诺底层已完成多集合切换；实现可分期补齐。
