# ADR 0011: 多集合存储布局与 legacy 兼容读取

**Status:** accepted  
**Date:** 2026-09-06  
**Related:** ADR 0009（集合作用域语义）

## Context

S1–S5 已将 **集合** 作用域接到提问、文档管理、运维概览等读侧，并把 catalog、ingestion_history、image_index、Chroma、BM25、配图目录改成按 `collection_id` 物理隔离。部署中仍可能存在升级前写入的**旧单集合布局**（sidecar 与向量目录直接落在 `settings.paths.*` 根路径，无 `{collection_id}` 子目录）。

本轮（S6）不引入启动时全量自动迁移，也不新增迁移脚本框架；需要把读写兼容规则写清楚，并用测试锁住关键行为。

## Decision

### 新布局（写入目标）

对任意 `collection_id`，新写入一律落到带集合子目录的路径（由 `collection_storage_bindings()` 解析）：

| 存储 | 新布局路径 |
| --- | --- |
| catalog | `{catalog_parent}/{collection_id}/catalog.json` |
| ingestion_history | `{history_parent}/{collection_id}/ingestion_history.db` |
| image_index | `{image_index_parent}/{collection_id}/image_index.db` |
| Chroma | `{chroma_root}/{collection_id}/` |
| BM25 | `{bm25_root}/{collection_id}/` |
| 配图文件 | `{images_root}/{collection_id}/` |

`catalog_parent` 等取自 `settings.paths.*` 的父目录；与旧布局根文件/目录共用同一配置项族。

### 读取兼容（仅默认集合）

`legacy_read_fallback` 仅在 `collection_id == settings.default_collection_id`（即 `settings.product.collection`）时为 `True`。

读取规则（`storage/compat.py`）：

1. **文件型 sidecar**（catalog、ingestion_history、image_index）：若新布局路径**存在**则读新布局；否则在 `legacy_read_fallback` 为真时读旧布局根路径。
2. **Chroma 目录**：若新布局目录**非空**则用新布局；否则在 `legacy_read_fallback` 为真时，若旧布局目录非空则用旧布局。
3. **写入**始终落新布局路径，不因读到旧数据而回写旧路径。

非默认集合**不**做 legacy fallback：旧布局数据对其它 `collection_id` 不可见。

### 幂等与删除隔离

`ingestion_fingerprints` 按集合各自的 `ingestion_history.db` 维护；同一 `source_path` 在不同集合中互不影响 skip / rebuild / delete。

### 明确不做

- 启动时或后台全量自动迁移旧数据到新布局。
- 将旧布局数据复制到新集合目录。
- 修改默认集合 ID 时自动重映射磁盘路径。

运维如需迁移，应显式导出/再入库或另行编写一次性脚本（超出本 ADR 范围）。

## Consequences

- 升级后默认集合在**未写入新布局前**仍可读到旧数据；首次写入某 sidecar 起，该 sidecar 新数据进入新布局，读取以新布局优先。
- 多集合并存时物理路径与指纹互不干扰；契约测试见 `tests/test_storage_compat.py`。
- 非默认集合只有在新布局下产生过存储后才可通过 `resolve_routable_collection_scope()` 等「已有存储即允许」规则路由（见 S3+ 实现）。
