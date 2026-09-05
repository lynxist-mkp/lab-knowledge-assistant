# ADR 0010: 文档管理 Facade 分层

**Status:** accepted  
**Date:** 2026-09-05  
**Related:** ADR 0009（集合作用域语义）

## Context

HTTP 运维看板与 MCP 工具各自直接调用 `Knowledge` 的读/写/审阅方法，调用方需要了解 `document_card`、`list_pending_review_documents`、`approve_review` 等分散入口。`ReadPath` / `WritePath` 作为深 module 应保留；但对外「列文档、看详情、删文档、审阅、集合统计」需要更高一层的稳定门面。

**入库编排**（`pipelines/ingestion.py`）负责 prepare → commit 两阶段，不应并入文档管理 facade。

## Decision

1. 新增 `DocumentManagement` facade，位于 `knowledge/document_management.py`，委托：
   - `Knowledge`：删文档、审阅通过/驳回、文档卡片与片段读取
   - `CollectionReadModel`：集合级统计
   - `ImageReferenceService`：配图引用与按需取内容
2. Facade 第一版范围固定为：
   - 列文档 / 取文档详情
   - 按文化域读取库览分组
   - 删文档
   - 列待审 / 通过 / 驳回
   - 集合级统计
   - 文档配图引用 / 按需取配图内容
3. 不将 `prepare_ingest_source()` / `run_prepare_commit()` 移入 facade；入库仍只经 **入库编排**。
4. MCP 管理类工具当前以 `DocumentManagement` 作为正式服务层 seam：`documents.*` 与 `reviews.*` 只调用 `DocumentManagement`，不直接触达 `WritePath`。
5. MCP `images.*` 也只经 `DocumentManagement` 取 **配图引用** / 内容，不直接自建 `ImageReferenceService`。
6. `collections.*` / `documents.*` / `reviews.*` / `images.*` 可各自暴露 tool entry，但对下统一停在 `DocumentManagement` 这一层，不再各自拼 deeper read/write 配方。
7. 现有 `Knowledge` 公共 API 保留，供提问编排与入库继续使用；HTTP 与 MCP 可共享同一服务层 seam，本切片以 MCP 与新合约为重点。

## Consequences

- 对外文档生命周期操作有单一入口，便于契约测试与 envelope 统一。
- HTTP 运维看板的库览读侧也可经同一 facade，减少直接触达 `Knowledge` 的页面配方。
- `ReadPath` / `WritePath` 内部演化不影响 MCP/管理类调用方。
- 配图内容默认以 **配图引用** 返回；取字节为显式 opt-in，避免路径泄漏。
- 管理类 MCP tool 先统一到 `DocumentManagement` 这一 seam；是否再往上抽新的 MCP service helper，可后续再评估，不阻塞当前契约落地。
