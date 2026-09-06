# 福云·文脉助手架构地图

这份地图面向第一次进入仓库的人，也面向要讲项目的人。目标不是罗列全部文件，而是先回答三件事：

1. 从哪个入口讲系统。
2. 一次**提问**和一次**入库**分别穿过哪些 seam。
3. 想改某个能力时，应该停在哪一层。

术语以 `CONTEXT.md` 为准；若此文和 ADR 冲突，以 ADR 为准。

## 先讲哪六个概念

- **提问服务**: HTTP 与 MCP 共用的一次提问入口，位置在 `src/wenmai/http/ask_service.py` 的 `run_ask()`
- **提问编排**: 提问深 module，负责预处理、检索、精排、生成，位置在 `src/wenmai/pipelines/query_orchestration.py`
- **入库编排**: 一份材料的 prepare -> commit 深 module，位置在 `src/wenmai/ingestion/orchestrator.py` 和 `src/wenmai/pipelines/ingestion.py`
- **知识库**: read/write facade，位置在 `src/wenmai/knowledge/store.py`
- **文档管理**: 运维看板与 MCP 管理类调用的文档生命周期入口，位置在 `src/wenmai/knowledge/document_management.py`
- **运维观测**: 概览、Trace、任务进展的统一读侧，位置在 `src/wenmai/ops/observation.py`

如果先记住这六个名词，再看代码会快很多。

## 系统分层

### 1. 服务层

对外契约的稳定入口，负责收口参数和返回值。

- `src/wenmai/http/ask_service.py`: **提问服务**
- `src/wenmai/http/ops_service.py`: **运维看板服务**
- `src/wenmai/http/*.py`: HTTP 路由
- `src/wenmai/mcp/tools/*.py`: MCP tools

规则：页面和 MCP 优先经服务层，不直接各自拼流水线。

### 2. 深 module 层

真正承载业务复杂度的地方。

- `src/wenmai/pipelines/query_orchestration.py`: **提问编排**
- `src/wenmai/ingestion/orchestrator.py`: **入库编排**
- `src/wenmai/knowledge/document_management.py`: **文档管理**
- `src/wenmai/ops/observation.py`: **运维观测**
- `src/wenmai/tracing/query_trace.py`: **QueryTrace**
- `src/wenmai/task_progress.py`: **任务进展**模型与落盘

规则：要改行为，优先改这一层；不要把新逻辑散落回 handler。

### 3. 组件与适配层

负责具体实现和可替换 provider。

- `src/wenmai/components/*`: LLM、Embedding、Reranker、Vision、Transform 等 provider
- `src/wenmai/factories/*`: provider 选择
- `src/wenmai/storage/*`: catalog、images、fingerprints 等存储实现

规则：这一层可换实现，但不应该重新定义上层概念。

## 一次提问怎么走

1. `http/workbench.py` 或 `mcp/tools/ask.py` 收到请求。
2. 两者统一进入 `http/ask_service.py::run_ask()`。
3. `run_ask()` 创建 work，委托给 `pipelines/query_orchestration.py`。
4. **提问编排**依次完成：
   - **提问预处理**
   - dense 检索
   - sparse 检索
   - RRF 融合
   - rerank
   - 生成 / 拒答
5. `QueryTrace` 负责写入与读取查询 Trace。
6. 服务层把结果回给 HTTP 或 MCP。

面试时可以把这条链概括成一句话：`提问服务` 收口对外契约，`提问编排` 承担检索与生成复杂度，`QueryTrace` 让链路可追。

## 一次入库怎么走

1. `http/ingest.py` 收到请求。
2. 委托 `pipelines/ingestion.py`。
3. `ingestion/orchestrator.py` 负责 prepare -> commit 两段：
   - **入库准入**: 质量门 / 灰区复判 / 待审决策
   - load
   - split
   - transform
   - embed
   - upsert
4. `PrepareTraceRecorder` 记录阶段证据。
5. `task_progress` 写入长任务运行态。
6. `knowledge/store.py` 委托 `ReadPath` / `WritePath` 落到实际存储。

面试时可以把这条链概括成一句话：`入库编排` 统一控制准入、切块、增强和提交，底层存储对上隐藏在 `知识库` seam 后面。

## 运维看板读什么

- 页面入口: `templates/ops.html`
- 路由入口: `src/wenmai/http/ops.py`
- 服务入口: `src/wenmai/http/ops_service.py`

读侧分工：

- 文档、库览、待审、配图: 走 **文档管理**
- 概览、Trace、任务进展、联查: 走 **运维观测**

这就是为什么 `ops.py` 应该尽量薄，只翻译 HTTP，不承载业务配方。

## 常见改动该落在哪

- 想改提问契约或并发治理: `提问服务`
- 想改检索阶段顺序、候选、拒答: `提问编排`
- 想改入库元数据、待审逻辑、质量门: `入库编排`
- 想改文档删除、审阅、库览文档详情: `文档管理`
- 想改概览、Trace、任务进展聚合: `运维观测`
- 想换模型或检索实现: `components/*` + `factories/*`

## 当前最重要的 ADR

- `docs/adr/0003-dual-surface-ui-editor-and-ops.md`: 编辑工作台 / 运维看板双面隔离
- `docs/adr/0006-deep-module-consolidation.md`: 第二轮深 module 收口
- `docs/adr/0007-deep-module-round3.md`: 第三轮深 module 收口
- `docs/adr/0010-document-management-facade.md`: 文档管理 facade

## 一句话总览

这个仓库不是“一个 RAG 脚本加几个页面”，而是按领域术语收口成几条稳定 seam：**提问服务 -> 提问编排**，**运维看板服务 -> 文档管理 / 运维观测**，**入库编排 -> 知识库**。
