# ADR 0008: 任务进展与快速溯源证据模型

**Status:** accepted  
**Date:** 2026-09-05  
**Parent:** [Spec: task progress and troubleshooting evidence for ops observation #59](https://github.com/lynxist-mkp/wenmai-assistant/issues/59)  
**Issue:** [#60 ADR: 任务进展与快速溯源证据模型](https://github.com/lynxist-mkp/wenmai-assistant/issues/60)

## Context

现有项目已经有 **Trace**、评测 artifact 与 **运维观测**，但长任务的运行态与排障证据仍然散落。入库或评测失败时，后续 Agent 往往需要重跑、翻日志或回读实现细节，才能知道卡在哪个阶段、影响了哪个对象、使用了什么配置。

我们需要补的不是一个泛化监控平台，而是一套服务 **任务进展** 与 **快速溯源** 的结构化证据模型，并且要遵守既有 seam：

- **运维观测** 继续做统一读侧入口。
- **提问** 继续使用 `QueryTrace`，不改造成长任务模型。
- **评测** 继续遵守 ADR 0004，不写生产 `QueryTrace`。

## Decision

1. 引入新的 `task_progress` 深 module，作为长任务运行态与排障证据的统一 seam。
2. `task_progress` 采用独立 JSONL 落盘，不混入现有 trace 文件，也不塞进 handler。
3. 顶层统一状态机固定为：`queued`、`running`、`succeeded`、`failed`、`partial_success`、`blocked`、`cancelled`。
4. `partial_success` 表示任务完成，但至少一个子项失败或降级；`blocked` 只表达“不再推进”，不等同于“跑得慢”。
5. 统一证据包固定为三层：
   - 任务头：任务类型、状态、时间戳、`last_progress_at`、计数器、触发来源、所属界面、配置指纹、顶层链接。
   - 阶段事件：阶段名、状态、耗时、失败分类、错误摘要、上下游摘要、输入/输出摘要、降级标记。
   - 子项证据：文档级或样本级对象的状态、失败分类、稳定 ID、联查链接与任务特有细节。
6. 顶层失败分类统一为：`input`、`dependency`、`model`、`storage`、`timeout`、`config`、`unknown`；任务可在子层保留更细的领域细节。
7. **入库** 通过 **入库编排** 汇总现有阶段证据并写入 `task_progress`；**评测** 通过 run 持久化 seam 汇总 run/group/item 证据并写入 `task_progress`。
8. **运维观测** 负责列出、筛选和读取 `task_progress`，并保留与 trace、评测 artifact、文档/样本标识的联查路径。

## Consequences

- 后续 Agent 可以先读结构化现场证据，再决定是否需要重跑。
- 长任务与单次提问分工更清楚：`task_progress` 负责任务级排障，`QueryTrace` 负责单次提问追因。
- 评测仍不污染生产 query trace，但不再只是孤立 artifact。
- 将来若要导出到外部观测平台，可从 `task_progress` 的统一模型继续扩展，而不是从 handler 或日志回推。
