# 课题组知识助手

面向课题组成员的知识检索助手：材料按研究主题组织，回答必须带出处；库里没有的事实就拒答。它是课题组知识沉淀层，不是实验跑数工具，也不能替代实验资料核对。

## Language

**课题组知识助手**:
面向课题组成员的检索助手，帮助沉淀与复用论文、实验记录、项目文档与组会纪要等组内知识。
_Avoid_: 闽问, Minwen, Ask My Docs, 企业知识库 RAG, 福云, 文脉助手

**研究主题**:
知识库的分片标签，对齐课题组常见资料维度，例如自然语言处理、多模态、检索增强、强化学习。
_Avoid_: 文化域, 分类, 标签云, topic

**提问**:
课题组成员用自然语言提出的一句问题，可选研究主题；结果是带出处的回答或拒答。
_Avoid_: ask, query, 问答请求

**入库**:
把一份材料送进知识库：切成片段、写上出处；内容没变就跳过。切分后的清洗、补元数据、图转文由入库后处理一门完成。PDF 等格式在 load 前过**入库质量门**。
_Avoid_: ingest, load, ingestion pipeline

**入库质量门**:
材料进入 load 前的文档级体检：按有效字符率分档——过低硬拒、过高放行、中间走**灰区复判**。与块级规则清洗（Refiner 丢低质块）是两层防线。
_Avoid_: 质量预检, quality gate, 文档体检

**灰区复判**:
有效字符率落在中间档时，由模型二次判断这份材料是否值得入库：数字版 PDF 看抽文，扫描件看页图。判不过不拦流水线，但产出**待审片段**。
_Avoid_: 软告警, 人工抽检, LLM 复判

**审阅状态**:
写在片段 metadata 上的标签：`已通过` 可参与检索；`待审` 已入库但检索与生成一律筛掉，直到维护者在运维看板对该文档点通过或驳回。
_Avoid_: review_status, 低置信度, 降权

**待审片段**:
审阅状态为待审的片段：已在知识库存着，提问时检索不到；库览里可见并标出，供维护者核对后放行或驳回整份文档。
_Avoid_: 隔离区, quarantine chunk, 软删除

**知识库**:
可检索的片段集合。一份材料写入后，稠密和稀疏同时可查；**待审片段**除外，须先审阅通过。删掉则两边和配图一起没。内容没变就跳过，变了就整份替换。可按研究主题浏览、看概览与片段详情。
_Avoid_: 向量库, 双索引, chroma

**集合**:
知识材料所属的作用域容器：界定一批文档、配图、索引与统计落在哪个边界里；可在同一系统中并存多个集合。**研究主题**是集合内的内容切分维度，不等同于集合。
_Avoid_: 研究主题（作集合名）, 租户（未明确多租户时）, namespace（作产品名）

**配图**:
文档解析抽出的插图资产；正文用占位引用，图转文后替换为说明；删文档时与稠密/稀疏索引一并删除。
_Avoid_: 图片库, ImageStore, CLIP

**配图引用**:
对外稳定指向一张**配图**的引用对象：至少能标明它属于哪份文档、该取哪张图，并供界面、接口或脚本按需再取内容；不是本地路径泄漏，也不要求每次都内嵌整张图片内容。
_Avoid_: 图片路径, 临时文件名, base64 blob（作正式概念）

**库览**:
按研究主题查看知识库里有哪些文档与片段；含概览计数。完整库览只在运维看板；检索工作台仅能从回答出处深链到只读片段。
_Avoid_: 数据浏览（UI 文案可保留）, browse service

**文档目录**:
知识库按研究主题汇总的文档与片段计数，供库览与运维概览读取；随入库提交或删除同步更新。
_Avoid_: catalog index, browse cache

**文档管理**:
围绕单份文档生命周期的统一入口：列文档、看详情、删文档、看统计、做审阅动作；对上给界面、脚本与接口稳定调用，不要求调用方分别理解底层读写细节。
_Avoid_: 文档函数袋, 杂项管理接口, browse service

**检索**:
按检索方式从知识库取出排好的片段，可含精排。
_Avoid_: retrieve, search pipeline

**生成**:
用对话 LLM 根据检索到的片段写出带引用的回答，或执行拒答。
_Avoid_: 嵌入, 视觉模型, Embedding

**嵌入**:
把文本或图片变成向量，供稠密检索。
_Avoid_: 生成, 图转文, 多模态大模型（生成用）

**图转文**:
用视觉模型给文档里的图生成自然语言描述，再写回文本块，之后走同一条文本检索链。
_Avoid_: CLIP 多模态向量, 以图搜图主路径, 图像解析, 用 OCR 替代说明

**文档解析**:
把来源路径变成可入库正文（含 load 路由：数字版 PDF / 扫描件 / Markdown）；扫描件走版面解析，不是给插图写说明。
_Avoid_: 图像解析, 图转文, 拿 OCR 当 Captioner, choose_pdf_route

**扫描件**:
抽出文字密度低于阈值、整份走文档解析的 PDF。
_Avoid_: 图片 PDF, 非数字版

**数字版 PDF**:
抽出文字足够、整份走 MarkItDown 的 PDF。
_Avoid_: 非扫描件, 普通 PDF

**语音转写**:
用 Dolphin 把音频变成文本再入库；没有单独的音频向量空间。
_Avoid_: Qwen3-ASR, 声纹检索

**Trace**:
入库链路和查询链路上，每个阶段的输入输出、候选和耗时记录。
_Avoid_: 监控, APM, LangSmith

**拒答**:
检索到的片段撑不住该主张时，系统拒绝编造，并列出实际看到的来源。
_Avoid_: 空回复, 幻觉兜底

**拒答原因**:
Trace 里区分拒答来自哪一层：证据不足（检索后无可用片段）或模型拒答（有片段但生成判定撑不住）。只在运维追踪展示；检索工作台仍只呈现拒答与出处。
_Avoid_: refusal_reason, 拒答类型码

**演示音频**:
为打通语音转写链路而准备的可公开短音频，不是真实组会录音。
_Avoid_: 工作音频, 实验原始录音

**黄金集**:
人工标注的问答评测集，每条有证据文档和是否可答标记。
_Avoid_: 测试集（泛称）, 自动生成后直接当标准答案的题单

**评测**:
用黄金集量提问：按检索方式分组，得到 Hit@5、MRR、拒答是否判对、出处覆盖。
_Avoid_: Ragas 当主尺子, 测试集, 各 runner 自写 _runs_dir

**检索工作台**:
课题组成员用的提问界面：默认入口；研究主题作顶栏筛选；主区是单次提问与带出处回答/拒答（历史条目彼此独立、不把上轮当上下文）；本次 Trace 默认折叠可展开；点出处在页内抽屉只读查看片段。固定页脚标明人工智能生成合成，并声明不可直接作为实验资料核对或正式报告终稿。不承担入库与评测。完整多轮对话不在当前范围。
_Avoid_: 聊天机器人, ChatGPT 壳, 问答首页（泛称）, 多轮私教, 编辑工作台

**提问服务**:
服务层对外承接一次**提问**的稳定入口：统一 HTTP 与 MCP 的参数与返回契约，再委托到**提问编排**执行；当前入口是 `http/ask_service.py` 的 `run_ask()`。
_Avoid_: ask handler, 薄包装（作正式名称）, 直接调 pipeline

**运维看板服务**:
服务层对外承接**运维看板**读写动作的稳定入口：统一 `ops` 路由的参数与返回契约，再委托到**文档管理**或**运维观测**执行；当前入口是 `http/ops_service.py` 的 `OpsService`。
_Avoid_: ops handler, 仪表盘 helper, 直接调 observation 函数袋

**运维看板**:
给知识库维护者与系统开发者的界面：默认落地概览；另有库览、入库、**待审**（对待审文档通过或驳回）、统一追踪（入库/查询分 Tab）、评测；与检索工作台导航硬隔离，共享统一蓝白视觉语言。旧根路径页面（如 `/browse`）不再保留、不重定向。
_Avoid_: Dashboard（作产品名）, 监测中心, 管理后台（泛称）

**提问编排**:
一次**提问**从**检索**、精排、生成前准备（含邻块扩展）到**生成**的深 module；**检索工作台**与**评测**共用同一编排 interface，仅是否写 query **Trace** 可选；调用方经 `ask_work_from_job` / `eval_work_from_item` / `gen_retry_work` 建 work，不直接拼 `OrchestrationWork`。
_Avoid_: QueryOrchestrator, query pipeline, ask handler, 手填 OrchestrationWork 字段

**提问预处理**:
**提问编排** Phase 1：术语归一与 Multi-Query 产出额外检索路径，并带上 Trace 所需 rewriter 元数据与耗时；入口 `prepare_query_extras`。
_Avoid_: collect_extra_queries, CollectedExtras, ExtrasPhase 内二次拼装

**入库准入**:
材料进入 load 前的三态决策：硬拒、直接放行、或标**待审**；**入库质量门**与**灰区复判**的结果在此收敛为单一决策，**入库**流水线按决策分支。
_Avoid_: AdmissionGate（作产品名）, quality check pipeline

**入库编排**:
一份材料从**入库质量门**、load、transform 到 embed/upsert 的深 module；单条 `run_prepare_commit`、批处理 `run_prepare_commit_batch`；prepare 产出 `PrepareBody`，commit 写**知识库**。
_Avoid_: ingest pipeline, run_ingest_phases, IngestPrepareBody

**运维观测**:
**运维看板**读侧深 module：提问/入库 Trace 的列表与详情、降级列表、概览统计；概览经 `load_overview_stats(settings)` 一次产出。
_Avoid_: build_overview_stats, tracing/read 函数袋, ops/overview

**任务进展**:
给系统开发者与知识库维护者看的长任务运行态视图：面向**入库**与**评测**，展示一次任务当前处于哪个阶段、已完成哪些阶段、卡住或失败在哪、以及可继续钻取的文档级/样本级证据；不等同于 **Trace**，而是对排障更友好的任务级读侧。
_Avoid_: 监测性, 运行态（泛称）, 进度条（作正式概念）, task progress（不翻译时）

**快速溯源**:
出现错误、降级、卡顿或异常结果时，维护者能从任务级或单次**提问**入口迅速定位到失败阶段、输入对象、关键配置与上游摘要，并继续钻取到文档级、样本级或阶段级证据完成排障。
_Avoid_: 看日志排查, 大概定位, debug 一下（泛称）

**ReadPath**:
**知识库**的 read seam：稠密/稀疏检索、按文档或片段读取、列出**待审**文档；与 WritePath 对称。
_Avoid_: browse service, retrieve adapter

**WritePath**:
**知识库**的 write seam：plan/commit **入库**、删文档、审阅通过/驳回；经 ReadPath 校验文档存在。
_Avoid_: ingest writer, upsert adapter

**PrepareTraceRecorder**:
**入库** pipeline seam 上的 Trace 适配器：编排层写阶段与摘要，底层不直接碰 TraceContext。
_Avoid_: IngestTraceRecorder, ingestion trace shim

**QueryTrace**:
**提问** Trace 的读写合一 module：编排层 begin/finalize/save，运维读 summary/detail；写 seam 输入为 `AskTracePayload`。
_Avoid_: query_views, TraceRecorder（对外入口）, work: object
