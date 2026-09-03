# 福云·文脉助手

面向融媒编辑的知识检索助手：材料按文化域组织，回答必须带出处；库里没有的事实就拒答。它是企业知识层，不是成片工具，也不是播出终审。

## Language

**福云·文脉助手**:
面向融媒编辑的检索助手，对内全称是福云内容库·闽派文化检索助手。
_Avoid_: 闽问, Minwen, Ask My Docs, 企业知识库 RAG

**文化域**:
知识库的分片标签，对齐内容库闽派子库的切法，例如海丝、朱子、妈祖、船政。
_Avoid_: 分类, 标签云, topic

**提问**:
融媒编辑用自然语言提出的一句问题，可选文化域；结果是带出处的回答或拒答。
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
审阅状态为待审的片段：已在知识库存着，编辑提问时检索不到；库览里可见并标出，供维护者核对后放行或驳回整份文档。
_Avoid_: 隔离区, quarantine chunk, 软删除

**知识库**:
可检索的片段集合。一份材料写入后，稠密和稀疏同时可查；**待审片段**除外，须先审阅通过。删掉则两边和配图一起没。内容没变就跳过，变了就整份替换。可按文化域浏览、看概览与片段详情。
_Avoid_: 向量库, 双索引, chroma

**配图**:
文档解析抽出的插图资产；正文用占位引用，图转文后替换为说明；删文档时与稠密/稀疏索引一并删除。
_Avoid_: 图片库, ImageStore, CLIP

**库览**:
按文化域查看知识库里有哪些文档与片段；含概览计数。完整库览只在运维看板；编辑工作台仅能从回答出处深链到只读片段。
_Avoid_: 数据浏览（UI 文案可保留）, browse service

**文档目录**:
知识库按文化域汇总的文档与片段计数，供库览与运维概览读取；随入库提交或删除同步更新。
_Avoid_: catalog index, browse cache

**检索**:
按检索方式从知识库取出排好的片段（融合）；精排不在此 module。
_Avoid_: retrieve 浅包装, search pipeline

**生成**:
用对话 LLM 根据检索到的片段写出带引用的回答，或执行拒答；邻块扩展在同一 interface 内完成，**提问编排**与**评测**共用。
_Avoid_: 嵌入, 视觉模型, Embedding, prepare_generation_context

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
_Avoid_: Qwen3-ASR, 声纹检索, 成片检索

**Trace**:
入库链路和查询链路上，每个阶段的输入输出、候选和耗时记录。
_Avoid_: 监控, APM, LangSmith

**拒答**:
检索到的片段撑不住该主张时，系统拒绝编造，并列出实际看到的来源。
_Avoid_: 空回复, 幻觉兜底

**拒答原因**:
Trace 里区分拒答来自哪一层：证据不足（检索后无可用片段）或模型拒答（有片段但生成判定撑不住）。只在运维追踪展示；编辑工作台仍只呈现拒答与出处。
_Avoid_: refusal_reason, 拒答类型码

**演示音频**:
为打通语音转写链路而准备的可公开短音频，不是实习工作音频。
_Avoid_: 工作音频, 节目成片

**黄金集**:
人工标注的问答评测集，每条有证据文档和是否可答标记。
_Avoid_: 测试集（泛称）, 自动生成后直接当标准答案的题单

**评测**:
用黄金集量提问：按检索方式分组，得到 Hit@5、MRR、拒答是否判对、出处覆盖；对外以 run / 看板读取为 interface。
_Avoid_: Ragas 当主尺子, 测试集

**编辑工作台**:
融媒编辑用的提问界面：默认入口；文化域作顶栏筛选；主区是单次提问与带出处回答/拒答（历史条目彼此独立、不把上轮当上下文）；本次 Trace 默认折叠可展开；点出处在页内抽屉只读查看片段。固定页脚标明人工智能生成合成，并声明不可直接作为播出终审。不是成片工具，也不承担入库与评测。完整多轮对话不在当前范围。
_Avoid_: 聊天机器人, ChatGPT 壳, 问答首页（泛称）, 多轮私教

**运维看板**:
给知识库维护者与系统开发者的界面：默认落地概览；另有库览、入库、**待审**（对待审文档通过或驳回）、统一追踪（入库/查询分 Tab）、评测；与编辑工作台导航硬隔离，共享内容库族蓝白视觉语言。旧根路径页面（如 `/browse`）不再保留、不重定向。
_Avoid_: Dashboard（作产品名）, 监测中心, 管理后台（泛称）

**提问编排**:
一次**提问**从**检索**、精排、生成前准备（含邻块扩展）到**生成**的深 module；**编辑工作台**与**评测**共用同一编排 interface，仅是否写 query **Trace** 可选。
_Avoid_: QueryOrchestrator, query pipeline, ask handler

**入库准入**:
材料进入 load 前的三态决策：硬拒、直接放行、或标**待审**；**入库质量门**与**灰区复判**的结果在此收敛为单一决策与 Trace stage 载荷，**入库**流水线按决策分支。
_Avoid_: AdmissionGate（作产品名）, quality check pipeline

**入库编排**:
一份材料从**入库质量门**、load、transform 到 embed/upsert 的深 module；单条 `run_prepare_commit`、批处理 `run_prepare_commit_batch`；prepare 产出 `PrepareBody`，commit 写**知识库**。
_Avoid_: ingest pipeline, run_ingest_phases, IngestPrepareBody

**运维观测**:
**运维看板**读侧深 module：提问/入库 Trace 的列表与详情、降级列表、概览统计。
_Avoid_: build_overview_stats, tracing/read 函数袋, ops/overview

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
**提问** Trace 的读写合一 module：编排层 begin/finalize/save，运维读 summary/detail。
_Avoid_: query_views, TraceRecorder（对外入口）
