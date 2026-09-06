# 课题组知识助手 (Lab Knowledge Assistant)

> 面向高校科研课题组与 AI 研究生的模块化 RAG 知识检索系统与 MCP 服务端实现。  
> 融合“组内研究资料”与“个人文献库 (Zotero)”，支持混合多路召回、两阶段重排、多模态图表解析、学术证据门禁与标准化 MCP 工具协议。

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MCP Compliant](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介 (Introduction)

在高校科研团队与 AI 课题组的日常研究中，知识沉淀往往面临“双重割裂”：一方面，课题组内部的组会纪要、学术报告、项目方案与技术文档零散分布在共享网盘或内部 Wiki 中；另一方面，团队成员个人积累的大量论文文献多存储在 Zotero 等文献管理软件中，缺乏与组内资料的统一交叉检索能力。此外，通用大语言模型直接用于学术问答时，极易产生非真实存在的“学术幻觉”与错误引用。

**课题组知识助手 (Lab Knowledge Assistant)** 针对上述痛点设计，是一套高可靠、模块化解耦的检索增强生成 (Modular RAG) 系统与 Model Context Protocol (MCP) 服务。项目具备以下核心能力：

- **混合科研资料库体系**：系统原生区分“组内资料 (`group_doc`)”与“个人文献库 (`personal_literature`)”，不仅支持 Markdown、TXT 及 PDF 多格式解析，还能自动提取与注入论文题目、作者、发表年份等学术元数据。
- **工业级混合多路召回**：结合密集语义检索 (BAAI/bge-m3 + ChromaDB) 与学术专有词典加权的稀疏检索 (Jieba + BM25)，经倒数排序融合 (RRF, Reciprocal Rank Fusion) 与 Cross-Encoder 深度重排序，大幅提升长尾学术概念的召回精度。
- **严谨学术证据门禁**：通过关键词/语义重叠度计算证据置信度；证据不足时严格执行硬拒答 (Refusal Guard)，并支持命中块前后邻近上下文平滑扩展 (Adjacent Chunk Expansion)，杜绝张冠李戴。
- **双工作台与智能体接入**：提供面向科研人员的日常检索工作台 (`/`) 与面向维护者的运维看板 (`/ops`)；同时发布标准 stdio 协议的 MCP Server，使 Cursor、Claude Desktop 等现代智能体能直接调用本课题组知识库。
- **消费级本地算力友好**：针对 Apple Silicon (MLX) 等单机环境优化，内置单模型互斥调度 (Single Model Exclusive) 与空闲自动卸载机制，在 16G/24G 统一内存设备上即可流畅运行本地大模型与 OCR 视觉流水线。

---

## 🏛️ 系统架构 (Architecture)

```
                                  【数据输入层】
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │  组内资料 (Group Docs)  │                     │ 个人文献库 (Zotero 导出)│
    │  Markdown / Reports/ PDF│                     │  PDF Attachments + Meta │
    └────────────┬────────────┘                     └────────────┬────────────┘
                 │                                               │
                 ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 摄取流水线 (Ingestion Pipeline)                                              │
│  ├─ 格式加载器 (Loaders: MarkItDown / pypdfium2 / PaddleOCR-VL 自动路由)     │
│  ├─ 语义切分器 (Splitter: 递归字符切分，窗口 600 chars，重叠率 15%)          │
│  ├─ 三阶段转换 (Transform: Refiner 规则清洗 -> Enricher 抽取 -> Captioner)  │
│  ├─ 质量门禁与灰度审核 (Quality Gate: 自动准入 / 灰度人工审核 / 拦截丢弃)   │
│  └─ 幂等入库 (Fingerprints 比对，目录增量导入不重入)                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 存储与多集合层 (Storage & Collections Layer)                                │
│  ├─ 密集向量索引 (ChromaDB: BAAI/bge-m3 嵌入向量)                           │
│  ├─ 稀疏词法索引 (BM25: 领域词典 domain.txt + 停用词表)                      │
│  ├─ 元数据目录与证据库 (Catalog JSON / Ingestion History DB / SQLite)         │
│  └─ 多模态资产库 (Image Storage / Image Index DB)                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 检索与生成编排 (Retrieval & Generation Orchestration)                       │
│  ├─ 查询理解与改写 (Synonyms Lexicon 归一化 / Multi-Query 视角扩展)         │
│  ├─ 双路混合检索 (Dense k=20 + Sparse k=20)                                 │
│  ├─ 融合排序 (Reciprocal Rank Fusion, RRF k=60 -> Top 10)                   │
│  ├─ 深度重排序 (Cross-Encoder: ms-marco-MiniLM-L6-v2 -> Top 5)              │
│  ├─ 上下文扩展 (Adjacent Chunks Expansion: 命中块前后邻近片段拼接)          │
│  ├─ 证据门禁 (Evidence Overlap Guard: 证据覆盖不足触发严格拒答)              │
│  └─ 答案生成 (Local Gemma via MLX / Cloud LLM + 带标注学术引用)             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│ 双端交互界面 (Dual-Surface Web) │         │ Model Context Protocol (MCP)    │
│  ├─ 检索工作台 (/)              │         │  ├─ ask_lab_knowledge (问答)    │
│  ├─ 运维管理面板 (/ops)         │         │  ├─ documents.* (文献文档管理)  │
│  └─ 评测中心 (/eval)            │         │  ├─ reviews.* (灰度审核管理)    │
│  (FastAPI + Jinja2 + SSE 流)    │         │  └─ collections.* / images.*    │
└─────────────────────────────────┘         └─────────────────────────────────┘
```

---

## ✨ 核心特性 (Key Features)

### 1. 混合科研资料库与文献元数据建模
- **双源数据建模**：原生标记 `group_doc`（课题组规范、实验记录、项目方案）与 `personal_literature`（个人研究文献）。
- **学术元数据抽取**：解析并持久化论文标题 (`title`)、作者列表 (`authors`)、出版年份 (`year`) 与来源标签。
- **Zotero 目录批量增量导入**：支持指定本地文献文件夹递归扫描，自动提取 PDF 嵌入元数据与文件名学术特征，支持幂等去重跳过。

### 2. 多路混合检索与深度重排序流水线
- **Dense + Sparse 混合召回**：向量检索捕获语义抽象意图，BM25 结合科研专有词典锁定关键专业术语（如特定模型名、算法简称）。
- **RRF 倒数排序融合**：使用标准 RRF 算法 ($Score = \sum \frac{1}{k + rank}$) 消除两种评分标尺差异，均衡候选池。
- **Cross-Encoder 二次重排**：利用精细跨编码器模型计算问答相关度得分，精准截断无用噪音。

### 3. 上下文扩展与证据不足防幻觉拒答
- **相邻块动态拼接 (Adjacent Expansion)**：命中核心片段后，自动向文档前后延伸拼接 $\pm N$ 块相邻内容，保留公式推导与实验步骤的完整上下文。
- **证据重叠度硬门禁 (Refusal Guard)**：评估提问核心语义与召回证据的交集重叠度，当证据缺失或相关度低于阈值时，明确给出“证据不足拒答”，杜绝模型盲目编造。

### 4. 工业级数据清洗与多模态解析
- **PDF 智能分流路由**：快速文本型 PDF 走原生高性能抽取；扫描件或图表型 PDF 自动分流接入 PaddleOCR-VL 视觉解析。
- **三阶段数据转换 (Transform)**：
  - *Refiner*：去除无效格式符、乱码与冗余空白；
  - *Enricher*：自动抽取文档摘要、关键标签与所属研究主题；
  - *Captioner*：识别论文图表、架构图，调用多模态模型生成文字描述沉淀入库。
- **质量门禁与灰度审核**：计算切块有效字符率与可读性分数，高质量自动放行，存疑块进入 `/ops` 灰度审核队列由人工核准。

### 5. 双工作台界面 (Dual-Surface Web UI)
- **检索工作台 (`/`)**：专为科研成员设计，提供极简对话框、多研究主题（自然语言处理、多模态、检索增强、强化学习等）筛选、交互式文献引用抽屉卡片、原文证据定位与多模态配图浏览。
- **运维管理面板 (`/ops`)**：专为系统管理员与开发者设计，提供单文件/目录批量摄取实时进度（SSE 推送）、系统负载与健康快照、全链路查询与入库 Trace 审计日志、灰度待审列表操作。

### 6. 标准化 MCP 服务 (Model Context Protocol)
- 遵循 Anthropic 开放的 MCP 标准，通过标准输入输出 (stdio) 传输。
- 允许外部智能体（如 Cursor Agent、Claude Desktop）一键接入本知识库，调用学术问答、文献检索与文档管理工具。

### 7. 本地轻量化推理与资源保护
- **单模型互斥调度 (Single Model Exclusive)**：大模型生成、OCR 解析、向量嵌入互斥调度，避免多模型并发挤爆本地显存/统一内存。
- **空闲超时自动卸载 (Idle Unload)**：系统空闲达到配置超时后自动释放内存权重，按需即时重载。
- **防打满与长任务防护 (Ask Saturation Guard)**：内置请求并发限流与超时熔断，保障服务稳定。

---

## 🛠️ 技术栈 (Tech Stack)

| 层次 | 核心技术选型 | 说明 |
| :--- | :--- | :--- |
| **基础语言与环境** | Python 3.12+ / uv | 强类型现代化 Python 工程，支持 uv 极速包管理 |
| **Web 框架与渲染** | FastAPI / Uvicorn / Jinja2 | 异步高性能 REST API + 双端 HTML 模板渲染 + SSE 流式通信 |
| **向量存储与嵌入** | ChromaDB / BAAI/bge-m3 | 1024 维高精度中文及跨语言密集向量表征 |
| **稀疏词法检索** | Jieba / 自研 BM25 引擎 | 内置科研专有词典 (`domain.txt`) 与同义词表 (`synonyms.yaml`) |
| **重排序模型** | cross-encoder/ms-marco-MiniLM-L6-v2 | 跨编码器高相关度判别排序 |
| **本地推理框架** | Apple Silicon MLX | 本地运行 Gemma-4-E2B 模型与 PaddleOCR-VL |
| **文档加载与解析** | MarkItDown / pypdfium2 / PaddleOCR | 覆盖 Markdown、TXT、富文本 PDF 及图像图表 |
| **协议与智能体** | Model Context Protocol (MCP Python SDK) | 标准 stdio 协议工具服务 |
| **评测与质量工程** | Ragas / Pytest / Ruff | 自动化 Golden 集评测、消融实验框架与代码规范检查 |

---

## 📁 目录结构 (Directory Structure)

```text
lab-knowledge-assistant/
├── pyproject.toml              # 项目依赖声明、构建配置与工具链配置
├── settings.yaml               # 核心配置：模型、路径、分块、检索与资源保护
├── prompts/                    # 生产提示词模板 (QA、多查询扩展、多模态抽取)
│   ├── qa_v1.txt
│   ├── multi_query_v1.txt
│   ├── enrich_v1.txt
│   └── caption_v1.txt
├── src/
│   └── lab_knowledge/          # 核心代码包
│       ├── app.py              # FastAPI 应用入口与中间件装配
│       ├── config.py           # 配置加载器与路径自适应解析
│       ├── runtime.py          # 全局单例与生命周期管理
│       ├── models.py           # 核心数据模型 (Citation, Chunk, BulkResult)
│       ├── components/         # 模型守卫 (Model Guard) 与组件接入
│       ├── factories/          # 组件工厂 (LLM, Embedding, BM25, Reranker 等)
│       ├── ingestion/          # 文档解析、元数据抽取、质量门禁与灰度流水线
│       ├── retrieval/          # 密集检索、BM25 检索与 RRF 融合实现
│       ├── generation/         # 上下文扩展、证据门禁与带引用生成
│       ├── query_processing/   # 同义词归一化与 Multi-Query 生成
│       ├── storage/            # 向量库、元数据目录、图片与指纹持久化
│       ├── knowledge/          # 知识库外观门面 (Facade)、集合与卡片封装
│       ├── ops/                # 运维指标统计、健康快照与观测性服务
│       ├── tracing/            # 查询与摄取链路追踪器 (Traces)
│       ├── eval/               # 评测流水线、Ragas 适配器与消融实验引擎
│       ├── mcp/                # MCP 服务端实现、协议封装与 Stdio 安全守卫
│       └── http/               # 路由模块 (Workbench, Ops, Ingest, Eval, Schemas)
├── templates/                  # 双工作台前端 HTML 模板
│   ├── workbench.html          # 科研检索工作台页面
│   ├── ops.html                # 运维管理面板页面
│   └── _shell.html             # 通用顶栏与基础骨架
├── scripts/                    # 运维、测试与评测辅助脚本
│   ├── run_mcp_server.py       # 启动 MCP Server (stdio)
│   ├── run_eval_ablation.py    # 运行多路检索消融对比实验
│   ├── run_phase_b_batch.py    # 运行阶段性评测与 Bad Case 导出
│   ├── ingest_corpus_manifest.py # 按语料清单批量建库
│   ├── fetch_corpus.py         # 抓取测试语料脚本
│   ├── start_gemma_mlx.sh      # 启动本地 MLX Gemma 进程
│   └── stop_server.sh          # 停止后台服务与清理进程
├── data/                       # 数据资产与存储目录
│   ├── corpus/                 # 示例语料与清单 (manifest.yaml)
│   ├── eval/                   # Golden 测试集 (golden.jsonl) 与评测记录
│   ├── lexicon/                # 学术专有词典、同义词表与停用词
│   └── db/                     # Chroma 向量库、BM25 索引与元数据 (运行时生成)
├── docs/                       # 架构设计决策记录 (ADR) 与开发文档
│   └── adr/                    # ADR 0001 - 0016
└── tests/                      # 单元测试与集成测试集 (70+ 测试套件)
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
- 操作系统：macOS (推荐 Apple Silicon) 或 Linux
- Python 版本：`>= 3.12`
- 包管理工具：推荐使用 [`uv`](https://docs.astral.sh/uv/)（亦支持标准 `pip`）

### 2. 安装项目依赖

克隆仓库后进入项目根目录：

```bash
cd lab-knowledge-assistant

# 使用 uv 一键同步虚拟环境与依赖（推荐）
uv sync --extra dev

# 或者使用标准 pip 进行可编辑模式安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. 配置说明
系统通过 `settings.yaml` 驱动，支持开箱即用。核心配置项：
- `product.name`: 项目名称（默认：“课题组知识助手”）
- `retrieval.mode`: 检索模式，可选 `rrf`（默认融合）、`dense_only`、`sparse_only`
- `providers.embedding`: 嵌入模型提供方（默认 `bge_m3`）
- `providers.reranker`: 重排模型提供方（默认 `cross_encoder`）
- `resources.single_model_exclusive`: 是否开启单模型显存互斥管理（单机推荐 `true`）

如需使用云端评测大模型（如 Ragas 评估），可创建 `.env` 文件配置 API 密钥：
```env
ZHIPU_API_KEY=your_zhipu_api_key_here
# 或 OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 启动 Web 服务 (双工作台)

运行以下命令启动 FastAPI 后台与 Web 工作台：

```bash
python -m uvicorn lab_knowledge.app:app --host 127.0.0.1 --port 8000 --reload
```

启动完成后，可直接在浏览器中访问：
- 🔍 **检索工作台**：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)  
  面向课题组成员进行知识检索、主题筛选、文献证据追溯。
- ⚙️ **运维管理面板**：[http://127.0.0.1:8000/ops](http://127.0.0.1:8000/ops)  
  面向管理员监控入库状态、灰度审核、Trace 链路日志与系统健康。

### 5. 启动 MCP Server

如需供 Cursor 或 Claude Desktop 挂载使用，运行：

```bash
python scripts/run_mcp_server.py
```

服务将通过标准输入输出 (stdio) 保持监听。

---

## 📚 数据摄取与知识库构建 (Data Ingestion)

知识助手支持单文件摄取、目录批量导入与命令行批量建库。

### 方式一：通过 Web 运维面板导入 (/ops)
1. 打开浏览器访问 [http://127.0.0.1:8000/ops](http://127.0.0.1:8000/ops)。
2. 在“数据入库”表单中：
   - 输入本地绝对路径（单文件路径或文件夹路径）；
   - 选择 **数据来源类型**（`课题组资料 group_doc` 或 `个人文献库 personal_literature`）；
   - 若导入单篇文献，可显式覆盖填入文献标题、作者与年份；
   - 点击“开始入库”，页面将通过 Server-Sent Events (SSE) 实时显示每个文件的解析、切分与建库进度。

### 方式二：调用 HTTP API 批量导入 Zotero 文献目录
可直接发送 POST 请求导入个人 Zotero 导出的本地 PDF 目录：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/path/to/zotero/storage/papers",
    "source_kind": "personal_literature"
  }'
```

返回示例：
```json
{
  "code": "success",
  "data": {
    "summary": {
      "total": 12,
      "ingested": 10,
      "skipped": 2,
      "rebuilt": 0,
      "failed": 0
    }
  }
}
```
*注：系统自动计算 SHA-256 文件指纹，未发生变化的文件将自动跳过（Idempotent），不会重复切块与索引。*

### 方式三：通过清单批量构建示例语料
```bash
python scripts/ingest_corpus_manifest.py --manifest data/corpus/manifest.yaml
```

---

## 🔌 MCP 接入智能体 (Cursor / Claude Desktop)

本系统原生支持作为 Model Context Protocol (MCP) Server 挂载，无缝将课题组知识注入日常编码与学术写作流。

### 1. Cursor 接入配置
在项目根目录 `.cursor/mcp.json` 或 Cursor 全局 MCP 设置中加入：

```json
{
  "mcpServers": {
    "lab_knowledge": {
      "command": "python",
      "args": [
        "/absolute/path/to/lab-knowledge-assistant/scripts/run_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/lab-knowledge-assistant/src"
      }
    }
  }
}
```

### 2. Claude Desktop 接入配置
编辑 Claude Desktop 配置文件 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "lab_knowledge": {
      "command": "python3",
      "args": [
        "/absolute/path/to/lab-knowledge-assistant/scripts/run_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/lab-knowledge-assistant/src"
      }
    }
  }
}
```

### 3. 可用 MCP 工具一览
| 工具名称 | 功能描述 |
| :--- | :--- |
| `ask_lab_knowledge` (`ask.answer`) | 向课题组知识库提问，执行多路检索、两阶段重排与带证据引用的回答生成 |
| `documents.list` | 查询已入库的文档清单与摘要列表（支持集合过滤） |
| `documents.get` | 获取指定文档详情及其所属切块信息 |
| `documents.delete` | 从知识库与向量库中级联物理删除指定文档 |
| `collections.list` | 查看当前已注册的集合命名空间 |
| `collections.get_stats` | 获取集合中文档总数、切块总数及运行统计信息 |
| `reviews.list_pending` | 列出处于灰度状态待人工审核入库的文档切块 |
| `reviews.approve` / `reviews.reject` | 人工审核通过或驳回指定待审项 |
| `images.get_ref` / `images.get_content` | 查看多模态配图引用元数据与图片原始内容 |

---

## 🧪 评测与消融实验 (Evaluation & Benchmark)

系统内置标准化的 Golden 测试集与消融对比评测流水线，用于评估 RAG 链路在学术文献领域的表现。

### 1. 运行检索消融对比实验 (Ablation Study)
评测框架内置对四种不同阶段链路的指标度量：
- `dense_only`: 仅使用 BGE-M3 语义向量召回
- `sparse_only`: 仅使用领域分词 BM25 稀疏召回
- `rrf`: 密集与稀疏经 RRF 算法融合召回
- `rrf_rerank`: RRF 融合召回后叠加 Cross-Encoder 深度重排

运行指令：
```bash
# 运行全部配置组的 Hit@5, MRR 与拒答准确率测试
python scripts/run_eval_ablation.py

# 快速验证前 10 条样本（冒烟测试）
python scripts/run_eval_ablation.py --limit 10

# 评测同义词查询改写开启前后的增益对比
python scripts/run_eval_ablation.py --rewrite-compare
```

评测产物会自动持久化在 `data/eval/runs/` 目录下，包含详细的 JSON/Markdown 评测报告。

### 2. 核心评测指标
- **Hit@5**: 前 5 个召回片段中命中真实支撑证据的比例。
- **MRR (Mean Reciprocal Rank)**: 真实首个支撑证据所在排名的倒数均值。
- **Refusal Precision**: 针对域外提问/无证据提问时，系统正确触发拒答的准确率。
- **Faithfulness / Context Precision**: （可选）基于 Ragas 框架计算的回答忠实度与上下文精确率。

---

## 🧪 自动化测试与代码规范 (Testing & Linting)

项目拥有完善的测试覆盖（包含单元测试、端到端集成测试与架构契约测试）：

```bash
# 运行单元与集成测试套件
pytest

# 运行特定模块测试（如混合检索或摄取流水线）
pytest tests/test_retrieval.py tests/test_ingest_directory.py

# 代码质量检查与格式化 (Ruff)
ruff check .
ruff format --check .
```

---

## 🔒 隐私与安全说明 (Security & Privacy)

1. **本地化保障**：默认检索（BM25、ChromaDB、BGE-M3、Cross-Encoder）均在本地进程中执行，不向任何第三方服务上传组内内部文档或个人私有文献。
2. **凭据安全**：所有外部模型 API 密钥（如智谱、OpenAI 等）仅通过本地环境变量注入，配置及代码库中严禁明文硬编码。
3. **MCP Stdio 安全纪律**：MCP 服务端严格对标准输出流实施沙箱保护，底层日志统一重定向至标准错误流或日志文件，防止污染 MCP JSON-RPC 报文。

---

## 📄 开源许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。
欢迎学术交流、课题组二次开发与自由使用。
