# 课题组知识助手

一个由 AI 方向研究生独立完成的科研知识助手原型。它把论文、实验记录、项目文档、组会纪要等资料统一入库，在公开 demo 语料和私有科研资料之间维持清晰边界，并提供带出处的混合检索问答。

系统目标很克制：

- 回答必须带引用
- 库里没有的事实直接拒答
- 公开 demo 可复现，私有资料不进 Git
- 组内资料与个人文献库可以在同一知识库中共存

快速理解架构时，先看 `docs/architecture-legibility.md`；项目术语以 `CONTEXT.md` 为准；关键设计取舍记录在 `docs/adr/`。

## 这是什么

这不是一个“随便接个向量库”的通用文献 RAG，而是一个面向课题组日常资料沉淀的模块化检索系统，强调三件事：

- 混合资料源：同时支持 `group_doc` 与 `personal_literature`
- 来源可解释：引用里展示出处、来源类型、作者、年份等信息
- 本地可控：检索、入库、评测和 MCP 调用都能在本机闭环

## 当前能力

- 检索链路：`bge-m3` Dense + jieba BM25 + RRF 融合 + Cross-Encoder 精排
- 提问处理：术语归一、Multi-Query、证据不足拒答
- 入库链路：单文件导入、目录批量导入、内容幂等、基础文献元数据抽取
- 文档治理：质量门、灰区复判、待审状态、运维页审阅
- 交互表面：`/` 检索工作台、`/ops` 运维看板、MCP 工具集
- 评测：黄金集、消融、改写对照、可选 Ragas judge

## 项目结构

- `src/lab_knowledge/`: 主包，包含 HTTP、MCP、ingestion、retrieval、generation、eval 等模块
- `templates/`: 检索工作台与运维看板
- `data/corpus/`: 公开 demo 语料清单与抓取说明
- `data/eval/`: 公开评测基线
- `scripts/`: 启动、抓取、入库、评测与本地辅助脚本

## 快速开始

需要 Python 3.12。真实运行时依赖本地模型缓存与 MLX 服务；开发和测试默认使用 fake provider，不强依赖真实模型。

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv sync --extra dev --frozen
```

本地校验：

```bash
uv run ruff check .
uv run pytest
```

如果要启用本机 Gemma MLX：

```bash
./scripts/start_gemma_mlx.sh
# 或双击 scripts/start_gemma_mlx.command
```

启动服务：

```bash
uv run uvicorn lab_knowledge.app:app --factory --host 127.0.0.1 --port 8000
```

停止服务：

```bash
./scripts/stop_server.sh
```

## 两条资料入库路径

### 1. 公开 demo 复现

仓库只保留公开语料的抓取清单与评测尺子，不直接提交正文产物：

```bash
PYTHONPATH=src python scripts/fetch_corpus.py
PYTHONPATH=src python scripts/ingest_corpus_manifest.py
```

这一条路径对应 `data/corpus/manifest.yaml` 和 `data/eval/golden.jsonl`，适合演示公开科研资料上的检索、拒答与评测。

### 2. 私有科研资料导入

组内文档和个人文献不需要进入 Git。可通过 `source_kind` 显式区分来源：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/absolute/path/to/file.md",
    "source_kind": "group_doc"
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/absolute/path/to/paper.pdf",
    "source_kind": "personal_literature"
  }'
```

如果目录下是 Zotero 自动导入形成的附件树，也可以直接批量导入：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/absolute/path/to/zotero/storage",
    "source_kind": "personal_literature"
  }'
```

目录导入会递归扫描 `.md` 和 `.pdf`，沿用内容幂等语义跳过未变化文件，并返回 `total`、`attempted`、`ingested`、`rebuilt`、`skipped`、`failed` 与逐文件摘要。

对于论文类资料，系统会优先保留文件内可得的基础元数据；Markdown 也支持通过 YAML 头补充：

```yaml
title: Attention Is All You Need
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit
year: 2017
source_kind: personal_literature
```

## MCP 与兼容命名

当前主包名已经切换为 `lab_knowledge`，服务入口为 `lab_knowledge.app:app`。

MCP 方面：

- 推荐使用 `ask.answer`
- 兼容提供 `ask_lab_knowledge`
- 旧的 `ask_wenmai` 仍保留为 legacy alias，便于老脚本平滑过渡

## 评测

常规公开评测：

```bash
PYTHONPATH=src python scripts/ingest_corpus_manifest.py
PYTHONPATH=src python scripts/run_phase_b_batch.py
```

已完成入库时可加 `--skip-ingest`。

## 边界说明

- 本仓库不提交私有 PDF、组会纪要、组内文档正文
- 不依赖 Zotero 数据库本体，只利用本地附件目录与基础元数据
- 公开 demo 与私有资料共用同一套系统能力，但不混淆版本管理边界
