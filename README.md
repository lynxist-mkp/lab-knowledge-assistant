# 福云·文脉助手（wenmai-assistant）

实习场景下的检索助手原型：公开网页 / PDF 入库，混合检索，回答必须带出处。**未接入福云生产库。**

技术对齐 `fjgdAgent/RAG系统.md` 的模块边界（五块存储、Factory、Trace），语料与中文检索自写。

## 本周范围

- 文本 + 图转文（Caption 缝进 chunk）+ 本地 `bge-m3` Dense + jieba BM25 + RRF + Cross-Encoder 精排
- 生成：DeepSeek V4 Flash，回答带引用，无依据则拒答
- 服务与监测：FastAPI + Jinja2 六页（总览 / 数据浏览 / Ingestion 管理 / Ingestion 追踪 / Query 追踪 / 评估面板）
- 评测：Ragas Faithfulness + 自算 Hit@5、MRR，四组消融
- 音频（Dolphin 转写）放在文本、图、监测、评测都完成之后

实现规格见 `.scratch/fuyun-wenmai/spec.md`。

## 运行

需要 Python 3.12（系统自带的 3.9 太旧）与环境变量 `DEEPSEEK_API_KEY`。

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

已验证：MPS 可用，`bge-m3` 已在本机 HuggingFace 缓存中。服务启动与入库命令随实现补全。
