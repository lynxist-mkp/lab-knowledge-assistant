# 黄金集

评测用的固定尺子。每次改切块、融合权重或 prompt，都用同一份题量；好坏是数出来的。

路径由 `settings.yaml` 的 `evaluation.golden_set` 指向本文件：`data/eval/golden.jsonl`。加载器用 `lab_knowledge.eval.load_golden_set` / `load_golden_set_from_settings`。

## 条目格式

每行一条 JSON：

| 字段 | 含义 |
|---|---|
| `id` | 题号，如 `g001` |
| `question` | 问题 |
| `evidence_doc_ids` | 证据文档 ID，取值 `data/corpus/manifest.yaml` 的 `id`（与 `{id}.md` 文件名一致） |
| `answerable` | 库中有依据则为 `true`；明确不可答（用来量化**拒答**）为 `false` |
| `reference_answer` | 参考答案；不可答题写「应拒答。」 |
| `category` | `单跳事实` / `跨文档` / `图内信息` / `明确不可答` |
| `status` | 缺省 `ready`；图内信息可为 `draft`（题面与证据已绑定，待 PDF 入库与图转文核对）或历史 `placeholder`（空槽）；评测默认跳过 `draft` 与 `placeholder` |
| `evidence_quote` | 可选。从语料摘出的原句，方便人工核对，不参与 Hit@5 |

### 状态与升档

| 状态 | 含义 | 默认评测 |
|---|---|---|
| `ready` | 题面、证据、参考答案均已核对 | 包含 |
| `draft` | 图内题已写好并绑定 `evidence_doc_ids`，参考答案待图转文 spot-check | 跳过 |
| `placeholder` | 空槽，仅占位 | 跳过 |

`draft` → `ready`：对应 PDF 已入库、图转文产出经 spot-check 通过，且 `reference_answer` / `evidence_quote` 与配图一致后改标。加载参数 `include_placeholders=True` 可同时载入 `placeholder` 与 `draft`。

入库后的 `document_id` 是文件 SHA256，会随抓取快照变。评测对齐时用 `corpus_id_from_source_path(chunk.metadata["source_path"])`，不要拿 SHA256 当证据 ID。

## 初版构成

100 条，四类齐（课题组知识助手场景）：

- 45 单跳事实、20 跨文档：答案从公开 arXiv 摘要页与中文维基正文摘出，覆盖自然语言处理、多模态、检索增强、强化学习等研究主题；g051–g075 中约半数采用短问/口语写法
- 27 明确不可答（27%）：规格不入库的内容（内网实验日志、组会纪要、工作音频、保密协议数据、实时监控）或现有公开页不会写的细节（内部经费、门禁记录、私人聊天）；不要用「清单里有、只是还没抓到」的题，以免补抓后尺子漂掉
- 8 图内信息：g043–g050 绑定 arXiv 论文架构图或维基配图条目，全部标 `ready`（待重抓语料后 spot-check）

语料清单共 40 条，按研究主题各 8 条（自然语言处理 / 多模态 / 检索增强 / 强化学习 / 其他）。清单中未列入但可能存在的维基重定向/消歧义残页不当证据。

补抓 arXiv 或维基条目之后，抽查不可答题是否变成可答；变成可答的要改标，不要让尺子自己漂。

## 图内题 spot-check（待语料重抓）

场景改造后需重跑 `scripts/fetch_corpus.py` 并入库，再逐条核对 g043–g050：

| id | 证据 doc | 说明 |
|---|---|---|
| g043 | nlp-arxiv-attention | Encoder/Decoder 层数（N=6） |
| g044 | nlp-arxiv-attention | Multi-Head Attention 维度 d_model=512 |
| g045 | rag-arxiv-rag | 检索器从 Wikipedia 索引获取候选 |
| g046 | mm-arxiv-clip | 对比学习最大化余弦相似度 |
| g047 | rl-arxiv-dqn | 经验回放稳定训练 |
| g048 | rl-wiki-mdp | 状态-动作-奖励交互循环 |
| g049 | mm-wiki-vit | 图像切分为 patch 序列 |
| g050 | mm-wiki-diffusion | 前向过程加入高斯噪声 |

**注意：** 改造后旧闽派文化语料 `items/` 与评测基线不再对齐；重抓 + 入库 + spot-check 是 align-docs-and-tests 阶段的必做项。

## 元数据字段约定

- 清单与抓取产物 YAML 头中研究主题仍写字段名 `culture_domain`（与入库管线一致），取值为 `自然语言处理` / `多模态` / `检索增强` / `强化学习` / `其他`
- 集合作用域字段为 `space: lab_knowledge`（对应 `settings.yaml` 的 `product.collection`）
- 界面与文档中称「研究主题」，不叫「文化域」

## 混合资料库说明

这份黄金集只覆盖**可公开复现**的 demo 语料，不要求也不建议把私有 PDF、组会纪要或组内资料提交进 Git。

混合科研资料库的验证口径分两层：

- 仓库内评测样例：继续以 `data/corpus/manifest.yaml` 对应的公开语料为固定尺子，保证任何人都能复现。
- 私有科研资料验证：通过测试与运行时元数据覆盖 `group_doc` / `personal_literature` 两类来源共存的情形，确认来源语义、基础文献信息与幂等导入保持可用。

如果你要用自己的 Zotero 附件目录做批量导入，评测基线仍建议保留这份公开黄金集，避免把不可共享的个人文献耦合进仓库。
