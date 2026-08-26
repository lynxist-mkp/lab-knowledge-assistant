# 黄金集

评测用的固定尺子。每次改切块、融合权重或 prompt，都用同一份题量；好坏是数出来的。

路径由 `settings.yaml` 的 `evaluation.golden_set` 指向本文件：`data/eval/golden.jsonl`。加载器用 `wenmai.eval.load_golden_set` / `load_golden_set_from_settings`。

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
| `status` | 缺省 `ready`；图内信息本票先 `placeholder`，评测默认跳过 |
| `evidence_quote` | 可选。从语料摘出的原句，方便人工核对，不参与 Hit@5 |

入库后的 `document_id` 是文件 SHA256，会随抓取快照变。评测对齐时用 `corpus_id_from_source_path(chunk.metadata["source_path"])`，不要拿 SHA256 当证据 ID。

## 初版构成

50 条，四类齐：

- 20 单跳事实、10 跨文档：答案从本机已抓取的维基/UNESCO 正文摘出，不是模型编的
- 12 明确不可答（24%）：规格不入库的内容（节目成片、工作音频、经营数字、内部材料）或现有公开页不会写的细节（实时人数、门票、年俸）；不要用「清单里有、只是还没抓到」的题，以免补抓后尺子漂掉
- 8 图内信息：占位。图转文（#21）已落地，等带图 PDF 入库后再补题，不阻塞 #26/#27 先跑前三类

未采用的语料：`chuanzheng-wiki-chuanzheng-xuetang`、`zhuzi-wiki-zhuzixue`、`zhuzi-wiki-lixue`、`zhuzi-wiki-wuyishan` 是维基重定向/消歧义残页，不当证据。

补抓政府网页或妈祖专条之后，抽查不可答题是否变成可答；变成可答的要改标，不要让尺子自己漂。
