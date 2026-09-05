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
| `status` | 缺省 `ready`；图内信息可为 `draft`（题面与证据已绑定，待 PDF 入库与图转文核对）或历史 `placeholder`（空槽）；评测默认跳过 `draft` 与 `placeholder` |
| `evidence_quote` | 可选。从语料摘出的原句，方便人工核对，不参与 Hit@5 |

### 状态与升档

| 状态 | 含义 | 默认评测 |
|---|---|---|
| `ready` | 题面、证据、参考答案均已核对 | 包含 |
| `draft` | 图内题已写好并绑定 `evidence_doc_ids`，参考答案待图转文 spot-check（#48/#49） | 跳过 |
| `placeholder` | 空槽，仅占位 | 跳过 |

`draft` → `ready`：对应 PDF 已入库（#47）、图转文产出经 spot-check 通过（#48/#49），且 `reference_answer` / `evidence_quote` 与配图一致后改标。加载参数 `include_placeholders=True` 可同时载入 `placeholder` 与 `draft`（名称保留，兼管两类跳过项）。

入库后的 `document_id` 是文件 SHA256，会随抓取快照变。评测对齐时用 `corpus_id_from_source_path(chunk.metadata["source_path"])`，不要拿 SHA256 当证据 ID。

## 初版构成

100 条，四类齐：

- 45 单跳事实、20 跨文档：答案从本机已抓取的维基/UNESCO/政府公开页正文摘出，不是模型编的；g051–g100 中约半数采用短问/口语写法，为后续改写对照铺路
- 27 明确不可答（27%）：规格不入库的内容（节目成片、工作音频、经营数字、内部材料）或现有公开页不会写的细节（实时人数、门票、年俸）；不要用「清单里有、只是还没抓到」的题，以免补抓后尺子漂掉
- 8 图内信息：g043–g050 全部 `ready`。默认评测含 8 条图内题

未采用的语料：`chuanzheng-wiki-chuanzheng-xuetang`、`zhuzi-wiki-zhuzixue`、`zhuzi-wiki-lixue`、`zhuzi-wiki-wuyishan` 是维基重定向/消歧义残页，不当证据。

补抓政府网页或妈祖专条之后，抽查不可答题是否变成可答；变成可答的要改标，不要让尺子自己漂。

## 图内题 spot-check（#48/#49，2026-09-03）

入库后逐条核对：`evidence_quote` 是否来自配图 caption / OCR，或正文中与图件对应的可检索句。

| id | 证据 doc | 入库 | 图转文 | spot-check | 说明 |
|---|---|---|---|---|---|
| g043 | haishi-unesco-quanzhou-maps | ✅ 核对 md | 图例摘录 | **通过** | OCR 曾灰区待审不可检索；现用核对 md（含 Contributing Element 等图例） |
| g044 | haishi-unesco-quanzhou-maps | ✅ 核对 md | 窑址摘录 | **通过** | 含「屈斗宫」↔ Qudougon Kils；完整 PDF 仍在 `.tmp/` |
| g045 | zhuzi-wuyi-trail-plan | ✅ markdown | 正文图件章节 | **通过** | 建阳段「三级旅游服务区：黄坑镇杨梅岭、新历村、麻沙镇杜潭村」 |
| g046 | zhuzi-feiyi-handdrawn-map | ✅ 本地 PNG | ⚠️ caption 噪声 | **通过（正文）** | 十条线路名在正文；密图 caption 未稳定读出线路名 |
| g047 | mazu-unesco-nomination-00227 | ✅ DOC→md | 正文含 Mazu lanterns | **通过** | Downloads `25638-EN.doc`；句：itinerate with “Mazu lanterns” |
| g048 | mazu-unesco-consent-00227 | ✅ 抽检 md | 首页声明 | **通过** | Downloads `00927.pdf`；申报主体：湄洲妈祖祖庙董事会（未全量 OCR） |
| g049 | chuanzheng-fuzhou-museum-opening | ✅ | 展厅配图 | **通过（正文）** | 「万年清」在开馆报道正文；配图 caption 为展厅舵轮陈列 |
| g050 | chuanzheng-mawei-museum-news | ✅ | 船名牌照片 | **通过（正文+配图）** | 正文写明「古田」号船名牌；caption 识别为金属牌匾但未稳定 OCR「古田」 |

**结论：** 8 条 `ready`（g043–g050），默认评测 100 条。同意书以抽检正文入库，完整 59 页扫描件保留在 `.tmp/`。

## 图内检索复测（2026-09-03，retrieval-only）

全量 100 条消融曾显示 g043/g044 各组稳定 miss。根因：`haishi-unesco-quanzhou-maps` OCR 入库落在质量门灰区 → `审阅状态=待审` → `is_searchable` 过滤掉，证据对检索不可见；g044 另缺「屈斗宫」↔ OCR 英文窑名对齐。修复：核对 md + `prefer_pdf: false`，质量门直接 `approve`，回归见 `tests/test_haishi_maps_curated_evidence.py`，环见 `.scratch/fuyun-wenmai/loop_g043_g044_hit5.py`。

修复后仅跑图内 8 题 × 四组检索（无生成）：

| 组 | Hit@5 | MRR | misses |
|---|---|---|---|
| dense_only | **1.000** | 0.938 | — |
| sparse_only | 0.875 | 0.875 | g047 |
| rrf | **1.000** | 0.917 | — |
| rrf_rerank | 0.750 | 0.667 | g046, g047 |

相对修复前（图内 Hit@5：dense 0.75 / sparse 0.625 / rrf 0.75 / rrf_rerank 0.50），g043/g044 已各组 HIT。残留 g046（rerank 挤出）与 g047（稀疏词面弱）是独立问题，未纳入本次诊断闭环。

**若要防再发：** 图内题入库后应用检索环断言证据 doc 可 `is_searchable` 且 Hit@5；灰区待审且无人工审阅模型时不得静默当「已入库可搜」。
