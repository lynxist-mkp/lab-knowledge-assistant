# 课题组公开语料（本机）

本目录存放**课题组知识助手**知识库的公开语料来源与抓取产物。仓库里只提交：

- `manifest.yaml`：人工精选的 URL 清单（约 40 篇，覆盖自然语言处理、多模态、检索增强、强化学习等研究主题）
- `README.md`（本文件）
- `../scripts/fetch_corpus.py`：可重复执行的抓取脚本

抓下来的正文在 `items/` 子目录，已被 `.gitignore` 忽略，**不要 `git add` 正文文件**。换机器后重跑脚本即可复现。

## 怎么抓

在 `lab-knowledge-assistant` 根目录：

```bash
.venv/bin/python scripts/fetch_corpus.py
```

可选参数：

- `--dry-run`：只打印会抓/会跳过哪些条目，不写文件
- `--limit N`：只处理清单前 N 条（调试）
- `--output-dir PATH`：自定义输出目录（默认 `data/corpus/items`）

已存在的 `{id}.md` 会自动跳过。

## 许可与引用

| 来源类型 | 许可说明 | 本项目用法 |
|---|---|---|
| arXiv 摘要页 / PDF | arXiv 许可（各论文以页脚为准）；摘要页可引用，须保留原文 URL 与获取日期 | YAML 头 `license_note` 写「arXiv 公开论文，引用时保留 URL」；`retrieved_at` 与 `snapshot_sha256` 记录快照 |
| 中文维基百科条目 | 正文默认 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh)（兼 GFDL）；**必须署名，修改后须以相同条款再许可** | 脚本经 [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) 拉取纯文本（不爬 `/wiki/` HTML），`source_url` 仍指向条目页 |
| 开源项目文档（GitHub 等） | 以仓库 LICENSE 为准 | 保留 URL 与获取日期 |

每条抓取结果的 YAML 头字段（勿改字段名）：

```yaml
source_url: https://...
source_org: arXiv / 维基媒体基金会 / …
license_note: arXiv 公开论文，引用时保留 URL
culture_domain: 检索增强
space: lab_knowledge
retrieved_at: 2026-09-06
snapshot_sha256: <正文 SHA256>
```

> **说明：** 元数据字段名仍为 `culture_domain`（与入库管线一致），产品文案与界面中称「研究主题」。

## robots.txt 策略

抓取前对每条 URL 所在主机请求 `/robots.txt`：

- 仅当响应为 HTTP 200 且内容含 `User-agent` 行时，视为**可核验**，再判断路径是否允许本脚本 User-Agent 访问。
- **未能核验**的主机（返回 404 HTML、超时、或非 robots 内容）**不自动抓取**，脚本会跳过并在日志列出主机。
- `manifest.yaml` 中 `robots_blocked_hosts` 列出的主机一律不抓。

自动抓取以 arXiv、维基等 robots 可核验站点为主；课题组内部实验记录、组会纪要等本地材料不入此清单，由运维页手工入库。

## 与私有文献的边界

`data/corpus/` 只承担**公开 demo 语料**的复现职责，不承担个人 Zotero 文献或组内私有资料的版本管理。

- 公开语料：通过 `manifest.yaml` + 抓取脚本复现，适合给别人演示和跑共享评测。
- 私有资料：通过 `/ingest` 单文件或目录导入进入本地知识库，推荐对 Zotero 自动导入形成的附件目录使用 `source_kind=personal_literature`。

这样仓库可以同时支持“公开可复现 demo”和“私有科研资料沉淀”两条路径，而不需要把个人 PDF 或组内文档纳入 Git。

## 不收录

- 课题组内网实验原始数据、未脱敏工作笔记
- 登录墙后的期刊全文、付费数据库
- 实时 GPU 监控、训练日志流、Slack/微信聊天记录
- 合作方保密协议下的未公开结果
- 本仓库 `data/audio/` 等工作音频

## 研究主题

`culture_domain`（界面称「研究主题」）取值：`自然语言处理` / `多模态` / `检索增强` / `强化学习` / `其他`（见 `settings.yaml` 与入库 Enricher 枚举）。
