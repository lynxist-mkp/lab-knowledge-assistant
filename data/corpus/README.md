# 公开闽派文化语料（本机）

本目录存放**福云·文脉助手**知识库的公开语料来源与抓取产物。仓库里只提交：

- `manifest.yaml`：人工精选的 URL 清单（30–50 篇，覆盖海丝、朱子、妈祖、船政四域）
- `README.md`（本文件）
- `../scripts/fetch_corpus.py`：可重复执行的抓取脚本

抓下来的正文在 `items/` 子目录，已被 `.gitignore` 忽略，**不要 `git add` 正文文件**。换机器后重跑脚本即可复现。

## 怎么抓

在 `wenmai-assistant` 根目录：

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
| 政府网站公开信息（省文旅厅、市县文旅局、湄洲祖庙官网等） | 以页脚版权栏为准；一般可引用，须保留原文 URL 与获取日期 | YAML 头 `license_note` 写「政府网站公开信息，引用时保留 URL」；`retrieved_at` 与 `snapshot_sha256` 记录快照 |
| 中文维基百科条目 | 正文默认 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh)（兼 GFDL）；**必须署名，修改后须以相同条款再许可** | 脚本经 [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) 拉取纯文本（不爬 `/wiki/` HTML），`source_url` 仍指向条目页 |
| UNESCO 世界遗产说明 | 说明文本 CC BY-SA IGO 3.0 | 保留 URL 与获取日期 |

每条抓取结果的 YAML 头字段（勿改字段名）：

```yaml
source_url: https://...
source_org: 福建省文化和旅游厅
license_note: 政府网站公开信息，引用时保留 URL
culture_domain: 妈祖
space: minpai_culture
retrieved_at: 2026-08-25
snapshot_sha256: <正文 SHA256>
```

## robots.txt 策略

抓取前对每条 URL 所在主机请求 `/robots.txt`：

- 仅当响应为 HTTP 200 且内容含 `User-agent` 行时，视为**可核验**，再判断路径是否允许本脚本 User-Agent 访问。
- **未能核验**的主机（返回 404 HTML、超时、或非 robots 内容）**不自动抓取**，脚本会跳过并在日志列出主机。
- `manifest.yaml` 中 `robots_blocked_hosts` 列出的主机（含 `fjtv.net` / `www.fjtv.net`）一律不抓：调研记录其 robots 为客户端下载中间页，规则未知。

一手政府网页仍在清单中供编辑核对与手工保存；自动抓取以维基、UNESCO 等 robots 可核验站点为主。

## 不收录

- 节目成片、完整字幕、海博 TV APK / 直播流
- 登录墙、素材交易后台、内网福云内容库
- `音频清洗` 工作音频与本仓库 `data/audio/`
- 集团自报经营数字（不作本项目指标）

## 文化域

`culture_domain` 取值：`海丝` / `朱子` / `妈祖` / `船政` / `其他`（见 `settings.yaml` 与入库 Enricher 枚举）。
