#!/usr/bin/env python3
"""Fetch public Minpai-culture corpus items listed in data/corpus/manifest.yaml.

Respects robots.txt where it can be verified. Skips hosts with unverifiable
robots rules (including fjtv.net per ticket #32). Existing output files are
not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from markitdown import MarkItDown

USER_AGENT = "WenmaiCorpusFetcher/1.0 (+https://github.com/local/wenmai-assistant; corpus-research)"
REQUEST_DELAY_SEC = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0
DEFAULT_MANIFEST = Path("data/corpus/manifest.yaml")
DEFAULT_OUTPUT_DIR = Path("data/corpus/items")


@dataclass
class RobotsStatus:
    host: str
    verifiable: bool
    allowed: bool | None
    detail: str
    robots_body: str = ""


def http_get(client: httpx.Client, url: str, *, timeout: float = 60.0) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, timeout=timeout)
            if response.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt * 2)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    assert last_error is not None
    raise last_error


class RobotsChecker:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._cache: dict[str, RobotsStatus] = {}
        self._parsers: dict[str, RobotFileParser | None] = {}

    def check(self, url: str) -> RobotsStatus:
        host = urlparse(url).netloc.lower()
        if host in self._cache:
            return self._cache[host]

        robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
        try:
            response = http_get(self._client, robots_url, timeout=30.0)
        except httpx.HTTPError as exc:
            status = RobotsStatus(host, False, None, f"robots.txt 请求失败: {exc}")
            self._cache[host] = status
            return status

        body = response.text
        if response.status_code != 200 or "user-agent" not in body.lower():
            status = RobotsStatus(
                host,
                False,
                None,
                f"robots.txt 未能核验（HTTP {response.status_code}）",
            )
            self._cache[host] = status
            return status

        parser = RobotFileParser()
        parser.parse(body.splitlines())
        self._parsers[host] = parser
        allowed = parser.can_fetch(USER_AGENT, url)
        status = RobotsStatus(host, True, allowed, "robots.txt 已核验", body)
        self._cache[host] = status
        return status


def wikipedia_api_allowed(robots_body: str, api_url: str) -> bool:
    """Wikimedia robots 显式 Allow /w/api.php，但标准 parser 对带 query 的 URL 判断偏严。"""
    path = urlparse(api_url).path
    if path != "/w/api.php":
        return False
    lowered = robots_body.lower()
    return "/w/api.php" in lowered and "allow:" in lowered


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def blocked_hosts(manifest: dict[str, Any]) -> set[str]:
    blocked: set[str] = set()
    for entry in manifest.get("robots_blocked_hosts", []):
        blocked.add(str(entry["host"]).lower())
    return blocked


def body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def yaml_front_matter(item: dict[str, Any], retrieved_at: str, snapshot_sha256: str) -> str:
    fields = {
        "source_url": item["source_url"],
        "source_org": item["source_org"],
        "license_note": item["license_note"],
        "culture_domain": item["culture_domain"],
        "space": item.get("space", "minpai_culture"),
        "retrieved_at": retrieved_at,
        "snapshot_sha256": snapshot_sha256,
    }
    lines = ["---"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def convert_bytes(content: bytes, url: str, client: httpx.Client) -> str:
    converter = MarkItDown()
    result = converter.convert_stream(io.BytesIO(content), file_extension=_guess_extension(url))
    text = (result.text_content or "").strip()
    if text:
        return text
    return content.decode("utf-8", errors="replace").strip()


def _guess_extension(url: str) -> str | None:
    path = urlparse(url).path.lower()
    for ext in (".pdf", ".html", ".htm", ".txt"):
        if path.endswith(ext):
            return ext.lstrip(".")
    return None


def is_wikipedia(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("wikipedia.org")


def wikipedia_title_from_url(url: str) -> str:
    path = urlparse(url).path
    if not path.startswith("/wiki/"):
        raise ValueError(f"非维基条目 URL: {url}")
    return unquote(path[len("/wiki/") :])


def wikipedia_api_url(article_url: str) -> str:
    title = wikipedia_title_from_url(article_url)
    host = urlparse(article_url).netloc
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
    }
    query = "&".join(f"{key}={quote(value)}" for key, value in params.items())
    return f"https://{host}/w/api.php?{query}"


def fetch_wikipedia_markdown(article_url: str, client: httpx.Client) -> str:
    api_url = wikipedia_api_url(article_url)
    response = http_get(client, api_url, timeout=60.0)
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("维基 API 未返回页面")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"维基条目不存在: {article_url}")
    extract = (page.get("extract") or "").strip()
    if not extract:
        raise ValueError("维基正文为空")
    title = page.get("title", wikipedia_title_from_url(article_url))
    return f"# {title}\n\n{extract}"


def fetch_markdown(url: str, client: httpx.Client) -> str:
    if is_wikipedia(url):
        return fetch_wikipedia_markdown(url, client)
    response = http_get(client, url, timeout=60.0)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    is_html_or_pdf = (
        "text/html" in content_type
        or "application/pdf" in content_type
        or url.lower().endswith(".pdf")
    )
    if is_html_or_pdf:
        return convert_bytes(response.content, url, client)
    return response.text.strip()


def write_item(
    output_path: Path,
    item: dict[str, Any],
    body: str,
    retrieved_at: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml_front_matter(item, retrieved_at, body_hash(body))
    output_path.write_text(front + body + "\n", encoding="utf-8")


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    manifest = load_manifest(manifest_path)
    items: list[dict[str, Any]] = manifest.get("items", [])
    blocked = blocked_hosts(manifest)
    retrieved_at = date.today().isoformat()

    headers = {"User-Agent": USER_AGENT}
    skipped_hosts: dict[str, str] = {}
    fetched = 0
    skipped_existing = 0
    skipped_robots = 0
    errors = 0

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        checker = RobotsChecker(client)
        selected = items if limit is None else items[:limit]

        for index, item in enumerate(selected, start=1):
            item_id = item["id"]
            url = item["source_url"]
            host = urlparse(url).netloc.lower()
            output_path = output_dir / f"{item_id}.md"

            if host in blocked:
                skipped_hosts.setdefault(host, "manifest robots_blocked_hosts")
                print(f"[{index}/{len(selected)}] SKIP {item_id}: 主机在排除清单 ({host})")
                skipped_robots += 1
                continue

            if output_path.exists():
                print(f"[{index}/{len(selected)}] SKIP {item_id}: 已存在 {output_path}")
                skipped_existing += 1
                continue

            fetch_url = wikipedia_api_url(url) if is_wikipedia(url) else url
            robots = checker.check(fetch_url)
            if not robots.verifiable:
                skipped_hosts.setdefault(host, robots.detail)
                print(f"[{index}/{len(selected)}] SKIP {item_id}: {robots.detail}")
                skipped_robots += 1
                continue

            allowed = robots.allowed
            if is_wikipedia(url) and allowed is False:
                allowed = wikipedia_api_allowed(robots.robots_body, fetch_url)

            if allowed is False:
                skipped_hosts.setdefault(host, "robots.txt 禁止抓取该路径")
                print(f"[{index}/{len(selected)}] SKIP {item_id}: robots.txt 禁止 {fetch_url}")
                skipped_robots += 1
                continue

            print(f"[{index}/{len(selected)}] FETCH {item_id} <- {url}")
            if dry_run:
                continue

            try:
                body = fetch_markdown(url, client)
                if not body:
                    raise ValueError("正文为空")
                write_item(output_path, item, body, retrieved_at)
                fetched += 1
            except Exception as exc:  # noqa: BLE001 - CLI tool reports and continues
                print(f"  ERROR: {exc}", file=sys.stderr)
                errors += 1

            time.sleep(REQUEST_DELAY_SEC)

    print("\n=== 摘要 ===")
    print(f"抓取成功: {fetched}")
    print(f"已存在跳过: {skipped_existing}")
    print(f"robots/排除跳过: {skipped_robots}")
    print(f"失败: {errors}")
    if skipped_hosts:
        print("\n因 robots 未能核验或禁止而跳过的主机:")
        for host, reason in sorted(skipped_hosts.items()):
            print(f"  - {host}: {reason}")

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 manifest.yaml 中的公开闽派文化语料")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="语料清单路径（默认 data/corpus/manifest.yaml）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（默认 data/corpus/items，已被 .gitignore 忽略）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 robots 与跳过逻辑，不写入文件",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    if not manifest_path.exists():
        print(f"找不到清单: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)

    raise SystemExit(run(manifest_path, output_dir, dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()
