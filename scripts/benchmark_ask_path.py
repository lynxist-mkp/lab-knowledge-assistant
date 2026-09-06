#!/usr/bin/env python3
"""Repeatable local benchmark for the ask/MCP service path (Phase C evidence)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from lab_knowledge.config import Settings
from lab_knowledge.http.ask_governor import AskSaturationError, reset_ask_governor
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.knowledge import create_knowledge


@dataclass
class RequestOutcome:
    ok: bool
    elapsed_ms: float
    error_code: str | None = None


@dataclass
class WorkerResult:
    worker_id: int
    outcomes: list[RequestOutcome]


def _load_benchmark_settings(root: Path, tmp_dir: Path) -> Settings:
    settings_path = root / "settings.yaml"
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    raw["paths"] = {
        "chroma": str(tmp_dir / "chroma"),
        "bm25": str(tmp_dir / "bm25"),
        "ingestion_history": str(tmp_dir / "ingestion_history.db"),
        "catalog": str(tmp_dir / "catalog.json"),
        "images": str(tmp_dir / "images"),
        "image_index": str(tmp_dir / "image_index.db"),
        "traces": str(tmp_dir / "traces.jsonl"),
        "corpus": str(tmp_dir / "corpus"),
        "prompts": raw["paths"]["prompts"],
    }
    raw["observability"] = {
        "trace_file": str(tmp_dir / "traces.jsonl"),
        "task_progress_file": str(tmp_dir / "task_progress.jsonl"),
        "ask_evidence_file": str(tmp_dir / "ask_evidence.jsonl"),
    }
    raw["providers"]["multimodal"] = "fake"
    raw["providers"]["embedding"] = "fake"
    raw["providers"]["reranker"] = "fake"
    raw["fakes"] = {
        "multimodal": "ok",
        "embedding": "ok",
        "reranker": "ok",
    }
    raw["query_processing"] = {"rewriter": "none", "multi_query": False}
    raw["generation"] = {"min_question_overlap": 0.0}
    raw["resources"] = dict(raw.get("resources") or {})
    # Concurrent scenarios exercise ask-governor admission, not model_guard
    # phase-batch exclusivity (which raises "batch already active").
    raw["resources"]["query_phase_batch"] = False
    raw["resources"]["single_model_exclusive"] = False
    return Settings.from_dict(raw, root=root)


def _seed_doc(settings: Settings, knowledge) -> None:
    corpus_dir = settings.root / settings.paths.corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source = corpus_dir / "benchmark.md"
    source.write_text(
        "湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上。\n",
        encoding="utf-8",
    )
    from lab_knowledge.pipelines.ingestion import run_prepare_commit

    run_prepare_commit(source, settings, knowledge=knowledge)


def _run_one(
    settings: Settings,
    knowledge,
    question: str,
) -> RequestOutcome:
    started = time.perf_counter()
    try:
        run_ask(question, settings, knowledge=knowledge, entrypoint="benchmark")
    except AskSaturationError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return RequestOutcome(ok=False, elapsed_ms=elapsed_ms, error_code=exc.code)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return RequestOutcome(ok=False, elapsed_ms=elapsed_ms, error_code="error")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return RequestOutcome(ok=True, elapsed_ms=elapsed_ms)


def _worker(
    worker_id: int,
    settings: Settings,
    knowledge,
    question: str,
    requests: int,
) -> WorkerResult:
    outcomes = [
        _run_one(settings, knowledge, question)
        for _ in range(requests)
    ]
    return WorkerResult(worker_id=worker_id, outcomes=outcomes)


def _summarize(all_outcomes: list[RequestOutcome]) -> dict[str, Any]:
    ok_latencies = [item.elapsed_ms for item in all_outcomes if item.ok]
    failures = [item for item in all_outcomes if not item.ok]
    by_code: dict[str, int] = {}
    for item in failures:
        code = item.error_code or "unknown"
        by_code[code] = by_code.get(code, 0) + 1
    summary: dict[str, Any] = {
        "total": len(all_outcomes),
        "success": len(ok_latencies),
        "failure": len(failures),
        "failure_by_code": by_code,
    }
    if ok_latencies:
        summary["latency_ms"] = {
            "p50": statistics.median(ok_latencies),
            "mean": statistics.mean(ok_latencies),
            "max": max(ok_latencies),
        }
    return summary


def run_scenario(
    settings: Settings,
    knowledge,
    *,
    workers: int,
    requests_per_worker: int,
    question: str,
) -> dict[str, Any]:
    reset_ask_governor()
    started = time.perf_counter()
    if workers == 1:
        result = _worker(0, settings, knowledge, question, requests_per_worker)
        worker_results = [result]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_worker, worker_id, settings, knowledge, question, requests_per_worker)
                for worker_id in range(workers)
            ]
            worker_results = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter() - started) * 1000
    outcomes = [item for result in worker_results for item in result.outcomes]
    return {
        "workers": workers,
        "requests_per_worker": requests_per_worker,
        "elapsed_ms": elapsed_ms,
        "summary": _summarize(outcomes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ask/MCP service path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/phase-c-ask-benchmark.json"),
        help="JSON output path for evidence",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Concurrency levels to test",
    )
    parser.add_argument("--requests", type=int, default=3, help="Requests per worker")
    parser.add_argument(
        "--question",
        default="妈祖信仰的发源地在哪里？",
        help="Benchmark question",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tmp_dir = root / ".scratch" / "phase-c-benchmark"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    settings = _load_benchmark_settings(root, tmp_dir)
    knowledge = create_knowledge(settings)
    _seed_doc(settings, knowledge)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness": {
            "query_phase_batch": settings.resources.query_phase_batch,
            "single_model_exclusive": settings.resources.single_model_exclusive,
            "note": (
                "Phase-batch and exclusive model residency are disabled so concurrent "
                "failures reflect ask-governor saturation (busy/timeout/long_task_active), "
                "not model_guard batch mutex collisions."
            ),
        },
        "ask_config": asdict(settings.resources.ask),
        "scenarios": [],
    }
    for workers in args.workers:
        report["scenarios"].append(
            run_scenario(
                settings,
                knowledge,
                workers=workers,
                requests_per_worker=args.requests,
                question=args.question,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
