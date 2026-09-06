# Phase C Ask Path Baseline

Repeatable local evidence for stdio-first ask/MCP hardening.

## How to run

From `lab-knowledge-assistant/`:

```bash
python scripts/benchmark_ask_path.py --output research/phase-c-ask-benchmark.json
```

Optional concurrency sweep:

```bash
python scripts/benchmark_ask_path.py --workers 1 2 4 8 --requests 5
```

## What it measures

- Serial baseline latency (`--workers 1`)
- Bounded concurrency throughput at 1/2/N workers
- Failure rate under saturation (`busy`, `timeout`, `long_task_active`)
- Governor config snapshot included in JSON output

## Harness notes

The benchmark disables `query_phase_batch` and `single_model_exclusive` so concurrent
scenarios measure ask-governor admission (`busy`, `timeout`, `long_task_active`) instead
of tripping `model_guard` phase-batch mutex collisions (`batch already active`).

- Governor defaults: `max_in_flight=2`, `saturation_policy=wait`, `max_wait_seconds=30`
- Long-task guard reduces effective capacity to `long_task_max_in_flight=1` while ingestion/eval runs
- Saturation events are written to `logs/ask_evidence.jsonl` and surfaced in ops health signals

## Baseline placeholder

Run the benchmark after deployment and paste summary JSON here for future comparison.

```json
{
  "note": "Run scripts/benchmark_ask_path.py to populate research/phase-c-ask-benchmark.json"
}
```
