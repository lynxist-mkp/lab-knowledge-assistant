# Eval slowness root-cause analysis

**Issue:** #39 量化 eval 耗时结构与变慢根因  
**Date:** 2026-09-01  
**Evidence sources:** `src/lab_knowledge/eval/{runner,pipeline,ragas_metrics}.py`, `src/lab_knowledge/pipelines/query_core.py`, `settings.yaml`, `data/eval/runs/*.json`, `data/eval/golden.jsonl`, `.scratch/qa/eval-run.log`

---

## Executive summary

Current full eval is **~2.4× more work** than historical 42-item runs (400 vs 168 item×ablation operations), runs on a code path that **bypasses `query_core` phase batching**, and therefore **reloads heavy models per item** (280 BGE + 92 CrossEncoder loads observed in the successful retry segment alone). A Ragas judge phase adds up to **~146 sequential external API calls** with **~36% logged failure rate** (mostly `max_tokens`). The observed run **did not finish**: no new `data/eval/runs/*.json`, `eval-run-latest.json` is 0 bytes, log ends mid-Ragas errors.

---

## 1. Work multiplier: golden set × ablations

| Metric | Historical (2026-08-28 runs) | Current |
|--------|------------------------------|---------|
| Golden items | **42** (`item_count` in `20260828T042220Z.json`, `20260828T062404Z.json`) | **100** (`wc -l data/eval/golden.jsonl`) |
| Ablations | 4 (`dense_only`, `sparse_only`, `rrf`, `rrf_rerank`) | 4 (same, `settings.yaml`) |
| **item × ablation operations** | **42 × 4 = 168** | **100 × 4 = 400** |
| **Multiplier** | baseline | **2.38×** |

Each operation is one `eval_item()` call = one `retrieve()` + one `generate()` (ADR 0004).

Answerable/unanswerable split grew proportionally: historical 30/12 → current **73/27**.

Additional Ragas work (not in Aug-28 artifacts): up to **73 answerable items** scored on `rrf_rerank` group only, **2 API calls per item** (Faithfulness + ContextPrecision) → **≤146 judge API calls**.

---

## 2. Model reload count (CrossEncoder / BGE)

From `.scratch/qa/eval-run.log`, counting completed tqdm bars (`391/391` = BGE-M3, `105/105` = CrossEncoder MiniLM):

| Segment | BGE loads (391) | CE loads (105) | Notes |
|---------|-----------------|----------------|-------|
| Run 1 (before `EVAL_EXIT=241`) | 4 | 0 | Killed after ~2 `dense_only` items |
| Run 2 (retry) | **276** | **92** | Main eval body |
| **Total** | **280** | **92** | |

**Expected loads without batching** (exclusive `model_guard`, per-item `hold()`/`release()`):

| Group | Items needing BGE | Items needing CE |
|-------|-------------------|------------------|
| `dense_only` | 100 | 0 |
| `sparse_only` | 0 | 0 |
| `rrf` | 100 | 0 |
| `rrf_rerank` | 100 | 100 |
| **Total** | **300** | **100** |

Observed run-2 counts (276 BGE, 92 CE) are **~92% of expected**, consistent with nearing completion of `rrf_rerank` before Ragas started. The alternating `391`/`105` pattern in the log tail (lines ~350–385) confirms per-item BGE→CE eviction during `rrf_rerank`.

**Root cause:** `eval_item()` calls `retrieve()` → `generate()` directly. It does **not** enter `run_ask_pipeline()` / `begin_batch()`. Each retrieval embed holds `ModelResource.BGE_M3` then releases (unloads); each rerank holds `CROSS_ENCODER` then unloads; each `GemmaMlxClient.generate_text()` holds `MLX_VLM` then calls `shutdown_all_mlx_vlm_managers()` when not in batch — **~400 MLX subprocess cycles** for generation alone.

Contrast: `/ask` path in `query_core.py` batches BGE (phase 2), CrossEncoder (phase 3), and MLX (phase 4) across jobs when `phase_batch=True`.

---

## 3. Eval path vs `model_guard` batching

**Call chain (eval):**

```
run_eval() → _run_grouped_eval()
  for (group, item) in groups × items:     # 400 iterations, sequential
    eval_item()
      retrieve() → run_fusion() → embed (hold BGE, release)
                 → rerank_chunks() if rrf_rerank (hold CE, release)
      generate() → mlx_gemma.generate_text() (hold MLX, shutdown)
```

**`/ask` path (batched):**

```
run_ask_pipeline(jobs, phase_batch=True)
  Phase 1: MLX multi-query (begin_batch MLX)
  Phase 2: retrieve all jobs (begin_batch BGE)
  Phase 3: rerank all jobs (begin_batch CE)
  Phase 4: generate all jobs (begin_batch MLX)
```

| Capability | `/ask` (`query_core.py`) | Eval (`eval/pipeline.py`) |
|------------|--------------------------|---------------------------|
| `begin_batch` / phase batching | Yes (Strategy A) | **No** |
| `query_core` | Yes | **Bypassed** (by design, ADR 0004) |
| Per-item model eviction | Minimized in batch | **Every item** |
| Query trace | Written | Not written |

`settings.yaml` has `query_window_batch: false` (Strategy B); even if enabled, eval does not use the ask pipeline.

---

## 4. Ragas phase: API volume and failure rate

**Configuration** (`settings.yaml`):

```yaml
evaluation:
  ablations: ["dense_only", "sparse_only", "rrf", "rrf_rerank"]
  ragas_judge:
    provider: zhipu
```

`should_run_ragas()` defaults to **on** when `ZHIPUAI_API_KEY` is set (`probe_ragas_judge`). Scoring runs only on `rrf_rerank` group (`ragas_metrics.py`).

**API call structure** (`ragas_collections.py` → `evaluator.score()`):

- Per answerable golden item with successful non-refused generation: **1× Faithfulness** + **1× ContextPrecision** (sequential, synchronous).
- Skips: unanswerable (27), refused, missing generation/chunks.

**Upper bound:** 73 answerable × 2 = **146 API calls**.

**Logged failures** (`.scratch/qa/eval-run.log`, run 2 only):

| Failure type | Count |
|--------------|-------|
| `Faithfulness scoring failed` | **29** |
| `ContextPrecision scoring failed` | **23** |
| **Total failure log lines** | **52** |
| `max_tokens length limit` (Faithfulness) | 28 |
| `max_tokens length limit` (ContextPrecision) | 22 |
| Zhipu content-filter 400 (`code 1301`) | 3 mentions (1 `API call failed` + retry exceeded) |

**Failure rate (log lines / max API calls):** 52 / 146 ≈ **35.6%**. Dominant cause: judge output truncated at `max_tokens` (Ragas default LLM config, not overridden in repo).

Ragas runs **after** all 400 eval items (`attach_ragas_to_artifact` at end of `run_eval`) — sequential blocking phase on external API latency.

---

## 5. Why the process appears "never stopping"

| Observation | Interpretation |
|-------------|----------------|
| Log shows steady `Loading weights` progress through run 2 | **Not stuck** on a single local model call; progressing item-by-item |
| 280 BGE + 92 CE loads + ~400 MLX generations | Wall-clock dominated by **per-item cold model residency** |
| Log tail = Ragas `max_tokens` / content-filter errors | Entered **external API phase** after main loop |
| No new `data/eval/runs/*.json`; `eval-run-latest.json` = **0 bytes** | Run **did not complete** artifact write |
| No `eval run {timestamp} items=…` stdout in log | Process killed or hung before `scripts/run_eval_ablation.py` finished |
| `.scratch/qa/report.md` marks eval "运行中" | Consistent with incomplete run at time of QA report |

**Progression vs stuck:** The log shows forward progress through retrieval/rerank (model loads) into Ragas. "Never stopping" is explained by **multi-hour total duration** (400 × ~30–40s generation per QA ask benchmark + model reload overhead + Ragas API), not an infinite hang on one step. Final state: likely **interrupted during Ragas** or exited before JSON write.

---

## 6. `EVAL_EXIT=241`

| Fact | Detail |
|------|--------|
| Location | Line 12 of `.scratch/qa/eval-run.log` |
| Context | Immediately after run 1; preceded by `resource_tracker: leaked semaphore` warning |
| Run 1 progress | 4 BGE loads only (~2 items of `dense_only`) |
| Prior attempt | `.scratch/qa/report.md` notes previous **EXIT:143** (128+15 = **SIGTERM**) |

`EVAL_EXIT=241` is **not defined in the repo** (wrapper shell echo). `241 = 128 + 113` does not map to a standard POSIX signal (SIGTERM=143, SIGKILL=137). Most likely an **external termination** (parent timeout, manual kill, resource limit) between run 1 and automatic retry. Not a Python/application exit code from `run_eval_ablation.py`.

---

## 7. Structural cost model (current 100-item run)

| Phase | Operations | Dominant cost |
|-------|------------|---------------|
| `dense_only` | 100 retrieve+generate | BGE load ×100, MLX cycle ×100 |
| `sparse_only` | 100 retrieve+generate | MLX cycle ×100 |
| `rrf` | 100 retrieve+generate | BGE load ×100, MLX cycle ×100 |
| `rrf_rerank` | 100 retrieve+generate | BGE+CE load ×100 each, MLX ×100 |
| Ragas | ≤73 items × 2 API | Zhipu latency + 36% failures/retries |
| **Total eval_item ops** | **400** | |
| **Est. BGE loads** | **300** (observed 276 in incomplete run) | |
| **Est. CE loads** | **100** (observed 92) | |
| **Est. MLX restarts** | **~400** | |

Compared to historical 42-item run: **168 eval_item ops**, **~126 BGE + 42 CE** loads (same per-item pattern, smaller N).

---

## 8. Contributing factors (ranked)

1. **Golden set 42→100** (2.38× eval_item operations) — #41 expansion.
2. **No phase batching on eval path** — per-item `model_guard` eviction reloads BGE/CE/MLX; log proves hundreds of reloads.
3. **Ragas judge enabled by default** — adds ≤146 sequential API calls post-hoc; 35%+ failures add wasted latency.
4. **MLX subprocess shutdown after every `generate()`** — `GemmaMlxClient._finish_mlx_call()` when not `in_batch()`.
5. **Run instability** — EVAL_EXIT 241/143; incomplete artifact; possible concurrent QA load (report notes eval blocking `/ask`).

---

## 9. Evidence gaps

- No completed 100-item run artifact for wall-clock latency percentiles (`latency_ms` field).
- Exact Ragas `scored_count` / `skipped_count` unknown (artifact not written).
- `EVAL_EXIT=241` launcher script not in repo; signal semantics inferred from log context.

---

## References

- `src/lab_knowledge/eval/runner.py` — `_run_grouped_eval`: nested `for group in groups for item in items`
- `src/lab_knowledge/eval/pipeline.py` — `eval_item`: direct `retrieve` + `generate`
- `src/lab_knowledge/pipelines/query_core.py` — `run_ask_pipeline` phase batching
- `src/lab_knowledge/components/model_guard.py` — `hold()` evicts other resources; `end_batch()` unloads
- `docs/adr/0004-eval-no-trace.md` — eval bypasses `ask_question` by design
