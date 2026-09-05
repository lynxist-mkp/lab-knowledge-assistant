#!/usr/bin/env bash
# 高+中收口冒烟：跑通 grill 锁定的 8 条能力（单元/路由级，不替代全量 Phase B）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export no_proxy='*' NO_PROXY='*'

echo "== 1–3 质量门 / 待审 / 拒答硬门 =="
"$PY" -m pytest -q \
  tests/test_quality_gate.py \
  tests/test_knowledge.py::test_pending_chunks_excluded_from_dense_and_sparse_search \
  tests/test_knowledge.py::test_approved_after_pending_becomes_searchable \
  tests/test_generation.py \
  --tb=line

echo "== 4 Trace 写入与读取 =="
"$PY" -m pytest -q tests/test_service_query_traces.py tests/test_trace_recorder.py --tb=line

echo "== 5 双面 UI 入口 =="
"$PY" -m pytest -q tests/test_workbench.py --tb=line

echo "== 6 图转文 =="
"$PY" -m pytest -q tests/test_captioner.py --tb=line

echo "== 7 扫描件分流 =="
"$PY" -m pytest -q tests/test_paddleocr_adapter.py --tb=line

echo "== 8 Phase B 入口可调用 =="
"$PY" scripts/run_phase_b_batch.py --help >/dev/null
"$PY" scripts/run_eval_ablation.py --help >/dev/null
echo "phase_b_cli_ok"

echo "ALL_SMOKE_OK"
