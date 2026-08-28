#!/usr/bin/env python3
"""Probe Ragas collections wiring against the configured OpenAI-compatible judge."""

from __future__ import annotations

import json
import sys

from wenmai.components.evaluator.ragas_probe import build_ragas_judge_llm
from wenmai.config import Settings

_SAMPLE = {
    "user_input": "朱熹出生于哪一年？",
    "response": "朱熹出生于1130年。",
    "retrieved_contexts": ["朱熹（1130年－1200年），字元晦，号晦庵。"],
    "reference": "1130年。",
}


def main() -> int:
    settings = Settings.load()
    judge = settings.evaluation.ragas_judge
    print(
        f"judge provider={judge.provider!r} model={judge.model!r} "
        f"base_url={judge.base_url!r} api_key_env={judge.api_key_env!r}"
    )

    try:
        faithfulness, context_precision = build_ragas_judge_llm(settings)
    except Exception as exc:
        print(f"structured_output: no")
        print(f"reason: judge init failed: {exc}")
        return 1

    try:
        faith = faithfulness.score(
            user_input=_SAMPLE["user_input"],
            response=_SAMPLE["response"],
            retrieved_contexts=_SAMPLE["retrieved_contexts"],
        )
        precision = context_precision.score(
            user_input=_SAMPLE["user_input"],
            reference=_SAMPLE["reference"],
            retrieved_contexts=_SAMPLE["retrieved_contexts"],
        )
    except Exception as exc:
        print(f"structured_output: no")
        print(f"reason: metric scoring failed: {exc}")
        return 1

    payload = {
        "structured_output": "yes",
        "faithfulness": faith.value,
        "context_precision": precision.value,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
