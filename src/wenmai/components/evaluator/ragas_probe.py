from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

from wenmai.config import Settings

logger = logging.getLogger(__name__)


def build_ragas_judge_llm(settings: Settings) -> tuple[Faithfulness, ContextPrecision]:
    """Wire Ragas collections metrics with an OpenAI-compatible judge client."""
    judge = settings.ragas_judge
    api_key = os.environ.get(judge.api_key_env)
    if not api_key:
        raise ValueError(f"{judge.api_key_env} is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=judge.base_url)
    llm = llm_factory(judge.model, client=client)
    return Faithfulness(llm=llm), ContextPrecision(llm=llm)


def probe_ragas_judge(settings: Settings) -> tuple[bool, str]:
    """Return whether the configured Ragas judge can initialize."""
    try:
        build_ragas_judge_llm(settings)
    except Exception as exc:
        logger.debug("Ragas judge probe failed: %s", exc)
        return False, str(exc)
    return True, ""
