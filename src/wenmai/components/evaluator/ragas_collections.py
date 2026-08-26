from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

from wenmai.components.evaluator.base import BaseEvaluator
from wenmai.config import Settings
from wenmai.factories.evaluator import registry

logger = logging.getLogger(__name__)

_STATUS_OK = "ok"
_STATUS_UNAVAILABLE = "unavailable"


def _metric_result(value: float | None, status: str, reason: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"value": value, "status": status}
    if reason:
        payload["reason"] = reason
    return payload


def build_ragas_judge_llm(settings: Settings) -> tuple[Faithfulness, ContextPrecision]:
    """Wire Ragas collections metrics with an OpenAI-compatible judge client."""
    judge = settings.ragas_judge
    api_key = os.environ.get(judge.api_key_env)
    if not api_key:
        raise ValueError(f"{judge.api_key_env} is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=judge.base_url)
    llm = llm_factory(judge.model, client=client)
    return Faithfulness(llm=llm), ContextPrecision(llm=llm)


@registry.register("ragas_collections")
class RagasCollectionsEvaluator(BaseEvaluator):
    def __init__(self, settings: Settings | None = None, **kwargs: object) -> None:
        self._init_error = ""
        self._faithfulness: Faithfulness | None = None
        self._context_precision: ContextPrecision | None = None
        if settings is None:
            self._init_error = "settings not provided"
            return
        try:
            self._faithfulness, self._context_precision = build_ragas_judge_llm(settings)
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Ragas judge init failed: %s", exc)

    @property
    def provider_name(self) -> str:
        return "ragas_collections"

    @property
    def judge_available(self) -> bool:
        return self._faithfulness is not None and self._context_precision is not None

    @property
    def judge_unavailable_reason(self) -> str:
        return self._init_error

    def score(self, **kwargs: Any) -> dict[str, Any]:
        user_input = str(kwargs.get("user_input") or "")
        response = str(kwargs.get("response") or "")
        retrieved_contexts = list(kwargs.get("retrieved_contexts") or [])
        reference = str(kwargs.get("reference") or "")

        if not self.judge_available:
            reason = self._init_error or "Ragas judge not initialized"
            unavailable = _metric_result(None, _STATUS_UNAVAILABLE, reason)
            return {
                "faithfulness": unavailable,
                "context_precision": unavailable,
            }

        return {
            "faithfulness": self._score_faithfulness(user_input, response, retrieved_contexts),
            "context_precision": self._score_context_precision(
                user_input, reference, retrieved_contexts
            ),
        }

    def _score_faithfulness(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> dict[str, Any]:
        assert self._faithfulness is not None
        try:
            result = self._faithfulness.score(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            return _metric_result(float(result.value), _STATUS_OK)
        except Exception as exc:
            logger.warning("Faithfulness scoring failed: %s", exc)
            return _metric_result(None, _STATUS_UNAVAILABLE, str(exc))

    def _score_context_precision(
        self,
        user_input: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[str, Any]:
        assert self._context_precision is not None
        if not reference:
            return _metric_result(
                None,
                _STATUS_UNAVAILABLE,
                "reference is required for ContextPrecision",
            )
        try:
            result = self._context_precision.score(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            return _metric_result(float(result.value), _STATUS_OK)
        except Exception as exc:
            logger.warning("ContextPrecision scoring failed: %s", exc)
            return _metric_result(None, _STATUS_UNAVAILABLE, str(exc))
