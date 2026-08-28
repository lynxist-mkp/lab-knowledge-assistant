from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in data:
            kwargs[item.name] = data[item.name]
        elif item.default is not MISSING:
            kwargs[item.name] = item.default
        elif item.default_factory is not MISSING:
            kwargs[item.name] = item.default_factory()
    return cls(**kwargs)


@dataclass
class Product:
    name: str
    collection: str


@dataclass
class Paths:
    chroma: str
    bm25: str
    ingestion_history: str
    images: str
    image_index: str
    traces: str
    corpus: str
    prompts: str


@dataclass
class Chunking:
    size_chars: int
    overlap_ratio: float

    @property
    def overlap_chars(self) -> int:
        return int(self.size_chars * self.overlap_ratio)


@dataclass
class TransformConfig:
    domains: list[str]
    enricher_prompt: str
    captioner_prompt: str
    stages: list[str] = field(default_factory=list)
    refiner: str = "rule"
    enricher: str = "llm"
    captioner: str = "vision"
    refiner_min_ratio: float = 0.5


@dataclass
class Retrieval:
    dense_k: int
    sparse_k: int
    rrf_k: int
    fused_k: int
    rerank_top: int
    mode: str = "rrf"
    rerank_enabled: bool = True
    rerank_timeout_seconds: float = 30.0


@dataclass
class Bm25:
    k1: float
    b: float
    domain_dict: str = "data/lexicon/domain.txt"
    stopwords: str = "data/lexicon/stopwords.txt"


@dataclass
class Providers:
    multimodal: str
    multimodal_model: str
    embedding: str
    embedding_model: str
    reranker: str
    reranker_model: str
    splitter: str
    vector_store: str
    caption_model: str = ""


@dataclass
class Server:
    host: str
    port: int
    templates: str


@dataclass
class Observability:
    trace_file: str


@dataclass
class RagasJudge:
    provider: str
    provider_label: str
    model: str
    base_url: str
    api_key_env: str


@dataclass
class Evaluation:
    golden_set: str
    runs: str
    ablations: list[str]
    ragas_judge: RagasJudge


@dataclass
class Dolphin:
    python: str
    script: str
    model: str
    lang: str
    region: str


@dataclass
class PdfLoad:
    mode: str
    chars_per_page_threshold: int


@dataclass
class PaddleOCR:
    python: str
    script: str
    mlx_python: str
    mlx_model: str
    mlx_fallback_model: str
    server_port: int
    server_url: str
    vl_rec_api_model_name: str
    idle_timeout_seconds: int


@dataclass
class Gemma:
    mlx_python: str
    resolve_script: str
    model: str
    server_port: int
    server_url: str
    idle_timeout_seconds: int


@dataclass
class Settings:
    product: Product
    paths: Paths
    chunking: Chunking
    transform: TransformConfig
    retrieval: Retrieval
    bm25: Bm25
    providers: Providers
    server: Server
    observability: Observability
    evaluation: Evaluation
    dolphin: Dolphin
    pdf_load: PdfLoad
    paddleocr: PaddleOCR
    gemma: Gemma
    fakes: dict[str, str] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path | None = None) -> Settings:
        return cls(
            product=_build(Product, raw["product"]),
            paths=_build(Paths, raw["paths"]),
            chunking=_build(Chunking, raw["chunking"]),
            transform=_build(TransformConfig, _normalize_transform(raw["transform"])),
            retrieval=_build(Retrieval, raw["retrieval"]),
            bm25=_build(Bm25, raw["bm25"]),
            providers=_build(Providers, _normalize_providers(raw["providers"])),
            server=_build(Server, raw["server"]),
            observability=_build(Observability, raw["observability"]),
            evaluation=_build_evaluation(raw),
            dolphin=_build(Dolphin, raw["dolphin"]),
            pdf_load=_build(PdfLoad, raw["pdf_load"]),
            paddleocr=_build(PaddleOCR, raw["paddleocr"]),
            gemma=_build(Gemma, raw["gemma"]),
            fakes=dict(raw.get("fakes") or {}),
            root=Path(root) if root is not None else Path("."),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        settings_path = path or Path(__file__).resolve().parents[2] / "settings.yaml"
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, root=settings_path.parent)

    def fake_behavior(self, name: str) -> str:
        if name in self.fakes:
            return self.fakes[name]
        aliases = {
            "multimodal": ("llm",),
            "caption": ("vision", "multimodal", "llm"),
        }
        for alt in aliases.get(name, ()):
            if alt in self.fakes:
                return self.fakes[alt]
        return "ok"


_RAGAS_JUDGE_PRESETS: dict[str, dict[str, str]] = {
    "zhipu": {
        "provider": "zhipu",
        "provider_label": "智谱",
        "model": "glm-5.3-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "deepseek": {
        "provider": "deepseek",
        "provider_label": "DeepSeek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

_RAGAS_JUDGE_PROVIDER_ALIASES: dict[str, str] = {
    "zhipu": "zhipu",
    "ds": "deepseek",
    "deepseek": "deepseek",
}


def resolve_ragas_judge(raw: dict[str, Any] | None) -> RagasJudge:
    """Normalize Ragas judge config; default provider is 智谱 (zhipu)."""
    payload = dict(raw or {})
    provider_raw = str(payload.pop("provider", "") or "").strip().lower()
    if not provider_raw:
        canonical = "zhipu"
    else:
        canonical = _RAGAS_JUDGE_PROVIDER_ALIASES.get(provider_raw)
        if canonical is None:
            raise ValueError(f"unknown ragas_judge provider: {provider_raw!r}")
    resolved = dict(_RAGAS_JUDGE_PRESETS[canonical])
    for key in ("model", "base_url", "api_key_env"):
        if key in payload:
            resolved[key] = str(payload[key])
    return RagasJudge(**resolved)


def _build_evaluation(raw: dict[str, Any]) -> Evaluation:
    eval_raw = dict(raw["evaluation"])
    legacy_judge = raw.get("ragas_judge") or {}
    nested_judge = eval_raw.pop("ragas_judge", None) or {}
    eval_raw["ragas_judge"] = resolve_ragas_judge({**legacy_judge, **nested_judge})
    return _build(Evaluation, eval_raw)


def _normalize_transform(raw: dict[str, Any]) -> dict[str, Any]:
    """Restore provider keys used by the transform registry."""
    payload = dict(raw)
    payload.setdefault("refiner", "rule")
    payload.setdefault("enricher", "llm")
    payload.setdefault("captioner", "vision")
    return payload


def _normalize_providers(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept multimodal keys, or legacy llm+vision when they match."""
    payload = dict(raw)
    if "multimodal" in payload:
        multimodal = str(payload["multimodal"])
        model = str(payload.get("multimodal_model") or payload.get("llm_model") or "")
        caption = str(payload.get("caption_model") or payload.get("vision_model") or "")
    else:
        llm = str(payload.get("llm") or "")
        vision = str(payload.get("vision") or "")
        if not llm:
            raise ValueError("providers.multimodal (or legacy providers.llm) is required")
        if vision and vision != llm:
            raise ValueError(
                "providers.llm and providers.vision must match; "
                f"got llm={llm!r} vision={vision!r}. "
                "Use providers.multimodal instead."
            )
        multimodal = llm
        model = str(payload.get("llm_model") or "")
        caption = str(payload.get("vision_model") or "")
    cleaned = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "llm",
            "vision",
            "llm_model",
            "vision_model",
            "multimodal",
            "multimodal_model",
            "caption_model",
            "evaluator",
        }
    }
    cleaned["multimodal"] = multimodal
    cleaned["multimodal_model"] = model
    if caption and caption != model:
        cleaned["caption_model"] = caption
    else:
        cleaned["caption_model"] = ""
    return cleaned
