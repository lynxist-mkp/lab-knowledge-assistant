from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _build_resources(raw: dict[str, Any]) -> Resources:
    payload = dict(raw)
    ask_raw = payload.get("ask")
    if isinstance(ask_raw, dict):
        payload["ask"] = _build(AskConcurrencyConfig, ask_raw)
    return _build(Resources, payload)


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


def _resolve_declared_path(value: str, *, root: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return str(expanded)
    return str((root / expanded).resolve())


@dataclass
class Product:
    name: str
    collection: str


@dataclass(frozen=True)
class CollectionRegistration:
    collection_id: str
    display_name: str


def _build_collections(raw: list[Any]) -> list[CollectionRegistration]:
    registrations: list[CollectionRegistration] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "collections entries must be objects with collection_id and display_name"
            )
        registrations.append(
            CollectionRegistration(
                collection_id=str(item["collection_id"]),
                display_name=str(item["display_name"]),
            )
        )
    return registrations


@dataclass
class Paths:
    chroma: str
    bm25: str
    ingestion_history: str
    catalog: str
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
class QueryProcessing:
    rewriter: str = "lexicon"
    lexicon: str = "data/lexicon/synonyms.yaml"
    multi_query: bool = False
    multi_query_n: int = 3
    multi_query_timeout_seconds: float = 10.0
    multi_query_prompt: str = "prompts/multi_query_v1.txt"


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
    adjacent_n: int = 1


@dataclass
class Generation:
    """生成层拒答硬门：提问词与检索片段重叠过低则直接证据不足拒答。"""

    min_question_overlap: float = 0.0  # 0 = 关闭；建议 0.12–0.2


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
    evaluator: str = "ragas_collections"
    caption_model: str = ""


@dataclass
class Server:
    host: str
    port: int
    templates: str


@dataclass
class Observability:
    trace_file: str
    task_progress_file: str = "logs/task_progress.jsonl"
    ask_evidence_file: str = "logs/ask_evidence.jsonl"
    query_latency_recent_n: int = 50
    ask_evidence_recent_n: int = 100


@dataclass
class RagasJudge:
    provider: str
    provider_label: str
    model: str
    base_url: str
    api_key_env: str
    max_tokens: int = 4096


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
class QualityGate:
    reject_below: float = 0.60
    approve_above: float = 0.80
    gray_review: bool = False
    timeout_seconds: float = 30.0
    preview_chars: int = 2000
    preview_pages: int = 3


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
class AskConcurrencyConfig:
    enabled: bool = True
    max_in_flight: int = 2
    max_wait_seconds: float = 30.0
    saturation_policy: str = "wait"
    long_task_guard: bool = True
    long_task_max_in_flight: int = 1


@dataclass
class Resources:
    single_model_exclusive: bool = True
    query_phase_batch: bool = True
    query_window_batch: bool = False
    ingest_window_batch: bool = True
    batch_window_seconds: float = 3.0
    batch_window_max_size: int = 4
    process_idle_timeout_seconds: float = 60.0
    process_idle_unload: bool = True
    ask: AskConcurrencyConfig = field(default_factory=AskConcurrencyConfig)


@dataclass
class Settings:
    product: Product
    paths: Paths
    chunking: Chunking
    transform: TransformConfig
    retrieval: Retrieval
    generation: Generation
    query_processing: QueryProcessing
    bm25: Bm25
    providers: Providers
    server: Server
    observability: Observability
    evaluation: Evaluation
    dolphin: Dolphin
    pdf_load: PdfLoad
    quality_gate: QualityGate
    paddleocr: PaddleOCR
    gemma: Gemma
    resources: Resources = field(default_factory=Resources)
    collections: list[CollectionRegistration] = field(default_factory=list)
    fakes: dict[str, str] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path("."))
    default_collection_id: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path | None = None) -> Settings:
        resolved_root = Path(root) if root is not None else Path(".")
        product = _build(Product, raw["product"])
        dolphin_raw = dict(raw["dolphin"])
        for key in ("python", "script"):
            dolphin_raw[key] = _resolve_declared_path(
                str(dolphin_raw[key]),
                root=resolved_root,
            )
        paddleocr_raw = dict(raw["paddleocr"])
        for key in ("python", "script", "mlx_python"):
            paddleocr_raw[key] = _resolve_declared_path(
                str(paddleocr_raw[key]),
                root=resolved_root,
            )
        gemma_raw = dict(raw["gemma"])
        for key in ("mlx_python", "resolve_script"):
            gemma_raw[key] = _resolve_declared_path(
                str(gemma_raw[key]),
                root=resolved_root,
            )
        return cls(
            product=product,
            collections=_build_collections(raw.get("collections") or []),
            paths=_build(Paths, raw["paths"]),
            chunking=_build(Chunking, raw["chunking"]),
            transform=_build(TransformConfig, _normalize_transform(raw["transform"])),
            retrieval=_build(Retrieval, raw["retrieval"]),
            generation=_build(Generation, raw.get("generation") or {}),
            query_processing=_build(
                QueryProcessing, raw.get("query_processing") or {}
            ),
            bm25=_build(Bm25, raw["bm25"]),
            providers=_build(Providers, _normalize_providers(raw["providers"])),
            server=_build(Server, raw["server"]),
            observability=_build(Observability, raw["observability"]),
            evaluation=_build_evaluation(raw),
            dolphin=_build(Dolphin, dolphin_raw),
            pdf_load=_build(PdfLoad, raw["pdf_load"]),
            quality_gate=_build(QualityGate, raw.get("quality_gate") or {}),
            paddleocr=_build(PaddleOCR, paddleocr_raw),
            gemma=_build(Gemma, gemma_raw),
            resources=_build_resources(raw.get("resources") or {}),
            fakes=dict(raw.get("fakes") or {}),
            root=resolved_root,
            default_collection_id=product.collection,
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
    max_tokens_raw = payload.get("max_tokens", 4096)
    resolved["max_tokens"] = int(max_tokens_raw)
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
