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
    refiner: str
    enricher: str
    captioner: str
    domains: list[str]
    stages: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    dense_k: int
    sparse_k: int
    rrf_k: int
    fused_k: int
    rerank_top: int


@dataclass
class Bm25:
    k1: float
    b: float


@dataclass
class Providers:
    llm: str
    llm_model: str
    vision: str
    vision_model: str
    embedding: str
    embedding_model: str
    reranker: str
    reranker_model: str
    splitter: str
    vector_store: str
    evaluator: str


@dataclass
class Server:
    host: str
    port: int
    templates: str


@dataclass
class Observability:
    trace_file: str


@dataclass
class Evaluation:
    golden_set: str
    runs: str
    ablations: list[str]


@dataclass
class Dolphin:
    python: str
    script: str
    model: str
    lang: str
    region: str


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
    fakes: dict[str, str] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path | None = None) -> Settings:
        return cls(
            product=_build(Product, raw["product"]),
            paths=_build(Paths, raw["paths"]),
            chunking=_build(Chunking, raw["chunking"]),
            transform=_build(TransformConfig, raw["transform"]),
            retrieval=_build(Retrieval, raw["retrieval"]),
            bm25=_build(Bm25, raw["bm25"]),
            providers=_build(Providers, raw["providers"]),
            server=_build(Server, raw["server"]),
            observability=_build(Observability, raw["observability"]),
            evaluation=_build(Evaluation, raw["evaluation"]),
            dolphin=_build(Dolphin, raw["dolphin"]),
            fakes=dict(raw.get("fakes") or {}),
            root=Path(root) if root is not None else Path("."),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        settings_path = path or Path(__file__).resolve().parents[2] / "settings.yaml"
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, root=settings_path.parent)

    def fake_behavior(self, name: str) -> str:
        return self.fakes.get(name, "ok")
