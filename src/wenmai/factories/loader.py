from __future__ import annotations

import importlib
import pkgutil

_COMPONENT_PACKAGES = (
    "wenmai.components.embedding",
    "wenmai.components.evaluator",
    "wenmai.components.multimodal",
    "wenmai.components.reranker",
    "wenmai.components.splitter",
    "wenmai.components.transform",
    "wenmai.components.vector_store",
)

_loaded = False


def _import_implementations(package_name: str) -> None:
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        leaf = module_info.name.rsplit(".", 1)[-1]
        if leaf == "base":
            continue
        importlib.import_module(module_info.name)


def ensure_providers() -> None:
    """Import every implementation module so @registry.register runs.

    A later ticket adds a file under one of these packages; this walker
    picks it up. Do not add per-provider imports here.
    """
    global _loaded
    if _loaded:
        return
    for package_name in _COMPONENT_PACKAGES:
        _import_implementations(package_name)
    _loaded = True
