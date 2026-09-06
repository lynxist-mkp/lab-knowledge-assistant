from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ProviderRegistry[T]:
    """Name → implementation. New provider = register a class, no if/elif."""

    def __init__(self) -> None:
        self._providers: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            self._providers[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T] | None:
        return self._providers.get(name)

    def create(self, name: str, **kwargs: Any) -> T:
        impl = self._providers.get(name)
        if impl is None:
            known = ", ".join(sorted(self._providers)) or "(none)"
            raise KeyError(f"unknown provider {name!r}; registered: {known}")
        return impl(**kwargs)
