from __future__ import annotations


def apply_behavior(behavior: str, name: str) -> None:
    if behavior == "timeout":
        raise TimeoutError(f"fake {name} timeout")
    if behavior == "error":
        raise RuntimeError(f"fake {name} error")
