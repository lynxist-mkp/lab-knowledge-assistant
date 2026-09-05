"""Stdio safety discipline for MCP transport."""

from __future__ import annotations

import sys

from wenmai.mcp.stdio_safety import (
    assert_stdout_reserved_for_protocol,
    install_mcp_stdio_discipline,
    is_mcp_stdio_discipline_installed,
    redact_error_message,
)


def test_redact_error_message_bounds_and_redacts_paths_and_secrets() -> None:
    message = (
        "failed at /Users/secret/project/file.py with api_key=abc123 "
        + ("x" * 600)
    )
    redacted = redact_error_message(message, max_len=120)
    assert len(redacted) <= 120
    assert "/Users" not in redacted
    assert "abc123" not in redacted
    assert "[PATH]" in redacted
    assert "[REDACTED]" in redacted


def test_install_mcp_stdio_discipline_redirects_print_to_stderr() -> None:
    install_mcp_stdio_discipline()
    assert is_mcp_stdio_discipline_installed()

    captured_stderr = []

    class _StderrCapture:
        def write(self, text: str) -> int:
            captured_stderr.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    original_stderr = sys.stderr
    sys.stderr = _StderrCapture()  # type: ignore[assignment]
    try:
        print("operational-log-line")
    finally:
        sys.stderr = original_stderr

    assert any("operational-log-line" in item for item in captured_stderr)


def test_assert_stdout_reserved_for_protocol_passes_for_clean_action() -> None:
    assert_stdout_reserved_for_protocol(lambda: None)


def test_assert_stdout_reserved_for_protocol_fails_on_stdout_write() -> None:
    try:
        assert_stdout_reserved_for_protocol(lambda: sys.stdout.write("pollution"))
        raised = False
    except AssertionError:
        raised = True
    assert raised
