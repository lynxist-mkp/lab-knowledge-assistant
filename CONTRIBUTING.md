# Contributing

Thank you for contributing to `lab-knowledge-assistant`.

## Development Setup

1. Install Python 3.12 and `uv`.
2. Sync dependencies:

```bash
uv sync --extra dev
```

3. Copy environment variables if needed:

```bash
cp .env.example .env
```

## Local Checks

Run the standard checks before opening a pull request:

```bash
uv run ruff check .
uv run pytest
```

If you change behavior, add or update focused tests when they materially reduce regression risk.

## Coding Notes

- Keep terminology aligned with `CONTEXT.md`.
- Prefer small, reviewable changes.
- Do not commit private corpora, API keys, local model caches, or audio files.
- Preserve the refusal and citation contract for answers backed by the knowledge base.

## Pull Requests

When opening a pull request, include:

- what changed
- why it changed
- how you verified it
- any follow-up work or known limitations
