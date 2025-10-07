# Development Setup

## Prerequisites
- Python 3.11+
- uv package manager

## Installation

Install dependencies:
```bash
uv sync
```

Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Development Commands

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/

# Run all checks (format, lint, type check)
uv run pre-commit run --all-files
```

## Before Committing

Pre-commit hooks will automatically run ruff and mypy when you commit.
If checks fail, fix the issues and commit again.
