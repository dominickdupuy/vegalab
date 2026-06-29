.PHONY: check lint format typecheck test smoke

# Interpreter that has the dev tools installed. Override on the CLI if needed,
# e.g. `make check PY=python` when the venv is already activated.
PY ?= .venv/Scripts/python

check: lint typecheck test

lint:
	$(PY) -m ruff check optspread tests
	$(PY) -m ruff format --check optspread tests

format:
	$(PY) -m ruff format optspread tests
	$(PY) -m ruff check --fix optspread tests

typecheck:
	$(PY) -m mypy --strict optspread

test:
	$(PY) -m pytest tests/ --cov=optspread --cov-report=term-missing

smoke:
	$(PY) -m optspread.cli.smoke_run
