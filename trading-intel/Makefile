.PHONY: install fmt lint type test check

install:
	uv sync --all-extras

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src tests

test:
	uv run pytest -q --cov=src/trading_intel --cov-report=term-missing

check:
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) test
