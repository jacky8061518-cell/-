.PHONY: install fmt lint type test coverage-gates check

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

# CLAUDE.md 品質門檻：src/features、src/backtest、src/risk 不得低於 90%。
# 寫成自動檢查而非靠人記得——與 tests/guard/ 的原則一致。
coverage-gates:
	@for pkg in features backtest risk; do \
		uv run coverage report --include="src/trading_intel/$$pkg/*" --fail-under=90 > /dev/null \
			&& echo "覆蓋率門檻通過：src/$$pkg" \
			|| { echo "覆蓋率不足 90%：src/$$pkg"; \
			     uv run coverage report --include="src/trading_intel/$$pkg/*"; exit 1; }; \
	done

check:
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) test
	$(MAKE) coverage-gates
