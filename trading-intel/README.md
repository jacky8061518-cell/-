# trading-intel

Phase 0 skeleton. No data sources, no models, no agents — only the three things
that are expensive to change later:

1. **Temporal discipline** — `core.clock.utc_now` is the single entry point for
   "what time is it". Direct `datetime.now` / `utcnow` / `today` / `time.time`
   calls are banned by `ruff` and by an AST guard test.
2. **Type contracts** — every model in `core.types` is a frozen, `extra="forbid"`
   Pydantic model with its temporal invariants enforced at construction.
3. **Guard rails** — `core.sandbox.backtest_mode` freezes the clock and cuts the
   network, so a backtest cannot look ahead or phone home.

## Quick start

```bash
make install   # uv sync --all-extras
make check     # fmt -> lint -> type -> test
```

## Layout

```
src/trading_intel/core/
  clock.py     the only file allowed to read the current time
  enums.py     closed vocabularies (StrEnum / IntEnum)
  ids.py       deterministic identifiers
  errors.py    exception hierarchy, each carrying a context dict
  types.py     frozen Pydantic contracts
  events.py    event envelope + order-independent payload hashing
  sandbox.py   no_network() and backtest_mode()
  settings.py  layered YAML + env configuration
  logging.py   structlog wiring
tests/guard/     rules that enforce themselves
tests/unit/      per-module behaviour
tests/property/  hypothesis invariants
docs/ADR/        decision records
```
