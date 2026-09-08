# ADR 0003 — Ban direct clock access; inject time everywhere

- **Status**: Accepted
- **Date**: 2026-09-08
- **Phase**: 0 (`core/clock.py`, enforced by ruff and by `tests/guard/`)

## Context

`datetime.now()` is the most dangerous function in a trading system, precisely
because it looks harmless.

A single `datetime.now()` inside a feature calculation makes that feature
untestable (the result changes every run), unreproducible (a replay of the same
events produces different output), and unsafe in a backtest (the calculation
sees the real present while the rest of the system believes it is 2020). The
failure is silent: nothing errors, the numbers are simply wrong, and they are
wrong in the optimistic direction.

`datetime.utcnow()` is worse still — it returns a *naive* datetime that looks
like UTC, so comparing it against an aware timestamp raises at runtime, and
attaching a timezone to it later produces a value that is quietly off by the
local offset.

## Decision

1. `trading_intel.core.clock.utc_now()` is the only sanctioned way to read the
   current time. It always returns a tz-aware UTC datetime.
2. The active clock lives in a `ContextVar`, so a simulated clock installed by
   one task or test does not leak into another.
3. `datetime.now`, `datetime.utcnow`, `datetime.today` and `time.time` are
   banned project-wide by ruff's `flake8-tidy-imports` banned-api rule, with
   `core/clock.py` as the sole exemption.
4. `tests/guard/test_no_direct_time_access.py` walks the AST of every source
   file and fails on the same calls.

## Rationale

Rules 3 and 4 overlap on purpose, because they fail differently:

- ruff's rule works on the import graph. It catches the common spelling and it
  catches it in the editor, before commit.
- The AST guard catches calls reached through an alias — `import datetime as dt;
  dt.datetime.now()` — which the import-based rule does not see.

Neither alone is sufficient; together they make the rule hold without anyone
remembering it. That is the whole point: this is not a convention that reviewers
enforce, it is a property of the build.

`ContextVar` rather than a module-level global is a deliberate second decision.
A global clock is fine until two tests run in parallel, or an async task is
suspended mid-backtest, at which point one context's simulated time silently
becomes another's. The bug that produces is essentially undiagnosable, so the
cheaper structure was chosen up front.

`ensure_utc` raising `NaiveDatetimeError` rather than assuming UTC is the same
reasoning: a naive datetime is a question we cannot answer, and guessing is how
an off-by-8-hours bug reaches production.

## Consequences

**Accepted costs**

- Every module that needs the time takes a dependency on `core.clock`. This is
  a small, deliberate coupling.
- Third-party libraries call `datetime.now()` internally and we cannot stop
  them. `no_network()` and `backtest_mode()` limit the blast radius by cutting
  the data path, but a library that stamps its own records with wall time is
  outside what this ADR can enforce.
- Contributors will hit the ban and be briefly annoyed. The error message names
  the replacement, which is most of the fix.

**Benefit that pays for it**

`backtest_mode(asof)` can freeze the entire system at a point in time with one
context manager, because there is exactly one place that reads the clock.
