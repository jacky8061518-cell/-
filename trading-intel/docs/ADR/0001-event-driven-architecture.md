# ADR 0001 — Event-driven architecture over batch scheduling

- **Status**: Accepted
- **Date**: 2026-09-08
- **Phase**: 0 (contracts only; no bus implementation yet)

## Context

The system ingests prices, filings, and news, derives features and signals, and
turns those into order intents. Two shapes were available:

1. **Batch schedule** — a cron chain: fetch at 18:00, compute features at 18:30,
   score at 19:00, publish at 19:15.
2. **Event-driven** — each fact enters as an event and each stage reacts.

Batch is much simpler to build and much simpler to reason about at 3am. It is
also what most research code starts as, and there is real cost in walking away
from that.

## Decision

Facts move through the system as immutable events in an `EventEnvelope`, and
every stage is a function from events to events.

## Rationale

The deciding factor is not throughput, it is **replay**.

- A batch pipeline's state is "whatever the last run left in the tables". You
  cannot ask it what it believed on 2020-03-19 without rebuilding the database
  as of that date, which in practice means you never ask.
- An event log can be replayed. Combined with deterministic ids (ADR 0003's
  sibling concern — see `core/ids.py`) and a frozen contract set, replaying the
  same events reproduces the same signals, byte for byte. That property is what
  makes a backtest an argument rather than an anecdote.
- Latency-sensitive paths (a halt, a data-quality alert that must flip an entity
  to `NO_TRADE`) do not fit a nightly cadence. In batch they become special
  cases; in an event model they are ordinary events.
- Stage isolation: a broken news parser stops producing `Document` events. It
  does not corrupt the price path, because the price path never reads its
  tables.

## Consequences

**Accepted costs**

- Ordering and duplicate delivery are now our problem. Mitigated by
  `idempotency_key`, derived from a canonical hash of payload content, so a
  redelivered fact is a no-op rather than a double count.
- Debugging is harder: there is no single table to `SELECT *` from. Mitigated by
  `correlation_id` threaded through the envelope and injected into every log
  line by `core/logging.py`.
- Eventual consistency between stages must be tolerated by anything that reads
  across them.

**Deferred to later phases**

- No broker is chosen here. Phase 1 may run the "bus" as an in-process queue,
  and the envelope contract is deliberately transport-agnostic so that choice
  stays cheap.

## Alternatives rejected

- **Batch with snapshot tables per run.** Gets partway to replay, but the
  snapshot is of derived state, not of inputs, so a bug fix cannot be replayed
  against history — only re-derived by the same buggy code.
- **Streaming framework (Flink/Beam) from day one.** Operationally heavy for a
  system that has not yet proven a single signal.
