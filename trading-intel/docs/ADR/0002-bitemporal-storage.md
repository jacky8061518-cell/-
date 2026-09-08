# ADR 0002 — Bitemporal facts: `event_time` and `ingest_time`

- **Status**: Accepted
- **Date**: 2026-09-08
- **Phase**: 0 (enforced in `TemporalModel`; storage layer is Phase 1)

## Context

Every fact has two timestamps that are routinely conflated:

- **`event_time`** — when the fact became true in the world. An earnings release
  is dated to the moment it was published.
- **`ingest_time`** — when *we* learned it. Our scraper may have picked it up
  four hours later; a restated figure may arrive three weeks later.

A single-timestamp store forces a choice, and both choices are wrong:

- Store only `event_time`, and a backtest at date T sees data that had not
  reached us at T. That is lookahead bias, and it is invisible: the backtest
  simply looks better than reality.
- Store only `ingest_time`, and every fact is misdated. Any analysis keyed to
  when things actually happened becomes noise.

Restatements make it worse. A revenue figure published, corrected, and corrected
again is three facts with one `event_time` and three `ingest_times`. A
single-timestamp store silently overwrites, destroying the record of what we
believed at the time.

## Decision

Every event-shaped model inherits `TemporalModel`, which requires both
timestamps, requires both to be UTC, and rejects `ingest_time` more than
`MAX_CLOCK_SKEW` (5s) before `event_time`. Facts are append-only: a correction
is a new row, never an update.

## Rationale

The point-in-time query — "what did we know about X as of T" — is
`WHERE event_time <= T AND ingest_time <= T`, keeping the latest row per key.
Making that expressible is the entire justification.

The 5-second skew tolerance exists because `event_time` often comes from a
publisher's clock and `ingest_time` from ours. Zero tolerance would reject
legitimate data over NTP drift; a generous tolerance would let a genuinely
inverted pipeline through. Five seconds is small enough that no real ingest lag
fits inside it.

## Consequences

**Accepted costs**

- **Storage grows with revisions, not with facts.** Each correction is a new
  row. For fundamentals with heavy restatement this is a real multiple, not a
  rounding error.
- **Every query is more expensive.** Two predicates instead of one, and a
  latest-per-key selection on top. Expect a composite index on
  `(entity_id, event_time, ingest_time)` and expect point-in-time reads to cost
  meaningfully more than "current value" reads.
- **Every query is more complicated,** and a forgotten `ingest_time` predicate
  reintroduces exactly the lookahead the design exists to prevent. Phase 1 must
  put point-in-time reads behind a helper rather than leaving raw SQL to
  discipline.
- Live trading pays this cost for no direct benefit — it always wants the
  latest. The cost is accepted so that live and backtest read through one path;
  two paths would eventually disagree, and the disagreement would be discovered
  in production.

## Alternatives rejected

- **Single `event_time` plus a data-availability lag constant.** A fixed lag is
  a guess, it varies per source, and it cannot represent restatements at all.
- **Snapshot the whole database nightly.** Storage cost is far worse, resolution
  is a day, and it answers "what did the tables look like" rather than "what did
  we know".
