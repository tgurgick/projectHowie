# Howie Hardening Backlog

**Created:** 2026-08-22  
**Purpose:** Follow-up work after the current implementation is complete. This
document records review findings; it is not a claim that the fixes below have
already been implemented.

## Priority order

### P0 — Fix before relying on the current server

#### 1. Remove the policy import race

`serve()` starts the background Monte Carlo worker immediately, while
`service.pick_payload()` lazily imports `howie3.value.policy`. The first HTTP
request can therefore observe a partially initialized module and return an
error instead of a recommendation.

Relevant code:

- `howie3/server.py:349-355`
- `howie3/service.py:308-327`
- `howie3/value/policy.py`

Preferred correction:

- import the policy module during `_warm_imports()` before starting workers; or
- move the policy imports to stable module scope and remove the lazy import
  race entirely.

Acceptance criteria:

- the first `/api/pick` request after server startup returns rows;
- no partially initialized-module errors appear in server logs;
- the HTTP test verifies the first request, not only a later retry;
- Monte Carlo worker failures are exposed as structured API state.

### P1 — Fix before treating recommendations or model output as trustworthy

#### 2. Separate coaching from evaluation

The coaching loop scores every candidate rule set against realized 2025
results and chooses the winner using that same replay score.

Relevant code:

- `howie3/coach.py:119-149`
- `howie3/coach.py:237-285`

Correction:

- use one or more training seasons to develop rules;
- use a validation season to select the best rule set;
- reserve an untouched test season for the final report;
- report the number of seasons and paired replays used;
- select using unrounded scores and confidence intervals, not only a rounded
  mean total.

#### 3. Make TARGET rules reliable

The policy is applied after candidate evaluation has already truncated the
board. A target outside the top candidate window can silently have no effect.
This was reproduced with a low-ranked target that did not appear in the
returned top-five rows.

Relevant code:

- `howie3/service.py:322-327`
- `howie3/evals.py:315-322`
- `howie3/mocksim.py:53-56`

Correction:

- resolve target names to stable player IDs;
- inject a valid, available target into the candidate set before truncation;
- prefer exact normalized-name or UID matching over substring matching;
- add tests for targets outside the engine's normal top-K window;
- define whether `N POS BY R` applies only on the deadline round or on every
  round through the deadline until the requirement is met.

#### 4. Enforce the model egress boundary for coaching

The coach sends its digest directly to the model. The digest is currently
mostly derived strategy context, but it bypasses the central egress policy and
could become unsafe if new fields are added later.

Relevant code:

- `howie3/coach.py:173-194`
- `howie3/egress.py:78-88`

Correction:

- define an explicit allowlisted coach-digest schema;
- pass the serialized digest through the same egress boundary as other model
  calls;
- test that raw provider keys, raw stat rows, credentials, and scraped source
  records cannot enter the outbound payload.

#### 5. Recalibrate simulation uncertainty without leakage

`SEASON_SIGMA` is documented as measured from the 2025 backtest and is reused
when evaluating 2025. The p10/p90 bands therefore cannot be treated as an
independent test of calibration.

Relevant code:

- `howie3/value/distributions.py:27-32`
- `howie3/value/distributions.py:71-97`
- `howie3/evals.py:161-195`

Correction:

- fit uncertainty parameters only on seasons before the evaluated season;
- use rolling train/test windows as more seasons become available;
- calibrate by preseason-known rank, position, games projection, and tier;
- report coverage with confidence intervals by position and player tier.

#### 6. Correct the meaning of coach p10

`coach.simulate()` averages each draft's individual p10 values. The average of
per-draft p10s is not the p10 of the combined strategy distribution.

Relevant code:

- `howie3/coach.py:87-100`

Correction:

- retain raw simulation outcomes or a reproducible summary sufficient to
  recompute pooled percentiles;
- calculate p10/p50/p90 over the intended population;
- label per-draft and pooled statistics differently in the UI.

#### 7. Harden the autodraft bridge before real-room use

The browser bridge has useful safety gates—autopilot is opt-in and real-room
clicking requires `--real`—but clicking a third-party draft room is a
high-impact action.

Relevant code:

- `howie3/autodraft.py`
- `howie3/cli.py:187-219`

Before enabling real-room workflows:

- require an explicit confirmation immediately before the first real click;
- verify the allowed page origin and draft-room identity, not only the page
  title;
- make clicks idempotent against the synced event log and current player
  availability;
- refuse to click when the recommendation is stale, the room state changed,
  or the player search is ambiguous;
- keep browser profiles, autodraft logs, and room exports local and excluded
  from context artifacts and Git.

### P2 — Improve concurrency, maintainability, and coverage

#### 8. Isolate evaluation state

`evals.set_rule_effects()` updates global mutable state. Concurrent sessions can
therefore evaluate with another session's rules.

Pass effects explicitly through evaluation functions and remove the global
mutable default. If a transition period is needed, reset global state in a
`finally` block and add a concurrency test.

#### 9. Serialize coach session ownership

`run_in_background()` checks `STATUS["running"]` without a lock. Two callers
can race and start separate sessions that write the same session store.

Use a lock or an atomic state transition, and make session writes resilient to
concurrent access.

#### 10. Expand cross-season and league-shape evaluation

The current policy result is centered on one season and one primary league
shape. Add multiple seasons, roster formats, scoring formats, and draft slots.
Keep paired seeds and bootstrap intervals, and add ablations for market
anchor, availability, bench insurance, schedule effects, and strategy rules.

#### 11. Model correlated availability and scoring

The simulator independently samples player shocks, weekly noise, and
availability. Add multi-week injury durations, team/environment shocks, and
player correlation where the data supports it. Validate the resulting roster
tails against held-out seasons.

## Data ownership guardrails

These rules remain part of the implementation contract:

- raw scraped/provider data is constructed independently by each user;
- local databases and provider exports stay under the ignored `data/`
  directory;
- no raw provider rows, raw draft exports, or credentials are committed;
- shared artifacts contain only the minimum derived strategy context needed to
  power the tool;
- new outbound fields must pass the model egress tests before being added.

Before committing the current work, review all untracked files and stage only
source code, tests, documentation, and approved derived artifacts.

## Verification checklist

```bash
python3 -m compileall -q howie3 tests
python3 -m pytest -q
python3 -m mypy howie3 --ignore-missing-imports
git diff --check
```

The HTTP integration tests require permission to bind localhost ports in the
execution environment. Their assertions should be run in an environment where
that permission is available.
