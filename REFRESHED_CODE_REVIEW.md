# Refreshed Code Review and Correction Checklist

**Review date:** 2026-08-22  
**Repository state:** refreshed hardening and cockpit changes through commit `c5be213`  
**Scope:** runtime reliability, data ownership, model egress, context-only operation, caching, UI safety, and evaluation.

## Executive summary

The refresh is a substantial improvement over the previous version. The following areas are now materially stronger:

- raw data and credentials are excluded from package manifests;
- local research artifacts are ignored by Git;
- draft-log validation is much stricter;
- server mutations require a session token;
- request bodies have size limits;
- the UI now uses an auto-escaping HTML helper;
- strategy rules participate in recommendation cache keys;
- the strategy-context artifact now carries derived simulation parameters;
- context-only mode can run Monte Carlo without the local database;
- policy evaluation now uses paired replays and bootstrap confidence intervals.

The repository is not completely ready yet. The most important remaining issue is a background Monte Carlo import race that can silently prevent Monte Carlo recommendations from being produced. The model egress boundary also still allows rendered plain text to bypass structured redaction.

## Verification status

- 90 tests pass in the sandboxed run.
- The two HTTP tests are blocked in the sandbox because localhost port binding is disallowed.
- A focused rerun of both HTTP tests passes when local networking is allowed.
- During that focused run, the background Monte Carlo worker logs `ImportError` failures caused by a partially initialized `howie3.value.distributions` module.
- `compileall` passes.
- `mypy` still reports errors across service, server, MCP, roster valuation, simulations, and evaluation code.

## Corrections to implement

### 1. Fix the background Monte Carlo import race — P1

The server starts Monte Carlo work in a background thread at `howie3/server.py:64`. The worker can race with the first request while importing simulation modules. The resulting failure is:

```text
ImportError: cannot import name 'build_sim_players' from partially initialized module 'howie3.value.distributions'
```

The HTTP request still succeeds because the worker catches and prints the exception, but Monte Carlo data is not produced.

Relevant code:

- `howie3/server.py:64-81`
- `howie3/value/roster.py:162-189`
- `howie3/value/distributions.py:1-20`

#### Correction

Use one of these approaches:

1. Eagerly import the simulation modules before starting any background worker.
2. Add a module-level initialization lock around first-time simulation imports.
3. Move shared simulation imports to stable module scope where circular-import risk is eliminated.

The worker must also expose failure state to the API instead of only printing a traceback.

#### Acceptance criteria

- No partially initialized-module errors appear during server startup.
- `/api/pick` eventually includes Monte Carlo data after startup.
- A worker failure is visible through a structured status field.
- The HTTP integration test waits for and verifies the `mc` payload.

### 2. Make model egress structurally safe — P1

The shared serializer at `howie3/egress.py:78` only redacts JSON. Plain text passes through unchanged.

The agent’s `player_info` tool renders database-backed Rich tables into plain text at `howie3/agent.py:327-328`. A verified output includes projection data, ADP, and historical season data. These are mostly aggregates, but they bypass the structured egress contract.

The current egress test is incomplete because `tests/test_hardening.py:156-176` only deeply inspects output when it parses as JSON.

#### Correction

- Make model-facing tools return typed derived dictionaries rather than Rich-rendered text.
- Serialize approved fields only at the model boundary.
- Keep Rich rendering exclusively for CLI/TUI/UI presentation.
- Remove or permanently disable arbitrary `query_database` from the model-facing tool schema in production mode.
- Keep the local SQL tool available only to explicit local diagnostic workflows, never to a remotely invoked agent.

#### Acceptance criteria

- Every model-facing tool returns a structured, allowlisted payload.
- No model-facing tool calls `_render()`.
- Tests inspect both JSON and plain-text results for raw fields and raw record shapes.
- Raw database tables and provider payloads cannot be returned by setting an environment variable in a production configuration.

Relevant files:

- `howie3/egress.py:78`
- `howie3/agent.py:301-328`
- `tests/test_hardening.py:156-176`

### 3. Complete recommendation cache invalidation — P1

The generation key at `howie3/server.py:43-55` includes draft identity, active rules, and league configuration. It does not include every data source that can change a recommendation:

- strategy-context artifact contents or version;
- local database version or file identity;
- mock-lab availability results;
- data schema version;
- current data refresh state.

The `/api/sim/mock/run` route also returns early at `howie3/server.py:254-258`, before explicitly invalidating or starting a new Monte Carlo calculation.

#### Correction

Include hashes or versions for all recommendation inputs in the generation key. Explicitly invalidate recommendation and Monte Carlo caches after mock-lab results change.

#### Acceptance criteria

- Changing the context artifact changes the recommendation generation.
- Importing new mock-lab results changes availability and ranking output.
- Refreshing the local database invalidates old recommendations.
- A test verifies that identical draft state with changed context produces a different cache generation.

### 4. Complete context-only support for the agent — P1

Schema 2 of the context artifact is now good enough for draft-board and Monte Carlo calculations. `howie3/context_artifact.py:243-270` reconstructs simulation parameters, and `howie3/views.py:248-259` uses them.

However, the agent’s `player_info` and `entity_context` tools still depend on the local database. In a context-only installation, draft rankings can work while player-specific or knowledge-graph questions fail.

#### Correction

Choose one supported model:

- add context-native implementations of player and entity tools; or
- make the agent explicitly require a local knowledge-graph bundle in addition to the strategy artifact; or
- have the context export include the limited, derived player/team facts needed by those tools.

Do not silently fall back to raw database access when the application is operating in context-only mode.

Relevant files:

- `howie3/agent.py:327-340`
- `howie3/context_artifact.py:243-270`
- `howie3/views.py:18-47`

### 5. Close the remaining database connection leak — P2

`howie3/views.py:303-311` opens a database connection and returns early when no player is found without closing it.

#### Correction

Use a context manager or `try/finally` around the connection lifetime.

Add a test for a missing player lookup and verify that the connection is closed.

### 6. Reduce the remaining typing debt — P2

`mypy` still reports errors in:

- `howie3/service.py`;
- `howie3/server.py`;
- `howie3/mcp_server.py`;
- `howie3/value/roster.py`;
- `howie3/mocksim.py`;
- `howie3/views.py`;
- `howie3/evals.py`;
- source adapters with missing `requests` stubs.

#### Correction

- Define shared typed models for simulation results and API/MCP payloads.
- Replace dynamically typed dictionaries with typed aliases or dataclasses.
- Fix optional-value handling at database and numerical-library boundaries.
- Add `types-requests` to development dependencies.
- Make CI fail on new type errors while gradually reducing the existing baseline.

### 7. Extend evaluation beyond the current season — P2

Evaluation is materially better now. Paired policy replays and bootstrap confidence intervals are implemented in `howie3/evals.py:387-428`.

The remaining limitations are:

- evaluation is still centered on the 2025 season;
- it still depends on the legacy `fantasy_ppr.db` at `howie3/evals.py:56-60`;
- the evaluation does not yet prove that the strategy generalizes across seasons or league shapes.

#### Correction

- Add multiple historical seasons.
- Separate feature tuning seasons from evaluation seasons.
- Evaluate across several league configurations.
- Keep paired seeds and bootstrap intervals.
- Add ablations for availability, roster marginal value, schedule, strategy rules, and research facts.
- Replace the legacy database dependency with normalized derived evaluation inputs.

## What is now considered fixed

The following previous findings appear addressed and have regression coverage:

- package manifests no longer include raw databases, pickles, research output, or `.env`;
- `/data/` and `/research/` are ignored for local artifacts;
- draft logs validate sequencing, teams, positions, roster limits, and malformed files;
- server POST routes require the session token and enforce a body-size limit;
- the UI has centralized HTML escaping through `howie3/ui/lib.js`;
- context artifacts are strictly allowlisted and validated on read/write;
- context schema 2 includes derived simulation parameters and variance buckets;
- context-only Monte Carlo is covered by tests;
- traded-player graph aggregation is deterministic;
- strategy-rule changes participate in cache generation;
- policy evaluation uses paired replays and bootstrap confidence intervals.

## Recommended correction order

1. Fix the background Monte Carlo import race.
2. Make all model-facing tools return typed, allowlisted payloads.
3. Complete cache invalidation for context, database, and mock-lab changes.
4. Define the complete context-only agent contract.
5. Close the database connection leak.
6. Reduce the `mypy` baseline.
7. Expand evaluation across seasons and league configurations.

## Final assessment

The refreshed code is now a strong prototype with a substantially improved security and data-ownership posture. The packaging boundary and portable simulation artifact are in good shape.

The next release should wait until the Monte Carlo worker is reliable and the model egress contract is structural rather than dependent on the format of each tool’s output. Those two corrections are the remaining architectural risks; the rest is reliability, typing, and evaluation maturity.
