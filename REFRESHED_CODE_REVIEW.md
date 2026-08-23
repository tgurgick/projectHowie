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

### 8. Recalibrate simulation uncertainty and remove evaluation leakage — P1

The simulation preserves the input projection mean well. Direct 100,000-run checks produced values close to the projection input, for example:

- Josh Allen: projection `307.4`, simulated mean `307.39`;
- Jahmyr Gibbs: projection `306.3`, simulated mean `307.75`;
- Drake Maye: projection `299.4`, simulated mean `299.44`.

That validates the mean-preservation arithmetic, but it does not validate the probability ranges.

The current 2025 calibration check reports:

| Position | p10–p90 coverage | 8+ games coverage |
|---|---:|---:|
| QB | 60.7% | 68.0% |
| RB | 74.5% | 82.0% |
| WR | 58.6% | 67.2% |
| TE | 92.3% | 92.3% |
| Overall | 68.7% | 75.9% |

The target is 80%. QB and WR ranges are too narrow, while TE ranges are too wide. The current p10/p90 values should therefore not be presented as calibrated probabilities.

#### Calibration problems

Historical variance buckets are assigned using realized season-total rank in `howie3/value/distributions.py:73-87`, while current players are assigned using projected rank in `howie3/value/distributions.py:91-97`. This is not a consistent predictive calibration scheme.

In addition, `SEASON_SIGMA` is described as measured from the 2025 backtest in `howie3/value/distributions.py:27-32`, but is reused by the 2025 evaluation in `howie3/evals.py:161-195`. That leaks evaluation-season information into the uncertainty model.

#### Correction

- Calibrate historical players by preseason projection rank, preseason ADP, projected games, and prior-season information.
- Do not use 2025-derived uncertainty parameters when evaluating 2025.
- Create rolling train/test windows for every evaluated season.
- Report calibration by position, tier, and availability bucket.
- Add confidence intervals to coverage estimates.
- Use empirical or distributional calibration to target the desired coverage instead of hand-tuning `SEASON_SIGMA`.

### 9. Improve injury and availability modeling — P1

Weekly availability is currently sampled as independent Bernoulli draws in `howie3/value/simulate.py:74-79`. This does not represent clustered injuries or multi-week absences.

Historical availability buckets can also be optimistic because they are measured among the best realized players in each position tier, which selects for players who stayed healthy.

#### Correction

- Model injury duration and consecutive missed games.
- Use player-specific availability where enough history exists.
- Add team-level injury/environment shocks.
- Separate healthy, limited, and unavailable states.
- Validate predicted games missed against held-out seasons.

### 10. Fix incomplete ADP coverage — P1

The current 2026 half-PPR pool contains 547 projected players, but 328 have no matching ADP record. The current fallback in `howie3/value/availability.py:16-20` treats missing ADP as 100% availability.

That is acceptable for truly undrafted fringe players, but it is unsafe as a general fallback. It distorts late-round availability and replacement-value calculations.

#### Correction

- Add a position/round availability prior for missing ADP.
- Use another independent ADP source where possible.
- Mark availability as unknown rather than silently assigning 1.0.
- Report ADP coverage in the data-quality payload.
- Add tests for high-projection players with missing ADP.

### 11. Model cross-player and team correlation — P2

The simulation independently samples each player’s season shock, weekly scoring noise, and availability in `howie3/value/simulate.py:71-85`.

This omits important football correlations:

- quarterback and pass-catcher relationships;
- team offensive environment;
- game script;
- shared injuries;
- opponent and weather effects.

As a result, roster-level tails may overstate diversification and understate correlated downside.

#### Correction

Add at least:

- team-week environment shocks;
- QB/pass-catcher correlation;
- shared team availability effects;
- opponent/game-level scoring shocks.

### 12. Increase Monte Carlo sample size for close decisions — P2

Production recommendations use approximately 150 simulations through `howie3/service.py:255`. That is sufficient for a rough directional estimate, but not for candidates separated by only a few fantasy points.

Different seeds produced stable top-three choices in a basic check, but later candidates moved. The fixed seed makes results reproducible while hiding sampling uncertainty.

#### Correction

Use a two-stage simulation budget:

1. 150–300 simulations for the full board.
2. 1,000–5,000 simulations for shortlisted candidates.

Expose standard error, rank stability across seeds, probability of being the best choice, and confidence intervals on value differences.

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
8. Recalibrate uncertainty using strictly preseason information.
9. Replace missing-ADP availability defaults.
10. Add clustered injury and team/player correlation.
11. Increase simulation depth for close decisions.

## Final assessment

The refreshed code is now a strong prototype with a substantially improved security and data-ownership posture. The packaging boundary and portable simulation artifact are in good shape.

The next release should wait until the Monte Carlo worker is reliable, the model egress contract is structural, and the uncertainty ranges are demonstrably calibrated. The current means are coherent, but the probability ranges and availability model still need empirical correction before being treated as trustworthy forecasts.
