# Howie next-gen design

Direction settled Aug 2026 (supersedes the remaining items of
IMPROVEMENT_PLAN.md). The organizing idea is a **speed split**:

- **Slow thinking** (days before the draft): research, projections, intel.
  LLM territory — done by skills/agents that write *structured records*.
- **Fast thinking** (on the clock): pick evaluation in milliseconds, no LLM
  in the loop.

LLMs distill → the engine decides → surfaces display.

## Primitives

1. **Engine as a pure library** (`howie3/value/`): functions of
   (pool, draft state, league config, seed) → rankings. Deterministic under a
   seed. Monte Carlo refinement runs in the background, never blocks.
2. **Draft state as an event log** (`howie3/state.py`, `data/draft.json`):
   an ordered list of pick events plus the user's strategy sheet (pinned
   rules + notes). Every surface reads/writes this one document; undo = pop,
   replay = the log. Mock-draft bots are just another writer.
3. **Knowledge graph** (`howie3/graph.py`: entities/edges/facts + FTS5):
   - *Derived layer* (rebuilt each refresh): teams, position rooms,
     last-season target/carry shares, vacated volume, team pass-rate YoY.
   - *Researched layer* (`howie graph import`): facts with provenance,
     confidence, expiry — written by research skills, never prose.
   Search is single-digit-ms FTS across players/teams/rooms; `entity_context`
   returns the 1-hop neighborhood that player cards and agents consume.
4. **Strategy-context artifact** (`howie3/context_artifact.py`): the only
   thing meant to leave the machine — derived abstractions, strict field
   whitelists.

## Surfaces (all thin, all over `howie3/service.py`)

- **Web cockpit** (`howie serve`, stdlib HTTP + one static page): board with
  outcome-distribution bars, strategy tab (positional NOW-vs-WAIT + pinned
  rules + notes), knowledge-graph player cards, search-first pick marking,
  pick log with undo, mock-draft bots. The design canvas (artifact
  "Howie Draft Cockpit") is the reference.
- **MCP server** (`python3 -m howie3.mcp_server`): the same service functions
  as tools for Claude Desktop/Code — including `mark_pick`, so a chat
  assistant (or a Claude-in-Chrome observer watching the draft room) writes
  the same event log the cockpit displays.
- **In-repo agent** (`howie ask`): bounded tool-calling runtime for
  model-agnostic use; `entity_context` gives it the graph.
- **CLI**: everything scriptable; skills call it.

## The measurement layer (`howie eval run`)

Backtests on realized 2025 results, four tiers: input quality (projection
MAE/rank-corr), calibration (p10–p90 coverage vs ~80% target, buckets fit on
≤2024 only), policy replay (full 2025 drafts, **paired**: every policy drafts
against the same seeded opponents per (slot, rep); reported as mean paired
difference vs follow-ADP with a 95% bootstrap CI and win rate, against
follow-ADP, ADP-with-need, pure-projection and VORP baselines), and SoS.

Findings that shaped the engine (Aug 2026):

| Finding | Change |
|---|---|
| Late-round picks tie at zero marginal value → QB hoarding | Positional caps + bench-insurance tie-break in `evaluate_candidates` |
| p10–p90 covered only 35% of outcomes (weekly noise averages out) | Season-level projection-error shock (`SEASON_SIGMA`, measured per position from 2025) → 78% coverage |
| Pure-projection drafting lost to follow-ADP by ~190 pts, CI far from zero (winner's curse on proj-vs-market outliers) | **Market anchor**: shrink projections toward ADP-implied value (league config `market_anchor`, 0.75) |
| With honest paired seeding the old "+48 vs ADP" vanished: −22 [−79, +36]. Diagnosis: the objective counted starters only, so the engine drafted ONE QB and ate every bye/injury week, while a 10-line ADP-with-need heuristic beat ADP by +45 | **Bench insurance in the objective** (`expected_lineup_points`): a backup counts his points × P(the better players ahead of him leave a slot open), binomial per position, Poisson-binomial for flex. Paired replay: **+101 vs ADP [+42, +158], 65% win rate**, beating ADP-with-need. Anchor sweep under the new objective: 0.75 ≈ 0.90 > 0.50; default stays 0.75 |
| The ADP normal-curve availability model is off by 15–20 pts for some players (Mock Draft Lab) | Lab rates blend into `p_available` with weight n/(n+30) once 10+ drafts exist; the board marks blended cells `LAB` |
| Does preseason SoS predict anything? (tier D) | Season-level corr between projected schedule ease and beating projection: **−0.09** (≈0.1 within positions); weekly within-player corr: **0.02**. Decomposed: even hindsight defense-vs-position moves a player only 0.98×→1.05× of his own mean (r=0.03), and the preseason grade forecasts realized defense at r=0.14. Small effect × weak forecast. Neither is actionable → SoS stays normalized (reshapes weeks, never season totals) and `playoff_weight` defaults to neutral 1.0. The knob exists for an in-season mode where matchup data is real. |

Caveat: the anchor weight is tuned on one season of replays; re-run the sweep
when 2026 actuals exist, and add seasons as they accumulate.

## Draft-day flow

Week before: `howie data refresh` → run the `research-teams` workflow in
Claude Code (research → fact-check → `howie graph import`) → review the facts
in DATA › RESEARCH → mock drafts in the LAB (their availability rates feed
the engine). Draft night: `howie serve`, and the command line is the whole
interface on the clock: type a name, ⏎ marks him taken, ⇧⏎ drafts him to
you, ⇥ opens the card; the board stays compact (decision columns only,
MORE reveals the rest) and the card folds its prep sections while you're on
the clock. The engine re-ranks deterministically at once (the cache key
covers draft identity, strategy rules and league config) and refines with MC
in the background; Claude sits alongside via MCP for "talk me out of this"
moments, reading the same redacted state.

## Boundaries that are enforced in code, not prose

- **Draft log** (`state.py`): schema version, contiguous sequence, valid
  positions/teams, roster limits, completion, mock turn legality; malformed
  files fail loudly (HTTP 409) instead of silently starting a fresh draft.
- **Model egress** (`egress.py`): agent tool results, MCP responses and
  insight payloads all pass through `redact()` — per-game stat lines and raw
  provider records never cross; the agent's SQL tool is opt-in.
- **Server** (`server.py`): per-process session token on every POST (the
  page carries it in a meta tag), 1 MB body limit, CSP, static allowlist.
- **UI** (`ui/lib.js`): every HTML fragment built from data goes through the
  auto-escaping `h` tag; `tests/test_ui.py` lints for it and runs the node
  unit tests.
- **Packaging**: allowlisted manifests; an sdist is built and scanned in tests.

## Explicitly not built (yet)

- Chrome extension for auto-marking picks (the Claude-in-Chrome observer via
  MCP covers it; build the extension only if latency annoys).
- Weekly in-season mode (start/sit, waivers) — the primitives support it.
- Deleting legacy v2 — pending sign-off.
