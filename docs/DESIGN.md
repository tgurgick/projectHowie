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

Backtests on realized 2025 results, three tiers: input quality (projection
MAE/rank-corr), calibration (p10–p90 coverage vs ~80% target, buckets fit on
≤2024 only), and policy replay (full 2025 drafts vs follow-ADP and VORP
baselines, scored with realized weekly points).

Findings that shaped the engine (Aug 2026):

| Finding | Change |
|---|---|
| Late-round picks tie at zero marginal value → QB hoarding | Positional caps + bench-insurance tie-break in `evaluate_candidates` |
| p10–p90 covered only 35% of outcomes (weekly noise averages out) | Season-level projection-error shock (`SEASON_SIGMA`, measured per position from 2025) → 78% coverage |
| Pure-projection drafting lost to follow-ADP by ~200 pts (winner's curse on proj-vs-market outliers) | **Market anchor**: shrink projections toward ADP-implied value; anchor 0.75 beats ADP by +48 (league config `market_anchor`) |

Caveat: the anchor weight is tuned on one season of replays; re-run the sweep
when 2026 actuals exist, and add seasons as they accumulate.

## Draft-day flow

Week before: `howie data refresh` → run `skills/research-team.md` per team →
`howie graph import research/*.json` → review the facts diff → mock drafts in
the cockpit. Draft night: `howie serve`, mark picks as they happen (or let an
MCP client do it); the engine re-ranks deterministically at once and refines
with MC in the background; Claude sits alongside via MCP for "talk me out of
this" moments, reading the same state.

## Explicitly not built (yet)

- Chrome extension for auto-marking picks (the Claude-in-Chrome observer via
  MCP covers it; build the extension only if latency annoys).
- Weekly in-season mode (start/sit, waivers) — the primitives support it.
- Deleting legacy v2 — pending sign-off.
