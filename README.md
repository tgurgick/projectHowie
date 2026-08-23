# 🏈 Project Howie — Fantasy Football Draft Engine

> Named for Howie Roseman, GM of the Philadelphia Eagles — the franchise's
> master of extracting value from every pick. Go Birds. 🦅

Howie is a TUI-first fantasy football draft assistant built around **marginal
value**: every recommendation is the value of taking a player *now* versus
what you could still get by waiting, computed from live ADP availability
distributions and Monte Carlo season simulation.

```bash
pip install -e ".[ai]"
howie serve          # the draft-night cockpit → http://127.0.0.1:8787
```

## What it does

- **`howie serve`** — the draft cockpit: a local web app with the ranked
  board (marginal value with bench insurance + Monte Carlo floor–ceiling
  bars), a strategy tab (per-position draft-now-vs-wait impact, pinned
  rules, notes), knowledge-graph player cards, a Mock Draft Lab whose
  availability rates feed back into the engine, and a command line built
  for the clock: type a name, **⏎ marks him taken, ⇧⏎ drafts him to you,
  ⇥ opens the card**, `?…` asks Howie. Mock-draft bots (they react to
  positional runs and their own needs) let you practice a full draft solo.
- **`howie draft board / pick`** — the same engine from the CLI.
- **`howie graph search|context|import`** — millisecond search over players,
  teams, and position rooms; 1-hop context (room shares, vacated volume,
  team trends, researched facts with provenance).
- **`howie eval run`** — backtests on realized 2025 results: projection
  quality, calibration coverage, and paired full-draft replays (same seeded
  opponents for every policy) vs follow-ADP, ADP-with-need, pure-projection
  and VORP baselines, with bootstrap confidence intervals. Current
  scoreboard (n=40 paired replays, one season): **Howie +101 pts vs ADP,
  95% CI [+42, +158], wins 65%**; the bench-insurance objective, market
  anchor and variance model all came out of this harness. One season of
  preseason inputs exists, so the CI covers draft-to-draft variance, not
  season-to-season.
- **`howie coach run`** — an experimental strategy loop: Howie runs mock
  drafts under the current strategy sheet, scores roster structure and season
  simulations, asks Claude for bounded rule changes, and records the trace in
  `data/coach_sessions.json`. Treat its current 2025 replay score as
  diagnostic, not as an out-of-sample performance claim; follow-up work is
  tracked in [HARDENING_BACKLOG.md](HARDENING_BACKLOG.md).
- **`howie autodraft signin|run`** — an optional Playwright bridge for a draft
  room. It uses a persistent local browser profile, syncs room picks into the
  Howie event log, and can queue or click picks only when `--autopilot` is
  explicitly enabled. Real-room clicking additionally requires `--real`;
  credentials stay in the user's browser profile and are never handled by
  Howie.
- **`python3 -m howie3.mcp_server`** — the engine as MCP tools for Claude
  Desktop/Code: chat marks picks into the same draft log the cockpit shows.
- **`howie ask "..."`** — in-repo natural-language agent (needs
  `ANTHROPIC_API_KEY`; `ai` extra). Everything sent to a model or an MCP
  client passes through one redaction policy (`howie3/egress.py`): derived
  context only, never per-game stat lines; the agent's raw SQL tool is
  opt-in (`HOWIE_AGENT_SQL=1`).
- **Player status layer** — projections assume 17 games for everyone and
  ADP prices injuries only indirectly, so `player_status` holds the typed
  truth the engine acts on (injured + games out, out for season, suspended,
  holdout, cut risk, role). Two writers: the free nflverse roster feed
  (`howie data refresh --steps roster`, automatic) and the research
  workflow. Applied after the market anchor: an ACL leaves the board no
  matter what a stale ADP says; games missed and cut risk scale value; the
  board, cards and search show the chip and its source.
- **`skills/` + the `research-teams` workflow** — Claude Code subagents
  research every draft-relevant player on a team (status record each) plus
  the offense (facts with provenance), a skeptic validates, the result is
  imported. `howie research targets TEAM` is the checklist, `howie research
  coverage` the scoreboard, `howie research stale` what to re-run — a full
  pass before the draft, then weekly or when news breaks.

Architecture and design decisions: [docs/DESIGN.md](docs/DESIGN.md).

### Honest limits of the numbers

- **p10–p90 bands are only partly calibrated.** On 2025 actuals they cover
  74% of outcomes (82% for players with 8+ games) against an 80% target,
  and by position QB/WR run narrow while TE runs wide. The season-level
  shock (`SEASON_SIGMA`) was measured on the same 2025 season the backtest
  scores, because 2025 is the only season with preseason projections in
  the db — so tier B is in-sample for that parameter. Variance buckets are
  now tiered by a preseason-knowable proxy (prior-season rank) rather than
  realized rank. Treat the bands as ranges, not probabilities, until a
  second season of projections exists.
- **Availability is independent per week** (no multi-week injury clustering,
  no QB/receiver or team-environment correlation), so roster-level tails are
  somewhat too optimistic. Known injuries enter through the status layer.
- **Players the market never drafts** (328 of 547 in the 2026 pool) get an
  implied availability prior past the drafted range instead of a flat 100%.
- **The agent needs the local database.** Context-only mode (`--context`)
  covers the draft board and Monte Carlo; the agent's player and
  knowledge-graph tools are not available from an artifact alone.
- **Strategy rules are now active engine policy.** `WAIT`, `NO`, `TARGET`, and
  positional `BY` rules affect the cockpit, mock-draft lab, and policy
  replays. The policy layer is new and remains under hardening; validate target
  resolution before relying on a target outside the displayed candidate set.

## Data: build it locally

**No scraped or provider data ships with this repository.** You build your own
local database from documented sources:

```bash
howie data refresh          # nflverse history, ID crosswalk, live mock-draft ADP
howie data status           # what you have
```

Optional sources you can add yourself:

- **PFF projections/SoS** — export CSVs from your own PFF account into
  `data/pff_csv/` (see `howie3/README.md` for names); the refresh picks them up.
- League shape lives in `data/league_config.json`
  (see `howie3/defaults/league.example.json`).

Everything lands in one SQLite file (`data/howie.db`), which stays on your
machine — the `data/` directory is never committed.

### Sharing without raw data

`howie context export` writes a **strategy-context artifact**: a small,
versioned JSON of derived abstractions (tiers, availability probabilities,
outcome summaries, simulation parameters, variance buckets, and provenance)
with no provider rows or raw scraped tables. Each user can construct the
underlying local database independently from the documented source adapters.
Only the necessary strategy context should be shared; the local `data/`
directory, provider exports, raw draft data, and credentials stay on the
machine. The artifact can be imported elsewhere (`howie context import`) and
powers the draft views — Monte Carlo included — without a database.
Packaging is an explicit allowlist (`MANIFEST.in`, `setup.py`): databases,
`.env`, and `data/` never ship, and `tests/test_boundary.py` builds an sdist
to prove it.

## Architecture (v3)

```
howie3/
  config.py        # settings + league validation (HOWIE_DATA_DIR override)
  db.py            # schema migrations
  data/            # source adapters + refresh orchestrator + integrity checks
  value/           # the engine: availability, marginal value, distributions, MC
  commands.py      # one registry serving both frontends
  views.py         # command output as renderables
  state.py         # the draft event log (validated schema; every surface writes it)
  service.py       # the JSON contract every surface calls
  value/policy.py  # strategy-rule effects applied to candidate rankings
  autodraft.py     # optional local browser bridge for draft-room sync/clicks
  server.py        # cockpit HTTP server (session token on writes, CSP, body limit)
  ui/              # index.html (markup) + style.css + lib.js (pure helpers) + app.js
  egress.py        # the one redaction policy for model/MCP-bound payloads
  mcp_server.py    # MCP tools over the service layer
  agent.py         # Anthropic tool-calling agent (engine tools; SQL opt-in)
  coach.py         # experimental mock-draft strategy coaching loop
  evals.py         # paired backtests with bootstrap CIs
```

Tests: `python -m pytest tests/ -q` (backend, HTTP, boundary, and — with
node installed — the front-end helpers via `node --test`). Type check:
`python -m mypy howie3 --ignore-missing-imports` (clean; CI enforces both).

## Legacy (v2)

The previous implementation remains temporarily as `howie-legacy` /
`howie-cli` (deprecated, unsupported, prints a notice; requires
`pip install -e ".[legacy]"`). It will be removed.

## License

MIT
