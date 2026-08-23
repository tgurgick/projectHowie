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
outcome summaries, the simulation parameters and variance buckets, and
provenance) with no provider rows. It can be imported elsewhere
(`howie context import`) and powers the draft views — Monte Carlo
included — without a database. Packaging is an explicit allowlist
(`MANIFEST.in`, `setup.py`): databases, `.env`, and `data/` never ship, and
`tests/test_boundary.py` builds an sdist to prove it.

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
  server.py        # cockpit HTTP server (session token on writes, CSP, body limit)
  ui/              # index.html (markup) + style.css + lib.js (pure helpers) + app.js
  egress.py        # the one redaction policy for model/MCP-bound payloads
  mcp_server.py    # MCP tools over the service layer
  agent.py         # Anthropic tool-calling agent (engine tools; SQL opt-in)
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
