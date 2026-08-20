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
  board (marginal value + Monte Carlo floor–ceiling bars), a strategy tab
  (per-position draft-now-vs-wait impact, pinned rules, notes), knowledge-
  graph player cards, search-first pick marking with undo, and **mock-draft
  bots** — hit START MOCK and practice a full draft solo.
- **`howie draft board / pick`** — the same engine from the CLI.
- **`howie graph search|context|import`** — millisecond search over players,
  teams, and position rooms; 1-hop context (room shares, vacated volume,
  team trends, researched facts with provenance).
- **`howie eval run`** — backtests on realized 2025 results: projection
  quality, calibration coverage, and full draft replays vs follow-ADP and
  VORP baselines. (Current scoreboard: Howie +48 pts vs ADP; the market
  anchor and variance model came out of this harness.)
- **`python3 -m howie3.mcp_server`** — the engine as MCP tools for Claude
  Desktop/Code: chat marks picks into the same draft log the cockpit shows.
- **`howie ask "..."`** — in-repo natural-language agent (needs
  `ANTHROPIC_API_KEY`; `ai` extra).
- **`skills/`** — research playbooks whose only output is structured facts
  (`howie graph import`), never prose.

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
outcome summaries) with no provider rows. It can be imported elsewhere
(`howie context import`) and powers the draft views without a database.

## Architecture (v3)

```
howie3/
  config.py        # settings + league validation (HOWIE_DATA_DIR override)
  db.py            # schema migrations
  data/            # source adapters + refresh orchestrator + integrity checks
  value/           # the engine: availability, marginal value, distributions, MC
  commands.py      # one registry serving both frontends
  views.py         # command output as renderables
  tui/             # Textual app
  agent.py         # Anthropic tool-calling agent (read-only SQL + engine tools)
```

Tests: `python -m pytest tests/ -q`

## Legacy (v2)

The previous implementation remains temporarily as `howie-legacy` /
`howie-cli` (deprecated, unsupported, prints a notice; requires
`pip install -e ".[legacy]"`). It will be removed.

## License

MIT
