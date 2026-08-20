# 🏈 Project Howie — Fantasy Football Draft Engine

> Named for Howie Roseman, GM of the Philadelphia Eagles — the franchise's
> master of extracting value from every pick. Go Birds. 🦅

Howie is a TUI-first fantasy football draft assistant built around **marginal
value**: every recommendation is the value of taking a player *now* versus
what you could still get by waiting, computed from live ADP availability
distributions and Monte Carlo season simulation.

```bash
pip install -e ".[ai]"
howie                # TUI (type `help` inside)
```

## What it does

- **`howie draft board --round 3`** — per position: who's likely to still be
  there at your pick, the expected best available at your *next* pick, and the
  marginal value of acting now.
- **`howie draft pick --have "CMC, ARSB" --taken "..."`** — live draft help:
  candidates ranked by expected final starting-lineup points, Monte Carlo
  simulated (weekly lineups, injuries, byes, SoS), with floor–ceiling bands
  and a positional plan for the rest of your draft.
- **`howie player <name>`** — projection, ADP spread, history.
- **`howie ask "..."`** — natural-language agent over the same engine
  (requires `ANTHROPIC_API_KEY`, install with the `ai` extra).

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
