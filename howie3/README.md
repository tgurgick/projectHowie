# Howie v3

Rebuild of Howie for the 2026 season. Lives alongside the legacy `howie_cli`
package until feature parity, then the legacy code gets deleted.

## Why a rebuild

The v2 review (July 2026) found: three duplicated scoring databases for one
derived column, name-based joins across three ID namespaces, a 4,341-line
script as the de-facto command layer with a circular TUI import, and a
valuation engine whose objective (mean projection sums) bypassed its own
variance machinery while marginal value was display-only.

## Principles

1. **One database** (`data/howie.db`). Scoring formats are columns
   (`pts_std`, `pts_half`, `pts_ppr`), never separate files. The league's
   format comes from `data/league_config.json` (`half_ppr` today) and flows
   through everything.
2. **Stable identity.** `players.player_uid` (gsis id when known) plus a
   `player_ids` crosswalk seeded from DynastyProcess. Name matching happens
   only at ingest, via `data/names.py`; failures land in `unmatched_names`
   for triage and manual `name_aliases` fixes. Name-based joins across
   tables are banned.
3. **Pluggable sources, one orchestrator.** `howie3 data refresh` runs the
   dependency-ordered steps idempotently and records each in `refresh_log`.
   Projection sources (PFF CSV exports, FantasyPros, ...) are additive rows
   keyed by `source` — the engine picks/blends at query time.
4. **Stat-level storage.** Projections store stat lines, points are
   recomputed under our scoring in `data/scoring.py` — one scoring
   implementation for history and projections alike.
5. **One objective in the math** (Phase 2): expected optimal-lineup points
   from sampled outcomes. A pick's value is *marginal*:
   E[roster | draft X now] − E[roster | best expected alternative at the
   next pick], with replacement level derived from league config, not
   hardcoded ranks.

## Layout

```
howie3/
  config.py          # Settings + LeagueConfig — the only path/season/format resolver
  db.py              # sqlite connection + schema migrations (PRAGMA user_version)
  schema.sql
  cli.py             # `howie3` entry point (click)
  data/
    names.py         # name_key/team/position normalization + uid resolution
    scoring.py       # the one fantasy-points implementation (all formats)
    refresh.py       # orchestrator: crosswalk→players→dst→games→weekly→adp→pff
    sources/
      dynastyprocess.py   # ID crosswalk (backbone of identity)
      nflverse.py         # players, schedules, weekly stats (gsis-keyed, no names)
      fantasypros.py      # ADP scrape (header-driven parsing)
      pff.py              # manual CSV exports for projections
```

## Usage

```bash
howie                               # TUI (type `help` inside)
howie data refresh                  # rebuild/update howie.db (idempotent)
howie data status                   # row counts, coverage, refresh history
howie draft board --round 3         # marginal value vs waiting, per position
howie draft pick --round 4 \
  --have "CMC, ARSB" --taken "..."  # live: best pick now, Monte Carlo ranked
howie player Puka Nacua             # projection, ADP, history
howie ask "RB or WR at pick 8?"     # AI agent (needs ANTHROPIC_API_KEY)
```

The legacy v2 app remains available as `howie-legacy` / `howie-cli` until the
old code is deleted.

To load 2026 PFF projections: export the projection CSVs from PFF into
`data/pff_csv/` named like `offensive_projections_2026_preseason.csv`
(same for `def_`/`k_`), then `howie3 data refresh --steps pff`.

## Roadmap

- [x] Phase 1 — data foundation (single db, crosswalk, refresh pipeline)
- [x] Phase 2 — marginal-value engine (`howie3/value/`): availability model,
      expected-k-th-best, roster-conditional rollout, empirical outcome
      distributions, Monte Carlo season simulation
- [x] Phase 3 — TUI + CLI on a single command registry (`commands.py`,
      `views.py`, `tui/app.py`)
- [x] Phase 4 — native tool-calling agent (`agent.py`), intelligence port,
      `howie` entry-point cutover
- [ ] Legacy deletion (`howie_enhanced.py`, `howie_cli/`, old dbs) — pending
      sign-off; everything is ported

Tests: `python3 -m pytest tests/test_howie3.py`
