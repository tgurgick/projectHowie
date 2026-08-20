# Skill: research one team's offense (for Howie's knowledge graph)

You are researching ONE NFL team's 2026 offense for a fantasy-football draft
engine. Your only deliverable is a JSON facts file matching the contract in
`skills/README.md` — no prose report.

## Inputs

- Team: `$TEAM` (nflverse abbreviation, e.g. ARI, PHI)
- Today's date; the draft is in late August 2026.

## What to research (web search; prefer beat reporters and primary sources)

1. **Coaching / scheme change**: new HC/OC? What did that coordinator's
   offenses look like (pass rate over expected, play volume, personnel
   groupings, TE/RB usage)? → `team:` facts, kind `coach_change` /
   `scheme_note`, with a numeric `value` when you can (e.g. pass-rate delta).
2. **Volume redistribution**: who left, who arrived, who's hurt? Which
   players absorb vacated targets/carries? (The engine already knows LAST
   season's shares and vacated volume — add what data can't know: camp
   reports, stated plans, injuries.) → `unit:` and `player:` facts, kinds
   `role_note` / `injury_note` / `volume_prior`.
3. **Offensive line**: projected quality, returning starters, injuries.
   → `unit:$TEAM-OL` fact, kind `oline_grade`, `value` = projected rank.

## Rules

- Every fact: one claim, one entity, honest `confidence`, dated `source`
  naming how many sources agree, and an `expires` date (draft week + buffer).
- Quantitative claims carry `value`. No fact without a source. No filler —
  five strong facts beat twenty weak ones.
- Player names exactly as rostered (they resolve through an ID crosswalk).

## Finish

Write the file to `research/$TEAM.json`, then run:

```bash
howie graph import research/$TEAM.json
```

and report the import count. If any entity fails to resolve, fix the name and
re-import (the import is idempotent per fact only if unchanged — prefer fixing
before importing twice).
