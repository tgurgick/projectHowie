# Skill: research one team — every draft-relevant player, plus the offense

You are researching ONE NFL team for a fantasy-football draft engine. Your
deliverable is a JSON file matching the contract in `skills/README.md`: a
**status record for every player on the target list** and 6–12 team/unit
facts. No prose report.

## Inputs

- Team: `$TEAM` (nflverse abbreviation, e.g. ARI, PHI)
- The target list — run it first, it is the checklist you must complete:

```bash
python3 -m howie3 research targets $TEAM --json
```

Each target carries `known_status` (what the roster feed or earlier research
already says). Today's date matters: the draft is late August 2026, and the
same skill is re-run during the season when news breaks.

## Part 1 — player status (one record per target, no exceptions)

For EVERY player on the list answer, from beat reporters, team sites, injury
reports and depth charts:

- **Is he hurt right now?** → `status`: `questionable` (day-to-day / camp
  nick) · `injured` (will miss games) · `out_season` (IR, ACL, Achilles…).
  Give `games_out` (regular-season games expected missed, 0–17) and
  `injury` (short, e.g. "ACL", "hamstring").
- **Suspended / holdout / retired?** → `suspended` (+ `games_out`),
  `holdout`, `retired`.
- **Could he be cut or lose the job?** → `cut_risk` 0–1 (probability he is
  not on the 53 / not a fantasy factor), `status: cut_risk` when it is the
  headline. `released` if already gone.
- **Role** → `starter` · `committee` · `backup` · `depth`.
- Healthy, secure starters are `status: active` with a one-line `note`
  (e.g. "bell-cow, 75% snaps in camp"). Do not skip them — the engine treats
  a missing record as "unknown", not "healthy".
- `confidence` 0–1, `source` naming outlet + date.

## Part 2 — team facts (what projections cannot know)

- Coaching / scheme change and what that coordinator's offenses looked like
  → `team:$TEAM` facts, kinds `coach_change` / `scheme_note`, numeric `value`
  when quantitative (pass-rate delta, plays/game).
- Volume redistribution: who left, who arrived, who absorbs vacated targets
  and carries → `unit:$TEAM-POS` / `player:` facts, kinds `role_note` /
  `volume_prior`.
- Offensive line quality → `unit:$TEAM-OL`, kind `oline_grade`, `value` =
  projected rank.

## Rules

- Every record: one claim, honest confidence, dated source. Five strong
  facts beat twenty weak ones — but the player list must be complete.
- Player names exactly as rostered (they resolve through an ID crosswalk;
  an unresolvable name fails the import loudly — fix it, don't drop it).
- Stale news presented as current is the main failure mode: check the date.

## Finish

Write `data/research/$TEAM.json` with `season`, `as_of` (today), `facts`,
`players`, then:

```bash
python3 -m howie3 graph import data/research/$TEAM.json
python3 -m howie3 research coverage
```

Report the imported count and any unresolved names.
