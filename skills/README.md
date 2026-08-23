# Howie research skills

Playbooks for running deep research with an LLM host (Claude Code, Claude
Desktop) that ends in **structured records**, never prose reports. The output
contract is the knowledge-graph import format; everything imported carries
provenance, confidence, and an expiry date, so research is diffable,
re-runnable, and can be scored by the eval harness later.

## The contract

Research output is a JSON file passed to `howie graph import <file>`. It has
two halves: **`players`** — a typed status record per draft-relevant player
(the engine acts on these: out-for-season players leave the board, games
missed and cut risk scale value) — and **`facts`** — narrative context with
provenance (shown on cards, read by the agent).

```json
{
  "season": 2026,
  "as_of": "2026-08-22",
  "players": [
    {"name": "Quinshon Judkins", "status": "active", "role": "starter",
     "note": "RB1, no competition added", "confidence": 0.8, "source": "cleveland.com 2026-08-20"},
    {"name": "Tucker Kraft", "status": "injured", "games_out": 6, "injury": "ACL (Nov 2025)",
     "role": "starter", "confidence": 0.7, "source": "packers.com 2026-08-19"},
    {"name": "Some Veteran", "status": "cut_risk", "cut_risk": 0.6, "role": "depth",
     "note": "WR6 on the depth chart, non-guaranteed deal", "confidence": 0.6, "source": "..."}
  ],
  "facts": [
    {
      "entity": "team:ARI",
      "kind": "scheme_note",
      "text": "New OC keeps the 12-personnel base; pass rate over expected +3.1% YoY expected.",
      "value": 0.031,
      "confidence": 0.8,
      "source": "research 2026-08-20 (3 sources)",
      "expires": "2026-12-01"
    },
    {
      "entity": "player:Trey McBride",
      "kind": "role_note",
      "text": "Route rate 82%, top-3 among TEs; no target competition added.",
      "confidence": 0.85,
      "source": "research 2026-08-20"
    },
    {
      "entity": "unit:ARI-OL",
      "kind": "oline_grade",
      "text": "Pass-block projected top-10; both tackles return.",
      "value": 9,
      "confidence": 0.7,
      "source": "research 2026-08-20"
    }
  ]
}
```

- `players[].status`: `active` · `questionable` · `injured` · `out_season` ·
  `suspended` · `holdout` · `cut_risk` · `released` · `retired`;
  `games_out` 0–17; `cut_risk` 0–1; `role` `starter` · `committee` ·
  `backup` · `depth` · `unknown`. The latest `as_of` wins; on the same day
  research beats the automatic nflverse roster feed (`howie data refresh
  --steps roster`).
- `entity`: `team:ABBR`, `unit:ABBR-POS`, or `player:<name>` (names resolve
  through the ID crosswalk; unresolvable names fail the import loudly).
- `kind`: short slug — `scheme_note`, `role_note`, `injury_note`,
  `oline_grade`, `coach_change`, `volume_prior`.
- `value`: optional number when the fact is quantitative (a rate delta, a
  rank, a share). Facts with values can later be evaluated.
- `confidence`: 0-1, your honest read.
- Facts show up on player cards in the cockpit and in the agent's
  `entity_context` tool.

## Skills

- `research-team.md` — one team: every draft-relevant player's status + the
  offense (run 32×, parallelizable). The `research-teams` Claude Code
  workflow runs it with subagents: research → skeptical validation → import.
- Bookkeeping: `howie research targets TEAM` (the checklist),
  `howie research coverage` (per-team players researched / facts / latest),
  `howie research stale --days 7` (teams to hand back to the workflow).
- Cadence: a full pass before the draft ("run the research-teams workflow
  for all"), then `stale` weekly in-season or whenever news breaks.

## Draft night: the observer skill

`.claude/skills/draft-observer/SKILL.md` — Claude in Chrome reads the real
draft room (ESPN / Sleeper / Yahoo) and keeps Howie's log in sync with
`howie draft sync`; when you are on the clock it returns `howie draft pick`.
Reset with `howie draft reset --mode live --slot N` first; every finished
draft is archived into the Mock Draft Lab for availability learning.
