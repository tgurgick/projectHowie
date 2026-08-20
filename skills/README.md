# Howie research skills

Playbooks for running deep research with an LLM host (Claude Code, Claude
Desktop) that ends in **structured records**, never prose reports. The output
contract is the knowledge-graph import format; everything imported carries
provenance, confidence, and an expiry date, so research is diffable,
re-runnable, and can be scored by the eval harness later.

## The contract

Research output is a JSON file passed to `howie graph import <file>`:

```json
{
  "season": 2026,
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

- `research-team.md` — deep-dive one NFL team's offense (run 32×, parallelizable)
- Run before the draft; re-run when news breaks; commit the JSON outputs.
