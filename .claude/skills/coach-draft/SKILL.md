---
name: coach-draft
description: Coach Howie's draft strategy — the engine simulates drafts under the strategy sheet, you (Claude) read the digest and change the rules, repeat; or review one real draft. Use for "coach the strategy", "optimize my rules", "review my draft".
---

# Coach the draft (Claude as coach, the engine as the fast drafter)

Two ways to run this:

**A. Automatic** — the built-in loop calls the API model as coach:
```bash
python3 -m howie3 coach run --iterations 3 --drafts 12 --reps 6
```
It prints the trace (MC mean / floor, structural holes, 2025 replay vs ADP
per rule set) and keeps the best rule set on the sheet. The LAB tab has the
same button (COACHED SIMULATION) and shows the trace.

**B. You are the coach** (deeper reasoning, full context of this repo):
1. `python3 -m howie3 strategy show` — the current sheet.
2. Produce a digest for the current rules and the no-rules baseline:
   ```bash
   python3 - <<'EOF'
   import json; from howie3.config import Settings; from howie3 import coach
   from howie3.state import DraftState
   s = Settings(); st = DraftState.load(s)
   cur = coach.score(s, st.rules, n_drafts=12, seed=7, reps=6)
   base = coach.score(s, [], n_drafts=12, seed=7, reps=6)
   print(json.dumps(coach.digest_for(s, st.rules, st.notes, cur, base, []), default=str))
   EOF
   ```
3. Read it the way the system prompt in `howie3/coach.py` (COACH_SYSTEM)
   describes: the engine already maximizes expected points; you look for
   what it cannot see — empty starting slots and bye stacks in the heatmap
   summary, positional timing that loses on the 2025 replay, a rule that is
   hurting. Realized 2025 points decide; MC mean breaks ties.
4. Apply with `python3 -m howie3 strategy add "WAIT QB UNTIL R7"`,
   `strategy remove "..."`, `strategy note "..."` (rules must use the
   engine's patterns: WAIT/NO BEFORE/N POS BY R/TARGET). Re-score; keep the
   better set; stop when a round changes nothing.

**Reviewing a real draft** (after the draft-observer skill tracked a room):
`python3 -m howie3 coach review` scores your actual picks (MC season, weak
weeks, bye stacks) and returns learnings + suggested rule changes.
