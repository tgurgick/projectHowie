"""Strategy rules as an engine policy.

`active_rule_effects` parses the strategy sheet; this module makes those
effects change what the fast drafter actually picks — in the cockpit board,
the Mock Draft Lab's "howie" policy, the 2025 replay, and the coaching loop —
so a rule is a real lever, not a tag.

Effects on a ranked candidate list for the user's round `rnd`:
  wait / ban  — candidates at that position are excluded while rnd < round
                (if nothing is left, the original order stands)
  need        — "N POS by R": when the roster is still short at round R, that
                position's candidates move to the front
  target      — a named player still on the board moves to the front, but
                only when he is within TARGET_TOLERANCE points of the best
                candidate: a target is "take him when it's close", never
                "take him at any cost" (a pick-17 target stays a tag at pick 5)
  badges      — structural signals (value/badges.py) shift a candidate's
                position in the order by a few engine points, without
                touching his projection. Worth about half a target at most,
                so a badge settles a close call and never overrules a real
                gap in value.
"""

from typing import Dict, List, Sequence

from .badges import nudge_of

TARGET_TOLERANCE = 8.0   # engine value points a target may trail the best by and still jump the line


def apply_rules(results: Sequence, rnd: int, effects: Dict[str, list],
                roster_positions: Dict[str, int], roster: Sequence = ()) -> List:
    """Reorder PickPlan-like objects (have .player with .name/.position).
    `roster` (PoolPlayers) is needed for the bye-stack cap."""
    from ..data.names import name_key

    if not effects or not results:
        return list(results)
    blocked = {pos for pos, until in effects.get("wait", []) if rnd < until}
    blocked |= {pos for pos, before in effects.get("ban", []) if rnd < before}
    age_rules = [(pos, age, before) for pos, age, before in effects.get("age", []) if rnd < before]

    def too_old(p) -> bool:
        return any(p.position == pos and (p.age or 0) >= age for pos, age, _ in age_rules)

    allowed = [r for r in results if r.player.position not in blocked and not too_old(r.player)] or list(results)
    caps = effects.get("bye_cap", [])
    stacked: set = set()
    if caps and roster:
        cap = min(caps)
        bye_counts: Dict[int, int] = {}
        for p in roster:
            if getattr(p, "bye", None):
                bye_counts[p.bye] = bye_counts.get(p.bye, 0) + 1
        stacked = {r.player.uid for r in allowed
                   if getattr(r.player, "bye", None) and bye_counts.get(r.player.bye, 0) + 1 > cap}

    forced = set()
    for pos, n, by in effects.get("need", []):
        if rnd == by and roster_positions.get(pos, 0) < n:
            forced.add(pos)
    targets = [name_key(t) for t in effects.get("targets", [])]

    def value(r) -> float:
        sim = getattr(r, "sim", None)
        return float(sim.mean) if sim is not None else float(getattr(r, "final_value", 0.0))

    best_value = max((value(r) for r in allowed), default=0.0)

    def rank(r) -> tuple:
        is_target = any(t and t in name_key(r.player.name) for t in targets) and value(r) >= best_value - TARGET_TOLERANCE
        # Inside a band the order is engine value plus the badge stack. The
        # nudge is clamped in badges.py, so this bends close calls and leaves
        # genuine gaps in value alone.
        return (0 if is_target else 1,
                0 if r.player.position in forced else 1,
                1 if r.player.uid in stacked else 0,
                -(value(r) + nudge_of(getattr(r.player, "badges", None))))

    return sorted(allowed, key=rank)


def roster_counts(roster: Sequence) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in roster:
        out[p.position] = out.get(p.position, 0) + 1
    return out
