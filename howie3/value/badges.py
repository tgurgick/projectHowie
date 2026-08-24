"""Badges: structural signals a projection does not carry.

Five auto-derived badges plus the user's own favorites. A badge NEVER changes
`proj` — the value estimate stays the engine's honest number, and availability
(howie3.status) stays the only thing allowed to scale it. Instead each badge
carries a `nudge` in engine points that `value/policy.py` spends to bend the
*pick order* within a tolerance band, the same way a Target rule does. That
keeps marginal value comparable across players and stops a badge from
silently compounding with the injury discount.

    RISK   age past the position's curve, a major injury in the rear view, or
           a heavy career touch load — the part of risk `games_out` does not
           already price.
    VALUE  the source projection ranks materially above the market's ADP.
    PATH   cheap, and with a documented path to volume (a gutted room he
           leads, or researched spillover pointing at him).
    BOOST  line, game script and room share all point his way.
    DRAG   they point away.

RISK and DRAG carry negative nudges. Stacked badges sum, then clamp, so a
player can never accumulate as much pull as a single explicit Target.
"""

import re
import sqlite3
from typing import Dict, List, Optional, Sequence

# A badge is worth HALF an explicit target, and a whole stack still less than
# one. Both are stated relative to policy.TARGET_TOLERANCE rather than pinned
# to a number — that constant has moved before, and when it does these must
# move with it. test_badges.py asserts the relationship so it cannot drift
# silently again.
BADGE_NUDGE = 4.0          # engine points per badge — half of policy.TARGET_TOLERANCE (8.0)
NUDGE_CLAMP = 6.0          # a full stack still trails one explicit target

# Age at which the position's curve turns down, as of Sept 1.
AGE_CURVE = {"RB": 27.0, "WR": 30.0, "TE": 31.0, "QB": 36.0}
HEAVY_TOUCHES = 1300       # career rush attempts + receptions (RB only)
MAJOR_INJURY = re.compile(r"\b(acl|achilles|lisfranc|patellar|torn\b|rupture)", re.I)

# Within a position, ADP orders players almost exactly as the projections do
# — the market is efficient at that job, and measured gaps top out around ten
# ranks. So VALUE is a DECILE, not a fixed distance: the top tenth of each
# position by value-over-market, floored so a flat position badges nobody.
VALUE_DECILE = 0.10
VALUE_MIN_GAP = 4          # ranks within the position, below which it is noise
VALUE_POSITIONS = ("RB", "WR", "TE", "QB")   # kickers and defenses have no meaningful market edge
CHEAP_ADP = 100.0          # "cheap" for the PATH badge
VACATED_ROOM = 0.20        # share of a room's volume that left for it to count as gutted
STRUCT_THRESHOLD = 0.60    # |score| at which BOOST / DRAG fires

# How much each structural component moves a position. O-line and game script
# drive a back's floor; a receiver lives on his room's share of the targets.
WEIGHTS = {
    #        oline  script  share
    "RB":   (1.00,  0.80,  0.60),
    "WR":   (0.40,  0.25,  1.00),
    "TE":   (0.35,  0.25,  1.00),
    "QB":   (0.70,  0.30,  0.30),
}
LEAGUE_MEAN_SHARE = {"WR": 0.57, "RB": 0.17, "TE": 0.23}   # from the researched 2026 corpus
SHARE_SPREAD = {"WR": 0.06, "RB": 0.04, "TE": 0.04}


class Badge(dict):
    """{'code','label','nudge','why'} — dict so it serialises straight to the UI."""

    def __init__(self, code: str, label: str, nudge: float, why: str):
        super().__init__(code=code, label=label, nudge=round(nudge, 1), why=why)


def nudge_of(badges: Optional[Sequence[dict]]) -> float:
    """Total pick-order pull from a badge stack, clamped both ways."""
    if not badges:
        return 0.0
    total = sum(float(b.get("nudge") or 0.0) for b in badges)
    return max(-NUDGE_CLAMP, min(NUDGE_CLAMP, total))


# ---------------------------------------------------------------- graph inputs

def _team_structure(conn: sqlite3.Connection) -> Dict[str, dict]:
    """Per team: O-line rank, win total, and target share by position — the
    latest researched value for each, newest import wins."""
    out: Dict[str, dict] = {}

    def latest(entity: str, kinds: Sequence[str]) -> Optional[float]:
        q = ",".join("?" * len(kinds))
        row = conn.execute(
            f"SELECT value FROM facts WHERE entity_id = ? AND kind IN ({q}) "
            "AND value IS NOT NULL ORDER BY created DESC, id DESC LIMIT 1",
            (entity, *kinds)).fetchone()
        return float(row[0]) if row else None

    teams = [r[0].split(":", 1)[1] for r in conn.execute(
        "SELECT id FROM entities WHERE kind = 'team'")]
    for t in teams:
        oline = latest(f"unit:{t}-OL", ("oline_grade",))
        # a rank must look like a rank; some researchers stored a share here
        if oline is not None and not (1.0 <= oline <= 32.0):
            oline = None
        out[t] = {
            "oline": oline,
            "wins": latest(f"team:{t}", ("division_odds", "win_total")),
            "share": {pos: latest(f"unit:{t}-{pos}", ("target_share",)) for pos in ("WR", "RB", "TE")},
        }
    return out


def _rooms(conn: sqlite3.Connection) -> Dict[str, dict]:
    """uid -> {'share', 'rank_in_room', 'vacated'} from the derived room layer."""
    out: Dict[str, dict] = {}
    vacated = {r[0]: float(r[1] or 0.0) for r in conn.execute(
        "SELECT entity_id, value FROM facts WHERE kind = 'vacated_share'")}
    rows = conn.execute(
        "SELECT src, dst, value FROM edges WHERE kind = 'in_room' ORDER BY dst, COALESCE(value, 0) DESC")
    seen: Dict[str, int] = {}
    for src, dst, value in rows:
        seen[dst] = seen.get(dst, 0) + 1
        out[src.split(":", 1)[1]] = {
            "share": float(value) if value is not None else None,
            "rank_in_room": seen[dst],
            "vacated": vacated.get(dst, 0.0),
        }
    return out


def _volume_facts(conn: sqlite3.Connection) -> set:
    """uids carrying a researched fact that points at more work for them."""
    rows = conn.execute(
        "SELECT entity_id FROM facts WHERE kind IN ('role_note', 'volume_prior') "
        "AND entity_id LIKE 'player:%' AND COALESCE(confidence, 0) >= 0.5")
    return {r[0].split(":", 1)[1] for r in rows}


def _career_touches(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT player_uid, SUM(COALESCE(rush_attempts,0) + COALESCE(receptions,0)) "
        "FROM weekly_stats GROUP BY player_uid")
    return {uid: int(n or 0) for uid, n in rows}


# ---------------------------------------------------------------- the badges

def _structural_score(pos: str, team: Optional[str], struct: dict, room: Optional[dict]) -> tuple:
    """Signed score plus the reasons behind it. Positive = the offense helps."""
    w = WEIGHTS.get(pos)
    if not w or not team or team not in struct:
        return 0.0, []
    s = struct[team]
    w_oline, w_script, w_share = w
    score, why = 0.0, []

    if s["oline"] is not None:
        # rank 1 -> +1, rank 32 -> -1
        c = (16.5 - s["oline"]) / 15.5
        score += c * w_oline
        if abs(c * w_oline) >= 0.25:
            why.append(f"O-line {int(s['oline'])}")
    if s["wins"] is not None:
        c = max(-1.0, min(1.0, (s["wins"] - 8.5) / 2.5))
        score += c * w_script
        if abs(c * w_script) >= 0.25:
            why.append(f"{s['wins']:g}-win total")
    share = s["share"].get(pos)
    if share is not None and pos in LEAGUE_MEAN_SHARE:
        c = max(-1.5, min(1.5, (share - LEAGUE_MEAN_SHARE[pos]) / SHARE_SPREAD[pos]))
        # a room's share only reaches the man who leads the room
        lead = 1.0 if (room and room.get("rank_in_room", 9) == 1) else 0.35
        score += c * w_share * lead
        if abs(c * w_share * lead) >= 0.25:
            why.append(f"{pos} share {share:.0%}")
    return score, why


def _risk(p, touches: Dict[str, int]) -> Optional[Badge]:
    why: List[str] = []
    curve = AGE_CURVE.get(p.position)
    if curve is not None and p.age is not None and p.age >= curve:
        why.append(f"age {p.age:g}")
    if p.position == "RB":
        t = touches.get(p.uid, 0)
        if t >= HEAVY_TOUCHES:
            why.append(f"{t:,} career touches")
    st = p.status or {}
    # A major injury he is NOT currently being discounted for: availability
    # already prices games_out, so only the rear-view case is new information.
    # Read the STRUCTURED injury field only — research notes discuss injuries
    # in order to dismiss them ("the Achilles is 2024-25 history"), and a
    # regex over that prose fires on the negation as readily as on the claim.
    if not int(st.get("games_out") or 0):
        m = MAJOR_INJURY.search(st.get("injury") or "")
        if m:
            why.append(f"coming off {m.group(0).lower()}")
    return Badge("RISK", "RISK", -BADGE_NUDGE, " · ".join(why)) if why else None


def compute(conn: sqlite3.Connection, pool: Sequence) -> None:
    """Attach `.badges` to every PoolPlayer. Idempotent; safe to re-run.

    A database with no knowledge graph yet simply badges nobody — the pool is
    still fully usable, so this must never be the thing that breaks a draft."""
    from ..graph import ensure_graph_schema

    ensure_graph_schema(conn)
    struct = _team_structure(conn)
    rooms = _rooms(conn)
    vol = _volume_facts(conn)
    touches = _career_touches(conn)

    # VALUE needs both orderings over the same population.
    # Ranked WITHIN POSITION: across positions the two orderings measure
    # different things — a QB always projects high and drafts late because of
    # replacement level, which is scarcity, not a market mistake.
    value_rank: Dict[str, int] = {}
    adp_rank: Dict[str, int] = {}
    value_cut: Dict[str, int] = {}
    for pos in VALUE_POSITIONS:
        # Both orderings must run over the SAME population or the comparison
        # is meaningless: rank the projection only among players the market
        # has actually priced.
        at_pos = [p for p in pool if p.position == pos and p.adp is not None and (p.raw or p.proj)]
        for i, q in enumerate(sorted(at_pos, key=lambda x: -(x.raw or x.proj))):
            value_rank[q.uid] = i + 1
        for i, q in enumerate(sorted(at_pos, key=lambda x: x.adp)):
            adp_rank[q.uid] = i + 1
        gaps = sorted((adp_rank[q.uid] - value_rank[q.uid] for q in at_pos), reverse=True)
        if gaps:
            cut = gaps[min(int(len(gaps) * VALUE_DECILE), len(gaps) - 1)]
            value_cut[pos] = max(VALUE_MIN_GAP, cut)

    for p in pool:
        badges: List[Badge] = []
        room = rooms.get(p.uid)

        risk = _risk(p, touches)
        if risk:
            badges.append(risk)

        vr, ar = value_rank.get(p.uid), adp_rank.get(p.uid)
        if vr and ar and ar - vr >= value_cut.get(p.position, 10 ** 6):
            badges.append(Badge("VALUE", "VALUE", BADGE_NUDGE,
                                f"engine {p.position}{vr} vs market {p.position}{ar}"))

        cheap = p.adp is None or p.adp >= CHEAP_ADP
        if cheap and p.position in ("RB", "WR", "TE"):
            gutted = room and room["vacated"] >= VACATED_ROOM and room["rank_in_room"] <= 2
            researched = p.uid in vol
            if gutted or researched:
                why = (f"{room['vacated']:.0%} of the room's volume left" if gutted
                       else "researched path to volume")
                badges.append(Badge("PATH", "PATH", BADGE_NUDGE, why))

        score, why = _structural_score(p.position, p.team, struct, room)
        if score >= STRUCT_THRESHOLD:
            badges.append(Badge("BOOST", "BOOST", BADGE_NUDGE, " · ".join(why)))
        elif score <= -STRUCT_THRESHOLD:
            badges.append(Badge("DRAG", "DRAG", -BADGE_NUDGE, " · ".join(why)))

        p.badges = badges
