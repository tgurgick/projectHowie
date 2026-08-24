"""Badges: structural signals, and the nudge they spend on the pick order."""

import sqlite3

from howie3.state import DraftState
from howie3.value import badges
from howie3.value.badges import Badge, nudge_of
from howie3.value.board import PoolPlayer
from howie3.value.policy import apply_rules, roster_counts
from howie3.value.roster import PickPlan


def _pp(uid, pos, proj, adp=None, age=None, status=None, raw=None):
    p = PoolPlayer(uid, uid, pos, "DET", proj, adp=adp, stdev=None, bye=None, age=age)
    p.raw = raw if raw is not None else proj
    p.status = status
    return p


def test_a_badge_stays_worth_half_a_target():
    """These two constants live in different modules and TARGET_TOLERANCE has
    been retuned before (15 -> 8). Pin the relationship so the next retune
    cannot silently promote a badge above an explicit target."""
    from howie3.value.policy import TARGET_TOLERANCE

    assert badges.BADGE_NUDGE == TARGET_TOLERANCE / 2
    assert badges.NUDGE_CLAMP < TARGET_TOLERANCE, "a full stack must trail one target"


def test_nudge_clamps_a_badge_stack():
    one = [Badge("BOOST", "BOOST", badges.BADGE_NUDGE, "x")]
    assert nudge_of(one) == badges.BADGE_NUDGE
    assert nudge_of(None) == 0.0
    piled = [Badge("BOOST", "BOOST", badges.BADGE_NUDGE, "x")] * 4
    assert nudge_of(piled) == badges.NUDGE_CLAMP, "a stack never outweighs an explicit target"
    # RISK and BOOST cancel rather than compound
    assert nudge_of([Badge("RISK", "RISK", -badges.BADGE_NUDGE, "x"),
                     Badge("BOOST", "BOOST", badges.BADGE_NUDGE, "y")]) == 0.0


def test_risk_reads_the_structured_injury_field_not_the_prose():
    """Research notes discuss an injury in order to DISMISS it. A regex over
    that prose fires on the negation, so only the injury field may be read."""
    dismissed = _pp("dismissed", "WR", 200, age=24.0, status={
        "games_out": 0, "injury": None,
        "note": "INJURY FIELD CLEARED: the Achilles rupture is 2024-25 history, not current"})
    assert badges._risk(dismissed, {}) is None

    real = _pp("real", "WR", 200, age=24.0, status={
        "games_out": 0, "injury": "knee (ACL)", "note": "avoided PUP"})
    got = badges._risk(real, {})
    assert got and "coming off acl" in got["why"]


def test_risk_does_not_double_count_a_priced_injury():
    """`availability_factor` already scales value by games_out — a badge that
    also demoted for it would charge the same injury twice."""
    priced = _pp("priced", "RB", 200, age=24.0, status={
        "games_out": 4, "injury": "ACL", "note": ""})
    assert badges._risk(priced, {}) is None


def test_risk_fires_on_age_and_career_touches():
    old = _pp("old", "RB", 200, age=28.1, status=None)
    assert "age 28.1" in badges._risk(old, {})["why"]
    worn = _pp("worn", "RB", 200, age=25.0, status=None)
    assert "career touches" in badges._risk(worn, {"worn": 1500})["why"]
    fine = _pp("fine", "RB", 200, age=23.0, status=None)
    assert badges._risk(fine, {"fine": 200}) is None


def test_badges_bend_the_order_without_touching_projections():
    gap = badges.BADGE_NUDGE / 2          # comfortably inside one badge
    plain = _pp("plain", "RB", 200)
    badged = _pp("badged", "RB", 200 - gap)
    badged.badges = [Badge("BOOST", "BOOST", badges.BADGE_NUDGE, "O-line 3")]
    before = badged.proj
    results = [PickPlan(plain, 2000.0), PickPlan(badged, 2000.0 - gap)]
    out = apply_rules(results, 3, DraftState().active_rule_effects(), roster_counts([]), [])
    assert [r.player.uid for r in out] == ["badged", "plain"], "a sub-nudge gap flips"
    assert badged.proj == before, "a badge must never change the value estimate"


def test_a_badge_never_overrules_a_real_gap_in_value():
    great = _pp("great", "RB", 240)
    badged = _pp("badged", "RB", 200)
    badged.badges = [Badge("BOOST", "BOOST", badges.BADGE_NUDGE, "x"),
                     Badge("PATH", "PATH", badges.BADGE_NUDGE, "y")]
    results = [PickPlan(great, 2040.0), PickPlan(badged, 2000.0)]
    out = apply_rules(results, 3, DraftState().active_rule_effects(), roster_counts([]), [])
    assert [r.player.uid for r in out] == ["great", "badged"]


def test_compute_on_a_database_with_no_graph_badges_nobody():
    """A draft must survive an un-researched database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE weekly_stats (player_uid TEXT, rush_attempts INT, receptions INT)")
    pool = [_pp("a", "RB", 200, adp=10.0, age=24.0)]
    badges.compute(conn, pool)
    assert pool[0].badges == []


def test_value_is_ranked_within_the_position():
    """Across positions the two orderings measure different things: a QB
    always projects high and drafts late, which is replacement level, not a
    market mistake."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE weekly_stats (player_uid TEXT, rush_attempts INT, receptions INT)")
    # One QB projecting above every RB but drafted after all of them.
    pool = [_pp("qb", "QB", 400, adp=90.0, age=27.0)]
    pool += [_pp(f"rb{i}", "RB", 300 - i, adp=float(i + 1), age=24.0) for i in range(30)]
    badges.compute(conn, pool)
    codes = [b["code"] for p in pool if p.uid == "qb" for b in p.badges]
    assert "VALUE" not in codes, "positional scarcity is not a market edge"
