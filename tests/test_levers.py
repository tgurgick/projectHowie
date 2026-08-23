"""New strategy levers: bye-stack cap and age bans."""

from howie3.state import DraftState, Rule, rule_key
from howie3.value.board import PoolPlayer
from howie3.value.policy import apply_rules, roster_counts
from howie3.value.roster import PickPlan


def _pp(uid, pos, proj, bye=None, age=None):
    return PoolPlayer(uid, uid, pos, None, proj, adp=None, stdev=None, bye=bye, age=age)


def test_rules_parse():
    fx = DraftState(rules=[Rule("NO BYE STACK > 2"), Rule("NO WR AGE 29+ BEFORE R6"), Rule("no rb age over 28 before round 8")]).active_rule_effects()
    assert fx["bye_cap"] == [2] and fx["age"] == [("WR", 29, 6), ("RB", 28, 8)]
    assert rule_key("NO BYE STACK > 2") == ("bye_cap", "*") and rule_key("NO WR AGE 29+ BEFORE R6") == ("age", "WR")


def test_age_ban_and_bye_cap_reorder_candidates():
    old = _pp("old", "WR", 230, bye=7, age=30.2)
    young = _pp("young", "WR", 225, bye=9, age=24.0)
    stack = _pp("stack", "RB", 240, bye=7, age=25.0)
    fresh = _pp("fresh", "RB", 235, bye=11, age=25.0)
    results = [PickPlan(stack, 2000), PickPlan(old, 1990), PickPlan(fresh, 1980), PickPlan(young, 1970)]
    roster = [_pp("a", "WR", 200, bye=7), _pp("b", "TE", 150, bye=7)]   # two starters already on bye 7
    fx = DraftState(rules=[Rule("NO WR AGE 29+ BEFORE R6"), Rule("NO BYE STACK > 2")]).active_rule_effects()
    out = apply_rules(results, 3, fx, roster_counts(roster), roster)
    names = [r.player.uid for r in out]
    assert "old" not in names, "a 30-year-old WR is excluded before R6"
    assert names[:2] == ["fresh", "young"], "bye-7 stackers drop behind the rest"
    assert names[-1] == "stack"
    later = apply_rules(results, 7, fx, roster_counts(roster), roster)
    assert "old" in [r.player.uid for r in later], "the age ban expires at R6"
