"""Rules as an engine policy, and the coaching loop's scoring pieces."""

import pytest

from howie3.config import Settings
from howie3.state import DraftState, Rule
from howie3.value.board import PoolPlayer
from howie3.value.policy import apply_rules, roster_counts


class _Plan:
    def __init__(self, p, value=100.0):
        self.player = p
        self.final_value = value
        self.sim = None


def _pp(uid, name, pos, value=100.0):
    return _Plan(PoolPlayer(uid, name, pos, None, 100, adp=None, stdev=None, bye=None), value)


def test_apply_rules_blocks_forces_and_targets():
    results = [_pp("q", "Josh Allen", "QB"), _pp("r", "Bijan Robinson", "RB"), _pp("w", "Puka Nacua", "WR"), _pp("t", "Trey McBride", "TE")]
    fx = DraftState(rules=[Rule("WAIT QB UNTIL R6")]).active_rule_effects()
    assert [r.player.position for r in apply_rules(results, 2, fx, {})] == ["RB", "WR", "TE"]
    assert [r.player.position for r in apply_rules(results, 6, fx, {})][0] == "QB"        # expired
    fx = DraftState(rules=[Rule("NO QB BEFORE R3"), Rule("NO RB BEFORE R3"), Rule("NO WR BEFORE R3"), Rule("NO TE BEFORE R3")]).active_rule_effects()
    assert len(apply_rules(results, 1, fx, {})) == 4, "if every candidate is blocked the engine order stands"
    fx = DraftState(rules=[Rule("2 RBs BY R3")]).active_rule_effects()
    assert apply_rules(results, 3, fx, {"RB": 1})[0].player.position == "RB"                # need unmet at R3
    assert apply_rules(results, 3, fx, {"RB": 2})[0].player.position == "QB"                # need met
    fx = DraftState(rules=[Rule("TARGET McBride")]).active_rule_effects()
    assert apply_rules(results, 1, fx, {})[0].player.name == "Trey McBride"
    # a target far below the best candidate stays where the engine put him
    far = [_pp("q", "Josh Allen", "QB", 300), _pp("t", "Trey McBride", "TE", 240)]
    assert apply_rules(far, 1, fx, {})[0].player.name == "Josh Allen"
    close = [_pp("q", "Josh Allen", "QB", 300), _pp("t", "Trey McBride", "TE", 290)]
    assert apply_rules(close, 1, fx, {})[0].player.name == "Trey McBride"
    assert roster_counts([results[1].player, results[2].player]) == {"RB": 1, "WR": 1}


def test_rules_change_what_the_lab_engine_drafts(settings, tmp_path, monkeypatch, league12):
    from howie3 import mocksim

    monkeypatch.setattr(mocksim, "store_path", lambda st: tmp_path / "mock_sims.json")
    free = mocksim.run_mock_drafts(settings, 2, policy="howie", seed=3, effects={}, persist=False)["drafts_run"]
    fx = DraftState(rules=[Rule("WAIT RB UNTIL R8"), Rule("WAIT WR UNTIL R8")]).active_rule_effects()
    ruled = mocksim.run_mock_drafts(settings, 2, policy="howie", seed=3, effects=fx, persist=False)["drafts_run"]
    conn = __import__("howie3.service", fromlist=["_conn"])._conn(settings)
    pos = {r[0]: r[1] for r in conn.execute("SELECT player_uid, position FROM projections WHERE season = 2026")}
    conn.close()
    first_free = [pos[d["mine"][0]] for d in free]
    first_ruled = [pos[d["mine"][0]] for d in ruled]
    assert all(p in ("RB", "WR") for p in first_free)
    assert all(p not in ("RB", "WR") for p in first_ruled), first_ruled
    assert not (tmp_path / "mock_sims.json").exists(), "persist=False leaves the lab store alone"


def test_coach_score_and_digest_without_api(settings, tmp_path, monkeypatch, league12):
    from howie3 import coach, mocksim

    monkeypatch.setattr(mocksim, "store_path", lambda st: tmp_path / "mock_sims.json")
    monkeypatch.setattr(coach, "replay_2025", lambda *a, **k: None)   # keep the test fast
    sc = coach.score(settings, [Rule("WAIT QB UNTIL R6")], n_drafts=2, seed=5, reps=1)
    summ = sc["sim"]["summary"]
    assert summ["drafts"] == 2 and summ["mc_mean"] > 1000 and "pos_by_round" in summ
    assert all(d["picks"][0]["pos"] != "QB" for d in sc["sim"]["drafts"])
    dg = coach.digest_for(settings, [Rule("WAIT QB UNTIL R6")], "notes", sc, None, [])
    assert dg["rules"] == ["WAIT QB UNTIL R6"] and dg["sample_drafts"]
    assert coach.better(sc, None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert coach._coach_call(settings, dg)["available"] is False


def test_apply_changes_validates_rule_patterns(settings, tmp_path, monkeypatch):
    from howie3 import coach

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    DraftState(created="x", rules=[Rule("NO QB BEFORE R3")]).save(settings)
    rules = coach.apply_changes(settings, [Rule("NO QB BEFORE R3")], "", {
        "rules_add": ["WAIT QB UNTIL R7", "draft good players", "2 RBs BY R4"],
        "rules_remove": ["no qb before r3"], "note": "stack RBs early"})
    texts = [r.text for r in rules if r.on]
    assert texts == ["WAIT QB UNTIL R7", "2 RBs BY R4"], texts
    st = DraftState.load(settings)
    assert "coach" in st.notes and "stack RBs early" in st.notes


def test_ci_gate_keeps_the_incumbent_unless_the_paired_gain_is_clear():
    from howie3 import coach

    def sc(mc, rep):
        return {"sim": {"summary": {"mc_mean": sum(mc) / len(mc)}, "drafts": [{"mc_mean": x} for x in mc]},
                "replay": {"mean_total": sum(rep) / len(rep), "scores": rep}}
    base = sc([2000 + i * 10 for i in range(12)], [1600 + i * 5 for i in range(16)])
    noise = sc([2000 + i * 10 + (3 if i % 2 else -4) for i in range(12)], [1600 + i * 5 + (2 if i % 2 else -3) for i in range(16)])
    clear = sc([2000 + i * 10 + 40 + (i % 3) for i in range(12)], [1600 + i * 5 + 30 + (i % 2) for i in range(16)])
    worse = sc([2000 + i * 10 - 30 for i in range(12)], [1600 + i * 5 + 30 for i in range(16)])
    g = coach.paired_gain(noise, base)
    assert g["replay"]["ci"][0] < 0 < g["replay"]["ci"][1] and not coach.better(noise, base), "noise never wins"
    assert coach.paired_gain(clear, base)["replay"]["ci"][0] > 0 and coach.better(clear, base)
    assert not coach.better(worse, base), "a clear loss on either axis is never adopted"
    assert coach.better(base, None)


def test_candidate_rules_and_parallel_scoring_shape(settings, tmp_path, monkeypatch, league12):
    from howie3 import coach, mocksim
    from howie3.state import Rule

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    monkeypatch.setattr(mocksim, "store_path", lambda st: tmp_path / "mock_sims.json")
    rs = coach.candidate_rules([Rule("NO QB BEFORE R3")], {"rules_add": ["NO BYE STACK > 2", "nonsense rule"], "rules_remove": ["no qb before r3"]})
    assert [r.text for r in rs] == ["NO BYE STACK > 2"]
    a, b = coach.score_many(settings, [[], rs], n_drafts=2, seed=5, reps=1, workers=1)
    assert len(a["sim"]["drafts"]) == 2 == len(b["sim"]["drafts"])
    assert a["sim"]["drafts"][0]["seed"] == b["sim"]["drafts"][0]["seed"], "candidates share seeds"
    g = coach.paired_gain(b, a)
    assert "sim" in g and g["sim"]["n"] == 2
