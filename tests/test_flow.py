"""Draft-flow simulation and the sequence decision."""

import pytest

from howie3.config import Settings
from howie3.state import DraftState


@pytest.fixture
def settings():
    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    return s


def test_flow_is_conditioned_on_the_live_board(settings, tmp_path, monkeypatch, league12):
    from howie3 import service
    from howie3.value.flow import attach, draft_flow

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="x", mode="live"); st.save(settings)
    conn = service._conn(settings); pool = service._pool(settings, conn); conn.close()
    for p in pool[:7]:
        service.mark_pick(settings, p.uid, mine=False)
    st = DraftState.load(settings)
    flow = draft_flow(pool, st, settings.league, n=60, horizon=3)
    assert flow.picks == [17, 32, 41], "on the clock at 8: the NEXT three picks"
    gone = {p.uid for p in pool[:7]}
    assert all(uid not in flow.avail for uid in gone)
    top = pool[7]                           # best remaining player: gone before 17 in nearly every rollout
    assert flow.avail.get(top.uid, {}).get(17, 0.0) <= 0.2
    deep = next(p for p in pool if p.adp and p.adp > 150)
    assert flow.avail[deep.uid][17] >= 0.9 and flow.avail[deep.uid][41] >= 0.6
    assert all(0 <= v <= 1 for per in flow.avail.values() for v in per.values())
    assert set(flow.survivors[17]) == {"QB", "RB", "WR", "TE"}
    attach(pool, flow)
    assert top.p_available(17) == flow.avail.get(top.uid, {}).get(17, 0.0) and top.availability_source(17) == "flow"
    assert pool[0].flow_avail == {17: 0.0, 32: 0.0, 41: 0.0}     # taken: never survives
    assert top.availability_source(100) != "flow"                # beyond the horizon: analytic model


def test_sequence_payload_is_a_decision_over_the_next_picks(settings, tmp_path, monkeypatch, league12):
    from howie3 import service

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="y", mode="live"); st.save(settings)
    conn = service._conn(settings); pool = service._pool(settings, conn); conn.close()
    for p in pool[:7]:
        service.mark_pick(settings, p.uid, mine=False)
    q = service.sequence_payload(settings, DraftState.load(settings))
    assert q["current_pick"] == 8 and q["now"]["name"] and q["rollouts"] > 0
    assert [s["pick"] for s in q["next"]] == [17, 32, 41][:len(q["next"])] and q["next"]
    assert q["next"][0]["pick"] > q["current_pick"]
    names = [s["target"]["name"] for s in q["next"] if s["target"]]
    assert len(names) == len(set(names)) and q["now"]["name"] not in names, "a player is used once in the sequence"
    assert all(s["target"] is None or s["target"]["p"] >= 0.5 for s in q["next"])
    assert len(q["plan_prior"]) <= 3 and isinstance(q["overrides_plan"], bool)
    # the board's availability now comes from the same rollout
    rows = service.pick_payload(settings, DraftState.load(settings), top_n=5)["rows"]
    assert all(r["avail_src"] == "flow" for r in rows)
