"""TEAM report: official depth-chart ingestion and the fused team payload."""

import pandas as pd
import pytest

from howie3.config import Settings
from howie3.db import connect
from howie3.state import DraftState


def test_depth_chart_ingest_keeps_latest_snapshot_and_matches_ids(tmp_path):
    from howie3 import depth

    conn = connect(tmp_path / "d.db")
    conn.execute("INSERT INTO players (player_uid, name, name_key, position, team) VALUES ('00-0000001','A One','a one','WR','AAA')")
    frame = pd.DataFrame([
        {"dt": "2026-08-20T07:00:00Z", "team": "AAA", "player_name": "Old Guy", "gsis_id": "00-0000009", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1},
        {"dt": "2026-08-22T07:00:00Z", "team": "AAA", "player_name": "A One", "gsis_id": "00-0000001", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1},
        {"dt": "2026-08-22T07:00:00Z", "team": "AAA", "player_name": "Unknown Body", "gsis_id": "00-0000077", "pos_abb": "WR", "pos_slot": 8, "pos_rank": 2},
        {"dt": "2026-08-22T07:00:00Z", "team": "AAA", "player_name": "A Guard", "gsis_id": "00-0000078", "pos_abb": "OG", "pos_slot": 1, "pos_rank": 1},
    ])
    assert depth.refresh_depth_charts(conn, 2026, frame=frame) == 2
    chart = depth.team_depth(conn, 2026, "aaa")
    assert [r["name"] for r in chart["WR"]] == ["A One", "Unknown Body"]
    assert chart["WR"][0]["uid"] == "00-0000001" and chart["WR"][1]["uid"] is None
    assert chart["WR"][0]["slot"] == "X" and chart["WR"][1]["slot"] == "slot"
    assert chart["QB"] == []
    conn.close()


def test_team_payload_fuses_chart_projection_status_and_board(tmp_path, monkeypatch):
    from howie3 import service

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    r = service.team_payload(s, DraftState.load(s), "phi")
    assert r["team"] == "PHI" and r["name"] == "Philadelphia Eagles"
    assert set(r["rooms"]) == {"QB", "RB", "WR", "TE"}
    qb = r["rooms"]["QB"]["rows"]
    assert qb and qb[0]["name"] == "Jalen Hurts" and qb[0]["proj"] and qb[0]["rank"] == 1
    wr = r["rooms"]["WR"]["rows"]
    assert any(row["slot"] == "X" for row in wr)
    assert any(row["share"] is not None for row in wr), "last-season shares are fused in"
    assert r["coverage"]["team"] == "PHI" and "players_researched" in r["coverage"]
    assert r["next_pick"] > r["current_pick"]
    with pytest.raises(ValueError):
        service.team_payload(s, DraftState.load(s), "XYZ")


def test_resolve_team_accepts_codes_names_and_nicknames():
    from howie3.service import resolve_team

    assert resolve_team("phi") == "PHI"
    assert resolve_team("eagles") == "PHI"
    assert resolve_team("Philadelphia Eagles") == "PHI"
    assert resolve_team("philly") == "PHI"
    assert resolve_team("giants") == "NYG"
    with pytest.raises(ValueError, match="matches"):
        resolve_team("new york")
    with pytest.raises(ValueError):
        resolve_team("mars")


def test_season_grid_marks_byes_outs_and_empty_slots(tmp_path, monkeypatch, league12):
    from howie3 import service

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="x", mode="live"); st.save(s)
    empty = service.season_grid_payload(s, DraftState.load(s))
    assert empty["players"] == 0 and len(empty["weeks"]) == 17
    assert all(c["level"] == "grey" and c["reason"] == "not drafted" for r in empty["rows"] for c in r["cells"])
    conn = service._conn(s); pool = service._pool(s, conn); conn.close()
    rb = next(p for p in pool if p.position == "RB" and p.bye and p.draftable)
    service.mark_pick(s, rb.uid, mine=True)
    g = service.season_grid_payload(s, DraftState.load(s))
    row = next(r for r in g["rows"] if r["slot"] == "RB1")
    bye_cell = row["cells"][rb.bye - 1]
    assert bye_cell["level"] == "grey" and bye_cell["reason"] == "bye"
    live = [c for c in row["cells"] if c["name"]]
    assert len(live) == 16 and all(c["name"] == rb.name and c["pts"] > 0 for c in live)
    assert g["week_totals"][rb.bye - 1]["bye"] == [rb.name]
    assert {c["level"] for c in live} <= {"green", "yellow", "red"}


def test_plan_payload_covers_every_round(tmp_path, monkeypatch, league12):
    from howie3 import service

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="x", mode="live", rules=[__import__("howie3.state", fromlist=["Rule"]).Rule("WAIT QB UNTIL R6")]); st.save(s)
    p = service.plan_payload(s, DraftState.load(s))
    assert [r["round"] for r in p["rows"]] == list(range(1, 17))
    assert p["rows"][0]["state"] == "now" and p["rows"][0]["pos"] and p["rows"][0]["player"]
    assert all(r["state"] == "plan" for r in p["rows"][1:])
    assert all(set(r["depth"]) == {"QB", "RB", "WR", "TE"} for r in p["rows"])
    assert any(t["type"] == "wait" and t["pos"] == "QB" for t in p["rows"][0]["rules"])
    assert not any(t["pos"] == "QB" for t in p["rows"][6]["rules"]), "the wait rule expires at R6"
    # a completed pick shows as done and the plan advances
    conn = service._conn(s); pool = service._pool(s, conn); conn.close()
    for q in pool[:7]:
        service.mark_pick(s, q.uid, mine=False)
    service.mark_pick(s, pool[7].uid, mine=True)
    p2 = service.plan_payload(s, DraftState.load(s))
    assert p2["rows"][0]["state"] == "done" and p2["rows"][0]["player"] == pool[7].name
    assert p2["rows"][1]["state"] == "now" and p2["rows"][1]["pick"] == 17


def test_lookahead_gives_best_alt_and_likely_for_next_picks(tmp_path, monkeypatch, league12):
    from howie3 import service

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    fs = service.lookahead_payload(s, DraftState(created="x"), 3)
    assert [p["pick"] for p in fs["picks"]] == [8, 17, 32] and fs["picks"][0]["picks_away"] == 7
    for p in fs["picks"]:
        assert p["best"]["name"] and 0 <= p["best"]["avail"] <= 1 and len(p["candidates"]) >= 2
        if p["safe"]:
            assert p["safe"]["avail"] >= p["best"]["avail"]


def test_plan_adapts_to_the_roster_and_sim_adp_is_reported(tmp_path, monkeypatch, league12):
    from howie3 import mocksim, service

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    monkeypatch.setattr(mocksim, "store_path", lambda st: tmp_path / "mock_sims.json")
    st = DraftState(created="x", mode="live"); st.save(s)
    base = service.plan_payload(s, DraftState.load(s))
    base_wr = sum(1 for r in base["rows"] if r["state"] == "plan" and r["pos"] == "WR")
    # draft two WRs to me early (slot 8): the remaining plan must shed WRs
    conn = service._conn(s); pool = service._pool(s, conn); conn.close()
    wrs = [p for p in pool if p.position == "WR" and p.draftable][:2]
    for i, p in enumerate(pool[:7]):
        service.mark_pick(s, p.uid, mine=False)
    service.mark_pick(s, wrs[0].uid, mine=True)
    for p in pool[8:16]:
        if p.uid not in DraftState.load(s).taken_uids():
            service.mark_pick(s, p.uid, mine=False)
    service.mark_pick(s, wrs[1].uid, mine=True)
    p2 = service.plan_payload(s, DraftState.load(s))
    later_wr = sum(1 for r in p2["rows"] if r["state"] == "plan" and r["pos"] == "WR")
    assert [r["state"] for r in p2["rows"][:2]] == ["done", "done"]
    assert later_wr < base_wr, (base_wr, later_wr)
    # sim ADP flows onto board rows once drafts exist in the store
    mocksim.run_mock_drafts(s, 3, policy="adp", seed=3)
    rows = service.pick_payload(s, DraftState.load(s), top_n=5)["rows"]
    assert any(r["sim_adp"] is not None and r["sim_adp_n"] == 3 for r in rows)
