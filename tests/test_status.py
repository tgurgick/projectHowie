"""Player status layer: import contract, roster feed mapping, precedence,
and the engine's use of it (applied after the market anchor)."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from howie3.config import Settings
from howie3.db import connect
from howie3 import status as st


def _mini_db(tmp_path):
    conn = connect(tmp_path / "s.db")
    for uid, name, key, pos, team, proj, adp in [
        ("00-0000001", "Healthy Guy", "healthy guy", "RB", "AAA", 200, 20.0),
        ("00-0000002", "Torn Acl", "torn acl", "RB", "AAA", 210, 12.0),
        ("00-0000003", "Six Weeks", "six weeks", "WR", "BBB", 180, 30.0),
        ("00-0000004", "Bubble Guy", "bubble guy", "WR", "BBB", 120, 150.0),
    ]:
        conn.execute("INSERT INTO players (player_uid, name, name_key, position, team) VALUES (?,?,?,?,?)",
                     (uid, name, key, pos, team))
        conn.execute("INSERT INTO projections (season, source, player_uid, position, team, pts_half, games) "
                     "VALUES (2026, 'pff', ?, ?, ?, ?, 17)", (uid, pos, team, proj))
        conn.execute("INSERT INTO adp (season, source, format, player_uid, adp, rank, stdev) "
                     "VALUES (2026, 'ffc', 'half', ?, ?, 1, 3.0)", (uid, adp))
    for team in ("AAA", "BBB"):
        conn.execute("INSERT INTO sos (season, team, position, week, value) VALUES (2026, ?, 'RB', 1, 5)", (team,))
    conn.commit()
    return conn


def test_import_contract_and_validation(tmp_path):
    conn = _mini_db(tmp_path)
    doc = {"season": 2026, "as_of": "2026-08-22", "players": [
        {"name": "Torn Acl", "status": "out_season", "injury": "ACL", "role": "starter", "confidence": 0.95, "source": "beat writer"},
        {"name": "Six Weeks", "status": "injured", "games_out": 6, "injury": "hamstring", "role": "committee"},
        {"name": "Bubble Guy", "status": "cut_risk", "cut_risk": 0.5, "note": "WR6 on the depth chart"},
    ]}
    assert st.import_player_status(conn, doc, 2026) == 3
    cur = st.current_status(conn, 2026)
    assert cur["00-0000002"]["status"] == "out_season" and cur["00-0000002"]["games_out"] == 17
    assert cur["00-0000003"]["games_out"] == 6 and cur["00-0000003"]["source"] == "research 2026-08-22"
    assert "00-0000001" not in cur
    with pytest.raises(ValueError, match="status must be"):
        st.import_player_status(conn, {"players": [{"name": "Healthy Guy", "status": "hurt"}]}, 2026)
    with pytest.raises(ValueError, match="games_out"):
        st.import_player_status(conn, {"players": [{"name": "Healthy Guy", "status": "injured", "games_out": 30}]}, 2026)
    with pytest.raises(ValueError, match="Cannot resolve"):
        st.import_player_status(conn, {"players": [{"name": "Nobody Real", "status": "active"}]}, 2026)
    # graph import accepts facts + players in one document
    from howie3.graph import import_facts
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"season": 2026, "as_of": "2026-08-23", "facts": [
        {"entity": "team:AAA", "kind": "coach_change", "text": "new OC", "confidence": 0.8, "source": "x"}],
        "players": [{"name": "Six Weeks", "status": "active", "role": "starter", "source": "cleared"}]}))
    assert import_facts(conn, f, 2026) == 2
    assert "00-0000003" not in st.current_status(conn, 2026), "a later 'active' row clears the injury"
    conn.close()


def test_roster_feed_mapping_and_precedence(tmp_path):
    conn = _mini_db(tmp_path)
    frame = pd.DataFrame({"gsis_id": ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0009999"],
                          "status": ["ACT", "RES", "E14", "CUT", "RES"], "full_name": list("abcde")})
    n = st.refresh_roster_status(conn, 2026, frame=frame)
    assert n == 4, "only players in the draft pool are recorded"
    cur = st.current_status(conn, 2026)
    # a projected player missing from every roster is a cut risk, not invisible
    frame2 = frame[frame["gsis_id"] != "00-0000004"]
    st.refresh_roster_status(conn, 2026, frame=frame2)
    assert st.current_status(conn, 2026)["00-0000004"]["cut_risk"] == 0.7
    st.refresh_roster_status(conn, 2026, frame=frame)
    assert cur["00-0000002"]["status"] == "injured" and cur["00-0000002"]["games_out"] == 8
    assert cur["00-0000003"]["status"] == "suspended"          # exempt list
    assert cur["00-0000004"]["status"] == "released"
    assert "00-0000001" not in cur
    # same-day research beats the roster feed; an older research row loses to today's feed
    st.import_player_status(conn, {"as_of": st._today(), "players": [
        {"name": "Torn Acl", "status": "injured", "games_out": 3, "injury": "ankle", "confidence": 0.9}]}, 2026)
    cur = st.current_status(conn, 2026)
    assert cur["00-0000002"]["games_out"] == 3 and cur["00-0000002"]["source"].startswith("research")
    old = (date.today() - timedelta(days=10)).isoformat()
    st.import_player_status(conn, {"as_of": old, "players": [{"name": "Six Weeks", "status": "active"}]}, 2026)
    assert st.current_status(conn, 2026)["00-0000003"]["status"] == "suspended"
    conn.close()


def test_engine_applies_status_after_anchor(tmp_path):
    from howie3.value.board import load_pool
    from howie3.value.roster import evaluate_candidates
    from howie3.config import LeagueConfig

    conn = _mini_db(tmp_path)
    base = {p.uid: p.proj for p in load_pool(conn, 2026, "half")}
    st.import_player_status(conn, {"as_of": "2026-08-22", "players": [
        {"name": "Torn Acl", "status": "out_season", "injury": "ACL"},
        {"name": "Six Weeks", "status": "injured", "games_out": 6},
        {"name": "Bubble Guy", "status": "cut_risk", "cut_risk": 0.5}]}, 2026)
    pool = {p.uid: p for p in load_pool(conn, 2026, "half")}
    assert pool["00-0000002"].proj == 0 and not pool["00-0000002"].draftable
    assert pool["00-0000002"].raw and pool["00-0000002"].raw > 0, "display keeps the source projection"
    assert pool["00-0000003"].proj == pytest.approx(base["00-0000003"] * 11 / 17, abs=0.2)
    assert pool["00-0000004"].proj == pytest.approx(base["00-0000004"] * 0.5, abs=0.2)
    assert pool["00-0000001"].proj == base["00-0000001"]
    # the ACL is never a candidate, whatever his ADP says
    league = LeagueConfig(num_teams=2, roster_size=2, rb_slots=1, wr_slots=1, qb_slots=0, te_slots=0,
                          flex_slots=0, k_slots=0, dst_slots=0, bench_slots=0, draft_position=1)
    res = evaluate_candidates(list(pool.values()), [], 1, [4], league, frozenset(), top_n=10)
    assert res and "00-0000002" not in {r.player.uid for r in res}
    assert st.chip(pool["00-0000002"].status) == {"text": "OUT · ACL", "level": "out"}
    assert st.chip(pool["00-0000003"].status)["text"] == "OUT 6"
    assert st.chip(pool["00-0000004"].status)["text"] == "CUT? 50%"
    conn.close()


def test_targets_coverage_and_stale(tmp_path):
    conn = _mini_db(tmp_path)
    from howie3 import graph
    graph.TEAM_NAMES.setdefault("AAA", "Team A"); graph.TEAM_NAMES.setdefault("BBB", "Team B")
    try:
        targets = st.research_targets(conn, 2026, "aaa")
        assert [t["name"] for t in targets] == ["Torn Acl", "Healthy Guy"]   # ADP order
        assert all(t["known_status"] == "none" for t in targets)
        assert {"AAA", "BBB"} <= set(st.stale_teams(conn, 2026))
        st.import_player_status(conn, {"as_of": st._today(), "players": [
            {"name": "Torn Acl", "status": "out_season"}, {"name": "Healthy Guy", "status": "active"}]}, 2026)
        cov = {r["team"]: r for r in st.research_coverage(conn, 2026)}
        assert cov["AAA"]["players_researched"] == 2 and cov["AAA"]["targets"] == 2
        assert "AAA" not in st.stale_teams(conn, 2026) and "BBB" in st.stale_teams(conn, 2026)
        assert st.research_targets(conn, 2026, "AAA")[0]["known_status"].startswith("out_season")
    finally:
        graph.TEAM_NAMES.pop("AAA", None); graph.TEAM_NAMES.pop("BBB", None)
        conn.close()
