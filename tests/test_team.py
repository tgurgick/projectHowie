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
