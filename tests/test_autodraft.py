"""Autodraft bridge: room parsing and event log (no browser needed)."""

from howie3 import autodraft


ESPN_PANEL = """Picks
Jahmyr Gibbs / DET RB
R1, P1 - Ryan's Rowdy Team
Bijan Robinson / ATL RB
R1, P2 - Douglas's Daring Team
Jaxon Smith-Njigba / SEA WR
R1, P3 - Tony's Talented Team
LA D/ST / LAR D/ST
R2, P4 - Jose's Supreme Team
Jeremiyah Love / ARI RB
R2, P1 - Trevor's Top Team
"""


def test_parse_espn_pick_panel():
    picks = autodraft.parse_picks(ESPN_PANEL)
    # ESPN numbers picks within the round: R2 P1 comes before R2 P4, after all of R1
    assert [p["name"] for p in picks] == ["Jahmyr Gibbs", "Bijan Robinson", "Jaxon Smith-Njigba", "Jeremiyah Love", "LA D/ST"]
    assert picks[0] == {"name": "Jahmyr Gibbs", "team": "DET", "pos": "RB", "round": 1, "pick": 1, "owner": "Ryan's Rowdy Team"}
    assert picks[-1]["pos"] == "D/ST" and picks[-1]["round"] == 2
    assert autodraft.parse_picks("nothing here\nR1, P1 - x") == []


def test_on_clock_detection():
    assert autodraft.on_clock("... You are on the clock! Your autopick would be ...")
    assert autodraft.on_clock("You're on the clock!")
    assert not autodraft.on_clock("You're on the clock in: 6 Picks")


def test_event_log_roundtrip(tmp_path, monkeypatch):
    from howie3.config import Settings

    monkeypatch.setenv("HOWIE_DATA_DIR", str(tmp_path))
    s = Settings()
    autodraft.log_event(s, "start", title="Mock", autopilot=True)
    autodraft.log_event(s, "sync", picks=["1.1 Gibbs"], unresolved=[], next_pick=2)
    ev = autodraft.recent_events(s)
    assert [e["kind"] for e in ev] == ["start", "sync"] and ev[1]["next_pick"] == 2
    assert autodraft.recent_events(s, 1)[0]["kind"] == "sync"


def test_room_config_from_page_text():
    text = "Roster Limits\n0/16 Players\n...\nYour draft is about to start\n\nYour first pick: Round 1, Pick 10\n"
    cfg = autodraft.room_config(text, "ESPN Fantasy Football Draft - Beginner 10-Team H2H Points PPR Mock")
    assert cfg == {"num_teams": 10, "draft_position": 10, "roster_size": 16, "scoring_type": "ppr"}
    assert autodraft.room_config("nothing", "Half PPR 12-Team") == {"num_teams": 12, "scoring_type": "half_ppr"}
