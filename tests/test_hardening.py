"""Hardening pass (Aug 2026 review): draft-log validation, traded-player
graph aggregation, recommendation cache keys, model egress redaction, and
server request hardening."""

import json
import sqlite3

import pytest

from howie3.config import LeagueConfig, Settings
from howie3.state import DraftState, DraftStateError, Rule

LEAGUE = LeagueConfig()


@pytest.fixture(scope="module")
def settings():
    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    return s


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda s: tmp_path / "draft.json"))
    return Settings()


# ---------------- draft-state validation ----------------

def test_state_rejects_out_of_order_and_bad_positions(iso):
    st = DraftState(created="x")
    st.add_pick(1, 1, "u1", "A", "RB", "t", league=LEAGUE)
    with pytest.raises(DraftStateError):
        st.add_pick(3, 3, "u2", "B", "WR", "t", league=LEAGUE)   # skips pick 2
    with pytest.raises(DraftStateError):
        st.add_pick(2, 2, "u2", "B", "LB", "t", league=LEAGUE)   # not a fantasy position
    with pytest.raises(DraftStateError):
        st.add_pick(2, 99, "u2", "B", "WR", "t", league=LEAGUE)  # team out of range
    with pytest.raises(DraftStateError):
        st.add_pick(2, 2, "u2", "B", "WR", "t", mine=True, league=LEAGUE)  # mine but not my slot


def test_state_mock_mode_enforces_turn_legality(iso):
    st = DraftState(created="x", mode="mock")
    st.add_pick(1, 1, "u1", "A", "RB", "mock", league=LEAGUE)
    with pytest.raises(DraftStateError):
        st.add_pick(2, 5, "u2", "B", "WR", "mock", league=LEAGUE)  # pick 2 belongs to slot 2


def test_state_roster_limit_and_completion(iso):
    small = LeagueConfig(num_teams=2, draft_position=1, roster_size=2,
                         rb_slots=1, wr_slots=1, qb_slots=0, te_slots=0,
                         flex_slots=0, k_slots=0, dst_slots=0, bench_slots=0)
    st = DraftState(created="x")
    st.add_pick(1, 1, "u1", "A", "RB", "t", mine=True, league=small)
    st.add_pick(2, 2, "u2", "B", "RB", "t", league=small)
    st.add_pick(3, 2, "u3", "C", "WR", "t", league=small)
    with pytest.raises(DraftStateError):
        st.add_pick(4, 2, "u4", "D", "WR", "t", league=small)     # team 2 is full
    st.add_pick(4, 1, "u4", "D", "WR", "t", mine=True, league=small)
    with pytest.raises(DraftStateError):
        st.add_pick(5, 1, "u5", "E", "WR", "t", league=small)     # draft complete


def test_state_load_fails_loudly_on_malformed_file(iso, tmp_path):
    p = tmp_path / "draft.json"
    p.write_text("{not json")
    with pytest.raises(DraftStateError):
        DraftState.load(iso)
    p.write_text(json.dumps({"version": 99, "events": []}))
    with pytest.raises(DraftStateError):
        DraftState.load(iso)
    p.write_text(json.dumps({"version": 1, "events": [{"seq": 1, "pick_no": 2, "team": 1,
                                                        "player_uid": "u", "player_name": "n"}]}))
    with pytest.raises(DraftStateError):
        DraftState.load(iso)  # seq/pick_no not contiguous
    st = DraftState(created="x"); st.reset("mock"); st.save(iso)
    assert DraftState.load(iso).mode == "mock"
    assert json.loads(p.read_text())["version"] == 1


# ---------------- traded-player graph aggregation ----------------

def test_graph_attributes_traded_player_to_highest_volume_team(tmp_path):
    from howie3.db import connect
    from howie3 import graph

    conn = connect(tmp_path / "t.db")  # real schema
    conn.execute("INSERT INTO players (player_uid, name, name_key, position, team) VALUES ('w1','Traded Guy','traded guy','WR','ZZZ')")
    conn.execute("INSERT INTO players (player_uid, name, name_key, position, team) VALUES ('w2','Stayed Guy','stayed guy','WR','AAA')")
    for uid, team in (("w1", "ZZZ"), ("w2", "AAA")):
        conn.execute("INSERT INTO projections (season, source, player_uid, position, team, pts_half) "
                     "VALUES (2026,'pff',?, 'WR', ?, 170)", (uid, team))
    for team in ("AAA", "ZZZ"):
        conn.execute("INSERT INTO sos (season, team, position, week, value) VALUES (2026, ?, 'WR', 1, 5)", (team,))
    # 2025: 40 targets with AAA, then 10 with ZZZ after a deadline trade
    ins = ("INSERT INTO weekly_stats (season, week, player_uid, team, position, targets, rush_attempts, pts_half) "
           "VALUES (2025, ?, ?, ?, 'WR', ?, 0, 5)")
    for wk in range(1, 9):
        conn.execute(ins, (wk, "w1", "AAA", 5))
        conn.execute(ins, (wk, "w2", "AAA", 5))
    for wk in range(9, 14):
        conn.execute(ins, (wk, "w1", "ZZZ", 2))
    conn.commit()
    graph.rebuild_derived(conn, season=2026)
    row = conn.execute("SELECT attrs, value FROM edges WHERE src='player:w1' AND kind='in_room'").fetchone()
    attrs = json.loads(row["attrs"])
    assert attrs["last_team"] == "AAA", "attribute to the team with the most volume, not the last row"
    assert attrs["targets_last"] == 40
    assert abs(row["value"] - 0.5) < 1e-6          # 40 of AAA's 80 targets
    vac = conn.execute("SELECT value FROM facts WHERE entity_id='unit:AAA-WR' AND kind='vacated_share'").fetchone()
    assert vac is not None and abs(vac["value"] - 0.5) < 1e-6
    conn.close()


# ---------------- model egress ----------------

def test_redact_strips_stat_lines_and_raw_keys():
    from howie3 import egress

    card = {
        "name": "X", "proj": 200, "milestones": {"player": {"100+ rush yds": 0.4}},
        "games": [{"season": 2025, "week": 1, "rush_yds": 120, "pts": 21.0}],
        "room": {"members": [{"name": "Y", "share": 0.3, "targets_last": 80}]},
        "seasons": [{"season": 2025, "g": 17, "pts": 300}],
        "nested": {"team_intel": "scouting text", "fine": 1},
    }
    out = egress.redact(card)
    assert "games" not in out and out["_redacted"] == ["games"]
    assert out["milestones"]["player"]["100+ rush yds"] == 0.4
    assert out["room"]["members"][0]["targets_last"] == 80      # derived aggregate survives
    assert out["seasons"] == [{"season": 2025, "g": 17, "pts": 300}]
    assert "team_intel" not in out["nested"] and out["nested"]["fine"] == 1
    assert not egress.contains_raw(out)
    assert egress.contains_raw(card)
    # a bare list of stat lines is emptied
    assert egress.redact([{"rush_yards": 5, "week": 1}, {"pts": 3}]) == [{"pts": 3}]
    # text passthrough vs JSON redaction
    assert egress.for_model("plain table text") == "plain table text"
    assert "rush_yards" not in egress.for_model(json.dumps([{"rush_yards": 9}]))


def test_agent_sql_tool_is_opt_in(monkeypatch):
    from howie3 import agent, egress

    monkeypatch.delenv("HOWIE_AGENT_SQL", raising=False)
    assert not egress.sql_tool_enabled()
    assert all(t["name"] != "query_database" for t in agent.active_tool_schemas())
    assert "disabled" in agent._query_tool({"sql": "SELECT 1"}, Settings())
    monkeypatch.setenv("HOWIE_AGENT_SQL", "1")
    assert any(t["name"] == "query_database" for t in agent.active_tool_schemas())


def test_every_model_facing_tool_is_free_of_raw_records(monkeypatch):
    """Invoke every agent and MCP tool against the real data plane and
    assert nothing stat-line-shaped crosses the boundary."""
    from howie3 import agent, egress, mcp_server

    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: s.data_dir / "_egress_test.json"))
    monkeypatch.delenv("HOWIE_AGENT_SQL", raising=False)
    try:
        probe = {
            "draft_board": {"round": 1}, "draft_pick": {"round": 2},
            "player_info": {"name": "Bijan Robinson"}, "entity_context": {"query": "Bijan Robinson"},
            "query_database": {"sql": "SELECT * FROM weekly_stats LIMIT 3"},
        }
        for name, args in probe.items():
            out = egress.for_model(agent._run_tool(name, args, s))
            assert "rush_yards" not in out and '"games"' not in out, name
            if out.lstrip().startswith(("{", "[")):
                assert not egress.contains_raw(json.loads(out)), name
        mcp_probe = {
            "get_draft_state": {}, "search": {"query": "Bijan"}, "best_picks": {"sims": 0},
            "positional_impact": {}, "player_card": {"name": "Bijan Robinson"},
            "entity_context": {"query": "DAL WR room"},
        }
        for name, args in mcp_probe.items():
            resp = mcp_server.handle(s, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": name, "arguments": args}})
            text = resp["result"]["content"][0]["text"]
            assert not resp["result"].get("isError"), (name, text)
            assert not egress.contains_raw(json.loads(text)), name
        # the card still carries derived context the model needs
        card = json.loads(mcp_server.handle(s, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                               "params": {"name": "player_card", "arguments": {"name": "Bijan Robinson"}}})["result"]["content"][0]["text"])
        assert card["milestones"]["labels"] and "games" not in card
    finally:
        (s.data_dir / "_egress_test.json").unlink(missing_ok=True)
        (s.data_dir / "_egress_test.json.tmp").unlink(missing_ok=True)


# ---------------- server hardening ----------------

def _serve(settings, port):
    import threading
    from howie3 import server as srv

    httpd = srv.serve(settings, port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, srv.Handler.token


def _req(port, path, body=None, token=None, method=None):
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Howie-Token"] = token
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers, method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_server_token_size_limit_csp_and_rule_cache(tmp_path, monkeypatch):
    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    httpd, token = _serve(s, 8893)
    try:
        # page carries the token and a CSP; static assets are allowlisted
        code, headers, body = _req(8893, "/")
        assert code == 200 and token in body.decode() and "__HOWIE_TOKEN__" not in body.decode()
        assert "Content-Security-Policy" in headers
        assert _req(8893, "/ui/nope.js")[0] == 404
        # mutations need the token
        assert _req(8893, "/api/reset", {"mode": "live"})[0] == 403
        assert _req(8893, "/api/reset", {"mode": "live"}, token="wrong")[0] == 403
        assert _req(8893, "/api/reset", {"mode": "live"}, token=token)[0] == 200
        # oversized bodies are refused before parsing
        big = {"text": "x" * 1_100_000}
        assert _req(8893, "/api/sim/mock/import", big, token=token)[0] == 413
        # a strategy rule re-ranks the board immediately (cache key includes rules)
        code, _, raw = _req(8893, "/api/pick?top=5")
        before = json.loads(raw)["rows"]
        target = before[2]["name"]
        _req(8893, "/api/strategy", {"rules": [{"text": f"TARGET {target}", "on": True}], "notes": ""}, token=token)
        after = json.loads(_req(8893, "/api/pick?top=5")[2])["rows"]
        tagged = [r for r in after if r["name"] == target][0]
        assert any(t["type"] == "target" for t in tagged["rules"]), "rule must show up without a draft event"
        # malformed draft log -> 409 with a clear message, not a silent fresh draft
        (tmp_path / "draft.json").write_text("{broken")
        code, _, raw = _req(8893, "/api/state")
        assert code == 409 and "draft log" in json.loads(raw)["error"]
    finally:
        httpd.shutdown()


# ---------------- engine: availability objective, lab blend, bots, rules ----------------

def test_expected_lineup_values_insurance():
    from howie3.value.lineup import expected_lineup_points as E, lineup_points

    L = LeagueConfig()
    assert E({"QB": [300]}, L) == pytest.approx(300)
    assert E({"QB": [300, 250]}, L) > 300                       # QB2 has insurance value
    assert E({"QB": [300, 250]}, L) < 300 + 250 * 0.2           # but nowhere near a starter
    # starters alone reproduce the deterministic objective exactly
    full = {"QB": [300], "RB": [250, 200], "WR": [230, 210, 190], "TE": [150]}
    assert E(full, L) == pytest.approx(lineup_points(full, L))
    # a deeper bench is worth more, and a better backup is worth more
    assert E({"RB": [250, 200, 150, 120]}, L) > E({"RB": [250, 200, 150]}, L)
    assert E({"WR": [230, 210, 190, 170]}, L) > E({"WR": [230, 210, 190, 100]}, L)


def test_empirical_availability_blends_toward_lab_rate():
    from howie3.value.board import PoolPlayer, EMPIRICAL_PRIOR_N

    p = PoolPlayer("u", "X", "RB", "DAL", 200, adp=30.0, stdev=5.0, bye=7)
    model = p.p_available(32)
    p.emp_avail = {32: (0.95, EMPIRICAL_PRIOR_N)}       # lab says he's usually there
    assert p.p_available(32) == pytest.approx(0.5 * model + 0.5 * 0.95)
    assert p.availability_source(32).startswith("blend")
    assert p.availability_source(33) == "model"
    p.emp_avail = {32: (0.95, 10 * EMPIRICAL_PRIOR_N)}  # big sample dominates
    assert abs(p.p_available(32) - 0.95) < abs(model - 0.95)


def test_lab_availability_table_feeds_pool(settings, tmp_path, monkeypatch):
    from howie3 import mocksim, service

    monkeypatch.setattr(mocksim, "store_path", lambda st: tmp_path / "mock_sims.json")
    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    s2 = settings
    # too few drafts -> no table, pool stays on the model
    mocksim.run_mock_drafts(s2, n=3, policy="adp", seed=5)
    assert mocksim.availability_table(s2) == {}
    mocksim.run_mock_drafts(s2, n=mocksim.MIN_DRAFTS, policy="adp", seed=9)
    table = mocksim.availability_table(s2)
    assert table, "enough drafts should produce an availability table"
    conn = service._conn(s2)
    pool = service._pool(s2, conn)
    conn.close()
    with_emp = [p for p in pool if p.emp_avail]
    assert with_emp
    st = DraftState.load(s2)
    rows = service.pick_payload(s2, st, top_n=5)["rows"]
    assert all("avail_src" in r for r in rows)


def test_bots_react_to_runs_and_needs():
    import numpy as np
    from howie3.mock import bot_pick
    from howie3.value.board import PoolPlayer

    L = LeagueConfig()
    pool = [PoolPlayer(f"rb{i}", f"RB {i}", "RB", None, 200 - i, adp=10.0 + i, stdev=1.0, bye=None) for i in range(6)] + \
           [PoolPlayer(f"wr{i}", f"WR {i}", "WR", None, 200 - i, adp=10.5 + i, stdev=1.0, bye=None) for i in range(6)]
    def share(pos, extra=(), tp=None, rnd=3, **kw):
        picks = [bot_pick(pool + list(extra), frozenset(), dict(tp or {}), rnd, L,
                          np.random.default_rng(seed), **kw) for seed in range(60)]
        return sum(1 for p in picks if p.position == pos) / len(picks)

    assert share("RB") > 0.5, "market order alone: RBs go first"
    assert share("WR", recent_positions=["WR", "WR", "RB", "WR"]) > share("WR") + 0.25, "a WR run makes WR feel urgent"
    # round 5+: a team with no TE yet reaches for one
    te = [PoolPlayer("te0", "TE 0", "TE", None, 150, adp=14.0, stdev=1.0, bye=None)]
    assert share("TE", extra=te, tp={"RB": 2, "WR": 3}, rnd=6) > share("TE", extra=te, tp={"RB": 2, "WR": 3, "TE": 1}, rnd=6) + 0.12


def test_rule_reconciliation_keeps_latest_per_constraint():
    from howie3.state import reconcile_rules, rule_key

    assert rule_key("WAIT QB UNTIL R7") == ("wait", "QB") == rule_key("NO QB BEFORE R3")
    assert rule_key("2 RBs BY R4") == ("need", "RB")
    assert rule_key("TARGET Trey McBride @41") == ("target", "trey mcbride")
    assert rule_key("some free text") is None
    rules, notes = reconcile_rules([Rule("WAIT QB UNTIL R7"), Rule("free thought"),
                                    Rule("NO QB BEFORE R3"), Rule("WAIT RB UNTIL R9", on=False)])
    assert [r.on for r in rules] == [False, True, True, False]
    assert len(notes) == 1 and "superseded" in notes[0]


def test_pick_context_follows_the_draft_not_the_roster(settings, tmp_path, monkeypatch):
    """Marking your own slot as taken (live mode) must not rewind the engine
    to an earlier round; and a mock refuses 'taken' while you're on the clock."""
    from howie3 import service

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="x", mode="live"); st.save(settings)
    conn = service._conn(settings); pool = service._pool(settings, conn); conn.close()
    # slot 8: picks 8, 17, 32, 41 ... fill picks 1-7 as other teams, then skip pick 8 via 'taken'
    for p in pool[:7]:
        service.mark_pick(settings, p.uid, mine=False)
    service.mark_pick(settings, pool[7].uid, mine=False)       # own slot, live: allowed, team 0
    for p in pool[8:16]:
        service.mark_pick(settings, p.uid, mine=False)
    state = DraftState.load(settings)
    assert state.next_pick_no() == 17 and not state.my_uids(settings.league)
    rnd, cur, nxt, future = service._pick_context(settings, state)
    assert (rnd, cur, nxt) == (2, 17, 32), "round 2 at pick 17 even with an empty roster"
    payload = service.pick_payload(settings, state, top_n=3)
    assert payload["current_pick"] == 17 and payload["next_pick"] == 32
    # mock mode: 'taken' on your own turn is refused instead of corrupting the log
    service.start_mock(settings)
    state = DraftState.load(settings)
    assert state.next_pick_no() == 8
    avail = next(p for p in pool if p.uid not in state.taken_uids())
    with pytest.raises(ValueError, match="on the clock"):
        service.mark_pick(settings, avail.uid, mine=False)


def test_card_reports_taken_players(settings, tmp_path, monkeypatch):
    from howie3 import service

    monkeypatch.setattr(DraftState, "path", staticmethod(lambda st: tmp_path / "draft.json"))
    st = DraftState(created="x", mode="live"); st.save(settings)
    conn = service._conn(settings); pool = service._pool(settings, conn); conn.close()
    assert service.card_payload(settings, pool[0].uid)["taken"] is False
    service.mark_pick(settings, pool[0].uid, mine=False)
    c = service.card_payload(settings, pool[0].uid)
    assert c["taken"] is True and c["taken_pick"] == 1 and c["taken_by"] == "team 1"
    hit = [x for x in service.search_payload(settings, pool[0].name) if x.get("uid") == pool[0].uid][0]
    assert hit["taken"] is True


def test_kicker_and_dst_are_closing_round_candidates_only(settings):
    from howie3 import service
    from howie3.value.board import snake_picks
    from howie3.value.roster import evaluate_candidates

    conn = service._conn(settings); pool = service._pool(settings, conn); conn.close()
    picks = snake_picks(settings.league)
    # round 9 of 16 with an empty roster: no K/DST among the candidates
    res = evaluate_candidates(pool, [], picks[8], picks[9:], settings.league, frozenset(), top_n=40)
    assert res and not any(r.player.position in ("K", "DST") for r in res)
    # last 4 picks with K and DST still open: both positions are on the table
    res = evaluate_candidates(pool, [], picks[12], picks[13:], settings.league, frozenset(), top_n=60)
    assert {r.player.position for r in res} >= {"K", "DST"}
