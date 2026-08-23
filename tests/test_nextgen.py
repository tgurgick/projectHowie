"""Tests for the next-gen layer: draft state, graph, service, mock bots,
market anchor, MCP, and an end-to-end mock draft against the live server."""

import json
import threading
import urllib.request

import numpy as np
import pytest

from howie3.config import LeagueConfig, Settings
from howie3.state import DraftState, Rule, snake_team_for_pick
from howie3.value.board import PoolPlayer, apply_market_anchor

LEAGUE = LeagueConfig()


@pytest.fixture(scope="module")
def settings():
    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    return s


# ---------------- draft state ----------------

def test_snake_team_mapping():
    assert snake_team_for_pick(LEAGUE, 1) == 1
    assert snake_team_for_pick(LEAGUE, 8) == 8
    assert snake_team_for_pick(LEAGUE, 12) == 12
    assert snake_team_for_pick(LEAGUE, 13) == 12
    assert snake_team_for_pick(LEAGUE, 17) == 8
    assert snake_team_for_pick(LEAGUE, 25) == 1


def test_state_event_log_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HOWIE_DATA_DIR", str(tmp_path))
    s = Settings()
    state = DraftState.load(s)
    state.reset("mock")
    state.add_pick(1, 1, "u1", "Player One", "RB", "test")
    state.add_pick(2, 2, "u2", "Player Two", "WR", "test")
    with pytest.raises(ValueError):
        state.add_pick(3, 3, "u1", "Player One", "RB", "test")  # double-draft
    state.save(s)
    loaded = DraftState.load(s)
    assert [e.player_uid for e in loaded.events] == ["u1", "u2"]
    assert loaded.undo().player_uid == "u2"
    assert loaded.taken_uids() == frozenset({"u1"})


def test_rule_parsing():
    state = DraftState(rules=[
        Rule("WAIT QB UNTIL R6"), Rule("TARGET McBride @41"),
        Rule("NO K/DST BEFORE R14"), Rule("free-text thought", on=True),
        Rule("WAIT RB UNTIL R9", on=False),
    ])
    fx = state.active_rule_effects()
    assert ("QB", 6) in fx["wait"]
    assert ("RB", 9) not in fx["wait"]  # toggled off
    assert "McBride" in fx["targets"]
    assert ("K", 14) in fx["ban"] and ("DST", 14) in fx["ban"]


# ---------------- market anchor ----------------

def _pp(uid, pos, proj, adp):
    return PoolPlayer(uid, uid, pos, None, proj, adp, 2.0, None)


def test_market_anchor_blends_toward_adp_order():
    # projection says B > A, market says A > B; anchor=1 must follow market
    pool = [_pp("A", "RB", 200, 5.0), _pp("B", "RB", 260, 30.0)]
    pure = apply_market_anchor(pool, 0.0)
    assert max(pure, key=lambda p: p.proj).uid == "B"
    market = apply_market_anchor(pool, 1.0)
    assert max(market, key=lambda p: p.proj).uid == "A"
    half = {p.uid: p.proj for p in apply_market_anchor(pool, 0.5)}
    assert half["A"] == pytest.approx((200 + 260) / 2)


def test_market_anchor_preserves_players_without_adp():
    pool = [_pp("A", "WR", 180, 20.0), PoolPlayer("C", "C", "WR", None, 170, None, None, None)]
    out = {p.uid: p.proj for p in apply_market_anchor(pool, 0.8)}
    assert out["C"] == 170


# ---------------- engine regression: no late-round hoarding ----------------

def test_no_qb_hoarding_in_full_rollout(settings):
    from howie3.db import connect
    from howie3.value.board import load_pool, snake_picks
    from howie3.value.roster import evaluate_candidates

    conn = connect(settings.db_path)
    pool = load_pool(conn, settings.current_season, "half")
    conn.close()
    picks = snake_picks(LEAGUE)
    # simulate a full solo draft always taking the top recommendation
    roster, taken = [], set()
    for rnd in range(1, 15):
        res = evaluate_candidates(pool, roster, picks[rnd - 1], picks[rnd:],
                                  LEAGUE, frozenset(taken), top_n=1)
        assert res, f"no candidates at round {rnd}"
        roster.append(res[0].player)
        taken.add(res[0].player.uid)
    positions = [p.position for p in roster]
    assert positions.count("QB") <= 2
    assert positions.count("K") <= 1 and positions.count("DST") <= 1


# ---------------- mock bots ----------------

def _isolate_state(tmp_path, monkeypatch):
    """Redirect the draft event log to tmp without touching the real db."""
    monkeypatch.setattr(DraftState, "path",
                        staticmethod(lambda s: tmp_path / "draft.json"))


def test_bots_complete_draft_sanely(settings, tmp_path, monkeypatch, league12):
    from howie3.db import connect
    from howie3.mock import advance_bots
    from howie3.value.board import load_pool

    _isolate_state(tmp_path, monkeypatch)
    conn = connect(settings.db_path)
    pool = load_pool(conn, settings.current_season, "half")
    conn.close()

    state = DraftState.load(settings)
    state.reset("mock")
    made = advance_bots(settings, state, pool)
    assert len(made) == 7  # picks 1-7 before slot 8
    # replays identically for the same seed
    state2 = DraftState.load(settings)
    state2.reset("mock")
    made2 = advance_bots(settings, state2, pool)
    assert [m["name"] for m in made] == [m["name"] for m in made2]
    # no early K/DST from bots
    assert all(m["position"] not in ("K", "DST") for m in made)


# ---------------- graph ----------------

def test_graph_search_and_context(settings):
    from howie3.db import connect
    from howie3.graph import entity_context, search

    conn = connect(settings.db_path)
    hits = search(conn, "mcbride")
    assert hits and hits[0]["name"] == "Trey McBride"
    ctx = entity_context(conn, hits[0]["id"])
    assert ctx["room"] and ctx["room"]["unit"] == "unit:ARI-TE"
    shares = [m["share"] for m in ctx["room"]["members"] if m["share"]]
    assert shares and max(shares) > 0.2  # McBride's real 2025 target share
    unit_hits = search(conn, "ari te room")
    assert any(h["kind"] == "unit" for h in unit_hits)
    conn.close()


def test_graph_import_rejects_bad_contract(settings, tmp_path):
    from howie3.db import connect
    from howie3.graph import import_facts

    conn = connect(settings.db_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"facts": [{"entity": "player:Nobody Realname XYZ",
                                          "kind": "x", "text": "t",
                                          "confidence": 1, "source": "s"}]}))
    with pytest.raises(ValueError, match="Cannot resolve"):
        import_facts(conn, bad, 2026)
    bad.write_text(json.dumps({"facts": [{"entity": "team:ARI", "kind": "x"}]}))
    with pytest.raises(ValueError, match="missing fields"):
        import_facts(conn, bad, 2026)
    conn.close()


# ---------------- end-to-end: full mock draft over HTTP ----------------

def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return json.loads(r.read())


def _post(port, path, body, token=None):
    from howie3 import server as srv

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "X-Howie-Token": srv.Handler.token if token is None else token}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_full_mock_draft_over_http(settings, tmp_path, monkeypatch, league12):
    from howie3 import server as srv

    _isolate_state(tmp_path, monkeypatch)
    port = 8891
    httpd = srv.serve(settings, port=port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        start = _post(port, "/api/mock/start", {})
        assert start["started"] and len(start["bots"]) == 7

        rounds = 0
        while True:
            st = _get(port, "/api/state")
            if st["complete"]:
                break
            assert st["you_are_on_clock"]
            pick = _get(port, "/api/pick?top=3")
            assert pick["rows"], "no candidates offered"
            _post(port, "/api/mark", {"uid": pick["rows"][0]["uid"], "mine": True})
            rounds += 1
            assert rounds <= 16, "draft did not converge"

        # mid-draft undo semantics: undo rolls back bots AND the user's last
        # pick, leaving the user on the clock at the same pick number
        st_before = _get(port, "/api/state")
        _post(port, "/api/undo", {})
        st_undone = _get(port, "/api/state")
        assert st_undone["you_are_on_clock"] or st_undone["complete"] is False
        my_before = len([s for s in st_before["roster"] if s["name"]])
        my_after = len([s for s in st_undone["roster"] if s["name"]])
        assert my_after == my_before - 1
        # redo: take best again to finish the draft
        pick = _get(port, "/api/pick?top=1")
        _post(port, "/api/mark", {"uid": pick["rows"][0]["uid"], "mine": True})

        st = _get(port, "/api/state")
        mine = [s for s in st["roster"] if s["name"]]
        assert len(mine) == 16
        starters = {s["slot"] for s in st["roster"]}
        assert {"QB", "RB", "WR", "TE", "K", "DST"} <= starters
        # undo works after completion
        _post(port, "/api/undo", {})
        assert not _get(port, "/api/state")["complete"]
        # strategy round-trip
        _post(port, "/api/strategy", {"rules": [{"text": "WAIT QB UNTIL R6", "on": True}],
                                      "notes": "test note"})
        strat = _get(port, "/api/strategy")
        assert strat["notes"] == "test note" and strat["rules"][0]["on"]
        # search + card
        res = _get(port, "/api/search?q=nacua")
        assert res and res[0]["name"] == "Puka Nacua"
        card = _get(port, "/api/card?uid=" + res[0]["uid"])
        assert card["band"]["p90"] > card["band"]["p10"]
    finally:
        httpd.shutdown()


# ---------------- MCP ----------------

def test_mcp_handshake_and_tools(settings):
    from howie3.mcp_server import TOOLS, handle

    init = handle(settings, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["name"] == "howie"
    tools = handle(settings, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in tools["result"]["tools"]]
    assert "best_picks" in names and "mark_pick" in names
    assert names == [t["name"] for t in TOOLS]
    call = handle(settings, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": "search", "arguments": {"query": "gibbs"}}})
    payload = json.loads(call["result"]["content"][0]["text"])
    assert any("Gibbs" in p["name"] for p in payload)
    unknown = handle(settings, {"jsonrpc": "2.0", "id": 4, "method": "nope"})
    assert unknown["error"]["code"] == -32601


# ---------------- playoff weighting knob ----------------

def test_playoff_weight_scales_objective():
    from howie3.value.distributions import SimPlayer
    from howie3.value.simulate import simulate_roster

    roster = [SimPlayer(name="QB", position="QB", proj=300, weekly_mu=300 / 17,
                        cv=0.01, p_play=1.0, bye_week=None, sos_mult=np.ones(18))]
    neutral = simulate_roster(roster, LEAGUE, n_sims=50, seed=1, playoff_weight=1.0)
    heavy = simulate_roster(roster, LEAGUE, n_sims=50, seed=1, playoff_weight=2.0)
    # 3 playoff weeks of ~17.6 pts each counted twice → +~53 over a 300-pt season
    assert heavy.mean - neutral.mean == pytest.approx(3 * 300 / 17, rel=0.05)
    with pytest.raises(ValueError, match="playoff_weight"):
        LeagueConfig(playoff_weight=0.5).validate()


# ---------------- milestone anchors ----------------

def test_milestones_card_and_anchors(settings, tmp_path, monkeypatch):
    from howie3 import service
    from howie3.db import connect
    from howie3.value.milestones import MILESTONES, league_trend, player_games, player_rates

    _isolate_state(tmp_path, monkeypatch)
    uid = service.search_payload(settings, "gibbs")[0]["uid"]
    conn = connect(settings.db_path)
    games = player_games(conn, uid, "half", (2024, 2025))
    assert len(games) >= 25 and all("flags" in g for g in games)
    rates = player_rates(games, "RB")
    assert set(rates) == {label for label, _ in MILESTONES["RB"]}
    assert all(0.0 <= v <= 1.0 for v in rates.values())
    trend = league_trend(conn, "half", (2018, 2025))
    # the 300-yard QB game has roughly halved since 2018 — a real league drift
    assert trend["QB"]["300+ pass yds"][2018] > trend["QB"]["300+ pass yds"][2025] + 0.1
    conn.close()

    card = service.card_payload(settings, uid)
    assert card["games"] and card["milestones"]["player"]["100+ scrimmage"] > 0.3
    # roster anchors: empty roster -> zero booms; with Gibbs -> his boom rate
    state = DraftState.load(settings); state.reset("live"); state.save(settings)
    assert service.anchors_payload(settings, state)["roster"]["expected_booms_per_week"] == 0
    service.mark_pick(settings, uid, mine=True, source="test")
    a = service.anchors_payload(settings, DraftState.load(settings))
    assert a["roster"]["starters"][0]["name"] == "Jahmyr Gibbs"
    assert a["roster"]["p_any_boom"] == pytest.approx(a["roster"]["starters"][0]["boom_rate"])


# ---------------- data tab ----------------

def test_data_tab_payloads(settings, tmp_path, monkeypatch):
    from howie3 import service

    _isolate_state(tmp_path, monkeypatch)
    d = service.games_distribution(settings, "RB", "rush_yds", "starter")
    assert len(d["seasons"]) == 3 and len(d["rows"]) > 1000
    assert d["columns"][6] == "value" and all(len(r) == 8 for r in d["rows"][:50])
    with pytest.raises(ValueError):
        service.games_distribution(settings, "RB", "nope")
    uid = service.search_payload(settings, "gibbs")[0]["uid"]
    sim = service.sim_payload(settings, uid, n_sims=100)
    assert len(sim["samples"]) == 100 and sim["p10"] < sim["p50"] < sim["p90"]
    assert sim["actual"] and sim["actual"][0]["season"] == 2025
    q = service.query_payload(settings, "sql: SELECT COUNT(*) n FROM players")
    assert q["mode"] == "sql" and q["rows"][0]["n"] > 1000
    q = service.query_payload(settings, "sql: DELETE FROM players")
    assert "error" in q
    q = service.query_payload(settings, "puka nacua")
    assert q["entity"]["kind"] == "player" and q["detail"]["seasons"]
    assert service.query_payload(settings, "")["presets"]


# ---------------- query builder + mock draft lab ----------------

def test_query_builder_whitelists(settings):
    from howie3 import service
    r = service.build_query(settings, pos="RB", season="2025", measure="games_over",
                            stat="rush_yds", thr=100, min_games=8, limit=5)
    assert r["rows"] and r["rows"][0]["value"] >= r["rows"][-1]["value"]
    assert "rush_yards >= 100.0" in r["sql"]
    t = service.build_query(settings, entity="team", season="2025", measure="total", stat="pass_yds", limit=3)
    assert t["rows"] and "team" in t["columns"]
    for bad in [dict(stat="name; DROP"), dict(measure="x"), dict(pos="RB'--"), dict(order="desc; --")]:
        with pytest.raises(ValueError):
            service.build_query(settings, **bad)


def test_mock_draft_lab(settings, tmp_path, monkeypatch, league12):
    from howie3 import mocksim
    monkeypatch.setattr(mocksim, "store_path", lambda s: tmp_path / "mock_sims.json")
    agg = mocksim.run_mock_drafts(settings, 3, policy="adp", seed=5)
    assert agg["drafts"] == 3 and agg["local"] == 3
    first_pick = str(agg["my_picks"][0])
    rows = agg["per_pick"][first_pick]["rows"]
    assert rows and all(0 <= r["avail_sim"] <= 1 and 0 <= r["avail_model"] <= 1 for r in rows)
    # every stored draft is a full, duplicate-free pick order
    store = mocksim.load_store(settings)
    for d in store["drafts"]:
        assert len(d["picks"]) == LEAGUE.num_teams * LEAGUE.roster_size
        assert len(set(d["picks"])) == len(d["picks"])
    # external import: numbered lines, trailing pos/team, resolves through the crosswalk
    text = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(
        ["Jahmyr Gibbs RB DET", "Bijan Robinson", "Ja'Marr Chase WR", "Puka Nacua", "Christian McCaffrey",
         "Jonathan Taylor", "Justin Jefferson", "James Cook", "Josh Allen QB BUF", "Malik Nabers",
         "Brock Bowers TE", "Trey McBride", "Derrick Henry", "Drake London"]))
    res = mocksim.import_external(settings, text, "espn")
    assert res["stored"] == 14 and not res["unresolved"]
    assert mocksim.aggregates(settings)["external"] == 1
    with pytest.raises(ValueError, match="at least 12"):
        mocksim.import_external(settings, "1. Nobody Real\n2. Also Fake", "x")


# ---------------- insights + research plumbing (no LLM calls) ----------------

def test_insights_graceful_without_key(settings, monkeypatch):
    from howie3 import insights
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = insights.generate_insights(settings, "mock", {"data": {}})
    assert r["available"] is False and "ANTHROPIC_API_KEY" in r["reason"]
    r = insights.research_team(settings, "PHI")
    assert r["available"] is False


def test_research_status_and_facts(settings):
    from howie3 import insights
    st = insights.research_status(settings)
    assert len(st["teams"]) == 32 and st["teams"][0]["team"] == "ARI"
    f = insights.facts_for(settings, "eagles")
    assert f["entity"]["name"] == "Philadelphia Eagles"
    assert all(x["source"] != "derived" for x in f["facts"])


def test_query_detail_includes_projections(settings):
    from howie3 import service
    q = service.query_payload(settings, "puka nacua")
    assert q["detail"]["projection"]["proj"] > 200
    t = service.query_payload(settings, "eagles")
    rbs = [m for m in t["detail"]["rooms"] if m["position"] == "RB" and m["proj"]]
    assert rbs and rbs[0]["proj"] >= rbs[-1]["proj"]  # sorted by projection within the room


# ---------------- roster risk ----------------

def test_need_rule_and_roster_risk(settings, tmp_path, monkeypatch, league12):
    from howie3 import service
    _isolate_state(tmp_path, monkeypatch)
    st = DraftState(rules=[Rule("2 RB BY ROUND 2"), Rule("WAIT QB UNTIL R3")])
    assert ("RB", 2, 2) in st.active_rule_effects()["need"]
    st.reset("live"); st.rules = [Rule("2 RB BY ROUND 1")]; st.save(settings)
    r = service.roster_risk(settings, st)
    assert set(r["positions"]) >= {"QB", "RB", "WR", "TE"}
    rb = r["positions"]["RB"]
    assert rb["level"] in ("warn", "danger") and any("rule" in x for x in rb["reasons"])
    assert any(s.startswith("RB") for s in r["summary"])


# ---------------- league config endpoint ----------------

def test_config_roundtrip_and_validation(settings, tmp_path, monkeypatch):
    from howie3 import service
    monkeypatch.setenv("HOWIE_DATA_DIR", str(tmp_path))
    s2 = Settings()
    from dataclasses import asdict
    (tmp_path / "league_config.json").write_text(json.dumps(asdict(LeagueConfig())))  # the default shape, not the user's
    c = service.config_payload(s2)
    assert c["num_teams"] == 12 and c["scoring_type"] == "half_ppr"
    out = service.update_config(s2, {"draft_position": "3", "market_anchor": "0.5"})
    assert out["draft_position"] == 3 and out["market_anchor"] == 0.5
    with pytest.raises(ValueError, match="draft_position"):
        service.update_config(s2, {"draft_position": 40})
    assert service.config_payload(s2)["draft_position"] == 3  # invalid write left the file alone
