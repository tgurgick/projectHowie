"""The engine's JSON contract layer.

Every surface — web UI, MCP server, agent tools, CLI — calls these functions
and gets plain JSON-able dicts. Nothing here renders; nothing here holds
state beyond the draft event log it is handed. This module IS the API.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple

from .config import LeagueConfig, Settings
from .db import connect
from .state import DraftState, snake_team_for_pick, state_lock
from .value.board import POSITIONS, PoolPlayer, expected_kth_best, load_pool, snake_picks


# ------------------------------------------------------------ shared loading

def _conn(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path)


def _pool(settings: Settings, conn: sqlite3.Connection) -> List[PoolPlayer]:
    return load_pool(conn, settings.current_season, settings.league.scoring_format,
                     market_anchor=settings.league.market_anchor)


def _me(settings: Settings) -> int:
    return settings.league.draft_position


def _pick_context(settings: Settings, state: DraftState,
                  league: Optional[LeagueConfig] = None) -> Tuple[int, int, int, List[int]]:
    """(round, current_pick, next_pick, future_picks) for the user's turn."""
    league = league or settings.league
    picks = snake_picks(league)
    rnd = min(len(state.my_uids(league)) + 1, len(picks))
    current_pick = picks[rnd - 1]
    future = picks[rnd:]
    next_pick = future[0] if future else current_pick + league.num_teams
    return rnd, current_pick, next_pick, future


# ------------------------------------------------------------ state & picks

def state_payload(settings: Settings, state: DraftState) -> dict:
    league = settings.league
    picks = snake_picks(league)
    next_no = state.next_pick_no()
    total = league.num_teams * league.roster_size
    on_clock = snake_team_for_pick(league, next_no) if next_no <= total else None

    conn = _conn(settings)
    pool_by_uid = {p.uid: p for p in _pool(settings, conn)}
    conn.close()

    my = [e for e in state.events if e.mine]
    slots = _fill_slots(league, my, pool_by_uid)
    log = [
        {
            "seq": e.seq, "pick_no": e.pick_no, "team": e.team,
            "name": e.player_name, "position": e.position,
            "mine": e.mine, "source": e.source,
        }
        for e in reversed(state.events[-14:])
    ]
    return {
        "mode": state.mode,
        "round": (next_no - 1) // league.num_teams + 1 if next_no <= total else league.roster_size,
        "next_pick_no": next_no,
        "on_clock_team": on_clock,
        "you_are_on_clock": on_clock == _me(settings),
        "my_picks": picks,
        "my_next_picks": [p for p in picks if p >= next_no][:5],
        "roster": slots,
        "log": log,
        "total_picks": total,
        "complete": next_no > total,
        "league": {
            "teams": league.num_teams, "slot": _me(settings),
            "scoring": league.scoring_format,
        },
    }


def _fill_slots(league, my_events, pool_by_uid) -> List[dict]:
    order = (
        [("QB", league.qb_slots), ("RB", league.rb_slots), ("WR", league.wr_slots),
         ("TE", league.te_slots), ("FLX", league.flex_slots), ("K", league.k_slots),
         ("DST", league.dst_slots)]
    )
    remaining = []
    for e in my_events:
        p = pool_by_uid.get(e.player_uid)
        remaining.append({"name": e.player_name, "pos": e.position,
                          "proj": round(p.proj) if p else None})
    slots = []
    flex_ok = {"RB", "WR", "TE"}
    for slot_name, count in order:
        for _ in range(count):
            hit = None
            for i, entry in enumerate(remaining):
                pos = entry["pos"]
                if pos == slot_name or (slot_name == "FLX" and pos in flex_ok):
                    hit = remaining.pop(i)
                    break
            slots.append({"slot": slot_name, **(hit or {"name": None, "pos": None, "proj": None})})
    for entry in remaining:
        slots.append({"slot": "BN", **entry})
    return slots


def mark_pick(settings: Settings, player_uid: str, mine: bool,
              source: str = "ui") -> dict:
    """Record a pick (cross-process safe). In mock mode, bots then advance
    until the user is back on the clock. This is THE write path — every
    surface (UI, MCP, CLI, agents) goes through here."""
    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    player = pool_by_uid.get(player_uid)
    if player is None:
        raise ValueError(f"Unknown player uid {player_uid}")

    with state_lock(settings):
        state = DraftState.load(settings)
        pick_no = state.next_pick_no()
        me = _me(settings)
        if mine:
            team = me
        else:
            team = snake_team_for_pick(league, pick_no)
            if team == me:
                team = 0  # attribution unknown (log lagging); never the user's slot
        event = state.add_pick(pick_no, team, player.uid, player.name,
                               player.position, source, mine=mine)
        state.save(settings)
        result = {"seq": event.seq, "pick_no": event.pick_no,
                  "team": event.team, "name": player.name}
        if state.mode == "mock":
            from .mock import advance_bots

            result["bots"] = advance_bots(settings, state, pool, seed=state.seed)
    return result


def undo_pick(settings: Settings) -> Optional[dict]:
    """Undo the user's last action. In mock mode that means rolling back the
    bot picks made after it AND the user's own pick, leaving them on the
    clock at that pick again."""
    with state_lock(settings):
        state = DraftState.load(settings)
        if not state.events:
            return None
        if state.mode == "mock":
            while state.events and not state.events[-1].mine:
                state.undo()
            event = state.undo()  # the user's own pick (or None at draft start)
        else:
            event = state.undo()
        state.save(settings)
    return {"name": event.player_name, "pick_no": event.pick_no} if event else None


def start_mock(settings: Settings) -> dict:
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    with state_lock(settings):
        state = DraftState.load(settings)
        state.reset("mock")
        state.save(settings)
        from .mock import advance_bots

        bots = advance_bots(settings, state, pool, seed=state.seed)
    return {"started": True, "seed": state.seed, "bots": bots}


def reset_draft(settings: Settings, mode: str = "live") -> dict:
    with state_lock(settings):
        state = DraftState.load(settings)
        state.reset(mode)
        state.save(settings)
    return {"reset": True, "mode": mode}


# ------------------------------------------------------------ search

def search_payload(settings: Settings, q: str, limit: int = 8) -> List[dict]:
    from .graph import search as g_search

    conn = _conn(settings)
    hits = g_search(conn, q, limit=limit)
    taken: frozenset = DraftState.load(settings).taken_uids()
    pool_by_uid = {p.uid: p for p in _pool(settings, conn)}
    out = []
    for h in hits:
        row = {"id": h["id"], "kind": h["kind"], "name": h["name"],
               "team": h["team"], "position": h["position"]}
        if h["kind"] == "player":
            uid = h["id"].split(":", 1)[1]
            p = pool_by_uid.get(uid)
            if p is None:
                continue  # not draft-relevant this season
            row.update({
                "uid": uid, "proj": round(p.raw or p.proj), "adp": p.adp,
                "taken": uid in taken,
            })
        out.append(row)
    conn.close()
    return out


# ------------------------------------------------------------ recommendations

def pick_payload(settings: Settings, state: DraftState, sims: int = 0, top_n: int = 10) -> dict:
    from .value.roster import evaluate_candidates, mc_rerank

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)

    effects = state.active_rule_effects()
    results = evaluate_candidates(pool, roster, current_pick, future, league,
                                  taken, top_n=max(top_n, 12))
    if sims > 0:
        results = mc_rerank(conn, results, roster, pool, league,
                            settings.current_season, n_sims=sims)

    rows, span = [], _outcome_span(results, sims)
    for r in results:
        fired = _fired_rules(r.player, rnd, effects)
        value = r.sim.mean if (sims and r.sim) else r.final_value
        rows.append({
            "uid": r.player.uid, "name": r.player.name, "pos": r.player.position,
            "team": r.player.team, "proj": round(r.player.raw or r.player.proj),
            "adp": r.player.adp,
            "avail_now": round(r.player.p_available(current_pick), 2),
            "avail_next": round(r.player.p_available(next_pick), 2),
            "value": round(value),
            "p10": round(r.sim.p10) if (sims and r.sim) else None,
            "p90": round(r.sim.p90) if (sims and r.sim) else None,
            "plan": r.plan_positions[:8],
            "rules": fired,
        })
    demoted = [row for row in rows if any(f["type"] in ("wait", "ban") for f in row["rules"])]
    kept = [row for row in rows if row not in demoted]
    rows = kept + demoted
    best = rows[0]["value"] if rows else 0
    for row in rows:
        row["delta"] = row["value"] - best
    conn.close()
    return {
        "current_pick": current_pick, "next_pick": next_pick, "round": rnd,
        "sims": sims, "rows": rows[:top_n], "outcome_span": span,
    }


def _outcome_span(results, sims) -> Optional[List[int]]:
    if not sims:
        return None
    lows = [r.sim.p10 for r in results if r.sim]
    highs = [r.sim.p90 for r in results if r.sim]
    if not lows:
        return None
    return [int(min(lows) - 5), int(max(highs) + 5)]


def _fired_rules(player: PoolPlayer, rnd: int, effects: Dict[str, list]) -> List[dict]:
    from .data.names import name_key

    fired = []
    for target in effects["targets"]:
        if name_key(target) in name_key(player.name):
            fired.append({"type": "target", "text": f"TARGET {target}"})
    for pos, until in effects["wait"]:
        if player.position == pos and rnd < until:
            fired.append({"type": "wait", "text": f"WAIT {pos} UNTIL R{until}"})
    for pos, until in effects["ban"]:
        if player.position == pos and rnd < until:
            fired.append({"type": "ban", "text": f"NO {pos} BEFORE R{until}"})
    return fired


def positions_payload(settings: Settings, state: DraftState) -> dict:
    """Per-position: expected final roster drafting it NOW vs the same roster
    with the position player degraded to expected-best at the next pick."""
    from .value.roster import _rollout

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)

    pools: Dict[str, List[PoolPlayer]] = {pos: [] for pos in POSITIONS}
    roster_uids = {p.uid for p in roster}
    for p in pool:
        if p.uid not in taken and p.uid not in roster_uids:
            pools[p.position].append(p)
    roster_pts: Dict[str, List[float]] = {}
    for p in roster:
        roster_pts.setdefault(p.position, []).append(p.proj)

    out = []
    for pos in POSITIONS:
        likely = [p for p in pools[pos] if p.p_available(current_pick) >= 0.10]
        if not likely:
            continue
        cand = likely[0]
        take_pts = dict(roster_pts)
        take_pts[pos] = roster_pts.get(pos, []) + [cand.proj]
        pools_after = dict(pools)
        pools_after[pos] = [p for p in pools[pos] if p.uid != cand.uid]
        now_val, _ = _rollout(take_pts, pools_after, future, league, taken)

        eba_next = expected_kth_best(pools[pos], next_pick, 1, taken)
        wait_pts = dict(roster_pts)
        wait_pts[pos] = roster_pts.get(pos, []) + [eba_next]
        wait_val, _ = _rollout(wait_pts, pools_after, future, league, taken)

        top_avail_next = cand.p_available(next_pick)
        out.append({
            "pos": pos, "player": cand.name, "player_proj": round(cand.proj),
            "now": round(now_val), "wait": round(wait_val),
            "cost": round(now_val - wait_val),
            "avail_next": round(top_avail_next, 2),
            "tier_drop": round(cand.proj - eba_next),
        })
    out.sort(key=lambda r: -r["now"])
    return {"current_pick": current_pick, "next_pick": next_pick, "rows": out}


# ------------------------------------------------------------ player card

def card_payload(settings: Settings, uid: str) -> dict:
    from .graph import entity_context
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_player_totals

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    pool_by_uid = {p.uid: p for p in pool}
    player = pool_by_uid.get(uid)
    if player is None:
        conn.close()
        raise ValueError(f"Unknown player uid {uid}")

    state = DraftState.load(settings)
    rnd, current_pick, next_pick, _future = _pick_context(settings, state, league)

    # outcome band
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    sp = build_sim_players(conn, [player], settings.current_season,
                           league.scoring_format, proj_rank, games)[0]
    totals = simulate_player_totals(sp, n_sims=300, seed=7)

    # marginal value vs waiting at the position
    same_pos = [p for p in pool if p.position == player.position
                and p.uid not in state.taken_uids()]
    eba_next = expected_kth_best(same_pos, next_pick, 1)

    ctx = entity_context(conn, f"player:{uid}")
    sos_rows = conn.execute(
        "SELECT week, value FROM sos WHERE season = ? AND team = ? AND position = ? "
        "AND week BETWEEN 15 AND 17 ORDER BY week",
        (settings.current_season, player.team, player.position)).fetchall()
    trend = conn.execute(
        f"SELECT season, COUNT(*) g, ROUND(AVG(pts_{league.scoring_format}), 1) ppg "
        "FROM weekly_stats WHERE player_uid = ? GROUP BY season ORDER BY season DESC LIMIT 3",
        (uid,)).fetchall()

    # empirical anchors: every game of the last two seasons + milestone rates
    from .value.milestones import MILESTONES, player_games, player_rates, tier_rates

    log_seasons = (settings.current_season - 2, settings.current_season - 1)
    games = player_games(conn, uid, league.scoring_format, log_seasons)
    milestones = {
        "labels": [label for label, _ in MILESTONES.get(player.position, [])],
        "player": player_rates(games, player.position),
        "tier": tier_rates(conn, league.scoring_format, player.position,
                           settings.current_season - 1),
        "seasons": list(log_seasons),
    }
    conn.close()

    import numpy as np
    return {
        "uid": uid, "name": player.name, "pos": player.position, "team": player.team,
        "bye": player.bye, "proj": round(player.raw or player.proj, 1),
        "value": round(player.proj, 1),  # market-anchored (what the engine ranks on)
        "adp": player.adp, "adp_stdev": player.stdev,
        "avail_now": round(player.p_available(current_pick), 2),
        "avail_next": round(player.p_available(next_pick), 2),
        "next_pick": next_pick,
        "mv_vs_wait": round(player.proj - eba_next),
        "band": {"p10": round(float(np.percentile(totals, 10))),
                 "p50": round(float(np.percentile(totals, 50))),
                 "p90": round(float(np.percentile(totals, 90)))},
        "room": (ctx or {}).get("room"),
        "facts": (ctx or {}).get("facts", []),
        "team_facts": (ctx or {}).get("team_facts", []),
        "playoff_sos": [{"week": r["week"], "value": round(r["value"], 1)} for r in sos_rows],
        "trend": [{"season": r["season"], "games": r["g"], "ppg": r["ppg"]}
                  for r in reversed(trend)],
        "games": games,
        "milestones": milestones,
    }


# ------------------------------------------------------------ anchors (strategy tab)

def anchors_payload(settings: Settings, state: DraftState) -> dict:
    """League-era base rates + this roster's typical week, from box scores."""
    from .value.milestones import BOOM, league_trend, roster_anchors

    league = settings.league
    fmt = league.scoring_format
    conn = _conn(settings)
    pool_by_uid = {p.uid: p for p in _pool(settings, conn)}
    last = settings.current_season - 1
    starters = []
    for uid in state.my_uids(league):
        p = pool_by_uid.get(uid)
        if p and p.position in BOOM:
            starters.append({"uid": uid, "name": p.name, "position": p.position})
    roster = roster_anchors(conn, fmt, starters, (last - 1, last), last)
    trend = league_trend(conn, fmt, (last - 7, last - 3, last))
    conn.close()
    return {"roster": roster, "league": trend,
            "seasons": [last - 7, last - 3, last], "boom": BOOM}


# ------------------------------------------------------------ strategy sheet

def strategy_payload(state: DraftState) -> dict:
    """Rules carry their parsed effect — an `inert` rule matched no pattern
    and will NOT influence the board, so the UI can badge it."""
    from .state import DraftState as DS

    rows = []
    for r in state.rules:
        probe = DS(rules=[type(r)(text=r.text, on=True)])
        fx = probe.active_rule_effects()
        inert = not (fx["targets"] or fx["wait"] or fx["ban"])
        rows.append({"text": r.text, "on": r.on, "inert": inert})
    return {"rules": rows, "notes": state.notes}


def update_strategy(settings: Settings,
                    rules: Optional[List[dict]] = None,
                    notes: Optional[str] = None) -> dict:
    from .state import Rule

    with state_lock(settings):
        state = DraftState.load(settings)
        if rules is not None:
            state.rules = [Rule(text=str(r["text"])[:120], on=bool(r.get("on", True)))
                           for r in rules[:20]]
        if notes is not None:
            state.notes = str(notes)[:8000]
        state.save(settings)
    return strategy_payload(state)


# ------------------------------------------------------------ data tab

DIST_STATS = {
    "pts": "pts_{fmt}", "rush_yds": "rush_yards", "rec_yds": "rec_yards",
    "targets": "targets", "receptions": "receptions",
    "tds": "(rush_tds + rec_tds)", "pass_yds": "pass_yards", "pass_tds": "pass_tds",
    "carries": "rush_attempts",
}
_STARTER_N = {"QB": 14, "RB": 36, "WR": 48, "TE": 14}


def games_distribution(settings: Settings, position: str, stat: str = "pts",
                       tier: str = "starter", n_seasons: int = 3) -> dict:
    """Every player-game for a position over the last n seasons — the dots."""
    fmt = settings.league.scoring_format
    if position not in _STARTER_N or stat not in DIST_STATS:
        raise ValueError("position must be QB/RB/WR/TE and stat one of " + ", ".join(DIST_STATS))
    expr = DIST_STATS[stat].format(fmt=fmt)
    last = settings.current_season - 1
    seasons = list(range(last - n_seasons + 1, last + 1))
    conn = _conn(settings)
    rows = []
    for season in seasons:
        if tier == "starter":
            sql = f"""WITH s AS (SELECT player_uid FROM weekly_stats WHERE season=? AND position=?
                                GROUP BY player_uid ORDER BY SUM(pts_{fmt}) DESC LIMIT ?)
                      SELECT w.player_uid, p.name, w.season, w.week, w.opponent, w.team,
                             {expr} AS v, w.pts_{fmt} AS pts
                      FROM weekly_stats w JOIN s USING(player_uid) JOIN players p USING(player_uid)
                      WHERE w.season=? AND w.week<=18"""
            params = (season, position, _STARTER_N[position], season)
        else:
            sql = f"""SELECT w.player_uid, p.name, w.season, w.week, w.opponent, w.team,
                             {expr} AS v, w.pts_{fmt} AS pts
                      FROM weekly_stats w JOIN players p USING(player_uid)
                      WHERE w.season=? AND w.position=? AND w.week<=18
                        AND (w.rush_attempts + w.targets + w.pass_attempts) >= 3"""
            params = (season, position)
        for r in conn.execute(sql, params):
            rows.append([r["player_uid"], r["name"], r["season"], r["week"], r["opponent"],
                         r["team"], round(r["v"] or 0, 1), round(r["pts"] or 0, 1)])
    conn.close()
    return {"position": position, "stat": stat, "tier": tier, "seasons": seasons,
            "columns": ["uid", "name", "season", "week", "opp", "team", "value", "pts"],
            "rows": rows}


def sim_payload(settings: Settings, uid: str, n_sims: int = 400) -> dict:
    """A player's simulated season-total distribution next to his actual past seasons."""
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_player_totals

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    pool_by_uid = {p.uid: p for p in pool}
    player = pool_by_uid.get(uid)
    if player is None:
        conn.close()
        raise ValueError(f"Unknown player uid {uid}")
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    sp = build_sim_players(conn, [player], settings.current_season,
                           league.scoring_format, proj_rank, games)[0]
    totals = simulate_player_totals(sp, n_sims=n_sims, seed=7)
    actual = [{"season": r["season"], "total": round(r["t"], 1), "games": r["g"]} for r in conn.execute(
        f"SELECT season, SUM(pts_{league.scoring_format}) t, COUNT(*) g FROM weekly_stats "
        "WHERE player_uid = ? AND week <= 17 GROUP BY season ORDER BY season DESC LIMIT 3", (uid,))]
    conn.close()
    import numpy as np
    return {
        "uid": uid, "name": player.name, "position": player.position,
        "proj": round(player.raw or player.proj, 1), "value": round(player.proj, 1),
        "samples": [round(float(x), 1) for x in totals],
        "p10": round(float(np.percentile(totals, 10))), "p50": round(float(np.percentile(totals, 50))),
        "p90": round(float(np.percentile(totals, 90))),
        "actual": actual,
        "model": {"weekly_mu": round(sp.weekly_mu, 2), "cv": round(sp.cv, 3),
                  "p_play": round(sp.p_play, 3), "season_sigma": sp.season_sigma},
    }


def roster_sim_payload(settings: Settings, state: DraftState, n_sims: int = 300) -> dict:
    """The current roster's simulated season-total distribution."""
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_roster

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    pool_by_uid = {p.uid: p for p in pool}
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    if not roster:
        conn.close()
        return {"players": [], "samples": []}
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    sps = build_sim_players(conn, roster, settings.current_season, league.scoring_format,
                            proj_rank, games)
    conn.close()
    res = simulate_roster(sps, league, n_sims=n_sims, playoff_weight=league.playoff_weight)
    return {
        "players": [p.name for p in roster],
        "samples": [round(float(x), 1) for x in res.samples],
        "mean": round(res.mean), "p10": round(res.p10), "p90": round(res.p90),
    }


QUERY_PRESETS = [
    ("Top 20 RB by 2025 pts", "SELECT p.name, SUM(w.pts_{fmt}) pts, COUNT(*) g FROM weekly_stats w JOIN players p USING(player_uid) WHERE w.season=2025 AND w.position='RB' GROUP BY p.name ORDER BY pts DESC LIMIT 20"),
    ("WR 100-yd games, 2025", "SELECT p.name, COUNT(*) games_100 FROM weekly_stats w JOIN players p USING(player_uid) WHERE w.season=2025 AND w.position='WR' AND w.rec_yards>=100 GROUP BY p.name ORDER BY games_100 DESC LIMIT 20"),
    ("Team pass rate 2025", "SELECT team, ROUND(SUM(pass_attempts)*1.0/(SUM(pass_attempts)+SUM(rush_attempts)),3) pass_rate FROM weekly_stats WHERE season=2025 GROUP BY team ORDER BY pass_rate DESC"),
    ("2026 ADP vs projection (RB)", "SELECT p.name, a.adp, pr.pts_{fmt} proj FROM projections pr JOIN players p USING(player_uid) JOIN adp a ON a.player_uid=pr.player_uid AND a.season=pr.season AND a.format='{fmt}' WHERE pr.season=2026 AND pr.position='RB' ORDER BY a.adp LIMIT 30"),
    ("Vacated volume by room", "SELECT entity_id, text, value FROM facts WHERE kind='vacated_share' ORDER BY value DESC LIMIT 20"),
]


def query_payload(settings: Settings, q: str) -> dict:
    """Quick data access: `sql: SELECT ...` runs a sandboxed read-only query;
    anything else is a graph search whose top hit is expanded."""
    from .agent import safe_query
    from .graph import entity_context, search as g_search
    import json as _json

    fmt = settings.league.scoring_format
    q = q.strip()
    if not q:
        return {"mode": "presets", "presets": [{"label": l, "sql": s.format(fmt=fmt)} for l, s in QUERY_PRESETS]}
    if q.lower().startswith("sql:"):
        raw = safe_query(settings, q[4:].strip())
        if raw.startswith("Error"):
            return {"mode": "sql", "error": raw}
        rows = _json.loads(raw)
        return {"mode": "sql", "columns": list(rows[0].keys()) if rows else [], "rows": rows}

    conn = _conn(settings)
    hits = g_search(conn, q, limit=6)
    if not hits:
        conn.close()
        return {"mode": "search", "hits": [], "entity": None, "detail": None}
    top = hits[0]
    ctx = entity_context(conn, top["id"])
    detail: dict = {"context": ctx}
    pool_by_uid = {p.uid: p for p in _pool(settings, conn)}
    if top["kind"] == "player":
        uid = top["id"].split(":", 1)[1]
        pp = pool_by_uid.get(uid)
        detail["projection"] = ({"proj": round(pp.raw or pp.proj, 1), "value": round(pp.proj, 1),
                                 "adp": pp.adp, "bye": pp.bye} if pp else None)
        detail["seasons"] = [dict(r) for r in conn.execute(
            f"SELECT season, COUNT(*) g, ROUND(SUM(pts_{fmt}),1) pts, ROUND(AVG(pts_{fmt}),1) ppg, "
            "SUM(rush_yards) rush_yds, SUM(rec_yards) rec_yds, SUM(targets) tgt, "
            "SUM(rush_tds + rec_tds) tds FROM weekly_stats WHERE player_uid = ? "
            "GROUP BY season ORDER BY season DESC", (uid,))]
        detail["uid"] = uid
    elif top["kind"] == "team":
        team = top["team"]
        by_name = {p.name: p for p in pool_by_uid.values() if p.team == team}
        detail["rooms"] = []
        for r in conn.execute(
            "SELECT e.name, e.position, ed.value AS share FROM edges ed JOIN entities e ON e.id = ed.src "
            "WHERE ed.kind = 'in_room' AND ed.dst LIKE ? ORDER BY e.position, ed.value DESC", (f"unit:{team}-%",)):
            pp = by_name.get(r["name"])
            detail["rooms"].append({**dict(r), "proj": round(pp.raw or pp.proj) if pp else None,
                                    "adp": pp.adp if pp else None})
        detail["rooms"].sort(key=lambda m: (m["position"], -(m["proj"] or 0)))
    conn.close()
    return {"mode": "search", "hits": hits, "entity": top, "detail": detail}


# ------------------------------------------------------------ structured query builder

_BUILD_STATS = {
    "pts": "pts_{fmt}", "rush_yds": "rush_yards", "rec_yds": "rec_yards", "targets": "targets",
    "receptions": "receptions", "carries": "rush_attempts", "tds": "(rush_tds + rec_tds)",
    "pass_yds": "pass_yards", "pass_tds": "pass_tds", "interceptions": "interceptions",
}
_BUILD_MEASURES = {
    "total": "SUM({e})", "per_game": "AVG({e})", "max": "MAX({e})",
    "games_over": "SUM(CASE WHEN {e} >= {thr} THEN 1 ELSE 0 END)",
}


def build_query(settings: Settings, entity: str = "player", pos: str = "ALL",
                season: str = "2025", measure: str = "total", stat: str = "pts",
                thr: float = 100.0, min_games: int = 1, order: str = "desc",
                limit: int = 20) -> dict:
    """Compile dropdown choices into SQL (whitelisted pieces only) and run it."""
    fmt = settings.league.scoring_format
    if entity not in ("player", "team") or stat not in _BUILD_STATS \
            or measure not in _BUILD_MEASURES or order not in ("desc", "asc"):
        raise ValueError("invalid builder choice")
    if pos not in ("ALL", "QB", "RB", "WR", "TE"):
        raise ValueError("invalid position")
    expr = _BUILD_STATS[stat].format(fmt=fmt)
    agg = _BUILD_MEASURES[measure].format(e=expr, thr=float(thr))
    limit = max(1, min(int(limit), 200))
    min_games = max(1, min(int(min_games), 18))
    where = ["w.week <= 18"]
    if season != "all":
        where.append(f"w.season = {int(season)}")
    if pos != "ALL":
        where.append(f"w.position = '{pos}'")
    label = {"total": f"total {stat}", "per_game": f"{stat} per game", "max": f"best game {stat}",
             "games_over": f"games with {stat} >= {float(thr):g}"}[measure]
    if entity == "player":
        sql = (f"SELECT p.name, w.position AS pos, COUNT(*) AS games, ROUND({agg}, 1) AS value "
               f"FROM weekly_stats w JOIN players p USING (player_uid) WHERE {' AND '.join(where)} "
               f"GROUP BY w.player_uid HAVING games >= {min_games} ORDER BY value {order.upper()} LIMIT {limit}")
    else:
        sql = (f"SELECT w.team, COUNT(DISTINCT w.season || '-' || w.week) AS games, ROUND({agg}, 1) AS value "
               f"FROM weekly_stats w WHERE {' AND '.join(where)} "
               f"GROUP BY w.team ORDER BY value {order.upper()} LIMIT {limit}")
    from .agent import safe_query
    import json as _json
    raw = safe_query(settings, sql)
    if raw.startswith("Error"):
        return {"sql": sql, "error": raw}
    rows = _json.loads(raw)
    return {"sql": sql, "label": label, "columns": list(rows[0].keys()) if rows else [], "rows": rows}
