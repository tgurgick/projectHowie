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
    }


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
