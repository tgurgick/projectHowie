"""The engine's JSON contract layer.

Every surface — web UI, MCP server, agent tools, CLI — calls these functions
and gets plain JSON-able dicts. Nothing here renders; nothing here holds
state beyond the draft event log it is handed. This module IS the API.
"""

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .config import LeagueConfig, Settings
from .db import connect
from .payloads import JsonDict, RosterSimPayload
from .status import chip as status_chip
from .state import DraftState, snake_team_for_pick, state_lock
from .value.board import POSITIONS, PoolPlayer, expected_kth_best, load_pool, snake_picks


# ------------------------------------------------------------ shared loading

def _conn(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path)


def _pool(settings: Settings, conn: sqlite3.Connection) -> List[PoolPlayer]:
    """The draft pool with the Mock Draft Lab's empirical availability
    attached (players never seen taken before a pick keep the ADP model)."""
    from . import mocksim  # module import (not from-import): safe if another thread is mid-import

    pool = load_pool(conn, settings.current_season, settings.league.scoring_format,
                     market_anchor=settings.league.market_anchor)
    table = mocksim.availability_table(settings)
    if table:
        for p in pool:
            p.emp_avail = table.get(p.uid)
    return pool


def _me(settings: Settings) -> int:
    return settings.league.draft_position


_FLOW_CACHE: Dict[str, object] = {}
FLOW_ROLLOUTS = 250
FLOW_HORIZON = 3


def draft_flow_for(settings: Settings, state: DraftState, pool: List[PoolPlayer], plan=None):
    """The live board rolled forward to the user's next picks, cached per
    draft generation (identity + length) so every payload in a request
    cycle shares one simulation."""
    from .league_profile import load_profile
    from .value.flow import draft_flow

    key = f"{state.created}:{state.seed}:{len(state.events)}:{settings.league.draft_position}"
    flow = _FLOW_CACHE.get(key)
    if flow is None:
        flow = draft_flow(pool, state, settings.league, n=FLOW_ROLLOUTS, horizon=FLOW_HORIZON, my_plan=plan,
                          profile=load_profile(settings))
        _FLOW_CACHE.clear()
        _FLOW_CACHE[key] = flow
    return flow


def _pool_with_flow(settings: Settings, conn: sqlite3.Connection, state: DraftState) -> List[PoolPlayer]:
    """The pool with conditioned availability attached for the next picks."""
    from .value.flow import attach

    pool = _pool(settings, conn)
    flow = draft_flow_for(settings, state, pool)
    attach(pool, flow)
    return pool


def _pick_context(settings: Settings, state: DraftState,
                  league: Optional[LeagueConfig] = None) -> Tuple[int, int, int, List[int]]:
    """(round, current_pick, next_pick, future_picks) for the user's turn.

    Derived from the draft's actual position (the next overall pick), never
    from roster size: a user who skipped a pick, marked his own slot as
    taken, or imported a partial log must still be evaluated at the pick the
    draft is really at. `round` is the index of the user's pick (1-based),
    which is what WAIT/NO-BEFORE rules compare against."""
    league = league or settings.league
    picks = snake_picks(league)
    upcoming = [k for k in picks if k >= state.next_pick_no()]
    if not upcoming:  # the user's picks are all behind him
        return len(picks), picks[-1], picks[-1] + league.num_teams, []
    current_pick = upcoming[0]
    future = upcoming[1:]
    next_pick = future[0] if future else current_pick + league.num_teams
    return picks.index(current_pick) + 1, current_pick, next_pick, future


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
                if state.mode == "mock":
                    raise ValueError(
                        f"You're on the clock at pick {pick_no} — draft {player.name} or "
                        "someone else (in a mock the bots have already picked)")
                team = 0  # live: attribution unknown (log lagging); never the user's slot
        event = state.add_pick(pick_no, team, player.uid, player.name,
                               player.position, source, mine=mine, league=league)
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


def archive_draft(settings: Settings, state: DraftState) -> Optional[dict]:
    """Store the draft's pick order in the Mock Draft Lab before it is wiped,
    so every mock or live draft you ran teaches availability. Returns the
    stored record, or None when there was nothing worth keeping."""
    from .mocksim import load_store, save_store

    if len(state.events) < settings.league.num_teams:  # less than one round: noise
        return None
    record = {
        "source": "cockpit-" + state.mode, "policy": "user", "seed": state.seed,
        "created": state.created, "archived": _now_iso(),
        "picks": [e.player_uid for e in state.events],
        "mine": [e.player_uid for e in state.events if e.mine],
        "complete": len(state.events) >= settings.league.num_teams * settings.league.roster_size,
    }
    store = load_store(settings)
    if any(d.get("created") == state.created and d.get("source") == record["source"] for d in store["drafts"]):
        return None  # already archived (e.g. reset pressed twice)
    store["drafts"].append(record)
    save_store(settings, store)
    return record


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start_mock(settings: Settings) -> dict:
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    with state_lock(settings):
        state = DraftState.load(settings)
        archived = archive_draft(settings, state)
        state.reset("mock")
        state.save(settings)
        from .mock import advance_bots

        bots = advance_bots(settings, state, pool, seed=state.seed)
    return {"started": True, "seed": state.seed, "bots": bots, "archived": bool(archived)}


def reset_draft(settings: Settings, mode: str = "live") -> dict:
    with state_lock(settings):
        state = DraftState.load(settings)
        archived = archive_draft(settings, state)
        state.reset(mode)
        state.save(settings)
    return {"reset": True, "mode": mode, "archived": bool(archived)}


def sync_picks(settings: Settings, names: List[str], source: str = "chrome",
               pick_numbers: Optional[List[int]] = None) -> dict:
    """Bring the draft log up to a pick order observed elsewhere (an ESPN /
    Sleeper draft room). Idempotent: names already in the log are skipped,
    new ones are appended in order; a pick that falls on the user's snake
    slot is recorded as the user's. With `pick_numbers` (overall pick per
    name) the log is aligned to the room's numbering: picks the observer
    never saw become placeholder events so attribution cannot drift by a
    slot. Unresolved names are reported, never guessed."""
    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    from .data.names import name_key
    from .graph import TEAM_NAMES
    by_key: Dict[str, str] = {}
    for p in pool:
        by_key.setdefault(name_key(p.name), p.uid)
        if p.position == "DST" and p.team:
            full = TEAM_NAMES.get(p.team, "")
            for alias in (full, full.split()[-1] if full else "", p.team):
                if alias:
                    by_key.setdefault(name_key(f"{alias} D/ST"), p.uid)
                    by_key.setdefault(name_key(f"{alias} DST"), p.uid)
    added, skipped, unresolved, gaps = [], [], [], []
    numbers: List[Optional[int]] = list(pick_numbers) if pick_numbers else [None] * len(names)
    for raw, pick_no in zip(names, numbers):
        key = name_key(raw.strip())
        uid = by_key.get(key)
        if uid is None:
            m = [u for k, u in by_key.items() if key and (k.endswith(" " + key) or k.startswith(key + " "))]
            uid = m[0] if len(m) == 1 else None
        if uid is None:
            unresolved.append(raw)
            continue
        state = DraftState.load(settings)
        if uid in state.taken_uids():
            skipped.append(raw)
            continue
        if pick_no is not None and pick_no < state.next_pick_no():
            skipped.append(raw)   # an earlier pick we already have under another name
            continue
        while pick_no is not None and state.next_pick_no() < pick_no:
            gap_no = state.next_pick_no()
            with state_lock(settings):
                st = DraftState.load(settings)
                st.add_pick(gap_no, snake_team_for_pick(league, gap_no), f"gap:{gap_no}", "(unseen pick)",
                            None, "gap", mine=False, league=league)
                st.save(settings)
            gaps.append(gap_no)
            state = DraftState.load(settings)
        mine = snake_team_for_pick(league, state.next_pick_no()) == league.draft_position
        r = mark_pick(settings, uid, mine=mine, source=source)
        added.append({"name": r["name"], "pick_no": r["pick_no"], "mine": mine})
    state = DraftState.load(settings)
    return {"added": added, "skipped": len(skipped), "unresolved": unresolved, "gaps": gaps,
            "next_pick": state.next_pick_no(),
            "on_clock": snake_team_for_pick(league, state.next_pick_no()) == league.draft_position}


def reconcile_roster(settings: Settings, roster: List[dict]) -> dict:
    """Make the log's 'mine' flags match the room's roster panel
    ([{name: 'D. Prescott' | 'Dak Prescott', pos: 'QB'}]). Abbreviated names
    resolve by initial + surname + position within the pool. Players on the
    roster but attributed elsewhere in the log are flipped to the user; log
    picks marked mine that the roster does not show are un-flipped; roster
    players missing from the log entirely are appended as the user's."""
    from .data.names import name_key

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()

    def resolve(name: str, pos: Optional[str]) -> Optional[str]:
        key = name_key(name)
        exact = [p for p in pool if name_key(p.name) == key and (not pos or p.position == pos)]
        if len(exact) == 1:
            return exact[0].uid
        m = re.match(r"^([A-Za-z])\.?\s+(.+)$", name.strip())
        if m:
            initial, last = m.group(1).lower(), name_key(m.group(2))
            cands = [p for p in pool if (not pos or p.position == pos)
                     and name_key(p.name).startswith(initial) and name_key(p.name).endswith(last)]
            if len(cands) == 1:
                return cands[0].uid
            if cands:  # several ("A. Brown"): prefer the one the log already marks as ours,
                # then one the log holds at all, then the best projected
                st = DraftState.load(settings)
                # was he taken at one of OUR snake slots? (pick number, not the
                # mine flag — the flag is what we are correcting)
                at_slot = {e.player_uid for e in st.events
                           if snake_team_for_pick(league, e.pick_no) == league.draft_position}
                held = st.taken_uids()
                ranked = sorted(cands, key=lambda p: (p.uid not in at_slot, p.uid not in held, -(p.raw or p.proj)))
                return ranked[0].uid
        if pos == "DST":
            team = name.replace("D/ST", "").replace("DST", "").strip()
            from .graph import TEAM_NAMES
            for code, full in TEAM_NAMES.items():
                if team.lower() in (code.lower(), full.lower(), full.split()[-1].lower()):
                    d = [p for p in pool if p.position == "DST" and p.team == code]
                    if d:
                        return d[0].uid
        return None

    wanted: Dict[str, str] = {}
    unresolved = []
    for r in roster:
        uid = resolve(r["name"], r.get("pos"))
        if uid:
            wanted[uid] = r["name"]
        else:
            unresolved.append(r["name"])
    flipped_on, flipped_off, added = [], [], []
    with state_lock(settings):
        state = DraftState.load(settings)
        have = {e.player_uid: e for e in state.events}
        for uid, name in wanted.items():
            e = have.get(uid)
            if e is None:
                continue
            if not e.mine:
                e.mine, e.team = True, league.draft_position
                flipped_on.append(e.player_name)
        for e in state.events:
            if e.mine and e.player_uid not in wanted and not e.player_uid.startswith("gap:"):
                e.mine = False
                flipped_off.append(e.player_name)
        state.save(settings)
    # roster players the log never saw (picks made while the observer was
    # away): append them as the user's without the roster-limit check — the
    # log's earlier attribution is what was wrong, not the roster
    pool_by_uid = {p.uid: p for p in pool}
    with state_lock(settings):
        state = DraftState.load(settings)
        have_uids = {e.player_uid for e in state.events}
        for uid, name in wanted.items():
            if uid in have_uids:
                continue
            p = pool_by_uid[uid]
            state.add_pick(state.next_pick_no(), league.draft_position, uid, p.name, p.position,
                           "roster", mine=True)
            added.append(p.name)
        state.save(settings)
    return {"changed": bool(flipped_on or flipped_off or added), "flipped_on": flipped_on,
            "flipped_off": flipped_off, "added": added, "unresolved": unresolved}


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
                "taken": uid in taken, "status": status_chip(p.status),
            })
        out.append(row)
    conn.close()
    return out


# ------------------------------------------------------------ recommendations

def pick_payload(settings: Settings, state: DraftState, sims: int = 0, top_n: int = 10,
                 pos: Optional[str] = None) -> dict:
    """The board. Without `pos`: the engine's candidates at the current pick
    (up to ~8 per position, K/DST only in the closing rounds). With `pos`:
    a BROWSE list — every draftable player at that position ranked by value,
    so the position chips always show the full depth chart of the board."""
    if pos:
        return _browse_position(settings, state, pos.upper(), top_n)
    return _candidates_payload(settings, state, sims, top_n)


def _browse_position(settings: Settings, state: DraftState, pos: str, top_n: int) -> dict:
    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    taken = state.taken_uids()
    rnd, current_pick, next_pick, _future = _pick_context(settings, state, league)
    effects = state.active_rule_effects()
    picks = snake_picks(league)
    rows: List[JsonDict] = []
    avail_pool = [p for p in pool if p.position == pos and p.uid not in taken]
    best_val = avail_pool[0].proj if avail_pool else 0
    for p in avail_pool[:max(top_n, 30)]:
        adp_round = int((p.adp - 1) // league.num_teams + 1) if p.adp else None
        gone_by = None
        for i, k in enumerate(picks):
            if k > current_pick and p.p_available(k) < 0.5:
                gone_by = i + 1
                break
        rows.append({
            "adp_round": adp_round, "gone_by_round": gone_by,
            "uid": p.uid, "name": p.name, "pos": p.position, "team": p.team,
            "proj": round(p.raw or p.proj), "adp": p.adp,
            "avail_now": round(p.p_available(current_pick), 2),
            "avail_next": round(p.p_available(next_pick), 2),
            "avail_src": p.availability_source(next_pick),
            "status": status_chip(p.status),
            "value": round(p.proj), "delta": round(p.proj - best_val),
            "p10": None, "p90": None, "plan": [],
            "rules": _fired_rules(p, rnd, effects),
        })
    return {"current_pick": current_pick, "next_pick": next_pick, "round": rnd,
            "sims": 0, "rows": rows, "outcome_span": None, "browse": pos}


def _candidates_payload(settings: Settings, state: DraftState, sims: int = 0, top_n: int = 10) -> dict:
    from .value.roster import evaluate_candidates, mc_rerank

    league = settings.league
    conn = _conn(settings)
    pool = _pool_with_flow(settings, conn, state)
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)

    from .value.policy import apply_rules, roster_counts

    effects = state.active_rule_effects()
    results = evaluate_candidates(pool, roster, current_pick, future, league,
                                  taken, top_n=max(top_n, 24))
    if sims > 0:
        results = mc_rerank(conn, results, roster, pool, league,
                            settings.current_season, n_sims=sims)
    results = apply_rules(results, rnd, effects, roster_counts(roster), roster)

    rows: List[JsonDict] = []
    span = _outcome_span(results, sims)
    picks = snake_picks(league)
    for r in results:
        fired = _fired_rules(r.player, rnd, effects)
        value = r.sim.mean if (sims and r.sim) else r.final_value
        # market round, and the first of the user's remaining rounds where he is
        # more likely gone than there
        adp_round = int((r.player.adp - 1) // league.num_teams + 1) if r.player.adp else None
        gone_by = None
        for i, k in enumerate(picks):
            if k > current_pick and r.player.p_available(k) < 0.5:
                gone_by = i + 1
                break
        rows.append({
            "adp_round": adp_round, "gone_by_round": gone_by,
            "uid": r.player.uid, "name": r.player.name, "pos": r.player.position,
            "team": r.player.team, "proj": round(r.player.raw or r.player.proj),
            "adp": r.player.adp,
            "avail_now": round(r.player.p_available(current_pick), 2),
            "avail_next": round(r.player.p_available(next_pick), 2),
            "avail_src": r.player.availability_source(next_pick),
            "status": status_chip(r.player.status),
            "badges": r.player.badges,
            "value": round(value),
            "p10": round(r.sim.p10) if (sims and r.sim) else None,
            "p90": round(r.sim.p90) if (sims and r.sim) else None,
            "plan": r.plan_positions[:8],
            "rules": fired,
        })
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
    pool = _pool_with_flow(settings, conn, state)
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

    out: List[JsonDict] = []
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


# ------------------------------------------------------------ sequence: the next 2-3 picks as one decision

def sequence_payload(settings: Settings, state: DraftState, now_uid: Optional[str] = None) -> dict:
    """Explore/exploit over the user's next picks, from the live board:
    which position to take NOW because its tier is draining, what to wait
    for because it survives, with the conditioned probabilities, a
    fallback at each pick, and the run indicator. The plan (strategy prior)
    is reported next to the sequence so an override is explicit."""
    from .value.flow import attach
    from .value.policy import apply_rules, roster_counts
    from .value.roster import evaluate_candidates

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)
    effects = state.active_rule_effects()
    # 1. the strategy prior: the rollout WITHOUT conditioned availability
    prior = evaluate_candidates(pool, roster, current_pick, future, league, taken, top_n=6)
    prior = apply_rules(prior, rnd, effects, roster_counts(roster), roster)
    prior_plan = prior[0].plan_positions[:FLOW_HORIZON] if prior else []
    # 2. the same decision with the live board rolled forward (plan-aware at my own picks)
    flow = draft_flow_for(settings, state, pool, plan=prior_plan)
    attach(pool, flow)
    results = evaluate_candidates(pool, roster, current_pick, future, league, taken, top_n=6)
    results = apply_rules(results, rnd, effects, roster_counts(roster), roster)
    if not results:
        return {"now": None, "next": [], "runs": flow.runs, "horizon_picks": flow.picks}
    # follow the board's Monte Carlo best when the server has one for this generation
    best = next((r for r in results if now_uid and r.player.uid == now_uid), results[0])
    steps = []
    blocked = {pos for pos, until in effects.get("wait", []) + effects.get("ban", []) if rnd < until}
    used = {best.player.uid}
    later = [k for k in flow.picks if k > current_pick][:FLOW_HORIZON]
    for j, k in enumerate(later):
        pos = best.plan_positions[j] if j < len(best.plan_positions) else None
        if not pos or pos == "—":
            continue
        r_no = rnd + 1 + j
        cands = [p for p in pool if p.position == pos and p.uid not in taken and p.uid not in used
                 and p.draftable and p.flow_avail and k in p.flow_avail]
        likely = sorted(cands, key=lambda p: -p.proj)
        fa = lambda q: (q.flow_avail or {}).get(k, 0.0)  # noqa: E731
        target = next((p for p in likely if fa(p) >= 0.5), None)
        fallback = next((p for p in likely if fa(p) >= 0.75 and p is not target), None)
        best_hope = likely[0] if likely else None
        if target:
            used.add(target.uid)
        steps.append({
            "pick": k, "round": r_no, "pos": pos,
            "target": {"name": target.name, "p": round(fa(target), 2), "proj": round(target.raw or target.proj)} if target else None,
            "fallback": {"name": fallback.name, "p": round(fa(fallback), 2), "proj": round(fallback.raw or fallback.proj)} if fallback else None,
            "best_hope": {"name": best_hope.name, "p": round(fa(best_hope), 2)} if best_hope and best_hope is not target else None,
            "survivors": flow.survivors.get(k, {}),
        })
    # urgency by position right now (what the decision is made of)
    pos_rows = positions_payload(settings, state)["rows"]
    urgency = [{"pos": r["pos"], "cost_of_waiting": r["cost"], "avail_next": r["avail_next"], "player": r["player"]} for r in pos_rows[:4]]
    override = bool(prior_plan) and best.player.position != (prior[0].player.position if prior else None)
    return {
        "current_pick": current_pick, "round": rnd,
        "now": {"name": best.player.name, "pos": best.player.position, "uid": best.player.uid,
                "proj": round(best.player.raw or best.player.proj), "value": round(best.final_value)},
        "plan_prior": prior_plan, "plan_live": best.plan_positions[:FLOW_HORIZON],
        "overrides_plan": override,
        "prior_now": prior[0].player.name if prior else None,
        "prior_now_pos": prior[0].player.position if prior else None,
        "next": steps, "runs": flow.runs, "horizon_picks": flow.picks, "rollouts": flow.n,
        "urgency": urgency, "blocked_now": sorted(blocked),
    }


# ------------------------------------------------------------ lookahead (next N picks)

def lookahead_payload(settings: Settings, state: DraftState, n: int = 3) -> dict:
    """For each of the user's next n picks, given the board right now: what
    the engine would take there (the best candidate), the runner-up, and the
    safest fallback — the strongest candidate most likely to still be on the
    board. Deterministic (no Monte Carlo) so it refreshes between picks."""
    from .value.policy import apply_rules, roster_counts
    from .value.roster import evaluate_candidates

    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    picks = snake_picks(league)
    upcoming = [k for k in picks if k >= state.next_pick_no()][:n]
    effects = state.active_rule_effects()
    out = []
    for i, k in enumerate(upcoming):
        rnd = picks.index(k) + 1
        future = [x for x in picks if x > k]
        res = evaluate_candidates(pool, roster, k, future, league, taken, top_n=8)
        res = apply_rules(res, rnd, effects, roster_counts(roster))
        rows = [{"name": r.player.name, "pos": r.player.position, "value": round(r.final_value),
                 "avail": round(r.player.p_available(k), 2), "proj": round(r.player.raw or r.player.proj)}
                for r in res[:6]]
        best = rows[0] if rows else None
        safe = max(rows[:6], key=lambda r: (r["avail"], r["value"])) if rows else None
        out.append({"pick": k, "round": rnd, "picks_away": k - state.next_pick_no(),
                    "best": best, "alt": rows[1] if len(rows) > 1 else None,
                    "safe": safe if (safe and best and safe["name"] != best["name"]) else None,
                    "candidates": rows[:4]})
        # assume the engine takes its best there, so the following pick is evaluated with it on the roster
        if best:
            roster = roster + [next(p for p in pool if p.name == best["name"])]
    return {"next_pick": state.next_pick_no(), "picks": out}


# ------------------------------------------------------------ round-by-round plan (STRATEGY tab)

def plan_payload(settings: Settings, state: DraftState) -> dict:
    """What to optimize for at each of the user's picks, from the engine's
    rollouts under the current strategy: completed rounds show what was taken;
    the current pick shows the best candidate; future rounds show the
    position the top candidates' plans agree on (with the expected points
    that position carries at that pick), the runner-up, how deep each
    position still runs at that pick, and the rules in force that round."""
    from .value.roster import evaluate_candidates

    league = settings.league
    conn = _conn(settings)
    pool = _pool_with_flow(settings, conn, state)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)
    picks = snake_picks(league)
    effects = state.active_rule_effects()
    starters = {"QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots, "TE": league.te_slots}
    flex_share = league.flex_slots / 3.0

    # depth: starter-tier players at each position likely (>= 50%) to still be there at pick k
    def depth_at(k: int) -> Dict[str, int]:
        out = {}
        for pos, n_slots in starters.items():
            n = int(league.num_teams * (n_slots + (flex_share if pos != "QB" else 0)))
            tier = [p for p in pool if p.position == pos and p.draftable][:n]
            out[pos] = sum(1 for p in tier if p.uid not in taken and p.p_available(k) >= 0.5)
        return out

    mine_events = [e for e in state.events if e.mine]
    rows: List[dict] = []
    # completed rounds
    for i, e in enumerate(mine_events):
        p = pool_by_uid.get(e.player_uid)
        rows.append({"round": i + 1, "pick": e.pick_no, "state": "done", "pos": e.position,
                     "player": e.player_name, "pts": round(p.raw or p.proj) if p else None})
    # the engine's view from the current pick forward
    results = evaluate_candidates(pool, roster, current_pick, future, league, taken, top_n=6) if current_pick <= picks[-1] else []
    best = results[0] if results else None
    votes: Dict[int, Dict[str, List[float]]] = {}
    for r in results:
        for j, (pos, pts) in enumerate(r.plan):
            votes.setdefault(j, {}).setdefault(pos, []).append(pts)
    if best is not None:
        rows.append({"round": rnd, "pick": current_pick, "state": "now", "pos": best.player.position,
                     "player": best.player.name, "pts": round(best.player.raw or best.player.proj),
                     "alt": results[1].player.position if len(results) > 1 and results[1].player.position != best.player.position else None,
                     "depth": depth_at(current_pick), "rules": _round_rules(effects, rnd)})
    for j, k in enumerate(future):
        v = votes.get(j, {})
        tpos: Optional[str] = None
        talt: Optional[str] = None
        pts_list: List[float] = []
        agree = 0.0
        if v:
            ranked = sorted(v.items(), key=lambda kv: (-len(kv[1]), -max(kv[1])))
            tpos, pts_list = ranked[0]
            talt = ranked[1][0] if len(ranked) > 1 else None
            agree = len(pts_list) / max(len(results), 1)
        r_no = rnd + 1 + j
        rows.append({"round": r_no, "pick": k, "state": "plan", "pos": tpos,
                     "pts": round(sum(pts_list) / len(pts_list)) if pts_list else None,
                     "agree": round(agree, 2), "alt": talt, "depth": depth_at(k),
                     "rules": _round_rules(effects, r_no)})
    return {"rows": rows, "current_round": rnd, "positions": list(starters),
            "starters": starters, "roster_size": league.roster_size}


def _round_rules(effects: Dict[str, list], rnd: int) -> List[dict]:
    out = []
    for pos, until in effects.get("wait", []):
        if rnd < until:
            out.append({"type": "wait", "pos": pos, "text": f"no {pos} until R{until}"})
    for pos, before in effects.get("ban", []):
        if rnd < before:
            out.append({"type": "ban", "pos": pos, "text": f"no {pos} before R{before}"})
    for pos, n, by in effects.get("need", []):
        if rnd <= by:
            out.append({"type": "need", "pos": pos, "text": f"{n} {pos} by R{by}"})
    for cap in effects.get("bye_cap", []):
        out.append({"type": "wait", "pos": "*", "text": f"bye stack ≤ {cap}"})
    for pos, age, before in effects.get("age", []):
        if rnd < before:
            out.append({"type": "ban", "pos": pos, "text": f"no {pos} {age}+ before R{before}"})
    return out


# ------------------------------------------------------------ player card

def card_payload(settings: Settings, uid: str) -> dict:
    from .data.names import name_key
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
    games_proj = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    sp = build_sim_players(conn, [player], settings.current_season,
                           league.scoring_format, proj_rank, games_proj)[0]
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
    taken_event = next((e for e in state.events if e.player_uid == uid), None)
    return {
        "uid": uid, "name": player.name, "pos": player.position, "team": player.team,
        "taken": taken_event is not None,
        "status": status_chip(player.status),
        "status_detail": player.status,
        "taken_pick": taken_event.pick_no if taken_event else None,
        "taken_by": ("you" if taken_event.mine else f"team {taken_event.team}" if taken_event.team else "another team") if taken_event else None,
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
        "badges": getattr(player, "badges", []),
        "favorite": name_key(player.name) in favorite_keys(state),
        "facts": (ctx or {}).get("facts", []),
        "team_facts": (ctx or {}).get("team_facts", []),
        "playoff_sos": [{"week": r["week"], "value": round(r["value"], 1)} for r in sos_rows],
        "trend": [{"season": r["season"], "games": r["g"], "ppg": r["ppg"]}
                  for r in reversed(trend)],
        "games": games,
        "milestones": milestones,
    }


# ------------------------------------------------------------ TEAM report

def resolve_team(q: str) -> str:
    """'PHI', 'phi', 'eagles', 'Philadelphia Eagles', 'philly' -> 'PHI'."""
    from .graph import TEAM_NAMES

    text = (q or "").strip()
    if text.upper() in TEAM_NAMES:
        return text.upper()
    low = text.lower()
    aliases = {"philly": "PHI", "niners": "SF", "49ers": "SF", "jags": "JAX", "bucs": "TB",
               "pats": "NE", "skins": "WAS", "commanders": "WAS", "rams": "LA", "chargers": "LAC",
               "raiders": "LV", "jets": "NYJ", "giants": "NYG"}
    if low in aliases:
        return aliases[low]
    hits = [code for code, name in TEAM_NAMES.items()
            if low and (low in name.lower() or name.lower().split()[-1] == low)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"Unknown team {q!r}" + (f" (matches {', '.join(sorted(hits))})" if hits else ""))


def team_payload(settings: Settings, state: DraftState, team: str) -> dict:
    """Everything the TEAM tab shows: header facts, the official depth chart
    fused with projections / ADP / last-season share / status / researched
    role, vacated volume per room, engine view at the user's next pick, and
    research freshness."""
    import json as _json

    from .depth import team_depth
    from .graph import TEAM_NAMES, ensure_graph_schema
    from .status import research_coverage
    from .value.distributions import team_bye_weeks

    team = resolve_team(team)
    league = settings.league
    conn = _conn(settings)
    ensure_graph_schema(conn)
    pool = _pool(settings, conn)
    by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    rnd, current_pick, next_pick, _future = _pick_context(settings, state, league)
    # engine view: where these players sit on the current board
    board = {r["uid"]: i + 1 for i, r in enumerate(
        pick_payload(settings, state, sims=0, top_n=25)["rows"])}

    facts = [dict(r) for r in conn.execute(
        "SELECT entity_id, kind, text, value, confidence, source, created FROM facts "
        "WHERE entity_id = ? OR entity_id LIKE ? ORDER BY id DESC LIMIT 40",
        (f"team:{team}", f"unit:{team}-%"))]
    # newest first; keep the latest claim per (entity, kind) so repeated
    # research runs read as one report, not a history
    seen_kinds = set()
    deduped = []
    for f in facts:
        key = (f["entity_id"], f["kind"], f["source"] == "derived")
        if key in seen_kinds:
            continue
        seen_kinds.add(key)
        deduped.append(f)
    team_facts = [f for f in deduped if f["entity_id"] == f"team:{team}"]
    unit_facts: Dict[str, list] = {}
    for f in deduped:
        if f["entity_id"].startswith("unit:"):
            unit_facts.setdefault(f["entity_id"].rsplit("-", 1)[1], []).append(f)
    shares = {}
    for r in conn.execute(
        "SELECT src, value, attrs FROM edges WHERE kind = 'in_room' AND dst LIKE ?", (f"unit:{team}-%",)):
        attrs = _json.loads(r["attrs"]) if r["attrs"] else {}
        last_team = attrs.get("last_team")
        shares[r["src"].split(":", 1)[1]] = {
            "share": round(r["value"], 3) if r["value"] is not None else None,
            "other_team": last_team if (last_team and last_team != team) else None,
            "targets_last": attrs.get("targets_last"), "carries_last": attrs.get("carries_last")}
    from .status import current_status

    status_rows = current_status(conn, settings.current_season, include_active=True)
    depth = team_depth(conn, settings.current_season, team)
    rooms = {}
    for pos in ("QB", "RB", "WR", "TE"):
        rows = []
        seen = set()
        for d in depth.get(pos, []):
            p = by_uid.get(d["uid"]) if d["uid"] else None
            st = status_rows.get(d["uid"]) if d["uid"] else None
            role = (st or {}).get("role")
            official = "starter" if d["rank"] == 1 or (pos == "WR" and d["rank"] <= 3) else "depth"
            rows.append({
                "rank": d["rank"], "slot": d["slot"], "name": p.name if p else d["name"], "uid": d["uid"],
                "proj": round(p.raw or p.proj) if p else None, "value": round(p.proj) if p else None,
                "adp": p.adp if p else None, "taken": bool(d["uid"] and d["uid"] in taken),
                "avail_next": round(p.p_available(next_pick), 2) if p and p.adp else None,
                "board_rank": board.get(d["uid"]), "status": status_chip(st),
                "role": role if role and role != "unknown" else None,
                "role_disagrees": bool(role and role != "unknown" and (role in ("backup", "depth")) != (official == "depth")),
                **shares.get(d["uid"] or "", {"share": None, "other_team": None}),
            })
            if d["uid"]:
                seen.add(d["uid"])
        # projected players the official chart does not list (cut / unsigned / rookies not yet slotted)
        for p in pool:
            if p.team == team and p.position == pos and p.uid not in seen:
                st = status_rows.get(p.uid)
                rows.append({"rank": None, "slot": None, "name": p.name, "uid": p.uid,
                             "proj": round(p.raw or p.proj), "value": round(p.proj), "adp": p.adp,
                             "taken": p.uid in taken, "avail_next": round(p.p_available(next_pick), 2) if p.adp else None,
                             "board_rank": board.get(p.uid), "status": status_chip(st),
                             "role": (st or {}).get("role") if (st or {}).get("role") not in (None, "unknown") else None,
                             "role_disagrees": False, **shares.get(p.uid, {"share": None, "other_team": None})})
        vac = conn.execute(
            "SELECT value, text FROM facts WHERE entity_id = ? AND kind = 'vacated_share' ORDER BY id DESC LIMIT 1",
            (f"unit:{team}-{pos}",)).fetchone()
        rooms[pos] = {"rows": rows, "vacated": round(vac["value"], 3) if vac else None,
                      "facts": [f for f in unit_facts.get(pos, []) if f["kind"] != "vacated_share"][:4]}
    ol = unit_facts.get("OL", [])
    sos: Dict[str, list] = {}
    for r in conn.execute(
        "SELECT position, week, value FROM sos WHERE season = ? AND team = ? AND week BETWEEN 15 AND 17 ORDER BY position, week",
        (settings.current_season, team)):
        sos.setdefault(r["position"], []).append({"week": r["week"], "value": round(r["value"], 1)})
    coverage = next((c for c in research_coverage(conn, settings.current_season) if c["team"] == team), None)
    dt = next((d[0]["dt"] for d in depth.values() if d), None)
    bye = team_bye_weeks(conn, settings.current_season).get(team)
    conn.close()
    return {
        "team": team, "name": TEAM_NAMES[team], "bye": bye,
        "depth_as_of": dt, "current_pick": current_pick, "next_pick": next_pick,
        "team_facts": team_facts[:8], "ol_facts": ol[:3], "rooms": rooms, "playoff_sos": sos,
        "coverage": coverage,
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
        inert = not any(fx[k] for k in ("targets", "wait", "ban", "need", "bye_cap", "age"))
        rows.append({"text": r.text, "on": r.on, "inert": inert})
    return {"rules": rows, "notes": state.notes}


def update_strategy(settings: Settings,
                    rules: Optional[List[dict]] = None,
                    notes: Optional[str] = None) -> dict:
    from .state import Rule, reconcile_rules

    conflicts: List[str] = []
    with state_lock(settings):
        state = DraftState.load(settings)
        if rules is not None:
            parsed = [Rule(text=str(r["text"])[:120], on=bool(r.get("on", True)))
                      for r in rules[:20]]
            state.rules, conflicts = reconcile_rules(parsed)
        if notes is not None:
            state.notes = str(notes)[:8000]
        state.save(settings)
    payload = strategy_payload(state)
    payload["conflicts"] = conflicts
    return payload


def toggle_favorite(settings: Settings, name: str) -> dict:
    """Star or unstar a player. A favorite IS a Target rule — it lands in the
    strategy sheet, survives a draft reset the way every rule does, and gets
    the same "take him when it's close" treatment from the ranking layer."""
    from .data.names import name_key
    from .state import Rule, reconcile_rules

    name = str(name).strip()[:80]
    if not name:
        raise ValueError("a favorite needs a player name")
    key = name_key(name)
    with state_lock(settings):
        state = DraftState.load(settings)
        existing = [r for r in state.rules
                    if r.text.lower().startswith("target") and key == name_key(r.text.split(":", 1)[-1])]
        if existing:
            state.rules = [r for r in state.rules if r not in existing]
            on = False
        else:
            state.rules, _ = reconcile_rules(state.rules + [Rule(text=f"Target: {name}")])
            on = True
        state.save(settings)
    payload = strategy_payload(state)
    payload["favorite"] = on
    payload["name"] = name
    return payload


def favorite_keys(state: DraftState) -> set:
    """name_keys of every starred player, for rendering the star filled."""
    from .data.names import name_key

    return {name_key(r.text.split(":", 1)[-1]) for r in state.rules
            if r.on and r.text.lower().startswith("target")}


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
    params: Tuple[Any, ...]
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


def roster_sim_payload(settings: Settings, state: DraftState, n_sims: int = 300) -> RosterSimPayload:
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
    assert res.samples is not None  # only an empty roster yields no samples, handled above
    return {
        "players": [p.name for p in roster],
        "samples": [round(float(x), 1) for x in res.samples],
        "mean": round(res.mean), "p10": round(res.p10), "p90": round(res.p90),
    }


# ------------------------------------------------------------ season grid (ROSTER tab)

GRID_LEVELS = ((1.10, "green"), (0.85, "yellow"), (0.0, "red"))


def season_grid_payload(settings: Settings, state: DraftState) -> dict:
    """Week-by-week heatmap of the roster: one row per starting slot, one
    column per week 1-17. Each cell is the player the weekly-optimal lineup
    would start there (byes, known games out and matchup-adjusted weekly
    means all applied), colored by his expected points relative to a
    league-average starter at that position. Grey = nobody to start."""
    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    return season_grid_for(settings, pool, roster)


def season_grid_for(settings: Settings, pool: List[PoolPlayer], roster: List[PoolPlayer]) -> dict:
    """The grid for any roster drawn from `pool` (the coach scores simulated
    drafts with it)."""
    from .value.distributions import build_sim_players
    from .value.lineup import FLEX_ELIGIBLE
    from .value.simulate import FANTASY_WEEKS

    league = settings.league
    conn = _conn(settings)
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    # league-average starter, per position, in weekly points: the mean of the
    # top (teams x slots [+ flex share]) projections / 17
    slots = {"QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
             "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots}
    baseline: Dict[str, float] = {}
    for pos, n_slots in slots.items():
        n = league.num_teams * n_slots + (league.num_teams * league.flex_slots // 3 if pos in FLEX_ELIGIBLE else 0)
        top = [p.raw or p.proj for p in pool if p.position == pos][:max(n, 1)]
        baseline[pos] = (sum(top) / len(top) / FANTASY_WEEKS) if top else 1.0
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    sps = build_sim_players(conn, roster, settings.current_season, league.scoring_format,
                            proj_rank, games) if roster else []
    conn.close()

    # expected points per player per week; known games out are taken from week 1
    weekly: List[List[float]] = []
    byes: List[Optional[int]] = []
    out_until: List[int] = []
    for p, sp in zip(roster, sps):
        st = p.status or {}
        miss = int(st.get("games_out") or 0) if st.get("status") not in ("out_season", "released", "retired") else FANTASY_WEEKS
        out_until.append(miss)
        byes.append(sp.bye_week)
        # weekly_mu already divides the season by played weeks; use the raw
        # projection's per-week rate so a player out N games isn't inflated
        rate = (p.raw or p.proj) / FANTASY_WEEKS
        wk: List[float] = []
        for w in range(1, FANTASY_WEEKS + 1):
            if (sp.bye_week and w == sp.bye_week) or w <= miss:
                wk.append(0.0)
            else:
                wk.append(rate * float(sp.sos_mult[w - 1]))
        weekly.append(wk)

    slot_rows: List[dict] = []
    for pos, n_slots in slots.items():
        for k in range(n_slots):
            slot_rows.append({"slot": pos if n_slots == 1 else f"{pos}{k + 1}", "pos": pos, "cells": []})
    for k in range(league.flex_slots):
        slot_rows.append({"slot": "FLEX" if league.flex_slots == 1 else f"FLEX{k + 1}", "pos": "FLEX", "cells": []})

    def level(ratio: float) -> str:
        for cut, name in GRID_LEVELS:
            if ratio >= cut:
                return name
        return "grey"

    week_totals = []
    for w in range(FANTASY_WEEKS):
        order = sorted(range(len(roster)), key=lambda i: -weekly[i][w])
        used = set()
        assigned: Dict[str, List[int]] = {pos: [] for pos in slots}
        flex: List[int] = []
        for i in order:
            if weekly[i][w] <= 0:
                break
            pos = roster[i].position
            if len(assigned[pos]) < slots[pos]:
                assigned[pos].append(i); used.add(i)
        for i in order:
            if len(flex) >= league.flex_slots or weekly[i][w] <= 0:
                break
            if i not in used and roster[i].position in FLEX_ELIGIBLE:
                flex.append(i); used.add(i)
        total = 0.0
        for row in slot_rows:
            pos = row["pos"]
            idx = int(row["slot"][len(pos):] or 1) - 1 if pos != "FLEX" else (0 if row["slot"] == "FLEX" else int(row["slot"][4:]) - 1)
            picks = flex if pos == "FLEX" else assigned[pos]
            if idx < len(picks):
                i = picks[idx]
                pts = weekly[i][w]
                base = baseline[roster[i].position]
                ratio = pts / base if base else 0.0
                total += pts
                row["cells"].append({"week": w + 1, "name": roster[i].name, "pts": round(pts, 1),
                                     "ratio": round(ratio, 2), "level": level(ratio),
                                     "sub": roster[i].position != pos and pos != "FLEX"})
            else:
                # why empty: everyone at this position is on bye / out, or the slot was never drafted
                holders = [i for i in range(len(roster)) if roster[i].position == pos or (pos == "FLEX" and roster[i].position in FLEX_ELIGIBLE)]
                reason = "not drafted" if not holders else ("bye" if any(byes[i] == w + 1 for i in holders) else "out")
                row["cells"].append({"week": w + 1, "name": None, "pts": 0, "ratio": 0, "level": "grey", "reason": reason})
        bench = [i for i in range(len(roster)) if i not in used and weekly[i][w] > 0]
        on_bye = [roster[i].name for i in range(len(roster)) if byes[i] == w + 1]
        out = [roster[i].name for i in range(len(roster)) if w + 1 <= out_until[i]]
        league_avg = sum(baseline[pos] * n for pos, n in slots.items()) + league.flex_slots * (baseline["RB"] + baseline["WR"]) / 2
        week_totals.append({"week": w + 1, "pts": round(total, 1), "ratio": round(total / league_avg, 2) if league_avg else 0,
                            "level": level(total / league_avg) if league_avg else "grey",
                            "bench": len(bench), "bench_by_pos": {pos: sum(1 for i in bench if roster[i].position == pos) for pos in ("QB", "RB", "WR", "TE")},
                            "bye": on_bye, "out": out})
    return {"weeks": list(range(1, FANTASY_WEEKS + 1)), "rows": slot_rows, "week_totals": week_totals,
            "baseline": {pos: round(v, 1) for pos, v in baseline.items()},
            "players": len(roster), "legend": {"green": "≥ 110% of a league-average starter", "yellow": "85–110%", "red": "< 85%", "grey": "nobody to start (bye / out / not drafted)"}}


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
        import json as _json
        for r in conn.execute(
            "SELECT e.name, e.position, ed.value AS share, ed.attrs FROM edges ed JOIN entities e ON e.id = ed.src "
            "WHERE ed.kind = 'in_room' AND ed.dst LIKE ? ORDER BY e.position, ed.value DESC", (f"unit:{team}-%",)):
            pp = by_name.get(r["name"])
            attrs = _json.loads(r["attrs"]) if r["attrs"] else {}
            lt = attrs.get("last_team")
            detail["rooms"].append({"name": r["name"], "position": r["position"], "share": r["share"],
                                    "other_team": lt if (lt and lt != team) else None,
                                    "proj": round(pp.raw or pp.proj) if pp else None,
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


# ------------------------------------------------------------ roster risk (collapsible rail)

def roster_risk(settings: Settings, state: DraftState) -> dict:
    """Per position: are we at risk? 'warn' = a strategy rule says we should
    already have this (or we're past the wait window with nothing), 'danger' =
    the board is thin for an empty starting slot (few players likely at the
    next pick within 80% of the best available now)."""
    league = settings.league
    conn = _conn(settings)
    pool = _pool(settings, conn)
    conn.close()
    pool_by_uid = {p.uid: p for p in pool}
    taken = state.taken_uids()
    roster = [pool_by_uid[u] for u in state.my_uids(league) if u in pool_by_uid]
    rnd, current_pick, next_pick, future = _pick_context(settings, state, league)
    counts: Dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    starters = {"QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
                "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots}
    effects = state.active_rule_effects()
    out: Dict[str, dict] = {}
    for pos, need_slots in starters.items():
        have = counts.get(pos, 0)
        level, reasons = "ok", []
        for rpos, n, by_round in effects["need"]:
            if rpos == pos and rnd >= by_round and have < n:
                level = "warn"; reasons.append(f"rule: {n} {pos} by R{by_round} — have {have}")
        for wpos, until in effects["wait"]:
            if wpos == pos and rnd > until + 2 and have == 0:
                level = "warn"; reasons.append(f"waited on {pos} past R{until} — still none")
        empty = max(need_slots - have, 0)
        if empty and pos in ("QB", "RB", "WR", "TE"):
            # THIN means: starter-grade players (>= the league's replacement
            # level at the position, NOT the top tier — the top tier never
            # survives, that's just a draft) may not cover the empty slots
            # at your next pick.
            pos_pool = [p for p in pool if p.position == pos]
            n_starters = league.num_teams * need_slots + (league.num_teams * league.flex_slots // 3 if pos in ("RB", "WR", "TE") else 0)
            replacement = pos_pool[n_starters - 1].proj if len(pos_pool) >= n_starters else 0
            startable_next = [p for p in pos_pool if p.uid not in taken
                              and p.proj >= replacement and p.p_available(next_pick) >= 0.5]
            if len(startable_next) < empty:
                level = "danger"
                reasons.append(f"only {len(startable_next)} starter-grade {pos} likely left at pick {next_pick} for {empty} empty slot{'s' if empty > 1 else ''}")
            elif len(startable_next) < 2 * empty and level == "ok":
                level = "warn"
                reasons.append(f"{pos} getting thin: {len(startable_next)} starter-grade options likely at {next_pick}")
        out[pos] = {"level": level, "have": have, "need": need_slots, "reasons": reasons}
    summary = [f"{pos} {'⚠' if v['level'] == 'warn' else '✖'} {v['reasons'][0]}"
               for pos, v in out.items() if v["level"] != "ok"]
    return {"round": rnd, "positions": out, "summary": summary}


def ask_howie(settings: Settings, question: str) -> dict:
    """Run the in-repo agent synchronously; returns answer + tool trace."""
    from .agent import AgentEventType, run_agent_events

    answer, tools, errors = [], [], []
    for ev in run_agent_events(question, settings):
        if ev.kind == AgentEventType.TEXT:
            answer.append(ev.text)
        elif ev.kind == AgentEventType.TOOL_CALL:
            tools.append(ev.text)
        elif ev.kind in (AgentEventType.ERROR, AgentEventType.STOP):
            errors.append(ev.text)
    return {"answer": "\n".join(answer).strip(), "tools": tools, "notes": errors}


# ------------------------------------------------------------ league config (header ⚙)

CONFIG_FIELDS = ("num_teams", "draft_position", "scoring_type", "qb_slots", "rb_slots", "wr_slots",
                 "te_slots", "flex_slots", "k_slots", "dst_slots", "bench_slots", "roster_size",
                 "market_anchor", "playoff_weight")


def config_payload(settings: Settings) -> dict:
    lc = settings.league
    return {f: getattr(lc, f) for f in CONFIG_FIELDS}


def update_config(settings: Settings, values: dict) -> dict:
    """Validate through LeagueConfig and write data/league_config.json."""
    import json as _json
    from dataclasses import replace as dc_replace

    current = settings.league
    clean: Dict[str, Any] = {}
    for f in CONFIG_FIELDS:
        if f in values and values[f] is not None and values[f] != "":
            v = values[f]
            if f == "scoring_type":
                clean[f] = str(v)
            elif f in ("market_anchor", "playoff_weight"):
                clean[f] = float(v)
            else:
                clean[f] = int(v)
    new = dc_replace(current, **clean)
    new.validate()  # raises ValueError with a clear message
    path = settings.data_dir / "league_config.json"
    path.write_text(_json.dumps({f: getattr(new, f) for f in CONFIG_FIELDS}, indent=2))
    return config_payload(settings)
