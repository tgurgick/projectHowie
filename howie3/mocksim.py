"""Mock Draft Lab: batches of simulated drafts that teach availability.

Runs local mock drafts (ADP-noise bots for every other team, a chosen policy
for the user's picks), accepts imported external drafts (ESPN/Sleeper mock
results pasted as a pick order), and persists every draft's pick order to
data/mock_sims.json. Aggregates answer the draft-prep question the engine's
ADP model only estimates: at each of MY picks, who is usually still there?
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Settings
from .state import snake_team_for_pick
from .value.board import PoolPlayer, snake_picks

STATUS: Dict[str, object] = {"running": False, "done": 0, "total": 0, "error": None}
MIN_DRAFTS = 10  # below this the lab has nothing the ADP model doesn't


def store_path(settings: Settings) -> Path:
    return settings.data_dir / "mock_sims.json"


def load_store(settings: Settings) -> dict:
    p = store_path(settings)
    if p.exists():
        return json.loads(p.read_text())
    return {"drafts": [], "runs": []}


def save_store(settings: Settings, store: dict) -> None:
    p = store_path(settings)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store))
    os.replace(tmp, p)


def _user_pick(policy: str, pool: List[PoolPlayer], taken: set, roster: List[PoolPlayer],
               pick_no: int, future: List[int], league, rng) -> Optional[PoolPlayer]:
    from .mock import bot_pick
    from .value.roster import evaluate_candidates

    if policy == "howie":
        res = evaluate_candidates(pool, roster, pick_no, future, league, frozenset(taken), top_n=1)
        if res:
            return res[0].player
    tp: Dict[str, int] = {}
    for p in roster:
        tp[p.position] = tp.get(p.position, 0) + 1
    return bot_pick(pool, frozenset(taken), tp, (pick_no - 1) // league.num_teams + 1, league, rng)


def run_mock_drafts(settings: Settings, n: int, policy: str = "adp",
                    seed: Optional[int] = None) -> dict:
    """Run n local drafts, persisting each; returns the refreshed aggregates."""
    from .db import connect
    from .mock import bot_pick
    from .value.board import load_pool

    league = settings.league
    conn = connect(settings.db_path)
    pool = load_pool(conn, settings.current_season, league.scoring_format,
                     market_anchor=league.market_anchor)
    conn.close()
    me = league.draft_position
    total = league.num_teams * league.roster_size
    my_picks = snake_picks(league)
    base_seed = seed if seed is not None else int(datetime.now(timezone.utc).timestamp()) % 10**6

    STATUS.update({"running": True, "done": 0, "total": n, "error": None})
    store = load_store(settings)
    try:
        for d in range(n):
            taken: set = set()
            picks: List[str] = []
            roster: List[PoolPlayer] = []
            team_positions: Dict[int, Dict[str, int]] = {}
            recent: List[str] = []
            for pick_no in range(1, total + 1):
                team = snake_team_for_pick(league, pick_no)
                rng = np.random.default_rng((base_seed + d) * 100_000 + pick_no)
                if team == me:
                    future = [k for k in my_picks if k > pick_no]
                    choice = _user_pick(policy, pool, taken, roster, pick_no, future, league, rng)
                    if choice:
                        roster.append(choice)
                else:
                    tp = team_positions.setdefault(team, {})
                    choice = bot_pick(pool, frozenset(taken), tp,
                                      (pick_no - 1) // league.num_teams + 1, league, rng,
                                      recent_positions=recent)
                    if choice:
                        tp[choice.position] = tp.get(choice.position, 0) + 1
                if choice is None:
                    break
                taken.add(choice.uid)
                picks.append(choice.uid)
                recent = (recent + [choice.position])[-5:]
            store["drafts"].append({"source": "local", "policy": policy,
                                    "seed": base_seed + d, "picks": picks})
            STATUS["done"] = d + 1
            if (d + 1) % 5 == 0 or d + 1 == n:
                save_store(settings, store)
        store["runs"].append({"ts": _now(), "n": n, "policy": policy, "seed": base_seed})
        save_store(settings, store)
    except Exception as e:  # surfaced through status; partial results are kept
        STATUS["error"] = f"{e.__class__.__name__}: {e}"
        save_store(settings, store)
    finally:
        STATUS["running"] = False
    return aggregates(settings)


def run_in_background(settings: Settings, n: int, policy: str = "adp") -> bool:
    if STATUS["running"]:
        return False
    threading.Thread(target=run_mock_drafts, args=(settings, n, policy), daemon=True).start()
    return True


def import_external(settings: Settings, text: str, source: str = "external") -> dict:
    """Parse a pasted pick order (one player per line, optional leading
    numbers) and store it as an external draft. Unresolved names are reported,
    never silently dropped."""
    from .data.names import name_key
    from .db import connect
    from .value.board import load_pool

    conn = connect(settings.db_path)
    pool = load_pool(conn, settings.current_season, settings.league.scoring_format)
    conn.close()
    by_key: Dict[str, str] = {}
    for p in pool:
        by_key.setdefault(name_key(p.name), p.uid)
    picks, unresolved = [], []
    for line in text.splitlines():
        raw = re.sub(r"^\s*[\d]+[\.\)\-:\s]*", "", line).strip()
        raw = re.split(r"\s{2,}|\t|,\s*(?:QB|RB|WR|TE|K|DST)\b", raw)[0].strip()
        if not raw:
            continue
        uid = by_key.get(name_key(raw))
        if uid is None:
            m = re.match(r"^(.*?)\s+(?:QB|RB|WR|TE|K|DST|D/ST)\b", raw)
            if m:
                uid = by_key.get(name_key(m.group(1)))
        if uid is None:
            unresolved.append(raw)
        else:
            picks.append(uid)
    if len(picks) < 12:
        raise ValueError(f"Need at least 12 resolved picks, got {len(picks)} "
                         f"(unresolved: {unresolved[:5]})")
    store = load_store(settings)
    store["drafts"].append({"source": source, "policy": None, "seed": None, "picks": picks})
    save_store(settings, store)
    return {"stored": len(picks), "unresolved": unresolved, "drafts": len(store["drafts"])}


_AVAIL_CACHE: Dict[str, object] = {"sig": None, "table": {}}


def availability_table(settings: Settings) -> Dict[str, Dict[int, Tuple[float, int]]]:
    """uid -> {my_pick -> (availability rate, n drafts)} over every stored
    draft, for the engine's p_available blend. Cached on the store's mtime;
    empty when fewer than MIN_DRAFTS drafts exist."""
    p = store_path(settings)
    if not p.exists():
        return {}
    sig = (p.stat().st_mtime_ns, p.stat().st_size, settings.league.draft_position, settings.league.num_teams)
    if _AVAIL_CACHE["sig"] == sig:
        return _AVAIL_CACHE["table"]  # type: ignore[return-value]
    drafts = [d["picks"] for d in load_store(settings)["drafts"] if d["picks"]]
    table: Dict[str, Dict[int, Tuple[float, int]]] = {}
    if len(drafts) >= MIN_DRAFTS:
        for k in snake_picks(settings.league):
            eligible = [set(d[:k - 1]) for d in drafts if len(d) >= k - 1]
            if len(eligible) < MIN_DRAFTS:
                continue
            seen: Dict[str, int] = {}
            for gone in eligible:
                for uid in gone:
                    seen[uid] = seen.get(uid, 0) + 1
            n = len(eligible)
            for uid, cnt in seen.items():
                table.setdefault(uid, {})[k] = (round(1 - cnt / n, 3), n)
    _AVAIL_CACHE.update({"sig": sig, "table": table})
    return table


def aggregates(settings: Settings) -> dict:
    """Per user pick: availability of every market-relevant player across all
    stored drafts, next to the engine's ADP-model estimate; plus sim-ADP."""
    from .db import connect
    from .value.board import load_pool

    league = settings.league
    store = load_store(settings)
    drafts = [d["picks"] for d in store["drafts"] if d["picks"]]
    conn = connect(settings.db_path)
    pool = load_pool(conn, settings.current_season, league.scoring_format,
                     market_anchor=league.market_anchor)
    conn.close()
    by_uid = {p.uid: p for p in pool}
    my_picks = snake_picks(league)

    # sim ADP
    pos_sum: Dict[str, float] = {}
    pos_n: Dict[str, int] = {}
    for picks in drafts:
        for i, uid in enumerate(picks):
            pos_sum[uid] = pos_sum.get(uid, 0.0) + (i + 1)
            pos_n[uid] = pos_n.get(uid, 0) + 1
    sim_adp = {uid: pos_sum[uid] / pos_n[uid] for uid in pos_sum}

    per_pick = {}
    for k in my_picks:
        eligible = [d for d in drafts if len(d) >= k - 1]
        rows: List[Dict[str, Any]] = []
        if eligible:
            for p in pool:
                if p.adp is None or p.adp > k + 80:
                    continue
                avail = sum(1 for d in eligible if p.uid not in set(d[:k - 1])) / len(eligible)
                if avail < 0.02:
                    continue
                rows.append({
                    "uid": p.uid, "name": p.name, "pos": p.position,
                    "proj": round(p.raw or p.proj), "value": round(p.proj), "adp": p.adp,
                    "sim_adp": round(sim_adp[p.uid], 1) if p.uid in sim_adp else None,
                    "avail_sim": round(avail, 3),
                    "avail_model": round(p.p_available(k), 3),
                })
            rows.sort(key=lambda r: (-r["value"] * r["avail_sim"]))
        per_pick[str(k)] = {"n_drafts": len(eligible), "rows": rows[:40]}
    return {
        "drafts": len(drafts),
        "local": sum(1 for d in store["drafts"] if d["source"] == "local"),
        "external": sum(1 for d in store["drafts"] if d["source"] != "local"),
        "runs": store["runs"][-5:],
        "my_picks": my_picks,
        "per_pick": per_pick,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
