"""Coached simulation: the engine drafts fast, Claude coaches the strategy.

One iteration:
  1. simulate N mock drafts with the engine under the current rule set
     (ADP-noise bots that run positions and fill needs), no persistence;
  2. score every roster the ways the engine cannot see — the 2026 Monte
     Carlo season (mean / floor), the season heatmap's structural holes
     (weeks with an empty starting slot, bye stacks, thin bench), and the
     SAME rule set replayed on the 2025 season against realized results
     (paired seeds vs follow-ADP, bootstrap CI);
  3. hand Claude a structured digest (positions by round, rosters, scores,
     the rule set, the user's notes) and get back structured changes: rules
     to add / remove, a note, learnings;
  4. apply them (rules are reconciled), then re-score; the best rule set by
     realized 2025 points (MC mean as tie-break) is kept at the end.

Everything is recorded in data/coach_sessions.json so the LAB tab can show
the optimization trace and the cockpit's round-by-round reflects the result.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings
from .state import DraftState, Rule, reconcile_rules, state_lock

STATUS: Dict[str, object] = {"running": False, "phase": "", "iteration": 0, "total": 0, "error": None}


def store_path(settings: Settings) -> Path:
    return settings.data_dir / "coach_sessions.json"


def load_sessions(settings: Settings) -> dict:
    p = store_path(settings)
    return json.loads(p.read_text()) if p.exists() else {"sessions": []}


def save_sessions(settings: Settings, doc: dict) -> None:
    p = store_path(settings)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, p)


# ---------------------------------------------------------------- simulate + score

def simulate(settings: Settings, effects: Dict[str, list], n: int, seed: int) -> dict:
    """N unpersisted engine drafts under `effects`; each roster scored."""
    from . import service
    from .mocksim import run_mock_drafts
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_roster

    league = settings.league
    res = run_mock_drafts(settings, n, policy="howie", seed=seed, effects=effects, persist=False)
    conn = service._conn(settings)
    pool = service._pool(settings, conn)
    by_uid = {p.uid: p for p in pool}
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
        (settings.current_season,))}
    drafts = []
    for d in res["drafts_run"]:
        roster = [by_uid[u] for u in d["mine"] if u in by_uid]
        sps = build_sim_players(conn, roster, settings.current_season, league.scoring_format, proj_rank, games)
        sim = simulate_roster(sps, league, n_sims=200, playoff_weight=league.playoff_weight)
        grid = service.season_grid_for(settings, pool, roster)
        holes = sum(1 for row in grid["rows"] for c in row["cells"] if c["name"] is None and c.get("reason") != "not drafted")
        bye_stacks = sum(1 for t in grid["week_totals"] if len(t["bye"]) >= 3)
        thin = sum(1 for t in grid["week_totals"] if t["bench"] <= 1)
        drafts.append({
            "seed": d["seed"],
            "picks": [{"round": i + 1, "pos": p.position, "name": p.name, "proj": round(p.raw or p.proj)}
                      for i, p in enumerate(roster)],
            "by_pos": {pos: sum(1 for p in roster if p.position == pos) for pos in ("QB", "RB", "WR", "TE", "K", "DST")},
            "mc_mean": round(sim.mean), "mc_p10": round(sim.p10), "mc_p90": round(sim.p90),
            "holes": holes, "bye_stacks": bye_stacks, "thin_weeks": thin,
            "worst_week": min((t["pts"] for t in grid["week_totals"]), default=0),
        })
    conn.close()
    n_d = max(len(drafts), 1)
    summary = {
        "drafts": len(drafts),
        "mc_mean": round(sum(d["mc_mean"] for d in drafts) / n_d),
        "mc_p10": round(sum(d["mc_p10"] for d in drafts) / n_d),
        "holes": round(sum(d["holes"] for d in drafts) / n_d, 2),
        "bye_stacks": round(sum(d["bye_stacks"] for d in drafts) / n_d, 2),
        "thin_weeks": round(sum(d["thin_weeks"] for d in drafts) / n_d, 2),
        "worst_week": round(sum(d["worst_week"] for d in drafts) / n_d),
        "pos_by_round": _pos_by_round(drafts),
    }
    return {"summary": summary, "drafts": drafts}


def _pos_by_round(drafts: List[dict]) -> List[dict]:
    rounds: Dict[int, Dict[str, int]] = {}
    for d in drafts:
        for pk in d["picks"]:
            rounds.setdefault(pk["round"], {})
            rounds[pk["round"]][pk["pos"]] = rounds[pk["round"]].get(pk["pos"], 0) + 1
    out = []
    for r in sorted(rounds):
        total = sum(rounds[r].values())
        out.append({"round": r, **{pos: round(n / total, 2) for pos, n in sorted(rounds[r].items(), key=lambda kv: -kv[1])}})
    return out


def replay_2025(settings: Settings, effects: Optional[Dict[str, list]], reps: int = 6) -> Optional[dict]:
    """The rule set on last season's real results: paired vs follow-ADP.
    None when the legacy preseason inputs are not available."""
    from . import evals

    try:
        players = evals.load_eval_players(settings)
    except FileNotFoundError:
        return None
    evals.set_rule_effects(effects)
    r = evals.eval_policy(settings, players, reps=reps, policies=("howie+rules", "adp"))
    row = r["howie+rules"]
    return {"mean_total": round(row["mean_total"]), "delta_vs_adp": round(row["delta_vs_adp"]),
            "ci": [round(row["ci_low"]), round(row["ci_high"])], "win_rate": row["win_rate"], "n": row["n"]}


def score(settings: Settings, rules: List[Rule], n_drafts: int, seed: int, reps: int) -> dict:
    effects = DraftState(rules=rules).active_rule_effects()
    sim = simulate(settings, effects, n_drafts, seed)
    replay = replay_2025(settings, effects, reps=reps)
    return {"sim": sim, "replay": replay, "effects": effects}


def better(a: dict, b: Optional[dict]) -> bool:
    """Is score a better than b? Realized 2025 points decide; MC mean breaks ties."""
    if b is None:
        return True
    ra, rb = a.get("replay"), b.get("replay")
    if ra and rb and ra["mean_total"] != rb["mean_total"]:
        return ra["mean_total"] > rb["mean_total"]
    return a["sim"]["summary"]["mc_mean"] > b["sim"]["summary"]["mc_mean"]


# ---------------------------------------------------------------- the coach

COACH_SYSTEM = """You are a fantasy-football draft coach reviewing simulated drafts made by a
fast, deterministic engine that follows a strategy sheet. The engine already
maximizes expected lineup points under an availability model; your job is what
it cannot see: structural holes (bye stacks, empty starting slots, thin bench),
positional timing that fails against real drafters, and — most important —
how the rule set performed when replayed on LAST season's real results.

Respond with ONLY a JSON object:
{"learnings": [3 specific sentences citing the numbers],
 "rules_add": ["WAIT QB UNTIL R7", ...],      // ONLY these exact patterns:
 "rules_remove": ["NO QB BEFORE R3", ...],    //   'WAIT <POS> UNTIL R<n>', 'NO <POS> BEFORE R<n>',
 "note": "ONE line, <= 30 words, only if it adds something new",//   '<n> <POS> BY R<n>', 'TARGET <Player Name>'
 "round_targets": {"1": "RB", "2": "WR", ...}} // optional: position per round you'd aim for
Change at most 3 rules per iteration; removing a rule that hurt is as valuable
as adding one. If the replay says the current rules beat the baseline and the
structure is sound, say so and change nothing. Never suggest a rule the
engine cannot express."""


def _coach_call(settings: Settings, digest: dict) -> dict:
    from .insights import _client, _json_block, _model

    client, err = _client()
    if client is None:
        return {"available": False, "reason": err}
    league = settings.league
    user = (f"League: {league.num_teams}-team, slot {league.draft_position}, {league.scoring_format}, "
            f"roster {league.roster_size}.\n\nDIGEST:\n{json.dumps(digest, default=str)[:16000]}")
    try:
        resp = client.messages.create(model=_model(), max_tokens=1800, system=COACH_SYSTEM,
                                      messages=[{"role": "user", "content": user}])
    except Exception as e:
        return {"available": False, "reason": f"{e.__class__.__name__}: {e}"}
    text = "".join(getattr(b, "text", "") for b in resp.content)
    parsed = _json_block(text) or {}
    return {"available": True, "model": _model(),
            "learnings": [str(x) for x in parsed.get("learnings", [])][:5],
            "rules_add": [str(x) for x in parsed.get("rules_add", [])][:3],
            "rules_remove": [str(x) for x in parsed.get("rules_remove", [])][:3],
            "note": " ".join(str(parsed.get("note", "") or "").split())[:240],
            "round_targets": parsed.get("round_targets") or {}}


RULE_OK = re.compile(r"(?i)^(WAIT (QB|RB|WR|TE|K|DST) UNTIL R\d+|NO (QB|RB|WR|TE|K|DST) BEFORE R\d+|\d+ (QB|RB|WR|TE)S? BY R\d+|TARGET .+)$")


def digest_for(settings: Settings, rules: List[Rule], notes: str, current: dict,
               baseline: Optional[dict], history: List[dict]) -> dict:
    return {
        "rules": [r.text for r in rules if r.on],
        "notes": notes[:1500],
        "this_rule_set": {"simulated_2026": current["sim"]["summary"], "replay_2025": current["replay"]},
        "no_rules_baseline": {"simulated_2026": baseline["sim"]["summary"], "replay_2025": baseline["replay"]} if baseline else None,
        "sample_drafts": [{"picks": " ".join(f"R{p['round']}:{p['pos']}" for p in d["picks"]),
                           "mc_mean": d["mc_mean"], "mc_p10": d["mc_p10"], "holes": d["holes"],
                           "bye_stacks": d["bye_stacks"], "worst_week": d["worst_week"]}
                          for d in current["sim"]["drafts"][:6]],
        "previous_iterations": [{"rules": h["rules"], "replay_mean": (h["score"].get("replay") or {}).get("mean_total"),
                                 "mc_mean": h["score"]["sim"]["summary"]["mc_mean"], "learnings": h.get("learnings", [])[:2]}
                                for h in history[-4:]],
    }


def apply_changes(settings: Settings, rules: List[Rule], notes: str, changes: dict) -> List[Rule]:
    keep = [r for r in rules if r.text.strip().upper() not in {x.strip().upper() for x in changes.get("rules_remove", [])}]
    for text in changes.get("rules_add", []):
        text = text.strip()
        if RULE_OK.match(text) and text.upper() not in {r.text.upper() for r in keep}:
            keep.append(Rule(text=text, on=True))
    new_rules, _ = reconcile_rules(keep)
    note = (changes.get("note") or "").strip()
    with state_lock(settings):
        state = DraftState.load(settings)
        state.rules = new_rules
        if note:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            state.notes = (state.notes + "\n\n" if state.notes else "") + f"— coach {stamp}: {note}"
        state.save(settings)
    return new_rules


# ---------------------------------------------------------------- the loop

def run_session(settings: Settings, iterations: int = 3, n_drafts: int = 12,
                reps: int = 6, seed: int = 101) -> dict:
    """Coach for `iterations` rounds; keep the best rule set; record everything."""
    STATUS.update({"running": True, "phase": "baseline", "iteration": 0, "total": iterations, "error": None})
    session: Dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "iterations": [],
                               "n_drafts": n_drafts, "reps": reps, "seed": seed}
    try:
        state = DraftState.load(settings)
        rules, notes = list(state.rules), state.notes
        baseline = score(settings, [], n_drafts, seed, reps)
        session["baseline"] = {"sim": baseline["sim"]["summary"], "replay": baseline["replay"]}
        best: Optional[dict] = None
        best_rules = rules
        history: List[dict] = []
        for it in range(1, iterations + 1):
            STATUS.update({"phase": "simulating", "iteration": it})
            current = score(settings, rules, n_drafts, seed + it, reps)
            record = {"iteration": it, "rules": [r.text for r in rules if r.on],
                      "score": {"sim": {"summary": current["sim"]["summary"]}, "replay": current["replay"]}}
            if better(current, best):
                best, best_rules = current, rules
                record["best"] = True
            STATUS["phase"] = "coaching"
            changes = _coach_call(settings, digest_for(settings, rules, notes, current, baseline, history))
            record["coach"] = changes
            record["learnings"] = changes.get("learnings", [])
            if not changes.get("available"):
                session["iterations"].append(record)
                session["stopped"] = changes.get("reason")
                break
            rules = apply_changes(settings, rules, notes, changes)
            notes = DraftState.load(settings).notes
            record["rules_after"] = [r.text for r in rules if r.on]
            session["iterations"].append(record)
            history.append(record)
            doc = load_sessions(settings)
            doc["sessions"] = [x for x in doc["sessions"] if x.get("started") != session["started"]] + [session]
            save_sessions(settings, doc)
        # the final rule set is also scored so the best is chosen among all
        if session["iterations"] and "stopped" not in session:
            STATUS["phase"] = "final score"
            final = score(settings, rules, n_drafts, seed + iterations + 1, reps)
            session["final"] = {"rules": [r.text for r in rules if r.on], "sim": final["sim"]["summary"], "replay": final["replay"]}
            if better(final, best):
                best, best_rules = final, rules
        with state_lock(settings):
            state = DraftState.load(settings)
            state.rules = list(best_rules)
            state.save(settings)
        session["best_rules"] = [r.text for r in best_rules if r.on]
        session["best"] = {"sim": best["sim"]["summary"], "replay": best["replay"]} if best else None
        session["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as e:
        STATUS["error"] = f"{e.__class__.__name__}: {e}"
        session["error"] = STATUS["error"]
        raise
    finally:
        doc = load_sessions(settings)
        doc["sessions"] = [x for x in doc["sessions"] if x.get("started") != session["started"]] + [session]
        save_sessions(settings, doc)
        STATUS.update({"running": False, "phase": "done"})
    return session


def run_in_background(settings: Settings, **kw) -> bool:
    if STATUS["running"]:
        return False
    threading.Thread(target=run_session, args=(settings,), kwargs=kw, daemon=True).start()
    return True


def review_draft(settings: Settings, picks: List[dict]) -> dict:
    """Coach a single finished draft (e.g. a real room observed by Claude in
    Chrome): the same digest shape with one draft."""
    from . import service
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_roster

    league = settings.league
    conn = service._conn(settings)
    pool = service._pool(settings, conn)
    by_uid = {p.uid: p for p in pool}
    roster = [by_uid[p["uid"]] for p in picks if p.get("uid") in by_uid]
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'", (settings.current_season,))}
    sps = build_sim_players(conn, roster, settings.current_season, league.scoring_format, proj_rank, games)
    sim = simulate_roster(sps, league, n_sims=300) if roster else None
    grid = service.season_grid_for(settings, pool, roster)
    conn.close()
    state = DraftState.load(settings)
    digest = {
        "rules": [r.text for r in state.rules if r.on], "notes": state.notes[:1500],
        "this_draft": {"picks": [{"round": i + 1, "pos": p.position, "name": p.name, "proj": round(p.raw or p.proj)} for i, p in enumerate(roster)],
                       "mc_mean": round(sim.mean) if sim else None, "mc_p10": round(sim.p10) if sim else None,
                       "weak_weeks": [t["week"] for t in grid["week_totals"] if t["level"] == "red"],
                       "bye_stacks": [t["week"] for t in grid["week_totals"] if len(t["bye"]) >= 3]},
    }
    return {"digest": digest, "coach": _coach_call(settings, digest)}
