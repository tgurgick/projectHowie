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


_EVAL_PLAYERS: Dict[str, Any] = {}


def replay_2025(settings: Settings, effects: Optional[Dict[str, list]], reps: int = 6) -> Optional[dict]:
    """The rule set on last season's real results, replayed with the SAME
    seeded opponents for every rule set (common random numbers): per-replay
    realized points (paired by slot/rep) plus the summary vs follow-ADP.
    None when the legacy preseason inputs are not available."""
    from dataclasses import replace as dc_replace

    from . import evals
    from .value.board import apply_market_anchor

    try:
        players = _EVAL_PLAYERS.get("players") or evals.load_eval_players(settings)
        _EVAL_PLAYERS["players"] = players
    except FileNotFoundError:
        return None
    evals.set_rule_effects(effects)
    base_league = settings.league
    pool = [p for p in evals._top_pool(players) if p.adp is not None]
    anchored = {p.uid: p.proj for p in apply_market_anchor([evals._to_pool_player(p) for p in pool], base_league.market_anchor)}
    mine, adp = [], []
    for slot in evals.EVAL_SLOTS:
        league = dc_replace(base_league, draft_position=slot, k_slots=0, dst_slots=0, roster_size=evals.EVAL_ROSTER_SIZE)
        for rep in range(reps):
            r1, _ = evals.replay_draft("howie+rules", pool, league, slot, rep, anchored=anchored)
            r2, _ = evals.replay_draft("adp", pool, league, slot, rep, anchored=anchored)
            mine.append(evals._score_roster(r1, league)); adp.append(evals._score_roster(r2, league))
    summ = evals.summarize_paired({"howie+rules": mine, "adp": adp})["howie+rules"]
    return {"mean_total": round(summ["mean_total"]), "delta_vs_adp": round(summ["delta_vs_adp"]),
            "ci": [round(summ["ci_low"]), round(summ["ci_high"])], "win_rate": summ["win_rate"], "n": summ["n"],
            "scores": [round(x, 1) for x in mine]}


def score(settings: Settings, rules: List[Rule], n_drafts: int, seed: int, reps: int) -> dict:
    effects = DraftState(rules=rules).active_rule_effects()
    sim = simulate(settings, effects, n_drafts, seed)
    replay = replay_2025(settings, effects, reps=reps)
    return {"sim": sim, "replay": replay, "effects": effects}


def paired_gain(a: dict, b: dict) -> dict:
    """Candidate a vs incumbent b on common random numbers: mean paired
    difference and a 95% bootstrap CI, for the 2025 replay (realized points)
    and the 2026 simulation (MC mean per draft, same seeds)."""
    from .evals import bootstrap_mean_ci

    out: Dict[str, Any] = {}
    ra, rb = a.get("replay"), b.get("replay")
    if ra and rb and ra.get("scores") and rb.get("scores") and len(ra["scores"]) == len(rb["scores"]):
        d = [x - y for x, y in zip(ra["scores"], rb["scores"])]
        lo, hi = bootstrap_mean_ci(d)
        out["replay"] = {"mean": round(sum(d) / len(d)), "ci": [round(lo), round(hi)], "n": len(d)}
    da, db = a["sim"]["drafts"], b["sim"]["drafts"]
    if da and db and len(da) == len(db):
        d = [x["mc_mean"] - y["mc_mean"] for x, y in zip(da, db)]
        lo, hi = bootstrap_mean_ci(d)
        out["sim"] = {"mean": round(sum(d) / len(d)), "ci": [round(lo), round(hi)], "n": len(d)}
    return out


def better(a: dict, b: Optional[dict]) -> bool:
    """Accept candidate a over incumbent b only when a paired difference's CI
    excludes zero in its favour — realized 2025 points first, the 2026
    simulation otherwise — and never when either says it is worse. Ties
    keep the incumbent (the simpler sheet)."""
    if b is None:
        return True
    g = paired_gain(a, b)
    rep, sim = g.get("replay"), g.get("sim")
    if rep and rep["ci"][1] < 0:
        return False
    if sim and sim["ci"][1] < 0:
        return False
    if rep and rep["ci"][0] > 0:
        return True
    if sim and sim["ci"][0] > 0:
        return True
    return False


# ---------------------------------------------------------------- the coach

COACH_SYSTEM = """You are a fantasy-football draft coach reviewing simulated drafts made by a
fast, deterministic engine that follows a strategy sheet. The engine already
maximizes expected lineup points under an availability model; your job is what
it cannot see: structural holes (bye stacks, empty starting slots, thin bench),
positional timing that fails against real drafters, and — most important —
how the rule set performed when replayed on LAST season's real results.

Respond with ONLY a JSON object:
{"learnings": [3 specific sentences citing the numbers],
 "candidates": [                                 // 1-3 DISTINCT rule-set changes to test; each is scored
   {"label": "short name",                       //   on the same seeded drafts and the same 2025 replays,
    "rules_add": ["WAIT QB UNTIL R7", ...],      //   and only a change whose paired gain's 95% CI excludes
    "rules_remove": ["NO QB BEFORE R3", ...],    //   zero is adopted. ONLY these patterns:
    "why": "one line"}],                         //   'WAIT <POS> UNTIL R<n>', 'NO <POS> BEFORE R<n>',
 "note": "ONE line, <= 30 words, only if new"}  //   '<n> <POS> BY R<n>', 'TARGET <Player Name>',
                                                 //   'NO BYE STACK > <n>', 'NO <POS> AGE <a>+ BEFORE R<n>'
Make the candidates different hypotheses (timing, structure, targets), not
variants of one idea. Removing a rule that hurt is a valid candidate. If the
sheet already beats the baseline and the structure is sound, return an empty
candidates list and say so. Never suggest a rule the engine cannot express."""


def _coach_call(settings: Settings, digest: dict) -> dict:
    from .insights import _client, _json_block, _model

    client, err = _client()
    if client is None:
        return {"available": False, "reason": err}
    league = settings.league
    user = (f"League: {league.num_teams}-team, slot {league.draft_position}, {league.scoring_format}, "
            f"roster {league.roster_size}.\n\nDIGEST:\n{json.dumps(digest, default=str)[:16000]}")
    try:
        resp = client.messages.create(model=_model(), max_tokens=6000, system=COACH_SYSTEM,
                                      messages=[{"role": "user", "content": user}])
    except Exception as e:
        return {"available": False, "reason": f"{e.__class__.__name__}: {e}"}
    text = "".join(getattr(b, "text", "") for b in resp.content)
    parsed = _json_block(text) or {}
    cands = []
    for c in (parsed.get("candidates") or [])[:3]:
        if isinstance(c, dict):
            cands.append({"label": str(c.get("label", "candidate"))[:60],
                          "rules_add": [str(x) for x in c.get("rules_add", [])][:3],
                          "rules_remove": [str(x) for x in c.get("rules_remove", [])][:3],
                          "why": str(c.get("why", ""))[:200]})
    if not cands and (parsed.get("rules_add") or parsed.get("rules_remove")):   # single-change replies still work
        cands.append({"label": "coach", "rules_add": [str(x) for x in parsed.get("rules_add", [])][:3],
                      "rules_remove": [str(x) for x in parsed.get("rules_remove", [])][:3], "why": ""})
    return {"available": True, "model": _model(),
            "learnings": [str(x) for x in parsed.get("learnings", [])][:5],
            "candidates": cands,
            "note": " ".join(str(parsed.get("note", "") or "").split())[:240]}


RULE_OK = re.compile(r"(?i)^(WAIT (QB|RB|WR|TE|K|DST) UNTIL R\d+|NO (QB|RB|WR|TE|K|DST) BEFORE R\d+|\d+ (QB|RB|WR|TE)S? BY R\d+|TARGET .+|NO BYE STACK ?> ?\d+|NO (QB|RB|WR|TE) AGE \d+\+? BEFORE R\d+)$")


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
                                 "mc_mean": h["score"]["sim"]["summary"]["mc_mean"], "learnings": h.get("learnings", [])[:2],
                                 "decision": h.get("decision")}
                                for h in history[-4:]],
        "how_candidates_are_judged": "paired on the same seeds; adopted only if the 95% CI of the gain excludes 0 "
                                     "(2025 realized points first, then the 2026 simulation); ties keep the simpler sheet",
    }


def candidate_rules(rules: List[Rule], changes: dict) -> List[Rule]:
    """The rule set a candidate describes (no persistence)."""
    keep = [r for r in rules if r.text.strip().upper() not in {x.strip().upper() for x in changes.get("rules_remove", [])}]
    for text in changes.get("rules_add", []):
        text = text.strip()
        if RULE_OK.match(text) and text.upper() not in {r.text.upper() for r in keep}:
            keep.append(Rule(text=text, on=True))
    new_rules, _ = reconcile_rules(keep)
    return new_rules


def persist(settings: Settings, rules: List[Rule], note: str = "") -> None:
    with state_lock(settings):
        state = DraftState.load(settings)
        state.rules = list(rules)
        if note:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            state.notes = (state.notes + "\n\n" if state.notes else "") + f"— coach {stamp}: {note}"
        state.save(settings)


def apply_changes(settings: Settings, rules: List[Rule], notes: str, changes: dict) -> List[Rule]:
    new_rules = candidate_rules(rules, changes)
    persist(settings, new_rules, (changes.get("note") or "").strip())
    return new_rules


def _score_worker(args: tuple) -> dict:
    """Process-pool entry: score one rule set (texts) on fixed seeds."""
    from .config import Settings as _S

    texts, n_drafts, seed, reps = args
    return score(_S(), [Rule(text=t, on=True) for t in texts], n_drafts, seed, reps)


def score_many(settings: Settings, rule_sets: List[List[Rule]], n_drafts: int, seed: int, reps: int,
               workers: Optional[int] = None) -> List[dict]:
    """Score several rule sets on the SAME seeds, in parallel processes."""
    import os as _os
    from concurrent.futures import ProcessPoolExecutor

    jobs = [([r.text for r in rs if r.on], n_drafts, seed, reps) for rs in rule_sets]
    if len(jobs) == 1 or (workers or 0) == 1:
        return [_score_worker(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers or min(len(jobs), max(2, (_os.cpu_count() or 2) - 1))) as ex:
        return list(ex.map(_score_worker, jobs))


# ---------------------------------------------------------------- the loop

def run_session(settings: Settings, iterations: int = 3, n_drafts: int = 12,
                reps: int = 6, seed: int = 101, workers: Optional[int] = None) -> dict:
    """Coach for `iterations` rounds. Every rule set in a session is scored
    on the SAME seeded drafts and replays (common random numbers), so
    candidate-vs-incumbent differences are paired; a candidate is adopted
    only when its paired gain's 95% CI excludes zero. A fresh holdout seed
    at the end confirms the kept sheet against the starting one."""
    STATUS.update({"running": True, "phase": "baseline", "iteration": 0, "total": iterations, "error": None})
    session: Dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "iterations": [],
                               "n_drafts": n_drafts, "reps": reps, "seed": seed, "design": "paired seeds, CI-gated"}
    try:
        state = DraftState.load(settings)
        rules, notes = list(state.rules), state.notes
        start_rules = list(rules)
        STATUS["phase"] = "scoring baseline + current sheet"
        baseline, incumbent = score_many(settings, [[], rules], n_drafts, seed, reps, workers)
        session["baseline"] = {"sim": baseline["sim"]["summary"], "replay": _replay_summary(baseline["replay"])}
        session["start"] = {"rules": [r.text for r in rules if r.on], "sim": incumbent["sim"]["summary"],
                            "replay": _replay_summary(incumbent["replay"]), "vs_baseline": paired_gain(incumbent, baseline)}
        history: List[dict] = []
        for it in range(1, iterations + 1):
            STATUS.update({"phase": "coaching", "iteration": it})
            record: Dict[str, Any] = {"iteration": it, "rules": [r.text for r in rules if r.on],
                                      "score": {"sim": {"summary": incumbent["sim"]["summary"]}, "replay": _replay_summary(incumbent["replay"])}}
            changes = _coach_call(settings, digest_for(settings, rules, notes, incumbent, baseline, history))
            record["coach"] = {k: v for k, v in changes.items() if k != "candidates"}
            record["learnings"] = changes.get("learnings", [])
            if not changes.get("available"):
                record["decision"] = "coach unavailable: " + str(changes.get("reason"))
                session["iterations"].append(record); session["stopped"] = changes.get("reason")
                break
            cands = changes.get("candidates") or []
            if changes.get("note"):
                persist(settings, rules, changes["note"]); notes = DraftState.load(settings).notes
            if not cands:
                record["decision"] = "coach proposed no change"
                session["iterations"].append(record); history.append(record)
                _save(settings, session)
                continue
            STATUS["phase"] = f"scoring {len(cands)} candidates (iteration {it})"
            sets = [candidate_rules(rules, c) for c in cands]
            scores = score_many(settings, sets, n_drafts, seed, reps, workers)
            judged = []
            for c, rs, sc in zip(cands, sets, scores):
                g = paired_gain(sc, incumbent)
                judged.append({"label": c["label"], "why": c["why"], "rules": [r.text for r in rs if r.on],
                               "gain": g, "accept": better(sc, incumbent),
                               "sim": sc["sim"]["summary"], "replay": _replay_summary(sc["replay"])})
            record["candidates"] = judged
            winners = [(j, sc, rs) for j, sc, rs in zip(judged, scores, sets) if j["accept"]]
            if winners:
                def key(x):
                    g = x[0]["gain"]
                    return ((g.get("replay") or {}).get("mean", 0), (g.get("sim") or {}).get("mean", 0))
                j, sc, rs = max(winners, key=key)
                rules, incumbent = rs, sc
                persist(settings, rules)
                record["decision"] = f"adopted '{j['label']}': " + ", ".join(
                    f"{k} {v['mean']:+} [{v['ci'][0]:+}, {v['ci'][1]:+}]" for k, v in j["gain"].items())
            else:
                record["decision"] = "no candidate cleared the CI gate; sheet unchanged — " + "; ".join(
                    f"{j['label']}: " + ", ".join(f"{k} {v['mean']:+} [{v['ci'][0]:+}, {v['ci'][1]:+}]" for k, v in j["gain"].items())
                    for j in judged)
            record["rules_after"] = [r.text for r in rules if r.on]
            session["iterations"].append(record); history.append(record)
            _save(settings, session)
        # holdout: the kept sheet vs the starting sheet on seeds nobody optimized on
        if [r.text for r in rules if r.on] != [r.text for r in start_rules if r.on]:
            STATUS["phase"] = "holdout"
            hold_start, hold_kept = score_many(settings, [start_rules, rules], n_drafts, seed + 7919, reps, workers)
            session["holdout"] = {"gain": paired_gain(hold_kept, hold_start), "kept": [r.text for r in rules if r.on],
                                  "confirmed": better(hold_kept, hold_start)}
            if not session["holdout"]["confirmed"]:
                rules = start_rules
                persist(settings, rules)
                session["holdout"]["note"] = "holdout did not confirm the gain — starting sheet restored"
        session["best_rules"] = [r.text for r in rules if r.on]
        session["best"] = {"sim": incumbent["sim"]["summary"], "replay": _replay_summary(incumbent["replay"])}
        session["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as e:
        STATUS["error"] = f"{e.__class__.__name__}: {e}"
        session["error"] = STATUS["error"]
        raise
    finally:
        _save(settings, session)
        STATUS.update({"running": False, "phase": "done"})
    return session


def _replay_summary(rep: Optional[dict]) -> Optional[dict]:
    return {k: v for k, v in rep.items() if k != "scores"} if rep else None


def _save(settings: Settings, session: dict) -> None:
    doc = load_sessions(settings)
    doc["sessions"] = [x for x in doc["sessions"] if x.get("started") != session["started"]] + [session]
    save_sessions(settings, doc)


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


def review_recent(settings: Settings, n: int = 2) -> dict:
    """Coach the last n real drafts together: the current log plus the
    archived cockpit drafts (newest first). One call sees all of them, so it
    can say what repeated — the signal one draft cannot give."""
    from . import service
    from .mocksim import load_store
    from .value.distributions import build_sim_players
    from .value.simulate import simulate_roster

    league = settings.league
    state = DraftState.load(settings)
    drafts: List[dict] = []
    if state.events:
        drafts.append({"label": "current", "mine": [e.player_uid for e in state.events if e.mine],
                       "picks": [e.player_uid for e in state.events], "created": state.created})
    for d in reversed(load_store(settings)["drafts"]):
        if d.get("source", "").startswith("cockpit") and d.get("mine") and d.get("created") != state.created:
            drafts.append({"label": d["source"], "mine": d["mine"], "picks": d["picks"], "created": d.get("created")})
        if len(drafts) >= n:
            break
    conn = service._conn(settings)
    pool = service._pool(settings, conn)
    by_uid = {p.uid: p for p in pool}
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    games = {r["player_uid"]: r["games"] for r in conn.execute(
        "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'", (settings.current_season,))}
    summaries = []
    for d in drafts:
        roster = [by_uid[u] for u in d["mine"] if u in by_uid]
        order = {u: i + 1 for i, u in enumerate(d["picks"])}
        sps = build_sim_players(conn, roster, settings.current_season, league.scoring_format, proj_rank, games)
        sim = simulate_roster(sps, league, n_sims=300) if roster else None
        grid = service.season_grid_for(settings, pool, roster)
        summaries.append({
            "draft": d["label"], "created": (d.get("created") or "")[:16],
            "picks": [{"pick": order.get(p.uid), "round": (order.get(p.uid, 1) - 1) // league.num_teams + 1,
                       "pos": p.position, "name": p.name, "proj": round(p.raw or p.proj), "adp": p.adp} for p in roster],
            "by_pos": {pos: sum(1 for p in roster if p.position == pos) for pos in ("QB", "RB", "WR", "TE", "K", "DST")},
            "mc_mean": round(sim.mean) if sim else None, "mc_p10": round(sim.p10) if sim else None, "mc_p90": round(sim.p90) if sim else None,
            "weak_weeks": [t["week"] for t in grid["week_totals"] if t["level"] == "red"],
            "bye_stacks": [t["week"] for t in grid["week_totals"] if len(t["bye"]) >= 3],
            "worst_week": min((t["pts"] for t in grid["week_totals"]), default=0),
            "reaches": [f"{p.name} R{(order.get(p.uid, 1) - 1) // league.num_teams + 1} (ADP round {int((p.adp - 1) // league.num_teams + 1)})"
                        for p in roster if p.adp and (order.get(p.uid, 1) - 1) // league.num_teams + 1 < (p.adp - 1) // league.num_teams],
        })
    conn.close()
    digest = {"rules": [r.text for r in state.rules if r.on], "notes": state.notes[-1500:],
              "league": {"teams": league.num_teams, "slot": league.draft_position, "scoring": league.scoring_format},
              "drafts": summaries,
              "ask": "These are real ESPN rooms; the most recent is an expert room and the closest proxy for "
                     "the real draft. Say what REPEATED across them (timing, positions, structure), what the "
                     "engine got right, and the specific rule changes that would have improved BOTH."}
    return {"digest": digest, "coach": _coach_call(settings, digest)}
