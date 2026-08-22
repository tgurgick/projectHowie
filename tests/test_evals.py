"""Tier C (policy replay) evaluation: bootstrap CI helper, paired summary,
and the paired seeding of opponents across policies."""

from dataclasses import replace as dc_replace
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from howie3.config import LeagueConfig, Settings
from howie3.evals import (
    BASELINE_POLICY, EVAL_ROSTER_SIZE, POLICIES, EvalPlayer, bootstrap_mean_ci,
    eval_policy, load_eval_players, replay_draft, replay_seed, summarize_paired,
    _pick_adp_need,
)

# A tiny league so engine replays stay well under a second.
TINY = LeagueConfig(num_teams=4, draft_position=3, qb_slots=1, rb_slots=1, wr_slots=1,
                    te_slots=0, flex_slots=1, k_slots=0, dst_slots=0, bench_slots=2,
                    roster_size=6)


def _synthetic_pool() -> List[EvalPlayer]:
    """62 offense players with ADP ordered by projection (a little positional
    shuffle so ADP is not a pure projection rank). Realized points equal the
    projection spread evenly over 17 weeks."""
    specs = [("QB", 12, 380.0, 9.0), ("RB", 20, 300.0, 7.0),
             ("WR", 20, 290.0, 6.5), ("TE", 10, 200.0, 8.0)]
    players = []
    for pos, n, top, step in specs:
        for i in range(n):
            proj = round(top - step * i, 1)
            players.append(EvalPlayer(
                uid=f"{pos.lower()}{i}", name=f"{pos} {i}", position=pos, proj=proj,
                games_proj=17.0, adp=None, actual_total=proj,
                actual_weeks={w: round(proj / 17.0, 2) for w in range(1, 18)}))
    # market: QBs go later than projection alone would say
    order = sorted(players, key=lambda p: -(p.proj - (120.0 if p.position == "QB" else 0.0)))
    for rank, p in enumerate(order, start=1):
        p.adp = float(rank)
    players.sort(key=lambda p: -p.proj)
    return players


def _follow_adp(pool_avail, roster, league, current_pick, future):
    return min(pool_avail, key=lambda p: p.adp)


# ---------------- bootstrap CI ----------------

def test_bootstrap_ci_deterministic_and_brackets_mean():
    rng = np.random.default_rng(7)
    diffs = rng.normal(40.0, 150.0, size=120)
    lo1, hi1 = bootstrap_mean_ci(diffs, seed=0)
    lo2, hi2 = bootstrap_mean_ci(diffs, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert (lo1, hi1) != bootstrap_mean_ci(diffs, seed=1)
    m = float(diffs.mean())
    assert lo1 < m < hi1
    # a 95% CI on n=120 with sd 150 should be roughly +/- 27 wide, not degenerate
    assert 20 < (hi1 - lo1) / 2 < 40


def test_bootstrap_ci_degenerate_samples():
    assert bootstrap_mean_ci([5.0]) == (5.0, 5.0)
    assert bootstrap_mean_ci([3.0, 3.0, 3.0]) == (3.0, 3.0)
    lo, hi = bootstrap_mean_ci([])
    assert np.isnan(lo) and np.isnan(hi)


# ---------------- paired summary ----------------

def test_summarize_paired_structure():
    rng = np.random.default_rng(3)
    base = list(rng.normal(1500, 100, size=40))
    scores = {
        "howie": [b + 10.0 for b in base],                        # exactly +10 paired
        "noisy": [b + d for b, d in zip(base, rng.normal(0, 80, size=40))],
        BASELINE_POLICY: base,
    }
    rep = summarize_paired(scores)
    assert list(rep) == ["howie", "noisy", BASELINE_POLICY]
    for r in rep.values():
        assert {"mean_total", "delta_vs_adp", "ci_low", "ci_high", "n"} <= set(r)
        assert r["n"] == 40
    assert rep["howie"]["delta_vs_adp"] == 10.0
    assert (rep["howie"]["ci_low"], rep["howie"]["ci_high"]) == (10.0, 10.0)
    assert rep["howie"]["crosses_zero"] is False and rep["howie"]["win_rate"] == 1.0
    assert rep[BASELINE_POLICY]["delta_vs_adp"] == 0.0
    assert rep[BASELINE_POLICY]["crosses_zero"] is False
    nz = rep["noisy"]
    assert nz["ci_low"] <= nz["delta_vs_adp"] <= nz["ci_high"]
    assert nz["crosses_zero"] == (nz["ci_low"] <= 0 <= nz["ci_high"])


def test_summarize_paired_rejects_unpaired_lengths():
    with pytest.raises(ValueError):
        summarize_paired({"howie": [1.0, 2.0], BASELINE_POLICY: [1.0]})


# ---------------- ADP + positional need baseline ----------------

def test_adp_need_skips_filled_starters_until_bench_rounds():
    pool = _synthetic_pool()
    league = dc_replace(TINY, draft_position=1)
    qb = next(p for p in pool if p.position == "QB")
    best_adp = min(pool, key=lambda p: p.adp)
    assert best_adp.position == "RB"
    # starter rounds (1..4 here): with the RB slot + flex both taken by RBs,
    # the next RB by ADP is skipped for an open position
    rb1, rb2 = [p for p in pool if p.position == "RB"][:2]
    avail = [p for p in pool if p.uid not in (rb1.uid, rb2.uid)]
    pick = _pick_adp_need(avail, [rb1, rb2], league, current_pick=9)  # round 3
    assert pick.position in ("WR", "QB")
    # bench rounds: plain ADP, even if that is a 3rd RB
    pick = _pick_adp_need(avail, [rb1, rb2, qb], league, current_pick=17)  # round 5
    assert pick == min(avail, key=lambda p: p.adp)


# ---------------- paired seeding ----------------

def test_paired_seed_same_opponents_synthetic():
    pool = _synthetic_pool()
    league = dc_replace(TINY, draft_position=3)
    # two policies that make the same picks from the same (slot, rep) must see
    # identical opponent picks through the whole draft
    r1, opp_named = replay_draft("adp", pool, league, slot=3, rep=0)
    r2, opp_fn = replay_draft(_follow_adp, pool, league, slot=3, rep=0)
    assert [p.uid for p in r1] == [p.uid for p in r2]
    assert opp_named == opp_fn and len(opp_named) == 3 * 6
    # the engine is deterministic under the same seed
    h1, opp_h1 = replay_draft("howie", pool, league, slot=3, rep=0)
    h2, opp_h2 = replay_draft("howie", pool, league, slot=3, rep=0)
    assert [p.uid for p in h1] == [p.uid for p in h2] and opp_h1 == opp_h2
    # different policies from the same (slot, rep): opponents are identical
    # before my first pick (boards still coincide) — the pairing we rely on
    before_me = lambda opp: [o for o in opp if o[0] < 3]
    assert before_me(opp_h1) == before_me(opp_named) and len(before_me(opp_h1)) == 2
    # a different rep or slot is a different opponent draw
    assert replay_seed(3, 0) != replay_seed(3, 1) != replay_seed(5, 1)
    _, opp_rep1 = replay_draft("adp", pool, league, slot=3, rep=1)
    assert opp_rep1 != opp_named


def test_eval_policy_returns_structured_paired_report():
    players = _synthetic_pool()
    settings = SimpleNamespace(league=dc_replace(TINY, roster_size=EVAL_ROSTER_SIZE))
    rep = eval_policy(settings, players, slots_to_test=[2, 4], reps=2)
    assert list(rep) == list(POLICIES)
    for r in rep.values():
        assert r["n"] == 4
        assert r["ci_low"] <= r["delta_vs_adp"] <= r["ci_high"]
    assert rep[BASELINE_POLICY]["delta_vs_adp"] == 0.0
    assert rep["howie"]["mean_total"] > 0


def test_paired_seed_same_opponents_real_data():
    s = Settings()
    if not (s.data_dir / "fantasy_ppr.db").exists() or not s.db_path.exists():
        pytest.skip("legacy fantasy_ppr.db / howie.db not built")
    from howie3.evals import _top_pool

    pool = [p for p in _top_pool(load_eval_players(s)) if p.adp is not None]
    league = dc_replace(s.league, draft_position=5, k_slots=0, dst_slots=0,
                        roster_size=EVAL_ROSTER_SIZE)
    _, opp_named = replay_draft("adp", pool, league, slot=5, rep=2)
    _, opp_fn = replay_draft(_follow_adp, pool, league, slot=5, rep=2)
    assert opp_named == opp_fn and len(opp_named) == 11 * EVAL_ROSTER_SIZE
    _, opp_need = replay_draft("adp_need", pool, league, slot=5, rep=2)
    assert [o for o in opp_need if o[0] < 5] == [o for o in opp_named if o[0] < 5]
