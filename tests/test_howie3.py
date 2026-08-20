"""Test suite for the howie3 engine: math invariants, data-layer sanity, and
command smoke tests against the real database."""

import math
import random

import numpy as np
import pandas as pd
import pytest

from howie3.config import LeagueConfig, Settings, parse_seasons
from howie3.data.names import fix_position, fix_team, name_key
from howie3.data.scoring import add_points_columns, dst_points, kicker_points
from howie3.value.availability import p_available
from howie3.value.board import PoolPlayer, expected_kth_best, snake_picks
from howie3.value.lineup import lineup_points

LEAGUE = LeagueConfig()  # 12-team, slot 8, half-PPR, 1QB/2RB/3WR/1TE/1FLEX/1K/1DST


# ---------- scoring ----------

def test_reception_value_is_the_only_format_difference():
    df = pd.DataFrame([
        dict(pass_yards=300, pass_tds=2, interceptions=1, rush_yards=40,
             rush_tds=0, rec_yards=80, rec_tds=1, receptions=6, fumbles_lost=1,
             two_pt=1, st_tds=0),
    ])
    out = add_points_columns(df)
    assert out.loc[0, "pts_ppr"] - out.loc[0, "pts_half"] == pytest.approx(0.5 * 6)
    assert out.loc[0, "pts_half"] - out.loc[0, "pts_std"] == pytest.approx(0.5 * 6)
    # hand-computed standard points
    expected = 300 * 0.04 + 2 * 4 - 2 + 40 * 0.1 + 80 * 0.1 + 6 - 2 + 2
    assert out.loc[0, "pts_std"] == pytest.approx(expected)


def test_kicker_scoring():
    df = pd.DataFrame([dict(fg_made_0_19=1, fg_made_20_29=1, fg_made_30_39=1,
                            fg_made_40_49=2, fg_made_50_plus=1, pat_made=3)])
    assert kicker_points(df).iloc[0] == pytest.approx(3 * 3 + 2 * 4 + 5 + 3)


def test_dst_scoring():
    df = pd.DataFrame([dict(dst_sacks=3, dst_ints=2, dst_fumbles_rec=1,
                            dst_safeties=0, dst_tds=1, dst_return_tds=0,
                            dst_pa_7_13=1)])
    assert dst_points(df).iloc[0] == pytest.approx(3 + 4 + 2 + 6 + 4)


# ---------- lineup ----------

def test_lineup_fills_dedicated_slots_then_flex():
    by_pos = {
        "QB": [300, 250], "RB": [200, 180, 150], "WR": [190, 170, 160, 140],
        "TE": [120], "K": [140], "DST": [110],
    }
    # QB 300 + RB 200+180 + WR 190+170+160 + TE 120 + K 140 + DST 110 + flex max(150,140)
    assert lineup_points(by_pos, LEAGUE) == pytest.approx(
        300 + 380 + 520 + 120 + 140 + 110 + 150
    )


def test_lineup_missing_positions_score_zero():
    assert lineup_points({"QB": [300]}, LEAGUE) == pytest.approx(300)
    assert lineup_points({}, LEAGUE) == 0.0


# ---------- availability ----------

def test_p_available_bounds_and_monotonicity():
    assert p_available(None, None, 50) == 1.0
    assert p_available(10.0, 2.0, 10.0) == pytest.approx(0.5)
    picks = [1, 5, 10, 20, 40]
    probs = [p_available(10.0, 3.0, k) for k in picks]
    assert all(a >= b for a, b in zip(probs, probs[1:]))
    assert probs[0] > 0.99 and probs[-1] < 0.01


# ---------- expected k-th best (DP vs brute force) ----------

def _mk(uid, proj, adp, stdev):
    return PoolPlayer(uid, uid, "RB", None, proj, adp, stdev, None)


def test_expected_kth_best_matches_brute_force():
    players = [_mk(f"p{i}", 300 - 15 * i, 3 + 4 * i, 2 + 0.5 * i) for i in range(8)]
    pick = 15.0
    rng = random.Random(42)
    n = 40000
    sums = {1: 0.0, 2: 0.0}
    for _ in range(n):
        avail = [p for p in players if rng.random() < p.p_available(pick)]
        if len(avail) >= 1:
            sums[1] += avail[0].proj      # players sorted by proj already
        if len(avail) >= 2:
            sums[2] += avail[1].proj
    for k in (1, 2):
        dp = expected_kth_best(players, pick, k)
        assert dp == pytest.approx(sums[k] / n, rel=0.02)


# ---------- snake picks ----------

def test_snake_picks_12_team_slot_8():
    assert snake_picks(LEAGUE)[:6] == [8, 17, 32, 41, 56, 65]


def test_snake_picks_slot_1_and_12():
    lc = LeagueConfig(num_teams=12, draft_position=1)
    assert snake_picks(lc)[:4] == [1, 24, 25, 48]
    lc = LeagueConfig(num_teams=12, draft_position=12)
    assert snake_picks(lc)[:4] == [12, 13, 36, 37]


# ---------- names ----------

def test_name_key_normalization():
    assert name_key("Ja'Marr Chase") == name_key("JaMarr Chase")
    assert name_key("Odell Beckham Jr.") == name_key("Odell Beckham")
    assert name_key("Kenneth Walker III") == name_key("Kenneth Walker")
    assert name_key("A.J. Brown") == name_key("AJ Brown")


def test_team_and_position_fixes():
    assert fix_team("JAC") == "JAX" and fix_team("ARZ") == "ARI" and fix_team("LAR") == "LA"
    assert fix_position("RB12") == "RB" and fix_position("D/ST") == "DST" and fix_position("qb") == "QB"


def test_parse_seasons():
    assert parse_seasons("2018-2020") == [2018, 2019, 2020]
    assert parse_seasons("2024") == [2024]
    assert parse_seasons("2019,2021-2022") == [2019, 2021, 2022]


# ---------- integration against the real db ----------

@pytest.fixture(scope="module")
def settings():
    s = Settings()
    if not s.db_path.exists():
        pytest.skip("howie.db not built")
    return s


def test_db_has_all_datasets(settings):
    from howie3.db import connect
    conn = connect(settings.db_path)
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("players", "player_ids", "games", "weekly_stats",
                  "projections", "adp", "sos", "team_intel")
    }
    conn.close()
    assert counts["weekly_stats"] > 50000
    assert counts["projections"] >= 500
    assert counts["adp"] > 600
    assert counts["sos"] == 2560
    assert counts["team_intel"] > 200


def test_scoring_formats_consistent_in_db(settings):
    from howie3.db import connect
    conn = connect(settings.db_path)
    row = conn.execute(
        "SELECT SUM(pts_ppr - pts_half - 0.5 * receptions) AS d FROM weekly_stats"
    ).fetchone()
    conn.close()
    assert abs(row["d"]) < 1.0  # rounding dust only


def test_dispatch_smoke(settings):
    from howie3.commands import dispatch
    assert dispatch(settings, "help")
    assert dispatch(settings, "status")
    assert dispatch(settings, "board 2 top=3")
    assert dispatch(settings, "player Puka Nacua")
    out = dispatch(settings, "nonsense")
    assert "Unknown command" in out[0].plain


def test_pick_view_deterministic_and_mc(settings):
    from howie3 import views
    det = views.pick_view(settings, round_num=1, sims=0, top_n=4)
    mc = views.pick_view(settings, round_num=1, sims=50, top_n=4)
    assert det and mc


def test_bench_insurance_has_value(settings):
    """A backup RB must raise simulated season points when starters can miss games."""
    from howie3.value.distributions import SimPlayer
    from howie3.value.simulate import simulate_roster

    def player(name, pos, proj, p_play):
        return SimPlayer(name=name, position=pos, proj=proj, weekly_mu=proj / 16,
                         cv=0.35, p_play=p_play, bye_week=None, sos_mult=np.ones(18))

    base = [
        player("QB1", "QB", 300, 0.95), player("RB1", "RB", 250, 0.80),
        player("RB2", "RB", 200, 0.80), player("WR1", "WR", 220, 0.95),
        player("WR2", "WR", 200, 0.95), player("WR3", "WR", 180, 0.95),
        player("TE1", "TE", 150, 0.95), player("K1", "K", 150, 0.97),
        player("D1", "DST", 110, 1.0),
    ]
    without = simulate_roster(base, LEAGUE, n_sims=400, seed=3)
    with_backup = simulate_roster(base + [player("RB3", "RB", 140, 0.9)],
                                  LEAGUE, n_sims=400, seed=3)
    assert with_backup.mean > without.mean + 10


# ---------- improvement plan: config validation ----------

def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="scoring_type"):
        LeagueConfig(scoring_type="super_ppr").validate()
    with pytest.raises(ValueError, match="num_teams"):
        LeagueConfig(num_teams=1).validate()
    with pytest.raises(ValueError, match="draft_position"):
        LeagueConfig(draft_position=13).validate()
    with pytest.raises(ValueError, match="roster_size"):
        LeagueConfig(roster_size=3).validate()
    LeagueConfig().validate()  # defaults are valid


def test_config_load_rejects_unknown_keys(tmp_path):
    p = tmp_path / "league.json"
    p.write_text('{"num_teams": 12, "sacring_type": "ppr"}')
    with pytest.raises(ValueError, match="Unknown league config keys"):
        LeagueConfig.load(p)


# ---------- improvement plan: refresh hardening ----------

def test_refresh_rejects_unknown_steps():
    from howie3.data.refresh import run_refresh
    with pytest.raises(ValueError, match="Unknown refresh steps"):
        run_refresh(Settings(), steps=["bogus"])


def test_refresh_orders_steps_canonically():
    from howie3.data.refresh import STEP_ORDER
    assert STEP_ORDER.index("crosswalk") < STEP_ORDER.index("weekly")
    assert STEP_ORDER[-1] == "verify"


# ---------- improvement plan: simulation calibration ----------

def _sim_player(proj=250.0, cv=0.4, p_play=0.9, bye=None, sos=None):
    from howie3.value.distributions import SimPlayer
    playable = 17 - (1 if bye and bye <= 17 else 0)
    return SimPlayer(
        name="X", position="RB", proj=proj, weekly_mu=proj / (playable * p_play),
        cv=cv, p_play=p_play, bye_week=bye,
        sos_mult=sos if sos is not None else np.ones(18),
    )


def test_simulated_season_mean_matches_projection():
    from howie3.value.simulate import simulate_player_totals
    for cv, p_play, bye in [(0.3, 1.0, None), (0.5, 0.85, 7), (0.9, 0.7, None)]:
        p = _sim_player(proj=250.0, cv=cv, p_play=p_play, bye=bye)
        totals = simulate_player_totals(p, n_sims=6000, seed=11)
        assert totals.mean() == pytest.approx(250.0, rel=0.03), (cv, p_play, bye)


def test_bye_week_scores_zero():
    from howie3.value.simulate import simulate_player_totals
    p = _sim_player(bye=5, p_play=1.0, cv=0.01)
    # per-week contribution check: 16 playable weeks at proj/16 each
    totals = simulate_player_totals(p, n_sims=200, seed=1)
    assert totals.mean() == pytest.approx(250.0, rel=0.02)


def test_simulation_reproducible_for_fixed_seed():
    from howie3.value.simulate import simulate_roster
    roster = [_sim_player(), _sim_player(proj=180, cv=0.5)]
    a = simulate_roster(roster, LEAGUE, n_sims=100, seed=5)
    b = simulate_roster(roster, LEAGUE, n_sims=100, seed=5)
    assert a.mean == b.mean and a.p90 == b.p90


def test_sos_reshapes_weeks_but_preserves_season():
    from howie3.value.simulate import simulate_player_totals
    sos = np.ones(18)
    sos[:8] = 1.2   # easy first half...
    sos[8:17] = 17 / 9 - 8 / 9 * 1.2  # ...normalized so mean over 17 weeks = 1
    p = _sim_player(cv=0.3, p_play=1.0, sos=sos)
    totals = simulate_player_totals(p, n_sims=5000, seed=2)
    assert totals.mean() == pytest.approx(250.0, rel=0.03)


def test_truncation_factor_exact():
    from howie3.value.distributions import truncation_factor
    rng = np.random.default_rng(0)
    for cv in (0.3, 0.8, 1.2):
        draws = np.clip(rng.normal(100.0, cv * 100.0, 400_000), 0, None)
        assert draws.mean() / 100.0 == pytest.approx(truncation_factor(cv), rel=0.01)


# ---------- improvement plan: repo data boundary ----------

def test_repo_boundary_source_tracked_raw_data_ignored():
    import subprocess

    def ignored(path):
        return subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=Settings().repo_root,
        ).returncode == 0

    assert not ignored("howie3/data/refresh.py"), "source package must be trackable"
    assert not ignored("howie3/defaults/league.example.json")
    assert ignored("data/howie.db"), "local db must stay ignored"
    assert ignored("data/pff_csv/receiving_2024_reg.csv"), "raw exports must stay ignored"


# ---------- improvement plan: strategy-context artifact ----------

def test_artifact_validation_rejects_extra_fields():
    from howie3.context_artifact import validate_artifact
    good = {
        "schema_version": 1, "artifact_type": "strategy_context",
        "created_at": "2026-01-01T00:00:00Z",
        "league": {}, "simulation": {},
        "players": [{"uid": "x", "position": "RB", "projection": 100.0,
                     "outcomes": {"mean": 100.0}}],
    }
    validate_artifact(good)  # passes
    leaky = {**good, "players": [{**good["players"][0], "raw_rows": []}]}
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_artifact(leaky)
    with pytest.raises(ValueError, match="top-level"):
        validate_artifact({**good, "weekly_stats": []})
    with pytest.raises(ValueError, match="schema_version"):
        validate_artifact({**good, "schema_version": 99})


def test_artifact_roundtrip_and_redaction(settings, tmp_path):
    from howie3.context_artifact import PLAYER_FIELDS, export_context, load_context
    out = tmp_path / "ctx.json"
    artifact = export_context(settings, out, n_sims=50)
    # redaction: every player carries only whitelisted fields, no stat lines
    for p in artifact["players"]:
        assert set(p) <= PLAYER_FIELDS
        assert "pass_yards" not in p and "rec_yards" not in p
    league, pool = load_context(out)
    assert len(pool) == len(artifact["players"])
    assert league.num_teams == settings.league.num_teams


def test_views_run_from_artifact_without_db(settings, tmp_path, monkeypatch):
    from howie3 import views
    from howie3.context_artifact import export_context
    out = tmp_path / "ctx.json"
    export_context(settings, out, n_sims=50)
    board = views.board_view(settings, round_num=1, top_n=3, context=str(out))
    assert len(board) > 3
    pick = views.pick_view(settings, round_num=1, sims=200, top_n=3, context=str(out))
    # context mode forces deterministic and says so
    assert any("deterministic" in getattr(r, "plain", "") for r in pick)


# ---------- improvement plan: SQL security boundary ----------

def test_safe_query_boundary(settings):
    from howie3.agent import safe_query
    ok = safe_query(settings, "SELECT COUNT(*) AS n FROM players")
    assert ok.startswith("[")
    for bad in [
        "DELETE FROM players",
        "SELECT 1; DROP TABLE players",
        "SELECT 1 -- sneak",
        "PRAGMA user_version",
        "SELECT load_extension('evil')",
        "ATTACH DATABASE '/tmp/x' AS x",
    ]:
        out = safe_query(settings, bad)
        assert out.startswith("Error") or out.startswith("SQL error"), bad


def test_integrity_checks_pass_on_live_db(settings):
    from howie3.data.integrity import verify_integrity
    from howie3.db import connect
    conn = connect(settings.db_path)
    assert verify_integrity(conn) == 5
    conn.close()


def test_calibration_buckets_sane(settings):
    from howie3.db import connect
    from howie3.value.distributions import calibrate
    conn = connect(settings.db_path)
    buckets = calibrate(conn, "half")
    conn.close()
    assert len(buckets) >= 12
    for (pos, tier), b in buckets.items():
        assert 0.1 < b.cv < 1.5, (pos, tier, b.cv)
        assert 0.4 < b.p_play <= 1.0, (pos, tier, b.p_play)
    # elite players should be steadier than depth players
    assert buckets[("RB", 0)].cv < buckets[("RB", 3)].cv
