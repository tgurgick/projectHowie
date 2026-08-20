"""Portable strategy-context artifact: derived abstractions only.

The artifact is the ONLY thing meant to leave the user's machine. It carries
league shape, per-player derived values (tier, availability at the user's
picks, outcome summaries, market pick estimate), and simulation metadata —
never raw provider rows, stat lines, or scraped payloads. Field whitelists are
enforced on write AND read, so an artifact with unexpected fields fails
validation instead of leaking.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import LeagueConfig, Settings
from .db import connect

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "strategy_context"

# Strict whitelists — the redaction boundary.
PLAYER_FIELDS = {
    "uid", "name", "position", "team", "bye", "tier", "projection_band",
    "projection", "availability", "outcomes", "market",
}
OUTCOME_FIELDS = {"mean", "std", "p10", "p50", "p90"}
MARKET_FIELDS = {"pick", "spread"}
TOP_LEVEL_FIELDS = {
    "schema_version", "artifact_type", "created_at", "league", "players", "simulation",
}
BANDS = ("elite", "starter", "mid", "depth")


def export_context(
    settings: Settings, out_path: Path, n_sims: int = 300, seed: int = 7
) -> dict:
    from .value.board import load_pool, snake_picks
    from .value.distributions import build_sim_players, tier_of
    from .value.simulate import simulate_player_totals

    league = settings.league
    conn = connect(settings.db_path)
    fmt = league.scoring_format
    pool = load_pool(conn, settings.current_season, fmt)
    if not pool:
        raise RuntimeError("No projections in the local database — run `howie data refresh` first.")
    picks = snake_picks(league)

    games_by_uid = {
        r["player_uid"]: r["games"]
        for r in conn.execute(
            "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
            (settings.current_season,),
        )
    }
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]

    sim_players = build_sim_players(conn, pool, settings.current_season, fmt, proj_rank, games_by_uid)

    players = []
    for p, sp in zip(pool, sim_players):
        totals = simulate_player_totals(sp, n_sims=n_sims, seed=seed)
        tier = tier_of(p.position, proj_rank[p.uid])
        entry = {
            "uid": p.uid,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "bye": p.bye,
            "tier": tier,
            "projection_band": BANDS[min(tier, len(BANDS) - 1)],
            "projection": round(p.proj, 1),
            "availability": {
                str(k): round(p.p_available(k), 3) for k in picks
            },
            "outcomes": {
                "mean": round(float(totals.mean()), 1),
                "std": round(float(totals.std()), 1),
                "p10": round(float(_pct(totals, 10)), 1),
                "p50": round(float(_pct(totals, 50)), 1),
                "p90": round(float(_pct(totals, 90)), 1),
            },
        }
        if p.adp is not None:
            entry["market"] = {"pick": round(p.adp, 1), "spread": round(p.stdev or 0.0, 2)}
        players.append(entry)
    conn.close()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "league": {
            "teams": league.num_teams,
            "draft_position": league.draft_position,
            "scoring": league.scoring_format,
            "roster": {
                "QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
                "TE": league.te_slots, "FLEX": league.flex_slots,
                "K": league.k_slots, "DST": league.dst_slots,
                "BENCH": league.bench_slots,
            },
        },
        "players": players,
        "simulation": {"runs": n_sims, "seed": seed, "season": settings.current_season},
    }
    validate_artifact(artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1))
    return artifact


def _pct(arr, q):
    import numpy as np
    return np.percentile(arr, q)


def validate_artifact(artifact: dict) -> None:
    """Strict validation: schema version, required keys, and NO extra fields."""
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must be a JSON object")
    extra = set(artifact) - TOP_LEVEL_FIELDS
    if extra:
        raise ValueError(f"Unexpected top-level fields: {sorted(extra)}")
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Not a strategy_context artifact")
    version = artifact.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema_version {version} is not supported (expected {SCHEMA_VERSION})"
        )
    for key in ("league", "players", "simulation"):
        if key not in artifact:
            raise ValueError(f"Missing required field {key!r}")
    for i, p in enumerate(artifact["players"]):
        extra = set(p) - PLAYER_FIELDS
        if extra:
            raise ValueError(f"players[{i}] has unexpected fields: {sorted(extra)}")
        for req in ("uid", "position", "projection", "outcomes"):
            if req not in p:
                raise ValueError(f"players[{i}] missing {req!r}")
        if set(p["outcomes"]) - OUTCOME_FIELDS:
            raise ValueError(f"players[{i}].outcomes has unexpected fields")
        if "market" in p and set(p["market"]) - MARKET_FIELDS:
            raise ValueError(f"players[{i}].market has unexpected fields")


def load_context(path: Path) -> Tuple[LeagueConfig, List]:
    """Load an artifact into (league, pool) usable by the draft views."""
    from .value.board import PoolPlayer

    artifact = json.loads(Path(path).read_text())
    validate_artifact(artifact)
    lg = artifact["league"]
    roster = lg["roster"]
    fmt_to_type = {"std": "standard", "half": "half_ppr", "ppr": "ppr"}
    league = LeagueConfig(
        num_teams=lg["teams"], draft_position=lg["draft_position"],
        scoring_type=fmt_to_type.get(lg["scoring"], "half_ppr"),
        qb_slots=roster.get("QB", 1), rb_slots=roster.get("RB", 2),
        wr_slots=roster.get("WR", 3), te_slots=roster.get("TE", 1),
        flex_slots=roster.get("FLEX", 1), k_slots=roster.get("K", 1),
        dst_slots=roster.get("DST", 1), bench_slots=roster.get("BENCH", 6),
    )
    pool = []
    for p in artifact["players"]:
        market = p.get("market") or {}
        pool.append(
            PoolPlayer(
                uid=p["uid"], name=p.get("name", p["uid"]), position=p["position"],
                team=p.get("team"), proj=float(p["projection"]),
                adp=market.get("pick"), stdev=market.get("spread"),
                bye=p.get("bye"),
            )
        )
    pool.sort(key=lambda p: -p.proj)
    return league, pool


def inspect_context(path: Path) -> dict:
    artifact = json.loads(Path(path).read_text())
    validate_artifact(artifact)
    by_pos: Dict[str, int] = {}
    for p in artifact["players"]:
        by_pos[p["position"]] = by_pos.get(p["position"], 0) + 1
    return {
        "created_at": artifact["created_at"],
        "schema_version": artifact["schema_version"],
        "league": artifact["league"],
        "players": len(artifact["players"]),
        "by_position": by_pos,
        "simulation": artifact["simulation"],
    }


def default_context_path(settings: Settings) -> Path:
    return settings.data_dir / "strategy-context.json"
