"""The one refresh orchestrator. Steps run in dependency order, each is
idempotent, and every run is recorded in refresh_log."""

import sqlite3
import traceback
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from ..config import Settings
from ..db import connect
from .integrity import verify_integrity
from .names import TEAM_FIX
from .sources import dynastyprocess, ffcalculator, legacy_intel, nflverse, pff, pff_sos

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]


def _ensure_dst_players(conn: sqlite3.Connection) -> int:
    conn.executemany(
        "INSERT OR IGNORE INTO players (player_uid, name, name_key, position, team) "
        "VALUES (?, ?, ?, 'DST', ?)",
        [(f"dst:{t}", f"{t} D/ST", f"{t.lower()} dst", t) for t in NFL_TEAMS],
    )
    conn.commit()
    return len(NFL_TEAMS)


STEP_ORDER = ["crosswalk", "players", "dst", "games", "weekly", "adp", "pff", "roster", "depth", "sos", "intel", "graph", "verify"]


def _roster_status(conn, season: int) -> int:
    from ..status import refresh_roster_status
    return refresh_roster_status(conn, season)


def _depth_charts(conn, season: int) -> int:
    from ..depth import refresh_depth_charts
    return refresh_depth_charts(conn, season)


def _rebuild_graph(conn, season: int) -> int:
    from ..graph import rebuild_derived
    return rebuild_derived(conn, season)
# Steps that need earlier data present before they can run correctly
STEP_PRECONDITIONS = {
    "weekly": ("games", "Load games first (weekly stats attach to the schedule)."),
    "adp": ("players", "Load the crosswalk/players first (ADP resolves against them)."),
    "pff": ("players", "Load the crosswalk/players first (projections resolve against them)."),
    "roster": ("projections", "Load projections first (roster status is recorded for the draft pool)."),
    "graph": ("weekly_stats", "Load weekly stats first (shares/vacated volume derive from them)."),
}


def run_refresh(
    settings: Settings,
    seasons: Optional[List[int]] = None,
    steps: Optional[List[str]] = None,
) -> List[Tuple[str, str, int, str]]:
    """Returns [(step, status, rows, detail)]. Valid steps: STEP_ORDER.

    Unknown step names raise ValueError. Requested steps always execute in
    canonical dependency order regardless of how they were listed."""
    if steps:
        unknown = [s for s in steps if s not in STEP_ORDER]
        if unknown:
            raise ValueError(
                f"Unknown refresh steps {unknown} — valid: {', '.join(STEP_ORDER)}"
            )
        steps = [s for s in STEP_ORDER if s in steps]
    seasons = seasons or settings.hist_seasons
    conn = connect(settings.db_path)

    plan: List[Tuple[str, Callable[[], int]]] = [
        ("crosswalk", lambda: dynastyprocess.refresh_crosswalk(conn)),
        ("players", lambda: nflverse.refresh_players(conn)),
        ("dst", lambda: _ensure_dst_players(conn)),
        ("games", lambda: nflverse.refresh_games(conn, seasons)),
        ("weekly", lambda: nflverse.refresh_weekly(conn, seasons)),
        ("adp", lambda: ffcalculator.refresh_adp(
            conn, settings.current_season, settings.league.num_teams
        )),
        ("pff", lambda: pff.refresh_projections(conn, settings.pff_dir, settings.current_season)),
        ("roster", lambda: _roster_status(conn, settings.current_season)),
        ("depth", lambda: _depth_charts(conn, settings.current_season)),
        ("sos", lambda: pff_sos.refresh_sos(conn, settings.pff_dir, settings.current_season)),
        ("intel", lambda: legacy_intel.port_legacy_intel(
            conn, settings.data_dir / "fantasy_ppr.db"
        )),
        ("graph", lambda: _rebuild_graph(conn, settings.current_season)),
        ("verify", lambda: verify_integrity(conn)),
    ]
    if steps:
        plan = [(name, fn) for name, fn in plan if name in steps]

    results = []
    for name, fn in plan:
        try:
            precondition = STEP_PRECONDITIONS.get(name)
            if precondition:
                table, hint = precondition
                if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                    raise RuntimeError(f"Dependency missing: {table} table is empty. {hint}")
            rows = fn()
            status, detail = "ok", ""
        except FileNotFoundError as e:
            rows, status, detail = 0, "skipped", str(e)
        except Exception as e:
            rows, status, detail = 0, "error", f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
        conn.execute(
            "INSERT INTO refresh_log (step, seasons, rows, status, detail, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                ",".join(map(str, seasons)),
                rows,
                status,
                detail[:500],
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        results.append((name, status, rows, detail))
    conn.close()
    return results
