"""PFF projection CSV ingest (manual exports dropped into data/pff_csv/).

Files are matched by season, e.g. offensive_projections_2026_preseason.csv.
Points are recomputed from stat lines in all three formats — PFF's own
fantasyPoints column reflects their scoring, not ours.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..names import fix_position, fix_team, record_unmatched, resolve_uid
from ..scoring import add_points_columns, dst_points, kicker_points

_RENAME = {
    "playerName": "player_name",
    "teamName": "team",
    "byeWeek": "bye_week",
    "passYds": "pass_yards",
    "passTd": "pass_tds",
    "passInt": "interceptions",
    "rushYds": "rush_yards",
    "rushTd": "rush_tds",
    "recvTargets": "targets",
    "recvReceptions": "receptions",
    "recvYds": "rec_yards",
    "recvTd": "rec_tds",
    "fumblesLost": "fumbles_lost",
    "twoPt": "two_pt",
    # kickers
    "fgMade019": "fg_made_0_19",
    "fgMade2029": "fg_made_20_29",
    "fgMade3039": "fg_made_30_39",
    "fgMade4049": "fg_made_40_49",
    "fgMade50plus": "fg_made_50_plus",
    "patMade": "pat_made",
    # team defense
    "dstSacks": "dst_sacks",
    "dstInt": "dst_ints",
    "dstFumblesRecovered": "dst_fumbles_rec",
    "dstSafeties": "dst_safeties",
    "dstTd": "dst_tds",
    "dstReturnTd": "dst_return_tds",
    "dstPts0": "dst_pa_0",
    "dstPts16": "dst_pa_1_6",
    "dstPts713": "dst_pa_7_13",
    "dstPts1420": "dst_pa_14_20",
    "dstPts2127": "dst_pa_21_27",
    "dstPts2834": "dst_pa_28_34",
    "dstPts35plus": "dst_pa_35_plus",
}

_STAT_COLS = [
    "pass_yards", "pass_tds", "interceptions", "rush_yards", "rush_tds",
    "targets", "receptions", "rec_yards", "rec_tds", "fumbles_lost", "two_pt",
]


def projection_files(pff_dir: Path, season: int) -> List[Path]:
    return sorted(pff_dir.glob(f"*projections_{season}_*.csv"))


def refresh_projections(conn: sqlite3.Connection, pff_dir: Path, season: int) -> int:
    files = projection_files(pff_dir, season)
    if not files:
        raise FileNotFoundError(
            f"No PFF projection CSVs for {season} in {pff_dir} "
            f"(expected e.g. offensive_projections_{season}_preseason.csv)"
        )
    total = 0
    for path in files:
        df = pd.read_csv(path).rename(columns=_RENAME)
        df["team"] = df["team"].map(fix_team)
        df["position"] = df["position"].map(fix_position)
        df = add_points_columns(df)
        # K/DST score from their own stat lines, identical across formats
        for mask, scorer in [(df["position"] == "K", kicker_points), (df["position"] == "DST", dst_points)]:
            if mask.any():
                pts = scorer(df[mask]).values
                for c in ("pts_std", "pts_half", "pts_ppr"):
                    df.loc[mask, c] = pts

        rows = []
        for d in (r._asdict() for r in df.itertuples(index=False)):
            uid = _resolve(conn, d, season)
            if uid is None:
                continue
            rows.append(
                (
                    season, "pff", uid, d.get("position"), d.get("team"),
                    _num(d.get("bye_week")), _num(d.get("games")),
                    *[_num(d.get(c)) for c in _STAT_COLS],
                    d.get("pts_std"), d.get("pts_half"), d.get("pts_ppr"),
                )
            )
        conn.executemany(
            "INSERT OR REPLACE INTO projections "
            "(season, source, player_uid, position, team, bye_week, games, "
            " pass_yards, pass_tds, interceptions, rush_yards, rush_tds, "
            " targets, receptions, rec_yards, rec_tds, fumbles_lost, two_pt, "
            " pts_std, pts_half, pts_ppr) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        total += len(rows)
    conn.commit()
    return total


def _resolve(conn: sqlite3.Connection, d: dict, season: int) -> Optional[str]:
    name, pos, team = d.get("player_name"), d.get("position"), d.get("team")
    if not name or not isinstance(name, str):
        return None
    if pos == "DST":
        return f"dst:{team}"  # team defenses aren't people; synthetic uid
    uid = resolve_uid(conn, name, pos, team)
    if uid is None:
        record_unmatched(conn, "pff", name, season, pos, team, "projection row dropped")
    return uid


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
