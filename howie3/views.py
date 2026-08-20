"""View builders shared by the CLI and TUI.

Every command's output is built here as Rich renderables and RETURNED, never
printed — the CLI prints them to a terminal console, the TUI writes them into
its log widget. This is what keeps one command layer serving both frontends.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from rich.table import Table
from rich.text import Text

from .config import Settings
from .db import connect


def _pool_source(settings: Settings, context: Optional[str]) -> Tuple:
    """Resolve (conn, league, pool, source_label). conn is None in context mode.

    Priority: explicit --context file, then the local db, then an imported
    default artifact (data/strategy-context.json)."""
    from .context_artifact import default_context_path, load_context
    from .value.board import load_pool

    if context:
        league, pool = load_context(Path(context))
        return None, league, pool, f"context:{context}"
    if settings.db_path.exists():
        conn = connect(settings.db_path)
        pool = load_pool(conn, settings.current_season, settings.league.scoring_format)
        if pool:
            return conn, settings.league, pool, "local db"
        conn.close()
    default = default_context_path(settings)
    if default.exists():
        league, pool = load_context(default)
        return None, league, pool, f"context:{default.name}"
    raise RuntimeError(
        "No local data. Build it with `howie data refresh`, or import a "
        "strategy-context artifact with `howie context import <file>`."
    )


def status_view(settings: Settings) -> List:
    if not settings.db_path.exists():
        return [Text(f"No database at {settings.db_path} — run `refresh` first.", style="red")]
    conn = connect(settings.db_path)
    out: List = []

    schema_v = conn.execute("PRAGMA user_version").fetchone()[0]
    league = settings.league
    out.append(Text.from_markup(
        f"schema v{schema_v} · season {settings.current_season} · "
        f"{league.num_teams}-team {league.scoring_format} · slot {league.draft_position}"
    ))
    table = Table(title=f"howie.db — {settings.db_path}")
    table.add_column("table")
    table.add_column("rows", justify="right")
    table.add_column("coverage")
    checks = [
        ("players", None), ("player_ids", None), ("games", "season"),
        ("weekly_stats", "season"), ("projections", "season"),
        ("adp", "season"), ("sos", "season"), ("unmatched_names", None),
    ]
    for tbl, season_col in checks:
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        coverage = ""
        if season_col and n:
            lo, hi = conn.execute(f"SELECT MIN({season_col}), MAX({season_col}) FROM {tbl}").fetchone()
            coverage = f"{lo}–{hi}" if lo != hi else str(lo)
        table.add_row(tbl, f"{n:,}", coverage)
    out.append(table)

    log = conn.execute(
        "SELECT step, status, rows, finished_at FROM refresh_log ORDER BY id DESC LIMIT 8"
    ).fetchall()
    if log:
        lt = Table(title="Recent refreshes")
        for c in ("step", "status", "rows", "finished_at"):
            lt.add_column(c)
        for r in log:
            lt.add_row(r["step"], r["status"], str(r["rows"]), r["finished_at"])
        out.append(lt)
    conn.close()
    return out


def refresh_view(settings: Settings, seasons: Optional[List[int]] = None,
                 steps: Optional[List[str]] = None) -> List:
    from .data.refresh import run_refresh

    try:
        results = run_refresh(settings, seasons, steps)
    except ValueError as e:
        return [Text(str(e), style="red")]
    return render_refresh_results(results)


def render_refresh_results(results) -> List:
    table = Table(title="Refresh results")
    table.add_column("step")
    table.add_column("status")
    table.add_column("rows", justify="right")
    table.add_column("detail", overflow="fold", max_width=60)
    errors = 0
    for name, status, rows, detail in results:
        errors += status == "error"
        color = {"ok": "green", "skipped": "yellow", "error": "red"}.get(status, "white")
        table.add_row(name, f"[{color}]{status}[/{color}]", str(rows), detail)
    out: List = [table]
    if errors:
        out.append(Text(
            f"PARTIAL REFRESH: {errors} step(s) failed — the database is not fully updated.",
            style="bold red",
        ))
    return out


def board_view(settings: Settings, round_num: int = 1, top_n: int = 5,
               context: Optional[str] = None) -> List:
    from .value.board import marginal_values, snake_picks

    try:
        conn, league, pool, source = _pool_source(settings, context)
    except (RuntimeError, ValueError) as e:
        return [Text(str(e), style="red")]
    picks = snake_picks(league)
    if round_num < 1 or round_num > len(picks):
        return [Text(f"round must be 1..{len(picks)}", style="red")]
    current_pick = picks[round_num - 1]
    next_pick = picks[round_num] if round_num < len(picks) else current_pick + league.num_teams
    fmt = league.scoring_format

    out: List = [
        Text.from_markup(
            f"[bold]Pick {current_pick}[/bold] (your round {round_num}, "
            f"{league.num_teams}-team, slot {league.draft_position}, {fmt}, "
            f"data: {source}) — next pick: {next_pick}\n"
            f"Your picks: {', '.join(map(str, picks))}"
        )
    ]
    mv = marginal_values(pool, current_pick, next_pick, top_n=top_n)
    order = sorted(
        (pos for pos in mv if mv[pos]),
        key=lambda pos: -max(r["marginal"] for r in mv[pos]),
    )
    for pos in order:
        rows = mv[pos]
        table = Table(title=f"{pos} — expected best at pick {next_pick}: {rows[0]['eba_next']:.0f} pts")
        table.add_column("player")
        table.add_column("proj", justify="right")
        table.add_column("ADP", justify="right")
        table.add_column("avail now", justify="right")
        table.add_column("avail next", justify="right")
        table.add_column("MV now", justify="right", style="bold")
        for r in rows:
            pl = r["player"]
            mv_style = "green" if r["marginal"] > 10 else ("red" if r["marginal"] < 0 else "yellow")
            table.add_row(
                f"{pl.name} ({pl.team})",
                f"{pl.proj:.0f}",
                f"{pl.adp:.1f}" if pl.adp else "—",
                f"{r['p_now']:.0%}",
                f"{r['p_next']:.0%}",
                f"[{mv_style}]{r['marginal']:+.0f}[/{mv_style}]",
            )
        out.append(table)
    if conn is not None:
        conn.close()
    return out


def pick_view(
    settings: Settings,
    round_num: Optional[int] = None,
    have: str = "",
    taken: str = "",
    top_n: int = 10,
    sims: int = 200,
    context: Optional[str] = None,
) -> List:
    from .value.board import snake_picks
    from .value.roster import evaluate_candidates, mc_rerank, resolve_names

    try:
        conn, league, pool, source = _pool_source(settings, context)
    except (RuntimeError, ValueError) as e:
        return [Text(str(e), style="red")]

    out: List = []
    if conn is None and sims > 0:
        sims = 0
        out.append(Text(
            f"Running from {source}: Monte Carlo needs the local db, using deterministic ranking.",
            style="dim",
        ))
    have_names = [s for s in have.split(",") if s.strip()]
    taken_names = [s for s in taken.split(",") if s.strip()]
    roster, missing_h = resolve_names(conn, have_names, pool)
    taken_players, missing_t = resolve_names(conn, taken_names, pool)
    for name in missing_h + missing_t:
        out.append(Text(f"Unrecognized player: {name!r} — check spelling", style="yellow"))
    taken_uids = frozenset(p.uid for p in taken_players)

    picks = snake_picks(league)
    rnd = round_num or len(roster) + 1
    if rnd < 1 or rnd > len(picks):
        return out + [Text(f"round must be 1..{len(picks)}", style="red")]
    current_pick = picks[rnd - 1]
    future = picks[rnd:]

    if roster:
        out.append(Text("Your roster: " + ", ".join(f"{p.name} ({p.position})" for p in roster)))
    out.append(
        Text.from_markup(
            f"[bold]On the clock: pick {current_pick}[/bold] (your round {rnd}) — "
            f"{len(future)} picks remain after this"
        )
    )

    results = evaluate_candidates(pool, roster, current_pick, future, league, taken_uids, top_n=top_n)
    if not results:
        return out + [Text("No candidates found.", style="red")]

    use_mc = sims > 0
    if use_mc:
        results = mc_rerank(conn, results, roster, pool, league, settings.current_season, n_sims=sims)
    value_of = (lambda r: r.sim.mean) if use_mc else (lambda r: r.final_value)
    best = value_of(results[0])
    title = (
        f"Best picks now — Monte Carlo over {sims} seasons, seed 7 "
        f"(weekly lineups, injuries, byes, SoS)"
        if use_mc else "Best picks now (deterministic optimal-lineup points)"
    )
    table = Table(title=title)
    table.add_column("player")
    table.add_column("pos")
    table.add_column("proj", justify="right")
    table.add_column("ADP", justify="right")
    table.add_column("season pts", justify="right")
    if use_mc:
        table.add_column("floor–ceiling", justify="right")
    table.add_column("vs best", justify="right", style="bold")
    table.add_column("then draft…", overflow="fold", max_width=30)
    for r in results:
        delta = value_of(r) - best
        style = "green" if delta == 0 else ("yellow" if delta > -8 else "red")
        row = [
            f"{r.player.name} ({r.player.team})",
            r.player.position,
            f"{r.player.proj:.0f}",
            f"{r.player.adp:.1f}" if r.player.adp else "—",
            f"{value_of(r):.0f}",
        ]
        if use_mc:
            row.append(f"{r.sim.p10:.0f}–{r.sim.p90:.0f}")
        row.append(f"[{style}]{delta:+.0f}[/{style}]")
        row.append(" ".join(r.plan_positions[:8]))
        table.add_row(*row)
    out.append(table)
    if conn is not None:
        conn.close()
    return out


def player_view(settings: Settings, name: str) -> List:
    """One player: projection, ADP, SoS, recent seasons."""
    from .data.names import name_key

    conn = connect(settings.db_path)
    fmt = settings.league.scoring_format
    row = conn.execute(
        "SELECT p.player_uid, p.name, p.position, p.team FROM players p "
        "WHERE p.name_key = ? ORDER BY p.draft_year DESC LIMIT 1",
        (name_key(name),),
    ).fetchone()
    if row is None:
        return [Text(f"No player found for {name!r}", style="red")]
    uid = row["player_uid"]
    out: List = [Text.from_markup(f"[bold]{row['name']}[/bold] — {row['position']}, {row['team']}")]

    proj = conn.execute(
        f"SELECT source, games, pts_{fmt} AS pts, bye_week FROM projections "
        "WHERE season = ? AND player_uid = ?",
        (settings.current_season, uid),
    ).fetchall()
    adp = conn.execute(
        "SELECT format, adp, stdev, high, low FROM adp WHERE season = ? AND player_uid = ?",
        (settings.current_season, uid),
    ).fetchall()
    if proj:
        t = Table(title=f"{settings.current_season} projection ({fmt})")
        for c in ("source", "games", "points", "bye"):
            t.add_column(c, justify="right")
        for r in proj:
            t.add_row(r["source"], f"{r['games']:.0f}" if r["games"] else "—",
                      f"{r['pts']:.1f}" if r["pts"] else "—", str(r["bye_week"] or "—"))
        out.append(t)
    if adp:
        t = Table(title="ADP (live mock drafts)")
        for c in ("format", "adp", "stdev", "high", "low"):
            t.add_column(c, justify="right")
        for r in adp:
            t.add_row(r["format"], f"{r['adp']:.1f}", f"{r['stdev'] or 0:.1f}",
                      str(r["high"] or "—"), str(r["low"] or "—"))
        out.append(t)

    hist = conn.execute(
        f"SELECT season, COUNT(*) g, ROUND(SUM(pts_{fmt}),1) pts, ROUND(AVG(pts_{fmt}),1) avg "
        "FROM weekly_stats WHERE player_uid = ? GROUP BY season ORDER BY season DESC LIMIT 5",
        (uid,),
    ).fetchall()
    if hist:
        t = Table(title=f"History ({fmt})")
        for c in ("season", "games", "total", "per game"):
            t.add_column(c, justify="right")
        for r in hist:
            t.add_row(str(r["season"]), str(r["g"]), str(r["pts"]), str(r["avg"]))
        out.append(t)
    conn.close()
    return out
