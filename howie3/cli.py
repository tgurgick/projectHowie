"""howie3 command-line entry point. All output is built in views.py; this
module only parses options and prints."""

from typing import Optional

import click
from rich.console import Console

from . import views
from .config import Settings, parse_seasons

console = Console()


def _show(renderables) -> None:
    for r in renderables:
        console.print(r)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Howie v3 — fantasy football engine. Run with no command for the TUI."""
    if ctx.invoked_subcommand is None:
        from .tui.app import HowieApp

        HowieApp().run()


@main.group()
def data() -> None:
    """Build and inspect howie.db."""


@data.command()
@click.option("--seasons", "seasons_spec", default=None, help="e.g. 2018-2025 (default: all completed)")
@click.option("--steps", default=None, help="comma list: crosswalk,players,dst,games,weekly,adp,pff,sos")
def refresh(seasons_spec: Optional[str], steps: Optional[str]) -> None:
    """Refresh howie.db from all sources (idempotent). Exits nonzero if any step fails."""
    from .data.refresh import run_refresh

    settings = Settings()
    console.print(f"[bold]Refreshing[/bold] {settings.db_path}")
    seasons = parse_seasons(seasons_spec) if seasons_spec else None
    step_list = [s.strip() for s in steps.split(",")] if steps else None
    try:
        results = run_refresh(settings, seasons, step_list)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(2)
    _show(views.render_refresh_results(results))
    if any(status == "error" for _, status, _, _ in results):
        raise SystemExit(1)


@data.command()
def status() -> None:
    """Row counts, season coverage, and recent refresh history."""
    _show(views.status_view(Settings()))


@main.group()
def draft() -> None:
    """Draft analysis (marginal-value engine)."""


@draft.command()
@click.option("--round", "round_num", default=1, help="Which of your picks to analyze (1 = your first)")
@click.option("--top", "top_n", default=5, help="Candidates to show per position")
@click.option("--context", default=None, help="Run from a strategy-context artifact instead of the local db")
def board(round_num: int, top_n: int, context: Optional[str]) -> None:
    """Marginal value board for one of your picks."""
    _show(views.board_view(Settings(), round_num, top_n, context=context))


@draft.command()
@click.option("--round", "round_num", default=None, type=int,
              help="Which of your picks this is (default: inferred from roster size)")
@click.option("--have", default="", help="Comma list of players already on YOUR roster")
@click.option("--taken", default="", help="Comma list of players drafted by OTHERS")
@click.option("--top", "top_n", default=10, help="Candidates to show")
@click.option("--sims", default=200, help="Monte Carlo sims (0 = deterministic only)")
@click.option("--context", default=None, help="Run from a strategy-context artifact instead of the local db")
def pick(round_num: Optional[int], have: str, taken: str, top_n: int, sims: int,
         context: Optional[str]) -> None:
    """What to take right now, given your roster and who's gone."""
    _show(views.pick_view(Settings(), round_num, have, taken, top_n, sims, context=context))


@main.group()
def context() -> None:
    """Portable strategy-context artifacts (derived data only, safe to share)."""


@context.command("export")
@click.option("--out", "out_path", default="strategy-context.json", help="Output file")
@click.option("--sims", default=300, help="Simulation runs per player")
def context_export(out_path: str, sims: int) -> None:
    """Export the derived strategy context (no raw provider data)."""
    from pathlib import Path

    from .context_artifact import export_context

    artifact = export_context(Settings(), Path(out_path), n_sims=sims)
    console.print(
        f"[green]Wrote {out_path}[/green] — {len(artifact['players'])} players, "
        f"schema v{artifact['schema_version']}, derived fields only."
    )


@context.command("inspect")
@click.argument("path")
def context_inspect(path: str) -> None:
    """Validate an artifact and summarize its contents."""
    import json as _json
    from pathlib import Path

    from .context_artifact import inspect_context

    console.print_json(_json.dumps(inspect_context(Path(path))))


@context.command("import")
@click.argument("path")
def context_import(path: str) -> None:
    """Install an artifact as this machine's default data source fallback."""
    import shutil
    from pathlib import Path

    from .context_artifact import default_context_path, inspect_context

    info = inspect_context(Path(path))  # validates before install
    dest = default_context_path(Settings())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dest)
    console.print(f"[green]Imported to {dest}[/green] ({info['players']} players). "
                  "Draft views will use it whenever the local db is absent.")


@main.group()
def graph() -> None:
    """Knowledge graph: entities, edges, facts, fast search."""


@graph.command("rebuild")
def graph_rebuild() -> None:
    """Rebuild the derived layer (rooms, shares, vacated volume, team YoY)."""
    from .db import connect
    from .graph import rebuild_derived

    settings = Settings()
    conn = connect(settings.db_path)
    n = rebuild_derived(conn, settings.current_season)
    conn.close()
    console.print(f"[green]Graph rebuilt: {n} derived entities/edges/facts[/green]")


@graph.command("search")
@click.argument("query", nargs=-1, required=True)
def graph_search(query) -> None:
    """Lightning search over players, teams, and rooms."""
    from .db import connect
    from .graph import search as g_search

    conn = connect(Settings().db_path)
    for r in g_search(conn, " ".join(query)):
        console.print(f"[dim]{r['kind']:<7}[/dim] {r['name']}"
                      + (f"  [dim]{r['position'] or ''} {r['team'] or ''}[/dim]"))
    conn.close()


@graph.command("context")
@click.argument("query", nargs=-1, required=True)
def graph_context(query) -> None:
    """1-hop context for an entity (what the agent and player card consume)."""
    import json as _json

    from .db import connect
    from .graph import entity_context, search as g_search

    conn = connect(Settings().db_path)
    hits = g_search(conn, " ".join(query), limit=1)
    if not hits:
        console.print("[red]No entity found[/red]")
        raise SystemExit(1)
    console.print_json(_json.dumps(entity_context(conn, hits[0]["id"]), default=str))
    conn.close()


@graph.command("import")
@click.argument("path")
def graph_import(path: str) -> None:
    """Import researched facts (the research-skill output contract)."""
    from pathlib import Path

    from .db import connect
    from .graph import import_facts

    settings = Settings()
    conn = connect(settings.db_path)
    n = import_facts(conn, Path(path), settings.current_season)
    conn.close()
    console.print(f"[green]Imported {n} facts/edges from {path}[/green]")


@main.command()
@click.argument("name", nargs=-1, required=True)
def player(name) -> None:
    """One player's projection, ADP, and history."""
    _show(views.player_view(Settings(), " ".join(name)))


@main.group(name="eval")
def eval_group() -> None:
    """Backtests against realized results."""


@eval_group.command("run")
@click.option("--reps", default=3, help="Draft replays per slot per policy")
@click.option("--skip-policy", is_flag=True, help="Only run input/calibration tiers")
def eval_run(reps: int, skip_policy: bool) -> None:
    """Evaluate projections, calibration, and draft policy on 2025 actuals."""
    from rich.table import Table as RTable

    from .evals import (eval_calibration, eval_inputs_report, eval_policy,
                        eval_sos, load_eval_players)

    settings = Settings()
    players = load_eval_players(settings)
    console.print(f"Loaded {len(players)} 2025 players with preseason projections + actuals\n")

    t = RTable(title="A · 2025 preseason projection quality (top-of-pool)")
    for c in ("pos", "n", "MAE (pts)", "rank corr"):
        t.add_column(c, justify="right")
    for r in eval_inputs_report(players):
        t.add_row(r["pos"], str(r["n"]), str(r["proj_mae"]), str(r["rank_corr"]))
    console.print(t)

    cal = eval_calibration(settings, players)
    console.print(
        f"\nB · calibration: p10–p90 coverage {cal['coverage_all']:.0%} all / "
        f"{cal['coverage_8plus_games']:.0%} with 8+ games (target ~{cal['target']:.0%}, "
        f"n={cal['n']}; buckets fit on ≤2024 only)\n"
    )

    sos = eval_sos(settings, players)
    if sos.get("available"):
        console.print(
            f"D · does preseason SoS predict anything? season-level corr "
            f"{sos['season_corr']:+.3f} (n={sos['season_n']}; by pos "
            + ", ".join(f"{k} {v:+.2f}" for k, v in sos["season_by_pos"].items())
            + f") · weekly within-player corr {sos['weekly_corr']:+.3f} "
            f"(n={sos['weekly_n']}). ~0 on both = keep SoS normalized, playoff_weight neutral.\n"
        )

    if not skip_policy:
        t = RTable(title=f"C · 2025 draft replay, realized weekly scoring ({reps} reps × 4 slots)")
        for c in ("policy", "mean pts", "std", "vs ADP", "drafts"):
            t.add_column(c, justify="right")
        for r in eval_policy(settings, players, reps=reps):
            style = "green" if r["policy"] == "howie" and r["vs_adp"] > 0 else ""
            t.add_row(r["policy"], str(r["mean"]), str(r["std"]),
                      f"[{style}]{r['vs_adp']:+}[/{style}]" if style else f"{r['vs_adp']:+}",
                      str(r["n_drafts"]))
        console.print(t)


@main.command()
@click.option("--port", default=8787, help="Localhost port")
def serve(port: int) -> None:
    """Launch the draft-night cockpit (local web UI)."""
    from .server import main as serve_main

    serve_main(port=port)


@main.command()
def tui() -> None:
    """Launch the Howie TUI."""
    from .tui.app import HowieApp

    HowieApp().run()


@main.command()
@click.argument("question", nargs=-1, required=True)
def ask(question) -> None:
    """Ask Howie a question in natural language (needs ANTHROPIC_API_KEY)."""
    from .agent import AgentEventType, run_agent_events

    for event in run_agent_events(" ".join(question), Settings()):
        if event.kind == AgentEventType.TEXT:
            console.print(event.text)
        elif event.kind == AgentEventType.TOOL_CALL:
            console.print(f"[dim]→ {event.text}[/dim]")
        elif event.kind == AgentEventType.RETRY:
            console.print(f"[yellow]{event.text}[/yellow]")
        elif event.kind in {AgentEventType.ERROR, AgentEventType.STOP}:
            console.print(f"[yellow]{event.text}[/yellow]")


if __name__ == "__main__":
    main()
