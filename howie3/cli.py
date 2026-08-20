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


@main.command()
@click.argument("name", nargs=-1, required=True)
def player(name) -> None:
    """One player's projection, ADP, and history."""
    _show(views.player_view(Settings(), " ".join(name)))


@main.command()
def tui() -> None:
    """Launch the Howie TUI."""
    from .tui.app import HowieApp

    HowieApp().run()


@main.command()
@click.argument("question", nargs=-1, required=True)
def ask(question) -> None:
    """Ask Howie a question in natural language (needs ANTHROPIC_API_KEY)."""
    from .agent import run_agent

    for chunk in run_agent(" ".join(question), Settings()):
        console.print(chunk)


if __name__ == "__main__":
    main()
