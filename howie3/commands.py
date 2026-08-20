"""The single command registry shared by the CLI and TUI.

Each command is a spec: name, usage, help, and a runner that takes tokens
(shlex-style) and returns Rich renderables from views.py. The TUI dispatches
typed lines here; the click CLI wraps the same views with typed options.
"""

import shlex
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from rich.table import Table
from rich.text import Text

from .config import Settings, parse_seasons
from . import views


@dataclass
class CommandSpec:
    name: str
    usage: str
    help: str
    run: Callable[[Settings, List[str]], List]


def _kwargs(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Split tokens into positionals and key=value options."""
    pos, kw = [], {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kw[k.strip().lower()] = v
        else:
            pos.append(t)
    return pos, kw


def _run_status(settings: Settings, tokens: List[str]) -> List:
    return views.status_view(settings)


def _run_refresh(settings: Settings, tokens: List[str]) -> List:
    pos, kw = _kwargs(tokens)
    seasons = parse_seasons(kw["seasons"]) if "seasons" in kw else None
    steps = pos or None
    return views.refresh_view(settings, seasons, steps)


def _run_board(settings: Settings, tokens: List[str]) -> List:
    pos, kw = _kwargs(tokens)
    round_num = int(pos[0]) if pos else 1
    return views.board_view(settings, round_num, int(kw.get("top", 5)),
                            context=kw.get("context"))


def _run_pick(settings: Settings, tokens: List[str]) -> List:
    pos, kw = _kwargs(tokens)
    round_num = int(pos[0]) if pos else None
    return views.pick_view(
        settings,
        round_num=round_num,
        have=kw.get("have", ""),
        taken=kw.get("taken", ""),
        top_n=int(kw.get("top", 10)),
        sims=int(kw.get("sims", 200)),
        context=kw.get("context"),
    )


def _run_context(settings: Settings, tokens: List[str]) -> List:
    from pathlib import Path

    from .context_artifact import export_context, inspect_context

    pos, kw = _kwargs(tokens)
    action = pos[0] if pos else "export"
    if action == "export":
        out = Path(kw.get("out", "strategy-context.json"))
        artifact = export_context(settings, out, n_sims=int(kw.get("sims", 300)))
        return [Text(f"Wrote {out} — {len(artifact['players'])} players, derived fields only.",
                     style="green")]
    if action == "inspect" and len(pos) > 1:
        import json as _json
        return [Text(_json.dumps(inspect_context(Path(pos[1])), indent=1))]
    return [Text("usage: context export [out=file] | context inspect <file>", style="yellow")]


def _run_player(settings: Settings, tokens: List[str]) -> List:
    if not tokens:
        return [Text("usage: player <name>", style="yellow")]
    return views.player_view(settings, " ".join(tokens))


def _run_ask(settings: Settings, tokens: List[str]) -> List:
    if not tokens:
        return [Text("usage: ask <question>", style="yellow")]
    from rich.markdown import Markdown

    from .agent import AgentEventType, run_agent_events

    out: List = []
    for event in run_agent_events(" ".join(tokens), settings):
        if event.kind == AgentEventType.TEXT:
            out.append(Markdown(event.text))
        elif event.kind == AgentEventType.TOOL_CALL:
            out.append(Text(f"→ {event.text}", style="dim"))
        elif event.kind == AgentEventType.RETRY:
            out.append(Text(event.text, style="yellow"))
        elif event.kind in {AgentEventType.ERROR, AgentEventType.STOP}:
            out.append(Text(event.text, style="yellow"))
    return out


def _run_help(settings: Settings, tokens: List[str]) -> List:
    table = Table(title="Commands")
    table.add_column("command")
    table.add_column("what it does")
    for spec in REGISTRY.values():
        table.add_row(spec.usage, spec.help)
    return [table]


REGISTRY: Dict[str, CommandSpec] = {
    spec.name: spec
    for spec in [
        CommandSpec("help", "help", "Show this command list", _run_help),
        CommandSpec("status", "status", "Database row counts and refresh history", _run_status),
        CommandSpec(
            "refresh", "refresh [steps…] [seasons=2018-2025]",
            "Refresh howie.db from sources (all steps by default)", _run_refresh,
        ),
        CommandSpec(
            "board", "board [round] [top=5]",
            "Marginal value per position at one of your picks", _run_board,
        ),
        CommandSpec(
            "pick", "pick [round] [have=\"A, B\"] [taken=\"C, D\"] [sims=200] [top=10]",
            "Best picks right now, Monte Carlo over full seasons", _run_pick,
        ),
        CommandSpec("player", "player <name>", "One player's projection, ADP, and history", _run_player),
        CommandSpec("ask", "ask <question>", "Ask Howie in natural language (AI agent)", _run_ask),
        CommandSpec(
            "context", "context export [out=file] | context inspect <file>",
            "Portable strategy-context artifact (derived data only)", _run_context,
        ),
    ]
}


def dispatch(settings: Settings, line: str) -> Optional[List]:
    """Run one typed command line; returns renderables, or None for empty input."""
    line = line.strip().lstrip("/")
    if not line:
        return None
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        return [Text(f"Parse error: {e}", style="red")]
    name, args = tokens[0].lower(), tokens[1:]
    spec = REGISTRY.get(name)
    if spec is None:
        return [Text(f"Unknown command {name!r} — try `help`", style="yellow")]
    try:
        return spec.run(settings, args)
    except Exception as e:  # surface, don't crash the UI
        return [Text(f"{e.__class__.__name__}: {e}", style="red")]
