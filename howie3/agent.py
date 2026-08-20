"""Howie's AI layer: one Anthropic agent with native tool calling.

Replaces v2's two-generation agent stack (JSON-in-prose planning, keyword
classifiers, 500-char tool truncation) with the tools API. Tools are thin:
read-only SQL over howie.db plus the same views the CLI/TUI render.
"""

import json
import os
from typing import Iterator, List

from rich.console import Console

from .config import Settings
from .db import connect

DEFAULT_MODEL = os.environ.get("HOWIE_MODEL", "claude-sonnet-5")
MAX_TOOL_ROUNDS = 8

SYSTEM = """You are Howie, a fantasy football draft assistant backed by a local SQLite database (howie.db).

League: {league}. Scoring: {fmt}. Current season: {season}.

Schema (all player references join on player_uid; never join on names):
- players(player_uid, name, name_key, position, team, birthdate, draft_year, status)
- weekly_stats(season, week, player_uid, team, opponent, position, …stats…, pts_std, pts_half, pts_ppr) — 2018-2025 history
- projections(season, source, player_uid, position, team, bye_week, games, …stat lines…, pts_std, pts_half, pts_ppr) — source 'pff'
- adp(season, source='ffc', format std|half|ppr, player_uid, adp, rank, stdev, high, low) — live mock-draft ADP
- sos(season, team, position, week, value) — weekly matchup favorability, 0-10, higher = easier
- games(game_id, season, week, gameday, home_team, away_team, spread, total)
- team_intel(season, team, position, summary, usage_notes, coaching_style, injury_updates, confidence) — LLM scouting reports (2025 vintage unless newer)
- player_intel(season, player_uid, player_name, team, position, is_projected_starter, injury_risk, injury_details, usage_notes, confidence)

Prefer the draft tools for draft questions (they run the marginal-value engine); use SQL for stats lookups. Be concise and concrete; cite numbers from tool results only."""

TOOLS = [
    {
        "name": "query_database",
        "description": "Run a read-only SQL SELECT against howie.db. Returns up to 50 rows as JSON.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    },
    {
        "name": "draft_board",
        "description": "Marginal-value board for one of the user's draft rounds: per position, top available players with probability of availability and value of drafting now vs waiting until the next pick.",
        "input_schema": {
            "type": "object",
            "properties": {"round": {"type": "integer", "description": "Which of the user's picks (1-16)"}},
            "required": ["round"],
        },
    },
    {
        "name": "draft_pick",
        "description": "Best picks right now via Monte Carlo simulation. Optionally pass the user's current roster and players already drafted by others.",
        "input_schema": {
            "type": "object",
            "properties": {
                "round": {"type": "integer"},
                "have": {"type": "string", "description": "Comma-separated roster names"},
                "taken": {"type": "string", "description": "Comma-separated names drafted by others"},
            },
        },
    },
    {
        "name": "player_info",
        "description": "One player's projection, ADP, and recent-season history.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
]


_MAX_ROWS = 50
_MAX_VM_STEPS = 5_000_000  # progress-handler budget ~= query timeout

# SQLite authorizer action codes that a read-only query may perform
_ALLOWED_AUTH = {
    __import__("sqlite3").SQLITE_SELECT,
    __import__("sqlite3").SQLITE_READ,
    __import__("sqlite3").SQLITE_FUNCTION,
    __import__("sqlite3").SQLITE_RECURSIVE,
}


def safe_query(settings: Settings, sql: str) -> str:
    """Execute one read-only SELECT with defense in depth: statement
    validation, a read-only connection, an authorizer denying everything but
    reads, a VM-step budget, and a row cap."""
    import sqlite3

    sql = sql.strip().rstrip(";").strip()
    lowered = sql.lower()
    if not lowered.startswith("select"):
        return "Error: only a single SELECT statement is allowed."
    if any(marker in sql for marker in ("--", "/*", ";")):
        return "Error: comments and multiple statements are not allowed."
    for banned in ("pragma", "attach", "detach", "vacuum", "readfile", "writefile", "load_extension"):
        if banned in lowered:
            return f"Error: {banned.upper()} is not allowed."

    try:
        conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(
        lambda action, *_: sqlite3.SQLITE_OK if action in _ALLOWED_AUTH else sqlite3.SQLITE_DENY
    )
    steps = {"n": 0}

    def _budget():
        steps["n"] += 1
        return 1 if steps["n"] > _MAX_VM_STEPS // 1000 else 0

    conn.set_progress_handler(_budget, 1000)
    try:
        rows = conn.execute(sql).fetchmany(_MAX_ROWS)
        return json.dumps([dict(r) for r in rows], default=str)
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    finally:
        conn.close()


def _render(renderables) -> str:
    from io import StringIO

    console = Console(record=True, width=100, file=StringIO())
    for r in renderables:
        console.print(r)
    return console.export_text()


def _run_tool(name: str, args: dict, settings: Settings) -> str:
    from . import views

    if name == "query_database":
        return safe_query(settings, args.get("sql", ""))
    if name == "draft_board":
        return _render(views.board_view(settings, int(args.get("round", 1))))
    if name == "draft_pick":
        return _render(
            views.pick_view(
                settings,
                round_num=args.get("round"),
                have=args.get("have", ""),
                taken=args.get("taken", ""),
                sims=150,
            )
        )
    if name == "player_info":
        return _render(views.player_view(settings, args.get("name", "")))
    return f"Unknown tool {name}"


def run_agent(question: str, settings: Settings, model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Yields progress lines then the final answer."""
    try:
        import anthropic
    except ImportError:
        yield "[red]anthropic package not installed — pip install anthropic[/red]"
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield "[red]ANTHROPIC_API_KEY not set — add it to your environment or .env[/red]"
        return

    client = anthropic.Anthropic()
    league = settings.league
    system = SYSTEM.format(
        league=f"{league.num_teams}-team, draft slot {league.draft_position}, "
               f"{league.qb_slots}QB/{league.rb_slots}RB/{league.wr_slots}WR/"
               f"{league.te_slots}TE/{league.flex_slots}FLEX/{league.k_slots}K/{league.dst_slots}DST",
        fmt=league.scoring_format,
        season=settings.current_season,
    )
    messages: List[dict] = [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=model, max_tokens=1500, system=system, tools=TOOLS, messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            yield "".join(b.text for b in response.content if b.type == "text")
            return
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            yield f"[dim]→ {tu.name}({json.dumps(tu.input)[:90]})[/dim]"
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": _run_tool(tu.name, tu.input, settings)[:20000],
                }
            )
        messages.append({"role": "user", "content": results})
    yield "[yellow]Stopped after too many tool rounds.[/yellow]"
