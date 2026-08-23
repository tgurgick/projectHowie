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


@draft.command("reset")
@click.option("--mode", type=click.Choice(["live", "mock"]), default="live")
@click.option("--slot", type=int, default=None, help="Also set your draft slot in league_config.json")
def draft_reset(mode: str, slot: Optional[int]) -> None:
    """Wipe the board (the finished draft is archived into the Mock Draft Lab)."""
    from . import service

    settings = Settings()
    if slot is not None:
        service.update_config(settings, {"draft_position": slot})
    r = service.start_mock(settings) if mode == "mock" else service.reset_draft(settings, "live")
    console.print(f"[green]{mode} draft ready[/green]" + (" · previous draft archived to the lab" if r.get("archived") else ""))


@draft.command("mark")
@click.argument("name", nargs=-1, required=True)
@click.option("--mine", is_flag=True, help="Draft the player to your team (default: taken by another team)")
def draft_mark(name, mine: bool) -> None:
    """Record one pick in the shared draft log."""
    from . import service

    settings = Settings()
    hits = [h for h in service.search_payload(settings, " ".join(name)) if h.get("uid")]
    if not hits:
        raise click.ClickException(f"No player found for {' '.join(name)!r}")
    r = service.mark_pick(settings, hits[0]["uid"], mine=mine, source="cli")
    console.print(f"{'drafted' if mine else 'taken'}: {r['name']} (pick {r['pick_no']})")


@draft.command("sync")
@click.argument("names", nargs=-1)
@click.option("--file", "path", default=None, help="File with one player per line (numbers/positions ok); '-' for stdin")
def draft_sync(names, path: Optional[str]) -> None:
    """Bring the log up to an observed pick order (idempotent; your slot's
    picks are recorded as yours). Used by the draft-observer skill."""
    import re
    import sys

    from . import service

    lines = list(names)
    if path:
        text = sys.stdin.read() if path == "-" else open(path).read()
        lines += [ln for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\s*[\d]+[\.\)\-:\s]*", "", ln).strip()
        ln = re.split(r"\s{2,}|\t|,\s*(?:QB|RB|WR|TE|K|DST|D/ST)\b", ln)[0].strip()
        ln = re.sub(r"\s+(QB|RB|WR|TE|K|DST|D/ST)(\s+[A-Z]{2,3})?$", "", ln).strip()
        if ln:
            cleaned.append(ln)
    r = service.sync_picks(Settings(), cleaned)
    for a in r["added"]:
        console.print(f"  pick {a['pick_no']}: {a['name']}" + (" [green](you)[/green]" if a["mine"] else ""))
    console.print(f"added {len(r['added'])} · already logged {r['skipped']} · next pick {r['next_pick']}"
                  + (" · [green]YOU ARE ON THE CLOCK[/green]" if r["on_clock"] else ""))
    if r["unresolved"]:
        console.print(f"[yellow]unresolved: {', '.join(r['unresolved'])}[/yellow]")


@draft.command("best")
@click.option("--top", "top_n", default=5)
@click.option("--sims", default=150, help="Monte Carlo sims (0 = deterministic)")
def draft_best(top_n: int, sims: int) -> None:
    """Howie's ranked picks for the live draft log as JSON — what the
    autopilot clicks (top row) and the alternatives."""
    import json as _json

    from . import service
    from .state import DraftState

    settings = Settings()
    st = DraftState.load(settings)
    pk = service.pick_payload(settings, st, sims=sims, top_n=top_n)
    sp = service.state_payload(settings, st)
    click.echo(_json.dumps({
        "on_clock": sp["you_are_on_clock"], "pick": pk["current_pick"], "next_pick": pk["next_pick"], "round": pk["round"],
        "best": [{"name": r["name"], "pos": r["pos"], "team": r["team"], "value": r["value"], "delta": r["delta"],
                  "avail_next": r["avail_next"], "rules": [f["text"] for f in r["rules"]],
                  "status": (r.get("status") or {}).get("text")} for r in pk["rows"][:top_n]],
        "roster": [x["name"] for x in sp["roster"] if x["name"]],
    }))


@draft.command("log")
def draft_log() -> None:
    """The draft log as it stands."""
    from .state import DraftState

    settings = Settings()
    st = DraftState.load(settings)
    console.print(f"{st.mode} · {len(st.events)} picks · next pick {st.next_pick_no()}")
    for e in st.events[-15:]:
        console.print(f"  {e.pick_no:>3} {e.player_name} ({e.position}) " + ("[green]YOU[/green]" if e.mine else f"T{e.team}"))


@main.group()
def coach() -> None:
    """Coached simulation: the engine drafts, Claude coaches the strategy sheet."""


@coach.command("run")
@click.option("--iterations", default=3, help="Coaching rounds")
@click.option("--drafts", default=12, help="Simulated drafts per round")
@click.option("--reps", default=6, help="2025 replay reps per slot (paired vs ADP)")
@click.option("--seed", default=101)
def coach_run(iterations: int, drafts: int, reps: int, seed: int) -> None:
    """Simulate → score → coach → apply, keeping the best rule set."""
    from . import coach as coach_mod

    session = coach_mod.run_session(Settings(), iterations=iterations, n_drafts=drafts, reps=reps, seed=seed)
    _print_session(session)


def _print_session(session: dict) -> None:
    from rich.table import Table as RTable

    t = RTable(title="coaching session")
    for c in ("iter", "rules", "MC mean", "MC p10", "holes", "2025 replay", "vs ADP", "best"):
        t.add_column(c)
    b = session.get("baseline")
    if b:
        rp = b.get("replay") or {}
        t.add_row("base", "(no rules)", str(b["sim"]["mc_mean"]), str(b["sim"]["mc_p10"]), str(b["sim"]["holes"]),
                  str(rp.get("mean_total", "—")), f"{rp.get('delta_vs_adp', '—')}", "")
    for it in session.get("iterations", []):
        sc = it["score"]; rp = sc.get("replay") or {}
        t.add_row(str(it["iteration"]), " · ".join(it["rules"]) or "(none)", str(sc["sim"]["summary"]["mc_mean"]),
                  str(sc["sim"]["summary"]["mc_p10"]), str(sc["sim"]["summary"]["holes"]),
                  str(rp.get("mean_total", "—")), f"{rp.get('delta_vs_adp', '—')} {rp.get('ci', '')}",
                  "★" if it.get("best") else "")
    f = session.get("final")
    if f:
        rp = f.get("replay") or {}
        t.add_row("final", " · ".join(f["rules"]) or "(none)", str(f["sim"]["mc_mean"]), str(f["sim"]["mc_p10"]),
                  str(f["sim"]["holes"]), str(rp.get("mean_total", "—")), f"{rp.get('delta_vs_adp', '—')}", "")
    console.print(t)
    for it in session.get("iterations", []):
        for l in it.get("learnings", []):
            console.print(f"  [dim]iter {it['iteration']}[/dim] {l}")
    if session.get("stopped"):
        console.print(f"[yellow]stopped: {session['stopped']}[/yellow]")
    console.print(f"[green]kept rule set:[/green] {' · '.join(session.get('best_rules') or []) or '(none)'}")


@coach.command("review")
def coach_review() -> None:
    """Coach the current draft log (e.g. a real room observed with Claude in Chrome)."""
    import json as _json

    from . import coach as coach_mod
    from .state import DraftState

    settings = Settings()
    st = DraftState.load(settings)
    picks = [{"uid": e.player_uid, "name": e.player_name} for e in st.events if e.mine]
    if not picks:
        raise click.ClickException("no picks of yours in the draft log")
    r = coach_mod.review_draft(settings, picks)
    console.print_json(_json.dumps(r["digest"]["this_draft"]))
    c = r["coach"]
    if not c.get("available"):
        raise click.ClickException(c.get("reason", "coach unavailable"))
    for l in c.get("learnings", []):
        console.print(f"  • {l}")
    if c.get("rules_add") or c.get("rules_remove"):
        console.print(f"suggested: add {c.get('rules_add')} · remove {c.get('rules_remove')} — apply with `howie strategy add/remove`")


@main.group()
def strategy() -> None:
    """The strategy sheet from the command line (what the coach edits)."""


@strategy.command("show")
def strategy_show() -> None:
    from .state import DraftState

    st = DraftState.load(Settings())
    for r in st.rules:
        console.print(f"  {'●' if r.on else '○'} {r.text}")
    if st.notes:
        console.print(st.notes)


@strategy.command("add")
@click.argument("text", nargs=-1, required=True)
def strategy_add(text) -> None:
    from . import service

    st = __import__("howie3.state", fromlist=["DraftState"]).DraftState.load(Settings())
    rules = [{"text": r.text, "on": r.on} for r in st.rules] + [{"text": " ".join(text), "on": True}]
    out = service.update_strategy(Settings(), rules=rules)
    for c in out.get("conflicts", []):
        console.print(f"[yellow]{c}[/yellow]")
    console.print("added")


@strategy.command("remove")
@click.argument("text", nargs=-1, required=True)
def strategy_remove(text) -> None:
    from . import service

    st = __import__("howie3.state", fromlist=["DraftState"]).DraftState.load(Settings())
    key = " ".join(text).strip().upper()
    rules = [{"text": r.text, "on": r.on} for r in st.rules if r.text.strip().upper() != key]
    service.update_strategy(Settings(), rules=rules)
    console.print("removed" if len(rules) < len(st.rules) else "no such rule")


@strategy.command("note")
@click.argument("text", nargs=-1, required=True)
def strategy_note(text) -> None:
    from . import service

    st = __import__("howie3.state", fromlist=["DraftState"]).DraftState.load(Settings())
    notes = (st.notes + "\n\n" if st.notes else "") + " ".join(text)
    service.update_strategy(Settings(), notes=notes)
    console.print("noted")


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


@main.group()
def research() -> None:
    """Research bookkeeping: who to research, what is covered, what is stale."""


@research.command("targets")
@click.argument("team")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable (for the research workflow)")
def research_targets_cmd(team: str, as_json: bool) -> None:
    """Every draft-relevant player on TEAM with the status we already hold."""
    import json as _json

    from .db import connect
    from .status import research_targets

    settings = Settings()
    conn = connect(settings.db_path)
    rows = research_targets(conn, settings.current_season, team, settings.league.scoring_format)
    conn.close()
    if as_json:
        click.echo(_json.dumps(rows))
        return
    from rich.table import Table as RTable

    t = RTable(title=f"{team.upper()} — {len(rows)} players to cover")
    for c in ("pos", "player", "proj", "ADP", "known status"):
        t.add_column(c)
    for r in rows:
        t.add_row(r["position"], r["name"], str(r["proj"]), f"{r['adp']:.0f}" if r["adp"] else "—", r["known_status"])
    console.print(t)


@research.command("coverage")
def research_coverage_cmd() -> None:
    """Per-team research coverage and freshness."""
    from rich.table import Table as RTable

    from .db import connect
    from .status import research_coverage

    settings = Settings()
    conn = connect(settings.db_path)
    rows = research_coverage(conn, settings.current_season)
    conn.close()
    t = RTable(title="research coverage")
    for c in ("team", "players", "researched", "facts", "latest"):
        t.add_column(c, justify="right")
    for r in rows:
        style = "green" if r["players_researched"] >= r["targets"] and r["targets"] else ("yellow" if r["facts"] else "red")
        t.add_row(f"[{style}]{r['team']}[/{style}]", str(r["targets"]), str(r["players_researched"]),
                  str(r["facts"]), r["latest"] or "—")
    console.print(t)


@research.command("stale")
@click.option("--days", default=7, help="Research older than this is stale")
def research_stale_cmd(days: int) -> None:
    """Teams that need (re)research — paste the list into the workflow."""
    from .db import connect
    from .status import stale_teams

    settings = Settings()
    conn = connect(settings.db_path)
    teams = stale_teams(conn, settings.current_season, days)
    conn.close()
    click.echo(" ".join(teams) if teams else "nothing stale")


@main.command()
@click.argument("name", nargs=-1, required=True)
def player(name) -> None:
    """One player's projection, ADP, and history."""
    _show(views.player_view(Settings(), " ".join(name)))


@main.group(name="eval")
def eval_group() -> None:
    """Backtests against realized results."""


@eval_group.command("run")
@click.option("--reps", default=10, help="Paired draft replays per slot (each policy drafts every one)")
@click.option("--skip-policy", is_flag=True, help="Only run input/calibration tiers")
def eval_run(reps: int, skip_policy: bool) -> None:
    """Evaluate projections, calibration, and draft policy on 2025 actuals."""
    from rich.table import Table as RTable

    from .evals import (BASELINE_POLICY, BOOTSTRAP_RESAMPLES, EVAL_SLOTS,
                        eval_calibration, eval_inputs_report, eval_policy,
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
            f"(n={sos['weekly_n']}).\n    Decomposition — hindsight matchup quality → weekly scoring: "
            f"{sos['hindsight_corr']:+.3f} (hardest vs softest quartile: "
            f"{sos['hindsight_hard_vs_soft'][0]}x vs {sos['hindsight_hard_vs_soft'][1]}x own mean); "
            f"preseason grade → realized defense: {sos['forecast_corr']:+.3f}.\n"
            "    Small effect × weak forecast → keep SoS normalized, playoff_weight neutral.\n"
        )

    if not skip_policy:
        report = eval_policy(settings, players, reps=reps)
        n = next(iter(report.values()))["n"]
        t = RTable(
            title=f"C · 2025 draft replay, realized weekly scoring "
                  f"({reps} reps × {len(EVAL_SLOTS)} slots = {n} paired drafts per policy)")
        for c in ("policy", "mean pts", "std", "vs ADP", "95% CI", "win rate", "n", "verdict"):
            t.add_column(c, justify="right", no_wrap=c in ("95% CI", "verdict"))
        for policy, r in report.items():
            if policy == BASELINE_POLICY:
                t.add_row(policy, str(r["mean_total"]), str(r["std_total"]),
                          "baseline", "—", "—", str(r["n"]), "")
                continue
            ci = f"[{r['ci_low']:+.0f}, {r['ci_high']:+.0f}]"
            if r["ci_low"] > 0:
                style, verdict = "green", "beats ADP"
            elif r["ci_high"] < 0:
                style, verdict = "red", "loses to ADP"
            else:
                style, verdict = "yellow", "CI crosses 0"
            t.add_row(policy, str(r["mean_total"]), str(r["std_total"]),
                      f"[{style}]{r['delta_vs_adp']:+.0f}[/{style}]",
                      f"[{style}]{ci}[/{style}]",
                      f"{r['win_rate']:.0%}", str(r["n"]),
                      f"[{style}]{verdict}[/{style}]")
        console.print(t)
        for policy, r in report.items():
            if policy != BASELINE_POLICY:
                console.print(
                    f"  {policy:<9}{r['delta_vs_adp']:+.0f}  "
                    f"[95% CI  {r['ci_low']:+.0f}, {r['ci_high']:+.0f}]  n={r['n']}"
                    + ("  (crosses zero)" if r["crosses_zero"] else ""))
        console.print(
            f"\n  Paired: every policy drafts the same {n} (slot, rep) replays against the same "
            f"seeded opponents; vs ADP is the mean paired difference, CI is a percentile "
            f"bootstrap ({BOOTSTRAP_RESAMPLES} resamples) of that mean, win rate is the share "
            f"of replays the policy out-scored ADP. One season of preseason inputs (2025) — "
            f"reps vary opponent noise, not the year.\n"
        )


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
