"""Howie's voice: LLM insights over engine/lab outputs, and deep research.

Two jobs, both optional (they need ANTHROPIC_API_KEY; everything else in the
app works without it):

- insights(kind, payload): 3 key learnings + concrete strategy suggestions
  (pinned rules / notes) from a lab result, a player sim, or the live board.
- research_team / research_player: web-search-backed research that ends in
  STRUCTURED facts (the skills/ contract) imported into the knowledge graph,
  never prose — so cards and the agent's entity_context pick them up.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .config import Settings


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set — add it to .env to enable Howie's takes"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed (pip install -e '.[ai]')"
    return anthropic.Anthropic(), None


def _model() -> str:
    return os.environ.get("HOWIE_MODEL", "claude-sonnet-5")


def _json_block(text: str) -> Optional[dict]:
    """Find the first decodable JSON object in model output: whole text, a
    ```json fence, or a raw_decode from each '{' (tolerates trailing prose)."""
    text = text.strip()
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


_INSIGHT_PROMPTS = {
    "mock": "Results of a batch of mock drafts for the user's league: per pick, which players were "
            "usually still available (sim availability) versus the engine's ADP-model estimate. "
            "Find where the market behaves differently than the model assumes and what that means "
            "for the user's draft plan.",
    "player": "A player's simulated season-total distribution (p10/p50/p90), his actual past "
              "seasons, his milestone rates, and knowledge-graph context. Judge whether the "
              "projection looks right and what the risk profile implies.",
    "draft": "The live draft board: the user's roster, who is gone, the ranked candidates with "
             "expected final-lineup value and availability, and the positional now-vs-wait costs. "
             "Advise the pick and any strategy adjustments.",
}


def _compact_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """What a model needs from the sheet: ON rules and the newest notes,
    capped — the sheet is the user's, the context budget is not."""
    rules = [r["text"] for r in strategy.get("rules", []) if r.get("on") and not r.get("inert")]
    notes = [n.strip() for n in (strategy.get("notes") or "").split("\n") if n.strip()]
    return {"rules": rules, "notes": notes[::-1][:6], "notes_chars_cap": True} if len("".join(notes)) > 900 else {"rules": rules, "notes": notes}


def generate_insights(settings: Settings, kind: str, payload: Dict[str, Any]) -> dict:
    from . import egress

    client, err = _client()
    if client is None:
        return {"available": False, "reason": err}
    payload = egress.redact(payload)  # the one egress boundary for insight requests
    league = settings.league
    system = (
        "You are Howie, a sharp, concise fantasy-football draft analyst. League: "
        f"{league.num_teams}-team, draft slot {league.draft_position}, {league.scoring_format}. "
        "Respond with ONLY a JSON object: {\"learnings\": [exactly 3 separate strings, one specific sentence each], "
        "\"suggestions\": [{\"type\": \"rule\"|\"note\", \"text\": \"...\", \"why\": \"...\"}]}. "
        "Rules must use these exact patterns so the engine can enforce them: "
        "'WAIT <POS> UNTIL R<n>', 'TARGET <Player Name>', 'NO <POS> BEFORE R<n>'. "
        "Notes are free text (one line). Zero suggestions is fine if nothing should change. "
        "Keep each learning under 35 words and at most 3 suggestions. Cite numbers from the data. No preamble."
    )
    user = (
        f"{_INSIGHT_PROMPTS.get(kind, 'Analyze this data.')}\n\n"
        f"Current strategy sheet: {json.dumps(_compact_strategy(payload.get('strategy', {})))}\n\n"
        f"DATA:\n{json.dumps(payload.get('data', {}), default=str)[:14000]}"
    )
    try:
        resp = client.messages.create(model=_model(), max_tokens=2500, system=system,
                                      messages=[{"role": "user", "content": user}])
    except Exception as e:
        return {"available": False, "reason": f"{e.__class__.__name__}: {e}"}
    text = "".join(getattr(b, "text", "") for b in resp.content)
    parsed = _json_block(text) or _salvage(text)
    return {"available": True, "model": _model(), **_normalize(parsed)}


def _salvage(text: str) -> dict:
    """A truncated JSON reply still holds complete learning strings — keep those."""
    head = text.split('"suggestions"')[0]
    strings = [m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.){20,})"', head)]
    strings = [x.replace('\\"', '"') for x in strings if x != "learnings"]
    return {"learnings": strings[:3], "suggestions": []} if strings else {"learnings": [text.strip()[:300]], "suggestions": []}


def _normalize(parsed: dict) -> dict:
    """Models drift from the contract (one long learning string, suggestions
    as bare strings, different keys). Coerce to the shape the UI renders."""
    learnings = parsed.get("learnings") or parsed.get("insights") or parsed.get("takeaways") or []
    if not learnings:  # unknown key: take the first list of strings in the object
        for key, val in parsed.items():
            if isinstance(val, list) and val and all(isinstance(v, str) for v in val) \
                    and "suggest" not in key.lower() and "recommend" not in key.lower():
                learnings = val
                break
    if isinstance(learnings, str):
        learnings = [learnings]
    flat: List[str] = []
    for item in learnings:
        text = item if isinstance(item, str) else json.dumps(item)
        # split a single paragraph holding several sentences into separate points
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if len(flat) == 0 and len(learnings) == 1 else [text]
        flat.extend(p.strip() for p in parts if p.strip())
    suggestions = []
    for sg in parsed.get("suggestions") or parsed.get("recommendations") or []:
        if isinstance(sg, str):
            kind = "rule" if re.match(r"(?i)^(WAIT|TARGET|NO)\b", sg.strip()) else "note"
            suggestions.append({"type": kind, "text": sg.strip(), "why": ""})
        elif isinstance(sg, dict):
            suggestions.append({"type": "rule" if str(sg.get("type", "")).lower().startswith("rule") else "note",
                                "text": str(sg.get("text") or sg.get("suggestion") or "").strip(),
                                "why": str(sg.get("why") or sg.get("reason") or "")})
    return {"learnings": flat[:5], "suggestions": [s for s in suggestions if s["text"]][:5]}


# ------------------------------------------------------------ deep research

_RESEARCH_SYSTEM = (
    "You are an NFL research analyst feeding a fantasy-football knowledge graph for the "
    "{season} season. Use web search (beat reporters, team sites, injury reports, depth charts). "
    "Your ONLY output is a JSON object in this exact contract:\n"
    "{{\"facts\": [{{\"entity\": \"team:ABBR\" | \"unit:ABBR-POS\" | \"player:<Full Name>\", "
    "\"kind\": \"coach_change|scheme_note|role_note|injury_note|oline_grade|volume_prior\", "
    "\"text\": \"one claim, one sentence\", \"value\": <number or null>, "
    "\"confidence\": 0-1, \"source\": \"what and when\", \"expires\": \"YYYY-MM-DD\"}}]}}\n"
    "Rules: one claim per fact, dated sources, honest confidence, 5-10 strong facts, no filler. "
    "Player names exactly as rostered. Units: ABBR-QB/RB/WR/TE/OL. No text outside the JSON."
)


def _run_research(settings: Settings, subject_prompt: str) -> dict:
    from pathlib import Path

    from .db import connect
    from .graph import import_facts

    client, err = _client()
    if client is None:
        return {"available": False, "reason": err}
    try:
        resp = client.messages.create(
            model=_model(), max_tokens=2500,
            system=_RESEARCH_SYSTEM.format(season=settings.current_season),
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            messages=[{"role": "user", "content": subject_prompt}],
        )
    except Exception as e:
        return {"available": False, "reason": f"{e.__class__.__name__}: {e}"}
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    doc = _json_block(text)
    if not doc or "facts" not in doc:
        return {"available": True, "imported": 0, "error": "model returned no facts JSON",
                "raw": text[:600]}
    doc["season"] = settings.current_season
    out_dir = settings.data_dir / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", subject_prompt[:40]).strip("_")
    path = out_dir / f"{safe}.json"
    path.write_text(json.dumps(doc, indent=1))
    conn = connect(settings.db_path)
    skipped: List[str] = []
    # import fact-by-fact so one bad name doesn't void the batch
    imported = 0
    for f in doc["facts"]:
        single = Path(str(path) + ".one.json")
        single.write_text(json.dumps({"season": doc["season"], "facts": [f]}))
        try:
            imported += import_facts(conn, single, settings.current_season)
        except ValueError as e:
            skipped.append(f"{f.get('entity')}: {e}")
        finally:
            single.unlink(missing_ok=True)
    conn.close()
    return {"available": True, "imported": imported, "skipped": skipped, "file": str(path),
            "facts": doc["facts"]}


def research_team(settings: Settings, team: str) -> dict:
    from .graph import TEAM_NAMES

    team = team.upper()
    name = TEAM_NAMES.get(team, team)
    return _run_research(settings, (
        f"Research the {name} ({team}) offense for the {settings.current_season} season: "
        "coaching/scheme changes and what the coordinator's past offenses looked like (pass rate, "
        "volume, personnel), who left/arrived and who absorbs vacated targets and carries, "
        "current injuries and camp battles, offensive line quality. Entity refs: "
        f"team:{team}, unit:{team}-QB/RB/WR/TE/OL, player:<Full Name>."
    ))


def research_player(settings: Settings, player_name: str, team: Optional[str] = None) -> dict:
    return _run_research(settings, (
        f"Research NFL player {player_name}{' (' + team + ')' if team else ''} for the "
        f"{settings.current_season} fantasy season: role and depth-chart standing, injury status, "
        "usage expectations, scheme fit, anything that changes his projection. Entity ref: "
        f"player:{player_name}; also team:/unit: facts where relevant."
    ))


def research_status(settings: Settings) -> dict:
    """Per team: researched fact count and latest date (derived facts excluded)."""
    from .db import connect
    from .graph import TEAM_NAMES, ensure_graph_schema

    conn = connect(settings.db_path)
    ensure_graph_schema(conn)
    rows = conn.execute(
        "SELECT COALESCE(e.team, SUBSTR(f.entity_id, 6, 3)) AS team, COUNT(*) n, MAX(f.created) latest "
        "FROM facts f LEFT JOIN entities e ON e.id = f.entity_id "
        "WHERE f.source != 'derived' GROUP BY team"
    ).fetchall()
    conn.close()
    by_team = {r["team"]: {"facts": r["n"], "latest": r["latest"]} for r in rows if r["team"]}
    return {"teams": [{"team": t, "name": TEAM_NAMES.get(t, t), **by_team.get(t, {"facts": 0, "latest": None})}
                      for t in sorted(TEAM_NAMES)]}


def facts_for(settings: Settings, query: str) -> dict:
    from .db import connect
    from .graph import search as g_search

    from .graph import TEAM_NAMES

    conn = connect(settings.db_path)
    if query.strip().upper() in TEAM_NAMES:  # abbreviation -> the team itself, never a player on it
        abbr = query.strip().upper()
        hits = [{"id": f"team:{abbr}", "kind": "team", "name": TEAM_NAMES[abbr], "team": abbr, "position": None}]
    else:
        hits = g_search(conn, query, limit=1)
    if not hits:
        conn.close()
        return {"entity": None, "facts": []}
    eid = hits[0]["id"]
    ids = [eid]
    if hits[0]["kind"] == "team":
        ids += [f"unit:{hits[0]['team']}-{p}" for p in ("QB", "RB", "WR", "TE", "OL")]
    qmarks = ",".join("?" * len(ids))
    facts = [dict(r) for r in conn.execute(
        f"SELECT entity_id, kind, text, value, confidence, source, created, expires FROM facts "
        f"WHERE entity_id IN ({qmarks}) AND source != 'derived' ORDER BY created DESC LIMIT 40", ids)]
    conn.close()
    return {"entity": hits[0], "facts": facts}
