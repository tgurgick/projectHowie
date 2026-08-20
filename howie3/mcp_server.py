"""MCP server: Howie's engine as tools for any MCP client (Claude Desktop,
Claude Code, etc.). stdio transport, newline-delimited JSON-RPC, no deps.

Register in a client, e.g. Claude Code:
    claude mcp add howie -- python3 -m howie3.mcp_server
The tools call the same service layer as the web UI, and mark_pick writes the
same draft event log — chat and cockpit stay in sync during a draft.
"""

import json
import sys
from typing import Any, Dict

from .config import Settings
from .state import DraftState

PROTOCOL = "2024-11-05"

TOOLS = [
    {
        "name": "get_draft_state",
        "description": "Current draft state: round, pick, who's on the clock, the user's roster, recent picks, and their strategy notes/rules.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "search",
        "description": "Fast search over players, teams, and position rooms. Returns entities with projection/ADP for players.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
    },
    {
        "name": "best_picks",
        "description": "Ranked best picks right now given the live draft state (marginal-value engine; value = expected final starting-lineup points). Set sims>0 for Monte Carlo refinement.",
        "inputSchema": {"type": "object", "properties": {
            "sims": {"type": "integer", "minimum": 0, "maximum": 500}}},
    },
    {
        "name": "positional_impact",
        "description": "Per position: expected final roster drafting it NOW vs WAITing until the next pick, with the cost of waiting.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "player_card",
        "description": "One player's full card: projection, outcome band, availability, room shares, knowledge-graph facts, playoff SoS, trend.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
    },
    {
        "name": "entity_context",
        "description": "1-hop knowledge-graph context for a player, team, or room (facts carry provenance and confidence).",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
    },
    {
        "name": "mark_pick",
        "description": "Record a draft pick in the shared event log. mine=true drafts the player to the user's team; false marks them taken by another team. In mock mode, bots then pick until the user is back on the clock.",
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string"}, "mine": {"type": "boolean"}},
            "required": ["name", "mine"]},
    },
    {
        "name": "undo_pick",
        "description": "Undo the most recent pick in the draft log.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _resolve_uid(settings: Settings, name: str) -> str:
    from . import service

    hits = [h for h in service.search_payload(settings, name) if h["kind"] == "player"]
    if not hits:
        raise ValueError(f"No player found for {name!r}")
    return hits[0]["uid"]


def call_tool(settings: Settings, name: str, args: Dict[str, Any]) -> Any:
    from . import service

    state = DraftState.load(settings)
    if name == "get_draft_state":
        payload = service.state_payload(settings, state)
        payload["strategy"] = service.strategy_payload(state)
        return payload
    if name == "search":
        return service.search_payload(settings, str(args.get("query", "")))
    if name == "best_picks":
        return service.pick_payload(settings, state, sims=int(args.get("sims", 0)), top_n=8)
    if name == "positional_impact":
        return service.positions_payload(settings, state)
    if name == "player_card":
        return service.card_payload(settings, _resolve_uid(settings, str(args["name"])))
    if name == "entity_context":
        from .db import connect
        from .graph import entity_context, search as g_search

        conn = connect(settings.db_path)
        hits = g_search(conn, str(args.get("query", "")), limit=1)
        out = entity_context(conn, hits[0]["id"]) if hits else {}
        conn.close()
        return out
    if name == "mark_pick":
        # service owns the cross-process lock and mock-bot advancement
        return service.mark_pick(settings, _resolve_uid(settings, str(args["name"])),
                                 mine=bool(args.get("mine")), source="mcp")
    if name == "undo_pick":
        return {"undone": service.undo_pick(settings)}
    raise ValueError(f"Unknown tool {name}")


def handle(settings: Settings, req: dict) -> dict:
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "howie", "version": "3.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = req.get("params", {})
        try:
            payload = call_tool(settings, params.get("name", ""), params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main() -> None:
    settings = Settings()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("method", "").startswith("notifications/"):
            continue
        response = handle(settings, req)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
