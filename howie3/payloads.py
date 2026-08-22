"""Shared JSON payload shapes for the contract layer (service, MCP server).

The engine's own result objects live next to the code that produces them
(``SimResult`` in value/simulate.py, ``PickPlan`` in value/roster.py). This
module names the JSON-able dict shapes those results are serialized into so
every surface (web UI, MCP, CLI) agrees on them. TypedDicts are plain dicts at
runtime — nothing here changes behavior.
"""

from typing import Any, Dict, List, TypedDict

JsonDict = Dict[str, Any]
"""A plain JSON-able object payload: what every service function returns."""


class SimSummary(TypedDict, total=False):
    """Rounded season-total summary of a ``SimResult``. Keys are absent when
    nothing was simulated (e.g. an empty roster)."""
    mean: int
    p10: int
    p90: int


class RosterSimPayload(SimSummary, total=False):
    """service.roster_sim_payload: the current roster's simulated season."""
    players: List[str]
    samples: List[float]


class ToolSpec(TypedDict):
    """One MCP tool declaration (tools/list)."""
    name: str
    description: str
    inputSchema: JsonDict


class JsonRpcResponse(TypedDict, total=False):
    """A JSON-RPC 2.0 response: exactly one of ``result`` / ``error`` is set."""
    jsonrpc: str
    id: Any
    result: JsonDict
    error: JsonDict
