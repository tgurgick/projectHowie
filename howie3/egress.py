"""Model-egress policy: ONE serializer for everything that leaves the local
data plane toward a model API or an MCP client.

The rule is structural, not a prompt: per-game stat lines (weekly_stats rows
and anything shaped like them) and raw provider records never cross. Derived
context — projections, fantasy-point totals, shares, milestone rates,
distribution summaries, rankings, researched facts with provenance — does.
Every egress path (agent tool results, MCP responses, insights payloads)
calls `redact` so no feature decides for itself what is safe to send.
"""

import json
import os
from typing import Any, List

# Keys whose values are raw per-game box-score rows or provider payloads
RAW_KEYS = frozenset({
    "games", "game_log", "weekly", "weekly_stats", "weeks", "box_score",
    "intel", "team_intel", "player_intel", "raw", "source_row", "provider_row",
})
# Keys that only occur on a raw stat line (a record carrying any of these is
# a box score, wherever it lives)
STAT_LINE_KEYS = frozenset({
    "pass_yards", "pass_yds", "pass_attempts", "pass_tds", "interceptions",
    "rush_attempts", "rush_yards", "rush_yds", "rush_tds",
    "targets", "receptions", "rec_yards", "rec_yds", "rec_tds", "fumbles",
})
REDACTED_MARK = "_redacted"


def sql_tool_enabled() -> bool:
    """Raw SQL from the agent is an opt-in: HOWIE_AGENT_SQL=1. Off by
    default because arbitrary SELECTs can return stat lines verbatim."""
    return os.environ.get("HOWIE_AGENT_SQL", "").strip().lower() in ("1", "true", "yes", "on")


def _is_stat_line(d: dict) -> bool:
    return any(k in STAT_LINE_KEYS for k in d)


def redact(obj: Any) -> Any:
    """Return a deep copy of obj with raw records removed. Dict keys in
    RAW_KEYS are replaced by a marker; list items that look like stat lines
    are dropped; everything else is preserved."""
    if isinstance(obj, dict):
        out = {}
        dropped: List[str] = []
        for k, v in obj.items():
            if k in RAW_KEYS:
                dropped.append(k)
                continue
            out[k] = redact(v)
        if dropped:
            out[REDACTED_MARK] = sorted(set(dropped + list(out.get(REDACTED_MARK, []))))
        return out
    if isinstance(obj, (list, tuple)):
        kept = []
        for item in obj:
            if isinstance(item, dict) and _is_stat_line(item):
                continue
            kept.append(redact(item))
        return kept
    return obj


def contains_raw(obj: Any) -> bool:
    """True if a payload still carries raw keys or stat lines (used by tests
    and as an assertion on every egress path)."""
    if isinstance(obj, dict):
        if any(k in RAW_KEYS for k in obj) or _is_stat_line(obj):
            return True
        return any(contains_raw(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_raw(v) for v in obj)
    return False


def for_model(text: str) -> str:
    """Tool results are strings. If one is JSON, redact it structurally;
    plain text (rendered tables of season aggregates) passes through."""
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "[{":
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(redact(parsed), default=str)
