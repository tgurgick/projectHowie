"""League profile: how this room actually drafts, learned from its own draft
history (the ESPN draft recap), so the bots — and therefore the draft-flow
simulation and the mock lab — model your league rather than an average one.

The profile is the share of picks by position in each round. Bots read it as
an urgency shift: a position this room takes more than the market would is
perceived a few picks earlier, and vice versa. Modest by design — the bots
stay market-driven, tilted toward the room's habits."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
MAX_SHIFT = 5.0   # ADP picks of tilt at the largest league-vs-market share gap


def profile_path(settings: Settings) -> Path:
    return settings.data_dir / "league_profile.json"


def build_profile(draft_docs: List[dict], source: str = "league history") -> dict:
    """draft_docs: parsed draft recaps — {"picks": [{"round", "pos"}...]}."""
    by_round: Dict[int, Dict[str, int]] = {}
    n = 0
    for doc in draft_docs:
        for pk in doc.get("picks", []):
            pos = (pk.get("pos") or "").replace("?", "")
            if pos not in POSITIONS:
                continue
            by_round.setdefault(int(pk["round"]), {})
            by_round[int(pk["round"])][pos] = by_round[int(pk["round"])].get(pos, 0) + 1
            n += 1
    rounds = {}
    for r, cnt in sorted(by_round.items()):
        total = sum(cnt.values())
        rounds[str(r)] = {pos: round(cnt.get(pos, 0) / total, 3) for pos in POSITIONS}
    first = {}
    for pos in POSITIONS:
        rs = [r for r, cnt in by_round.items() if cnt.get(pos)]
        first[pos] = min(rs) if rs else None
    return {"source": source, "drafts": len(draft_docs), "picks": n, "by_round": rounds, "first_round": first}


def load_profile(settings: Settings) -> Optional[dict]:
    p = profile_path(settings)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def position_shift(profile: Optional[dict], rnd: int, window_share: Dict[str, float]) -> Dict[str, float]:
    """Urgency shift per position for this round: MAX_SHIFT x (league share -
    market share in the bot's candidate window), clipped. Empty when no
    profile or the round is beyond the history."""
    if not profile:
        return {}
    row = profile.get("by_round", {}).get(str(rnd))
    if not row:
        return {}
    out = {}
    for pos in POSITIONS:
        gap = row.get(pos, 0.0) - window_share.get(pos, 0.0)
        if abs(gap) >= 0.05:
            out[pos] = max(-MAX_SHIFT, min(MAX_SHIFT, gap * 2 * MAX_SHIFT))
    return out
