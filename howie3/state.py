"""Draft state as an event log — the shared primitive every surface reads
and writes.

One JSON document (data/draft.json): an ordered list of pick events plus the
user's strategy sheet (pinned rules + free notes). The engine is stateless
against it; UI, CLI, MCP, agent, and mock bots are all just writers to this
log. Undo is popping the last event; replay is the log itself.
"""

import contextlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .config import LeagueConfig, Settings


@contextlib.contextmanager
def state_lock(settings: Settings) -> Iterator[None]:
    """Cross-process lock for draft.json mutations: the cockpit server and the
    MCP server are separate processes writing the same event log."""
    import fcntl

    lock_path = settings.data_dir / "draft.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@dataclass
class PickEvent:
    seq: int
    pick_no: int              # overall pick number, 1-based
    team: int                 # 1..num_teams (snake slot that made the pick)
    player_uid: str
    player_name: str
    position: Optional[str] = None
    source: str = "ui"        # ui | mock | mcp | chrome | cli
    ts: str = ""
    mine: bool = False        # explicit ownership — never inferred from team


@dataclass
class Rule:
    text: str
    on: bool = True


@dataclass
class DraftState:
    mode: str = "live"        # live | mock
    events: List[PickEvent] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    notes: str = ""
    created: str = ""
    seed: int = 11            # bot RNG seed; new value per mock draft

    # ---- persistence ----

    @staticmethod
    def path(settings: Settings) -> Path:
        return settings.data_dir / "draft.json"

    @classmethod
    def load(cls, settings: Settings) -> "DraftState":
        p = cls.path(settings)
        if not p.exists():
            return cls(created=_now())
        raw = json.loads(p.read_text())
        return cls(
            mode=raw.get("mode", "live"),
            events=[PickEvent(**e) for e in raw.get("events", [])],
            rules=[Rule(**r) for r in raw.get("rules", [])],
            notes=raw.get("notes", ""),
            created=raw.get("created", _now()),
            seed=raw.get("seed", 11),
        )

    def save(self, settings: Settings) -> None:
        p = self.path(settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {
                "mode": self.mode,
                "events": [asdict(e) for e in self.events],
                "rules": [asdict(r) for r in self.rules],
                "notes": self.notes,
                "created": self.created,
                "seed": self.seed,
            },
            indent=1,
        ))
        os.replace(tmp, p)

    # ---- event log ----

    def add_pick(self, pick_no: int, team: int, player_uid: str,
                 player_name: str, position: Optional[str], source: str,
                 mine: bool = False) -> PickEvent:
        if any(e.player_uid == player_uid for e in self.events):
            raise ValueError(f"{player_name} is already drafted")
        event = PickEvent(
            seq=len(self.events) + 1, pick_no=pick_no, team=team,
            player_uid=player_uid, player_name=player_name,
            position=position, source=source, ts=_now(), mine=mine,
        )
        self.events.append(event)
        return event

    def undo(self) -> Optional[PickEvent]:
        return self.events.pop() if self.events else None

    def reset(self, mode: str, seed: Optional[int] = None) -> None:
        self.mode = mode
        self.events = []
        self.created = _now()
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 10**6)

    # ---- derived views ----

    def taken_uids(self) -> frozenset:
        return frozenset(e.player_uid for e in self.events)

    def my_uids(self, league: LeagueConfig) -> List[str]:
        return [e.player_uid for e in self.events if e.mine]

    def next_pick_no(self) -> int:
        return len(self.events) + 1

    # ---- strategy rules (parsed, machine-actionable subset) ----

    def active_rule_effects(self) -> Dict[str, list]:
        """Parse ON rules into effects the ranking layer applies:
        targets: [player name], wait: [(POS, round)], ban: [(POS, round)]."""
        effects: Dict[str, list] = {"targets": [], "wait": [], "ban": [], "need": []}
        for rule in self.rules:
            if not rule.on:
                continue
            text = rule.text.strip()
            m = re.match(r"(?i)target[:\s]+(.+?)(?:\s+@\d+)?$", text)
            if m:
                effects["targets"].append(m.group(1).strip())
                continue
            m = re.match(r"(?i)wait\s+(QB|RB|WR|TE|K|DST)\s+until\s+r(?:ound)?\s*(\d+)", text)
            if m:
                effects["wait"].append((m.group(1).upper(), int(m.group(2))))
                continue
            m = re.match(r"(?i)(\d+)\s+(QB|RB|WR|TE)s?\s+by\s+r(?:ound)?\s*(\d+)", text)
            if m:
                effects["need"].append((m.group(2).upper(), int(m.group(1)), int(m.group(3))))
                continue
            m = re.match(r"(?i)no\s+([A-Z/\s]+?)\s+before\s+r(?:ound)?\s*(\d+)", text)
            if m:
                for pos in re.split(r"[/\s]+", m.group(1).strip()):
                    if pos.upper() in {"QB", "RB", "WR", "TE", "K", "DST"}:
                        effects["ban"].append((pos.upper(), int(m.group(2))))
        return effects


def snake_team_for_pick(league: LeagueConfig, pick_no: int) -> int:
    """Which draft slot owns overall pick pick_no in a snake draft."""
    t = league.num_teams
    rnd = (pick_no - 1) // t + 1
    idx = (pick_no - 1) % t
    return idx + 1 if rnd % 2 == 1 else t - idx


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
