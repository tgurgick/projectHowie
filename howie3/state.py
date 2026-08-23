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
from typing import Dict, Iterator, List, Optional, Tuple

from .config import LeagueConfig, Settings

SCHEMA_VERSION = 1
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
VALID_MODES = {"live", "mock"}


class DraftStateError(ValueError):
    """The draft log on disk is malformed, from an incompatible schema, or a
    requested transition would make the draft impossible."""


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
        """Load and validate the log. A malformed or incompatible file raises
        DraftStateError instead of silently starting a fresh draft — losing a
        live draft log is worse than a loud failure (`reset` repairs it)."""
        p = cls.path(settings)
        if not p.exists():
            return cls(created=_now())
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise DraftStateError(f"{p} is not valid JSON ({e}); reset the draft to repair") from e
        if not isinstance(raw, dict):
            raise DraftStateError(f"{p} must hold a JSON object")
        version = raw.get("version", 1)
        if version != SCHEMA_VERSION:
            raise DraftStateError(
                f"{p} has draft schema version {version}; this build reads {SCHEMA_VERSION}")
        try:
            state = cls(
                mode=str(raw.get("mode", "live")),
                events=[PickEvent(**e) for e in raw.get("events", [])],
                rules=[Rule(**r) for r in raw.get("rules", [])],
                notes=str(raw.get("notes", "")),
                created=str(raw.get("created") or _now()),
                seed=int(raw.get("seed", 11)),
            )
        except (TypeError, ValueError) as e:
            raise DraftStateError(f"{p} has a malformed record: {e}") from e
        state.check_invariants()
        return state

    def check_invariants(self) -> None:
        """Structural invariants every persisted log must satisfy."""
        if self.mode not in VALID_MODES:
            raise DraftStateError(f"unknown draft mode {self.mode!r}")
        seen = set()
        for i, e in enumerate(self.events):
            if e.seq != i + 1 or e.pick_no != i + 1:
                raise DraftStateError(
                    f"event {i + 1} has seq/pick_no {e.seq}/{e.pick_no}; the log must be contiguous")
            if not e.player_uid or e.player_uid in seen:
                raise DraftStateError(f"event {e.seq}: duplicate or empty player {e.player_uid!r}")
            seen.add(e.player_uid)
            if e.position is not None and e.position not in VALID_POSITIONS:
                raise DraftStateError(f"event {e.seq}: unknown position {e.position!r}")
            if not isinstance(e.team, int) or e.team < 0:
                raise DraftStateError(f"event {e.seq}: invalid team {e.team!r}")
            if not isinstance(e.mine, bool):
                raise DraftStateError(f"event {e.seq}: 'mine' must be a boolean")

    def save(self, settings: Settings) -> None:
        self.check_invariants()
        p = self.path(settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {
                "version": SCHEMA_VERSION,
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
                 mine: bool = False, league: Optional[LeagueConfig] = None) -> PickEvent:
        """Append a pick. With a league, the transition is validated against
        the draft's shape: sequential pick number, draft not complete, team
        in range, roster limits, and — in mock mode, where every pick is
        attributed — turn legality for the snake slot."""
        if any(e.player_uid == player_uid for e in self.events):
            raise ValueError(f"{player_name} is already drafted")
        if pick_no != self.next_pick_no():
            raise DraftStateError(
                f"pick {pick_no} is out of order; the next pick is {self.next_pick_no()}")
        if position is not None and position not in VALID_POSITIONS:
            raise DraftStateError(f"unknown position {position!r}")
        if league is not None:
            total = league.num_teams * league.roster_size
            if pick_no > total:
                raise DraftStateError(f"draft is complete ({total} picks)")
            if not 0 <= team <= league.num_teams:
                raise DraftStateError(f"team {team} is outside 1..{league.num_teams}")
            if mine and team != league.draft_position:
                raise DraftStateError("a pick marked mine must belong to the user's slot")
            # a placeholder for a pick the observer never saw may sit on any slot
            if team:
                have = sum(1 for e in self.events if e.team == team)
                if have >= league.roster_size:
                    raise DraftStateError(f"team {team} already has a full roster")
            if self.mode == "mock" and team != snake_team_for_pick(league, pick_no):
                raise DraftStateError(
                    f"pick {pick_no} belongs to slot {snake_team_for_pick(league, pick_no)}, not {team}")
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
        if mode not in VALID_MODES:
            raise DraftStateError(f"unknown draft mode {mode!r}")
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
        effects: Dict[str, list] = {"targets": [], "wait": [], "ban": [], "need": [], "bye_cap": [], "age": []}
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
            m = re.match(r"(?i)no\s+bye\s+stack\s*(?:>|over|above)\s*(\d+)", text)
            if m:
                effects["bye_cap"].append(int(m.group(1)))   # at most N starters may share a bye
                continue
            m = re.match(r"(?i)no\s+(QB|RB|WR|TE)\s+age\s*(?:>=|over|above|\+)?\s*(\d+)\+?\s+before\s+r(?:ound)?\s*(\d+)", text)
            if m:
                effects["age"].append((m.group(1).upper(), int(m.group(2)), int(m.group(3))))
                continue
            m = re.match(r"(?i)no\s+([A-Z/\s]+?)\s+before\s+r(?:ound)?\s*(\d+)", text)
            if m:
                for pos in re.split(r"[/\s]+", m.group(1).strip()):
                    if pos.upper() in {"QB", "RB", "WR", "TE", "K", "DST"}:
                        effects["ban"].append((pos.upper(), int(m.group(2))))
        return effects


def rule_key(text: str) -> Optional[Tuple[str, str]]:
    """The constraint a rule expresses, so duplicates and contradictions can
    be reconciled: ('wait', POS) for WAIT/NO-BEFORE, ('need', POS) for
    'N POS BY Rn', ('target', name). Free text has no key."""
    t = text.strip()
    m = re.match(r"(?i)target[:\s]+(.+?)(?:\s+@\d+)?$", t)
    if m:
        return ("target", m.group(1).strip().lower())
    m = re.match(r"(?i)wait\s+(QB|RB|WR|TE|K|DST)\s+until\s+r", t)
    if m:
        return ("wait", m.group(1).upper())
    m = re.match(r"(?i)no\s+(QB|RB|WR|TE|K|DST)\s+before\s+r", t)
    if m:
        return ("wait", m.group(1).upper())
    m = re.match(r"(?i)(\d+)\s+(QB|RB|WR|TE)s?\s+by\s+r", t)
    if m:
        return ("need", m.group(2).upper())
    if re.match(r"(?i)no\s+bye\s+stack", t):
        return ("bye_cap", "*")
    m = re.match(r"(?i)no\s+(QB|RB|WR|TE)\s+age", t)
    if m:
        return ("age", m.group(1).upper())
    return None


def reconcile_rules(rules: List[Rule]) -> Tuple[List[Rule], List[str]]:
    """Keep one ON rule per constraint key — the most recently added wins,
    older ones are switched OFF (never deleted). Returns (rules, notes)."""
    notes: List[str] = []
    latest: Dict[Tuple[str, str], int] = {}
    for i, r in enumerate(rules):
        k = rule_key(r.text) if r.on else None
        if k is not None:
            latest[k] = i
    out = []
    for i, r in enumerate(rules):
        k = rule_key(r.text) if r.on else None
        if k is not None and latest[k] != i:
            out.append(Rule(text=r.text, on=False))
            notes.append(f"'{r.text}' switched off — superseded by '{rules[latest[k]].text}'")
        else:
            out.append(r)
    return out, notes


def snake_team_for_pick(league: LeagueConfig, pick_no: int) -> int:
    """Which draft slot owns overall pick pick_no in a snake draft."""
    t = league.num_teams
    rnd = (pick_no - 1) // t + 1
    idx = (pick_no - 1) % t
    return idx + 1 if rnd % 2 == 1 else t - idx


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
