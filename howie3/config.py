"""Central settings for Howie v3.

Every module gets paths, seasons, and league shape from here — nothing else
resolves paths or hardcodes a season/scoring format.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Scoring format short codes used throughout the schema (column suffixes, adp.format)
FORMATS = ("std", "half", "ppr")

_SCORING_TYPE_TO_FORMAT = {
    "standard": "std",
    "half_ppr": "half",
    "ppr": "ppr",
}


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from this file (or a given path) to the repo root."""
    p = (start or Path(__file__).resolve()).parent
    for candidate in [p, *p.parents]:
        if (candidate / "data").is_dir() and (candidate / "howie3").is_dir():
            return candidate
    raise RuntimeError("Could not locate projectHowie repo root")


@dataclass
class LeagueConfig:
    num_teams: int = 12
    draft_position: int = 8
    scoring_type: str = "half_ppr"
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 3
    te_slots: int = 1
    flex_slots: int = 1
    k_slots: int = 1
    dst_slots: int = 1
    bench_slots: int = 6
    roster_size: int = 16
    # How far projections shrink toward market-implied value (0 = pure
    # projections, 1 = pure market). 0.75 won the 2025 replay backtest
    # (+48 pts vs follow-ADP with snake-correct opponents; pure projections
    # lost by ~200). Re-sweep with `howie eval run` as seasons accumulate.
    market_anchor: float = 0.75
    # Extra weight on fantasy-playoff weeks (15-17) in the simulated objective.
    # Default 1.0 (neutral): the 2025 backtest (`howie eval run`, tier D)
    # found preseason weekly SoS has ~zero predictive power, so weighting
    # playoff weeks only earns value once in-season matchup data exists.
    playoff_weight: float = 1.0

    @property
    def scoring_format(self) -> str:
        return _SCORING_TYPE_TO_FORMAT[self.scoring_type]

    def validate(self) -> None:
        if self.scoring_type not in _SCORING_TYPE_TO_FORMAT:
            raise ValueError(
                f"Unknown scoring_type {self.scoring_type!r} — expected one of "
                f"{sorted(_SCORING_TYPE_TO_FORMAT)}"
            )
        if not 2 <= self.num_teams <= 20:
            raise ValueError(f"num_teams must be 2-20, got {self.num_teams}")
        if not 1 <= self.draft_position <= self.num_teams:
            raise ValueError(
                f"draft_position must be 1-{self.num_teams}, got {self.draft_position}"
            )
        slots = [self.qb_slots, self.rb_slots, self.wr_slots, self.te_slots,
                 self.flex_slots, self.k_slots, self.dst_slots, self.bench_slots]
        if any(s < 0 for s in slots):
            raise ValueError("roster slot counts cannot be negative")
        starters = sum(slots) - self.bench_slots
        if self.roster_size < starters:
            raise ValueError(
                f"roster_size {self.roster_size} is smaller than {starters} starting slots"
            )
        if not 0.0 <= self.market_anchor <= 1.0:
            raise ValueError(f"market_anchor must be 0..1, got {self.market_anchor}")
        if not 1.0 <= self.playoff_weight <= 3.0:
            raise ValueError(f"playoff_weight must be 1..3, got {self.playoff_weight}")

    @classmethod
    def load(cls, path: Path) -> "LeagueConfig":
        if not path.exists():
            cfg = cls()
        else:
            raw = json.loads(path.read_text())
            known = {f for f in cls.__dataclass_fields__}
            unknown = set(raw) - known
            if unknown:
                raise ValueError(f"Unknown league config keys in {path}: {sorted(unknown)}")
            cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.validate()
        return cfg


@dataclass
class Settings:
    repo_root: Path = field(default_factory=find_repo_root)
    hist_start: int = 2018
    current_season: int = 2026

    def __post_init__(self) -> None:
        # Load the repo's .env (API keys, HOWIE_MODEL) for every entry point —
        # CLI, server, MCP — without overriding variables already set
        try:
            from dotenv import load_dotenv

            load_dotenv(self.repo_root / ".env", override=False)
        except ImportError:
            pass

    @property
    def data_dir(self) -> Path:
        override = os.environ.get("HOWIE_DATA_DIR")
        return Path(override).expanduser() if override else self.repo_root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "howie.db"

    @property
    def pff_dir(self) -> Path:
        return self.data_dir / "pff_csv"

    @property
    def league(self) -> LeagueConfig:
        return LeagueConfig.load(self.data_dir / "league_config.json")

    @property
    def hist_seasons(self) -> List[int]:
        # Completed seasons only; current_season is draft-prep territory
        return list(range(self.hist_start, self.current_season))


def parse_seasons(spec: str) -> List[int]:
    """'2018-2025' or '2024' or '2022,2024' -> list of ints."""
    seasons: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seasons.extend(range(int(lo), int(hi) + 1))
        elif part:
            seasons.append(int(part))
    return sorted(set(seasons))
