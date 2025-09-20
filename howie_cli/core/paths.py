"""
Common path utilities for Howie.

Goals:
- Resolve data directory and database paths independent of CWD or install location.
- Prefer user-writable directory by default: ~/.howie/data
- Allow overrides via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_data_dir() -> Path:
    """Return the directory to store Howie data (creates if missing).

    Priority:
    1) HOWIE_DATA_DIR environment variable
    2) ~/.howie/data
    """
    env_dir = os.getenv("HOWIE_DATA_DIR")
    base = Path(env_dir).expanduser() if env_dir else (Path.home() / ".howie" / "data")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _db_filename(scoring: str = "ppr") -> str:
    scoring = scoring.lower()
    if scoring in {"ppr", "half_ppr", "standard"}:
        suffix = {
            "ppr": "fantasy_ppr.db",
            "half_ppr": "fantasy_halfppr.db",
            "standard": "fantasy_standard.db",
        }[scoring]
        return suffix
    return "fantasy_ppr.db"


def get_db_path(scoring: str = "ppr") -> Path:
    """Return absolute path to the sqlite DB file for a scoring format."""
    return get_data_dir() / _db_filename(scoring)


def get_db_url(scoring: str = "ppr") -> str:
    """Return a database URL, honoring DB_URL if provided.

    - If DB_URL is set, return it as-is.
    - Otherwise, construct a sqlite:/// URL pointing to the per-user data dir.
    """
    env = os.getenv("DB_URL")
    if env:
        return env
    return f"sqlite:///{get_db_path(scoring)}"


def repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Best-effort detection of the repository root by walking up.

    Not required for DB resolution, but helpful for locating docs/scripts.
    """
    cur = Path(start or __file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists() or (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    return None

