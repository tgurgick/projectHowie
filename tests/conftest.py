"""Shared fixtures. `league12` pins the league to the default 12-team / slot 8
shape for tests that assert specific pick numbers — the real
data/league_config.json is the user's and changes."""

import pytest

from howie3.config import LeagueConfig, Settings


@pytest.fixture
def league12(monkeypatch):
    monkeypatch.setattr(Settings, "league", property(lambda self: LeagueConfig()))
    return LeagueConfig()
