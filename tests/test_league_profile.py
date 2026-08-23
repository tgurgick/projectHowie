"""League profile: the room's own draft habits as a bot tilt."""

import numpy as np

from howie3.config import LeagueConfig
from howie3.league_profile import MAX_SHIFT, build_profile, position_shift
from howie3.mock import bot_pick
from howie3.value.board import PoolPlayer


def test_profile_shares_and_shift():
    doc = {"picks": [{"round": 1, "pos": "RB"}] * 5 + [{"round": 1, "pos": "WR"}] * 4 + [{"round": 1, "pos": "QB"}]
                     + [{"round": 3, "pos": "QB"}] * 3 + [{"round": 3, "pos": "RB"}] * 7 + [{"round": 9, "pos": "K?"}]}
    prof = build_profile([doc])
    assert prof["by_round"]["1"] == {"QB": 0.1, "RB": 0.5, "WR": 0.4, "TE": 0.0, "K": 0.0, "DST": 0.0}
    assert prof["first_round"]["QB"] == 1 and prof["first_round"]["TE"] is None
    # a room that takes 30% QBs in R3 when the market window offers 8%: QB pulled earlier, up to the cap
    shift = position_shift(prof, 3, {"QB": 0.08, "RB": 0.5, "WR": 0.42})
    assert 0 < shift["QB"] <= MAX_SHIFT and shift["WR"] < 0
    assert position_shift(prof, 12, {"QB": 0.1}) == {} and position_shift(None, 1, {}) == {}


def test_bots_tilt_toward_the_rooms_habits():
    L = LeagueConfig()
    pool = [PoolPlayer(f"wr{i}", f"WR {i}", "WR", None, 200 - i, adp=20.0 + i, stdev=1.0, bye=None) for i in range(8)] + \
           [PoolPlayer(f"te{i}", f"TE {i}", "TE", None, 150 - i, adp=22.0 + 2 * i, stdev=1.0, bye=None) for i in range(3)]
    room = build_profile([{"picks": [{"round": 3, "pos": "TE"}] * 6 + [{"round": 3, "pos": "WR"}] * 4}])  # a TE-hungry room in R3
    def share(prof):
        picks = [bot_pick(pool, frozenset(), {}, 3, L, np.random.default_rng(s), profile=prof) for s in range(80)]
        return sum(1 for p in picks if p.position == "TE") / len(picks)
    assert share(room) > share(None) + 0.15
