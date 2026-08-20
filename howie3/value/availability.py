"""Pick-availability model.

P(player still on the board at pick k), from mock-draft ADP treated as a
normal distribution over the player's selection pick. FFC reports per-player
stdev from real drafts, so the spread is empirical, not assumed.
"""

import math
from typing import Optional

# Real drafts are messier than mocks; never let the model be more certain
# than about +/- 1 pick even for consensus top picks.
_MIN_STDEV = 1.0


def p_available(adp: Optional[float], stdev: Optional[float], pick: float) -> float:
    """Probability the player is undrafted when pick number `pick` comes up."""
    if adp is None:
        return 1.0  # outside drafted range in mocks — assume available
    sigma = max(stdev if stdev else 0.0, _MIN_STDEV)
    z = (pick - adp) / sigma
    # P(selection pick >= k) = 1 - CDF(k)
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
