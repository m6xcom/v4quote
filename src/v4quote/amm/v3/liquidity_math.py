"""Port of LiquidityMath.addDelta."""

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT128


def add_delta(x: int, y: int) -> int:
    """x + y for uint128 x and int128 y; reverts on under/overflow ('LS' / 'LA')."""
    z = x + y
    if y < 0 and z < 0:
        raise MathRevert("LS")
    if z > MAX_UINT128:
        raise MathRevert("LA")
    return z
