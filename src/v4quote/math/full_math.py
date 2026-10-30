"""Port of Uniswap FullMath / UnsafeMath.

Solidity's FullMath.mulDiv computes floor(a*b/denominator) with a 512-bit
intermediate and reverts if the denominator is zero or the result does not
fit into uint256. Python integers are arbitrary precision, so the exact
quotient is computed directly and the same revert conditions are enforced.
"""

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT256


def mul_div(a: int, b: int, denominator: int) -> int:
    """floor(a * b / denominator); reverts like FullMath.mulDiv."""
    if denominator == 0:
        raise MathRevert("mulDiv: denominator is zero")
    result = (a * b) // denominator
    if result > MAX_UINT256:
        raise MathRevert("mulDiv: overflow")
    return result


def mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    """ceil(a * b / denominator); reverts like FullMath.mulDivRoundingUp."""
    result = mul_div(a, b, denominator)
    if (a * b) % denominator > 0:
        if result >= MAX_UINT256:
            raise MathRevert("mulDivRoundingUp: overflow")
        result += 1
    return result


def div_rounding_up(x: int, y: int) -> int:
    """UnsafeMath.divRoundingUp: ceil(x / y). Division by zero returns 0 in the EVM."""
    if y == 0:
        return 0
    return x // y + (1 if x % y > 0 else 0)
