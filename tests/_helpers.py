"""Test-only helpers (not part of the swap path)."""

from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, localcontext


def encode_price_sqrt(reserve1: int, reserve0: int) -> int:
    """Replica of v3-core test/shared/utilities.ts encodePriceSqrt.

    bignumber.js is configured with DECIMAL_PLACES=40 and ROUND_HALF_UP, so
    the division and the square root are each rounded to 40 decimal places;
    the final value is floored (integerValue(3)). Test helper only.
    """
    dp = Decimal(1).scaleb(-40)
    with localcontext() as ctx:
        ctx.prec = 400
        ratio = (Decimal(reserve1) / Decimal(reserve0)).quantize(dp, rounding=ROUND_HALF_UP)
        root = ratio.sqrt().quantize(dp, rounding=ROUND_HALF_UP)
        return int((root * (Decimal(2) ** 96)).to_integral_value(rounding=ROUND_FLOOR))
