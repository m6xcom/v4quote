"""Port of Uniswap v3/v4 SqrtPriceMath (v3-core and v4-core produce identical results)."""

from v4quote.errors import MathRevert
from v4quote.math.full_math import div_rounding_up, mul_div, mul_div_rounding_up
from v4quote.math.uint import MAX_UINT160, MAX_UINT256, Q96, to_uint160

RESOLUTION = 96


def get_next_sqrt_price_from_amount0_rounding_up(
    sqrt_px96: int, liquidity: int, amount: int, add: bool
) -> int:
    if amount == 0:
        return sqrt_px96
    numerator1 = liquidity << RESOLUTION

    if add:
        product = (amount * sqrt_px96) & MAX_UINT256  # unchecked multiply
        if product // amount == sqrt_px96:
            denominator = (numerator1 + product) & MAX_UINT256  # unchecked add
            if denominator >= numerator1:
                # always fits in 160 bits
                return mul_div_rounding_up(numerator1, sqrt_px96, denominator)
        denom = numerator1 // sqrt_px96 + amount
        if denom > MAX_UINT256:
            raise MathRevert("add overflow")
        return div_rounding_up(numerator1, denom)

    product = (amount * sqrt_px96) & MAX_UINT256
    if not (product // amount == sqrt_px96 and numerator1 > product):
        raise MathRevert("PriceOverflow")
    denominator = numerator1 - product
    return to_uint160(mul_div_rounding_up(numerator1, sqrt_px96, denominator))


def get_next_sqrt_price_from_amount1_rounding_down(
    sqrt_px96: int, liquidity: int, amount: int, add: bool
) -> int:
    if add:
        if amount <= MAX_UINT160:
            quotient = (amount << RESOLUTION) // liquidity
        else:
            quotient = mul_div(amount, Q96, liquidity)
        return to_uint160(sqrt_px96 + quotient)

    if amount <= MAX_UINT160:
        quotient = div_rounding_up(amount << RESOLUTION, liquidity)
    else:
        quotient = mul_div_rounding_up(amount, Q96, liquidity)
    if not sqrt_px96 > quotient:
        raise MathRevert("NotEnoughLiquidity")
    return sqrt_px96 - quotient


def get_next_sqrt_price_from_input(sqrt_px96: int, liquidity: int, amount_in: int, zero_for_one: bool) -> int:
    if sqrt_px96 == 0 or liquidity == 0:
        raise MathRevert("InvalidPriceOrLiquidity")
    if zero_for_one:
        return get_next_sqrt_price_from_amount0_rounding_up(sqrt_px96, liquidity, amount_in, True)
    return get_next_sqrt_price_from_amount1_rounding_down(sqrt_px96, liquidity, amount_in, True)


def get_next_sqrt_price_from_output(sqrt_px96: int, liquidity: int, amount_out: int, zero_for_one: bool) -> int:
    if sqrt_px96 == 0 or liquidity == 0:
        raise MathRevert("InvalidPriceOrLiquidity")
    if zero_for_one:
        return get_next_sqrt_price_from_amount1_rounding_down(sqrt_px96, liquidity, amount_out, False)
    return get_next_sqrt_price_from_amount0_rounding_up(sqrt_px96, liquidity, amount_out, False)


def get_amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if sqrt_a == 0:
        raise MathRevert("InvalidPrice")
    numerator1 = liquidity << RESOLUTION
    numerator2 = sqrt_b - sqrt_a
    if round_up:
        return div_rounding_up(mul_div_rounding_up(numerator1, numerator2, sqrt_b), sqrt_a)
    return mul_div(numerator1, numerator2, sqrt_b) // sqrt_a


def get_amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if round_up:
        return mul_div_rounding_up(liquidity, sqrt_b - sqrt_a, Q96)
    return mul_div(liquidity, sqrt_b - sqrt_a, Q96)


def get_amount0_delta_signed(sqrt_a: int, sqrt_b: int, liquidity: int) -> int:
    """v3 signed helper: negative liquidity -> negative amount rounded down."""
    if liquidity < 0:
        return -get_amount0_delta(sqrt_a, sqrt_b, -liquidity, False)
    return get_amount0_delta(sqrt_a, sqrt_b, liquidity, True)


def get_amount1_delta_signed(sqrt_a: int, sqrt_b: int, liquidity: int) -> int:
    if liquidity < 0:
        return -get_amount1_delta(sqrt_a, sqrt_b, -liquidity, False)
    return get_amount1_delta(sqrt_a, sqrt_b, liquidity, True)
