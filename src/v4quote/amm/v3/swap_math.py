"""Ports of SwapMath.computeSwapStep.

v3-core and v4-core use opposite sign conventions for ``amount_remaining``
and differ slightly in how the fee/amountIn split is computed when the step
does not reach the target price. Both are ported verbatim:

* ``compute_swap_step_v3``: amount_remaining > 0 means exact input (v3-core).
* ``compute_swap_step_v4``: amount_remaining < 0 means exact input (v4-core).

Returns (sqrt_price_next_x96, amount_in, amount_out, fee_amount).
"""

from v4quote.errors import MathRevert
from v4quote.math.full_math import mul_div, mul_div_rounding_up
from v4quote.amm.v3.sqrt_price_math import (
    get_amount0_delta,
    get_amount1_delta,
    get_next_sqrt_price_from_input,
    get_next_sqrt_price_from_output,
)

MAX_SWAP_FEE = 1_000_000  # pips


def compute_swap_step_v3(
    sqrt_ratio_current_x96: int,
    sqrt_ratio_target_x96: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int,
) -> tuple[int, int, int, int]:
    zero_for_one = sqrt_ratio_current_x96 >= sqrt_ratio_target_x96
    exact_in = amount_remaining >= 0
    amount_in = 0
    amount_out = 0

    if exact_in:
        amount_remaining_less_fee = mul_div(amount_remaining, MAX_SWAP_FEE - fee_pips, MAX_SWAP_FEE)
        amount_in = (
            get_amount0_delta(sqrt_ratio_target_x96, sqrt_ratio_current_x96, liquidity, True)
            if zero_for_one
            else get_amount1_delta(sqrt_ratio_current_x96, sqrt_ratio_target_x96, liquidity, True)
        )
        if amount_remaining_less_fee >= amount_in:
            sqrt_ratio_next_x96 = sqrt_ratio_target_x96
        else:
            sqrt_ratio_next_x96 = get_next_sqrt_price_from_input(
                sqrt_ratio_current_x96, liquidity, amount_remaining_less_fee, zero_for_one
            )
    else:
        amount_out = (
            get_amount1_delta(sqrt_ratio_target_x96, sqrt_ratio_current_x96, liquidity, False)
            if zero_for_one
            else get_amount0_delta(sqrt_ratio_current_x96, sqrt_ratio_target_x96, liquidity, False)
        )
        if -amount_remaining >= amount_out:
            sqrt_ratio_next_x96 = sqrt_ratio_target_x96
        else:
            sqrt_ratio_next_x96 = get_next_sqrt_price_from_output(
                sqrt_ratio_current_x96, liquidity, -amount_remaining, zero_for_one
            )

    is_max = sqrt_ratio_target_x96 == sqrt_ratio_next_x96

    if zero_for_one:
        if not (is_max and exact_in):
            amount_in = get_amount0_delta(sqrt_ratio_next_x96, sqrt_ratio_current_x96, liquidity, True)
        if not (is_max and not exact_in):
            amount_out = get_amount1_delta(sqrt_ratio_next_x96, sqrt_ratio_current_x96, liquidity, False)
    else:
        if not (is_max and exact_in):
            amount_in = get_amount1_delta(sqrt_ratio_current_x96, sqrt_ratio_next_x96, liquidity, True)
        if not (is_max and not exact_in):
            amount_out = get_amount0_delta(sqrt_ratio_current_x96, sqrt_ratio_next_x96, liquidity, False)

    if not exact_in and amount_out > -amount_remaining:
        amount_out = -amount_remaining

    if exact_in and sqrt_ratio_next_x96 != sqrt_ratio_target_x96:
        fee_amount = amount_remaining - amount_in
    else:
        fee_amount = mul_div_rounding_up(amount_in, fee_pips, MAX_SWAP_FEE - fee_pips)

    return sqrt_ratio_next_x96, amount_in, amount_out, fee_amount


def get_sqrt_price_target(zero_for_one: bool, sqrt_price_next_x96: int, sqrt_price_limit_x96: int) -> int:
    """v4 SwapMath.getSqrtPriceTarget: max(next, limit) for zeroForOne, else min(next, limit)."""
    if zero_for_one:
        return sqrt_price_limit_x96 if sqrt_price_next_x96 < sqrt_price_limit_x96 else sqrt_price_next_x96
    return sqrt_price_limit_x96 if sqrt_price_next_x96 > sqrt_price_limit_x96 else sqrt_price_next_x96


def compute_swap_step_v4(
    sqrt_price_current_x96: int,
    sqrt_price_target_x96: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int,
) -> tuple[int, int, int, int]:
    zero_for_one = sqrt_price_current_x96 >= sqrt_price_target_x96
    exact_in = amount_remaining < 0

    if exact_in:
        amount_remaining_less_fee = mul_div(-amount_remaining, MAX_SWAP_FEE - fee_pips, MAX_SWAP_FEE)
        amount_in = (
            get_amount0_delta(sqrt_price_target_x96, sqrt_price_current_x96, liquidity, True)
            if zero_for_one
            else get_amount1_delta(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, True)
        )
        if amount_remaining_less_fee >= amount_in:
            sqrt_price_next_x96 = sqrt_price_target_x96
            fee_amount = (
                amount_in
                if fee_pips == MAX_SWAP_FEE
                else mul_div_rounding_up(amount_in, fee_pips, MAX_SWAP_FEE - fee_pips)
            )
        else:
            amount_in = amount_remaining_less_fee
            sqrt_price_next_x96 = get_next_sqrt_price_from_input(
                sqrt_price_current_x96, liquidity, amount_remaining_less_fee, zero_for_one
            )
            fee_amount = -amount_remaining - amount_in
        amount_out = (
            get_amount1_delta(sqrt_price_next_x96, sqrt_price_current_x96, liquidity, False)
            if zero_for_one
            else get_amount0_delta(sqrt_price_current_x96, sqrt_price_next_x96, liquidity, False)
        )
    else:
        if fee_pips >= MAX_SWAP_FEE:
            raise MathRevert("fee cannot be MAX_SWAP_FEE for exact output")
        amount_out = (
            get_amount1_delta(sqrt_price_target_x96, sqrt_price_current_x96, liquidity, False)
            if zero_for_one
            else get_amount0_delta(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, False)
        )
        if amount_remaining >= amount_out:
            sqrt_price_next_x96 = sqrt_price_target_x96
        else:
            amount_out = amount_remaining
            sqrt_price_next_x96 = get_next_sqrt_price_from_output(
                sqrt_price_current_x96, liquidity, amount_out, zero_for_one
            )
        amount_in = (
            get_amount0_delta(sqrt_price_next_x96, sqrt_price_current_x96, liquidity, True)
            if zero_for_one
            else get_amount1_delta(sqrt_price_current_x96, sqrt_price_next_x96, liquidity, True)
        )
        fee_amount = mul_div_rounding_up(amount_in, fee_pips, MAX_SWAP_FEE - fee_pips)

    return sqrt_price_next_x96, amount_in, amount_out, fee_amount
