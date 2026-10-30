"""Uniswap v3/v4 library math against the upstream test vectors.

Sources (copied verbatim, see comments on each test):
  * Uniswap/v3-core test/TickMath.spec.ts (+ snapshot), SqrtPriceMath.spec.ts, SwapMath.spec.ts
  * Uniswap/v4-core test/libraries/SwapMath.t.sol, test/utils/Constants.sol
"""

import json
import random
from pathlib import Path

import pytest

from v4quote.errors import MathRevert
from v4quote.math import MAX_UINT128, MAX_UINT256, mul_div, mul_div_rounding_up
from v4quote.amm.v3.sqrt_price_math import (
    get_amount0_delta,
    get_amount1_delta,
    get_next_sqrt_price_from_input,
    get_next_sqrt_price_from_output,
)
from v4quote.amm.v3.swap_math import compute_swap_step_v3, compute_swap_step_v4
from v4quote.amm.v3.tick_math import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# v4-core test/utils/Constants.sol (== encodePriceSqrt in v3-core tests)
SQRT_PRICE_1_1 = 79228162514264337593543950336
SQRT_PRICE_1_4 = 39614081257132168796771975168
SQRT_PRICE_121_100 = 87150978765690771352898345369
SQRT_PRICE_101_100 = 79623317895830914510639640423
SQRT_PRICE_1000_100 = 250541448375047931186413801569
SQRT_PRICE_10000_100 = 792281625142643375935439503360
E18 = 10**18


# ---------------------------------------------------------------- FullMath
def test_full_math_basic_and_reverts():
    # v3-core FullMath.spec.ts
    q128 = 1 << 128
    assert mul_div(q128, 50 * q128 // 100, 150 * q128 // 100) == q128 // 3
    assert mul_div(MAX_UINT256, MAX_UINT256, MAX_UINT256) == MAX_UINT256
    assert mul_div_rounding_up(q128, 50 * q128 // 100, 150 * q128 // 100) == q128 // 3 + 1
    with pytest.raises(MathRevert):
        mul_div(q128, 5, 0)
    with pytest.raises(MathRevert):
        mul_div(q128, q128, 1)
    with pytest.raises(MathRevert):  # rounding up would overflow
        mul_div_rounding_up(
            535006138814359, 432862656469423142931042426214547535783388063929571229938474969, 2
        )


# ---------------------------------------------------------------- TickMath
def test_tick_math_bounds():
    assert get_sqrt_ratio_at_tick(MIN_TICK) == 4295128739 == MIN_SQRT_RATIO
    assert get_sqrt_ratio_at_tick(MIN_TICK + 1) == 4295343490
    assert get_sqrt_ratio_at_tick(MAX_TICK - 1) == 1461373636630004318706518188784493106690254656249
    assert get_sqrt_ratio_at_tick(MAX_TICK) == 1461446703485210103287273052203988822378723970342 == MAX_SQRT_RATIO
    with pytest.raises(MathRevert):
        get_sqrt_ratio_at_tick(MIN_TICK - 1)
    with pytest.raises(MathRevert):
        get_sqrt_ratio_at_tick(MAX_TICK + 1)

    assert get_tick_at_sqrt_ratio(MIN_SQRT_RATIO) == MIN_TICK
    assert get_tick_at_sqrt_ratio(4295343490) == MIN_TICK + 1
    assert get_tick_at_sqrt_ratio(1461373636630004318706518188784493106690254656249) == MAX_TICK - 1
    assert get_tick_at_sqrt_ratio(MAX_SQRT_RATIO - 1) == MAX_TICK - 1
    with pytest.raises(MathRevert):
        get_tick_at_sqrt_ratio(MIN_SQRT_RATIO - 1)
    with pytest.raises(MathRevert):
        get_tick_at_sqrt_ratio(MAX_SQRT_RATIO)


def test_tick_math_snapshot_vectors():
    data = json.loads((FIXTURES / "uniswap_v3_tickmath_vectors.json").read_text(encoding="utf-8"))
    assert len(data["get_sqrt_ratio_at_tick"]) >= 20
    for tick, expected in data["get_sqrt_ratio_at_tick"]:
        assert get_sqrt_ratio_at_tick(tick) == int(expected), tick
    for ratio, expected in data["get_tick_at_sqrt_ratio"]:
        assert get_tick_at_sqrt_ratio(int(ratio)) == expected, ratio


def _tick_by_bisection(sqrt_price: int) -> int:
    lo, hi = MIN_TICK, MAX_TICK
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if get_sqrt_ratio_at_tick(mid) <= sqrt_price:
            lo = mid
        else:
            hi = mid - 1
    return lo


def test_get_tick_at_sqrt_ratio_is_floor_inverse():
    """getTickAtSqrtRatio == greatest tick whose sqrt ratio <= price (property from TickMath.spec)."""
    rng = random.Random(1234)
    for _ in range(300):
        tick = rng.randint(MIN_TICK, MAX_TICK - 1)
        p = get_sqrt_ratio_at_tick(tick)
        assert get_tick_at_sqrt_ratio(p) == tick
        # a price just below a tick boundary belongs to the previous tick
        if tick > MIN_TICK:
            assert get_tick_at_sqrt_ratio(p - 1) == tick - 1
    for _ in range(100):
        p = rng.randint(MIN_SQRT_RATIO, MAX_SQRT_RATIO - 1)
        assert get_tick_at_sqrt_ratio(p) == _tick_by_bisection(p)


# ---------------------------------------------------------------- SqrtPriceMath
def test_next_sqrt_price_from_input():
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_input(0, 0, E18 // 10, False)
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_input(1, 0, E18 // 10, True)
    with pytest.raises(MathRevert):  # input amount overflows the price
        get_next_sqrt_price_from_input(2**160 - 1, 1024, 1024, False)
    assert get_next_sqrt_price_from_input(1, 1, 2**255, True) == 1
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, E18 // 10, 0, True) == SQRT_PRICE_1_1
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, E18 // 10, 0, False) == SQRT_PRICE_1_1
    sqrt_p = 2**160 - 1
    max_amount_no_overflow = MAX_UINT256 - (MAX_UINT128 << 96) // sqrt_p
    assert get_next_sqrt_price_from_input(sqrt_p, MAX_UINT128, max_amount_no_overflow, True) == 1
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, E18, E18 // 10, False) == 87150978765690771352898345369
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, E18, E18 // 10, True) == 72025602285694852357767227579
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, 10 * E18, 2**100, True) == 624999999995069620
    assert get_next_sqrt_price_from_input(SQRT_PRICE_1_1, 1, MAX_UINT256 // 2, True) == 1


def test_next_sqrt_price_from_output():
    price = 20282409603651670423947251286016
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_output(0, 0, E18 // 10, False)
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_output(1, 0, E18 // 10, True)
    for amount_out, zfo in ((4, False), (5, False), (262145, True), (262144, True)):
        with pytest.raises(MathRevert):
            get_next_sqrt_price_from_output(price, 1024, amount_out, zfo)
    assert get_next_sqrt_price_from_output(price, 1024, 262143, True) == 77371252455336267181195264
    assert get_next_sqrt_price_from_output(SQRT_PRICE_1_1, E18 // 10, 0, True) == SQRT_PRICE_1_1
    assert get_next_sqrt_price_from_output(SQRT_PRICE_1_1, E18 // 10, 0, False) == SQRT_PRICE_1_1
    assert get_next_sqrt_price_from_output(SQRT_PRICE_1_1, E18, E18 // 10, False) == 88031291682515930659493278152
    assert get_next_sqrt_price_from_output(SQRT_PRICE_1_1, E18, E18 // 10, True) == 71305346262837903834189555302
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_output(SQRT_PRICE_1_1, 1, MAX_UINT256, True)
    with pytest.raises(MathRevert):
        get_next_sqrt_price_from_output(SQRT_PRICE_1_1, 1, MAX_UINT256, False)


def test_amount_deltas():
    assert get_amount0_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, 0, True) == 0
    assert get_amount0_delta(SQRT_PRICE_1_1, SQRT_PRICE_1_1, 0, True) == 0
    assert get_amount0_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, E18, True) == 90909090909090910
    assert get_amount0_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, E18, False) == 90909090909090909
    a = (1 << 45) << 96  # encodePriceSqrt(2**90, 1)
    b = (1 << 48) << 96  # encodePriceSqrt(2**96, 1)
    assert get_amount0_delta(a, b, E18, True) == get_amount0_delta(a, b, E18, False) + 1
    assert get_amount1_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, 0, True) == 0
    assert get_amount1_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, E18, True) == 100000000000000000
    assert get_amount1_delta(SQRT_PRICE_1_1, SQRT_PRICE_121_100, E18, False) == 99999999999999999


def test_swap_computation_sqrt_p_times_sqrt_q_overflows():
    sqrt_p = 1025574284609383690408304870162715216695788925244
    liquidity = 50015962439936049619261659728067971248
    sqrt_q = get_next_sqrt_price_from_input(sqrt_p, liquidity, 406, True)
    assert sqrt_q == 1025574284609383582644711336373707553698163132913
    assert get_amount0_delta(sqrt_q, sqrt_p, liquidity, True) == 406


# ---------------------------------------------------------------- SwapMath (v3 sign convention)
def test_swap_step_v3_vectors():
    # exact in capped at target, one for zero
    q, i, o, f = compute_swap_step_v3(SQRT_PRICE_1_1, SQRT_PRICE_101_100, 2 * E18, E18, 600)
    assert (i, f, o, q) == (9975124224178055, 5988667735148, 9925619580021728, SQRT_PRICE_101_100)
    assert q < get_next_sqrt_price_from_input(SQRT_PRICE_1_1, 2 * E18, E18, False)
    # exact out capped at target, one for zero
    q, i, o, f = compute_swap_step_v3(SQRT_PRICE_1_1, SQRT_PRICE_101_100, 2 * E18, -E18, 600)
    assert (i, f, o, q) == (9975124224178055, 5988667735148, 9925619580021728, SQRT_PRICE_101_100)
    # exact in fully spent
    q, i, o, f = compute_swap_step_v3(SQRT_PRICE_1_1, SQRT_PRICE_1000_100, 2 * E18, E18, 600)
    assert (i, f, o) == (999400000000000000, 600000000000000, 666399946655997866)
    assert q == get_next_sqrt_price_from_input(SQRT_PRICE_1_1, 2 * E18, E18 - f, False)
    # exact out fully received
    q, i, o, f = compute_swap_step_v3(SQRT_PRICE_1_1, SQRT_PRICE_10000_100, 2 * E18, -E18, 600)
    assert (i, f, o) == (2000000000000000000, 1200720432259356, E18)
    assert q == get_next_sqrt_price_from_output(SQRT_PRICE_1_1, 2 * E18, E18, False)
    # amount out capped at desired amount out
    q, i, o, f = compute_swap_step_v3(
        417332158212080721273783715441582, 1452870262520218020823638996, 159344665391607089467575320103, -1, 1
    )
    assert (i, f, o, q) == (1, 1, 1, 417332158212080721273783715441581)
    # target price of 1 uses partial input amount
    q, i, o, f = compute_swap_step_v3(2, 1, 1, 3915081100057732413702495386755767, 1)
    assert (i, f, o, q) == (39614081257132168796771975168, 39614120871253040049813, 0, 1)
    # entire input amount taken as fee (v3 behaviour)
    q, i, o, f = compute_swap_step_v3(2413, 79887613182836312, 1985041575832132834610021537970, 10, 1872)
    assert (i, f, o, q) == (0, 10, 0, 2413)
    # intermediate insufficient liquidity, zero for one exact output
    sqrt_p = 20282409603651670423947251286016
    q, i, o, f = compute_swap_step_v3(sqrt_p, sqrt_p * 11 // 10, 1024, -4, 3000)
    assert (o, q, i, f) == (0, sqrt_p * 11 // 10, 26215, 79)
    # intermediate insufficient liquidity, one for zero exact output
    q, i, o, f = compute_swap_step_v3(sqrt_p, sqrt_p * 9 // 10, 1024, -263000, 3000)
    assert (o, q, i, f) == (26214, sqrt_p * 9 // 10, 1, 1)


# ---------------------------------------------------------------- SwapMath (v4 sign convention)
def test_swap_step_v4_vectors():
    q, i, o, f = compute_swap_step_v4(SQRT_PRICE_1_1, SQRT_PRICE_101_100, 2 * E18, -E18, 600)
    assert (i, o, f, q) == (9975124224178055, 9925619580021728, 5988667735148, SQRT_PRICE_101_100)
    q, i, o, f = compute_swap_step_v4(SQRT_PRICE_1_1, SQRT_PRICE_101_100, 2 * E18, E18, 600)
    assert (i, o, f, q) == (9975124224178055, 9925619580021728, 5988667735148, SQRT_PRICE_101_100)
    q, i, o, f = compute_swap_step_v4(SQRT_PRICE_1_1, SQRT_PRICE_1000_100, 2 * E18, -E18, 600)
    assert (i, o, f) == (999400000000000000, 666399946655997866, 600000000000000)
    q, i, o, f = compute_swap_step_v4(SQRT_PRICE_1_1, SQRT_PRICE_10000_100, 2 * E18, E18, 600)
    assert (i, f, o) == (2000000000000000000, 1200720432259356, E18)
    q, i, o, f = compute_swap_step_v4(
        417332158212080721273783715441582, 1452870262520218020823638996, 159344665391607089467575320103, 1, 1
    )
    assert (i, f, o, q) == (1, 1, 1, 417332158212080721273783715441581)
    q, i, o, f = compute_swap_step_v4(2, 1, 1, -3915081100057732413702495386755767, 1)
    assert (i, f, o, q) == (SQRT_PRICE_1_4, 39614120871253040049813, 0, 1)
    # v4 differs from v3 here: amountIn = amountRemainingLessFee (9), fee = 1
    q, i, o, f = compute_swap_step_v4(2413, 79887613182836312, 1985041575832132834610021537970, -10, 1872)
    assert (i, f, o, q) == (9, 1, 0, 2413)
    sqrt_p = 20282409603651670423947251286016
    q, i, o, f = compute_swap_step_v4(sqrt_p, sqrt_p * 11 // 10, 1024, 4, 3000)
    assert (o, q, i, f) == (0, sqrt_p * 11 // 10, 26215, 79)
    q, i, o, f = compute_swap_step_v4(sqrt_p, sqrt_p * 9 // 10, 1024, 263000, 3000)
    assert (o, q, i, f) == (26214, sqrt_p * 9 // 10, 1, 1)


def test_swap_step_invariants_fuzz():
    """Invariants from v4-core test_fuzz_computeSwapStep, run on both ports."""
    rng = random.Random(42)
    for _ in range(2000):
        cur = rng.randint(MIN_SQRT_RATIO, MAX_SQRT_RATIO - 1)
        tgt = rng.randint(MIN_SQRT_RATIO, MAX_SQRT_RATIO - 1)
        liq = rng.randint(1, MAX_UINT128)
        amt = rng.choice([1, 10**6, 10**18, 10**30, rng.randint(1, 2**200)])
        fee = rng.choice([0, 100, 500, 3000, 10000, rng.randint(0, 999_999)])
        exact_in = rng.random() < 0.5
        zfo = cur >= tgt
        for v4 in (False, True):
            rem = (-amt if exact_in else amt) if v4 else (amt if exact_in else -amt)
            try:
                q, i, o, f = (compute_swap_step_v4 if v4 else compute_swap_step_v3)(cur, tgt, liq, rem, fee)
            except MathRevert:
                continue
            if exact_in:
                assert i + f <= amt
            else:
                assert o <= amt
            if zfo:
                assert tgt <= q <= cur
            else:
                assert cur <= q <= tgt
            if q != tgt:  # did not reach target => whole amount used
                if exact_in:
                    assert i + f == amt
                else:
                    assert o == amt
