"""Full swap loop against Uniswap v3-core's UniswapV3Pool.swaps.spec.ts snapshot (240 cases).

Each snapshot entry records the pool balance deltas, the resulting tick and
fee growth after executing the swap on the real Solidity pool. We rebuild the
same pools (positions -> liquidityNet) and compare exactly.
"""

import json
from pathlib import Path

import pytest

from v4quote.errors import InsufficientState, MathRevert
from v4quote.math import MAX_INT256, MAX_UINT128
from v4quote.amm.v3 import TickBitmapSnapshot, V3PoolState, quote_exact_input, quote_exact_output, swap
from v4quote.amm.v3.tick_math import MAX_SQRT_RATIO, MIN_SQRT_RATIO, get_tick_at_sqrt_ratio

from tests._helpers import encode_price_sqrt

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
E18 = 10**18
LOW, MEDIUM, HIGH = 500, 3000, 10000
SPACING = {LOW: 10, MEDIUM: 60, HIGH: 200}


def min_tick(ts: int) -> int:
    return -(887272 // ts) * ts


def max_tick(ts: int) -> int:
    return (887272 // ts) * ts


def max_liquidity_per_tick(ts: int) -> int:
    return MAX_UINT128 // ((max_tick(ts) - min_tick(ts)) // ts + 1)


def _pools():
    m = SPACING[MEDIUM]
    full = lambda fee: [(min_tick(SPACING[fee]), max_tick(SPACING[fee]), 2 * E18)]  # noqa: E731
    return {
        "low fee, 1:1 price, 2e18 max range liquidity": (LOW, encode_price_sqrt(1, 1), full(LOW)),
        "medium fee, 1:1 price, 2e18 max range liquidity": (MEDIUM, encode_price_sqrt(1, 1), full(MEDIUM)),
        "high fee, 1:1 price, 2e18 max range liquidity": (HIGH, encode_price_sqrt(1, 1), full(HIGH)),
        "medium fee, 10:1 price, 2e18 max range liquidity": (MEDIUM, encode_price_sqrt(10, 1), full(MEDIUM)),
        "medium fee, 1:10 price, 2e18 max range liquidity": (MEDIUM, encode_price_sqrt(1, 10), full(MEDIUM)),
        "medium fee, 1:1 price, 0 liquidity, all liquidity around current price": (
            MEDIUM,
            encode_price_sqrt(1, 1),
            [(min_tick(m), -m, 2 * E18), (m, max_tick(m), 2 * E18)],
        ),
        "medium fee, 1:1 price, additional liquidity around current price": (
            MEDIUM,
            encode_price_sqrt(1, 1),
            [(min_tick(m), max_tick(m), 2 * E18), (min_tick(m), -m, 2 * E18), (m, max_tick(m), 2 * E18)],
        ),
        "low fee, large liquidity around current price (stable swap)": (
            LOW,
            encode_price_sqrt(1, 1),
            [(-SPACING[LOW], SPACING[LOW], 2 * E18)],
        ),
        "medium fee, token0 liquidity only": (MEDIUM, encode_price_sqrt(1, 1), [(0, 2000 * m, 2 * E18)]),
        "medium fee, token1 liquidity only": (MEDIUM, encode_price_sqrt(1, 1), [(-2000 * m, 0, 2 * E18)]),
        "close to max price": (MEDIUM, encode_price_sqrt(2**127, 1), full(MEDIUM)),
        "close to min price": (MEDIUM, encode_price_sqrt(1, 2**127), full(MEDIUM)),
        "max full range liquidity at 1:1 price with default fee": (
            MEDIUM,
            encode_price_sqrt(1, 1),
            [(min_tick(m), max_tick(m), max_liquidity_per_tick(m))],
        ),
        "initialized at the max ratio": (MEDIUM, MAX_SQRT_RATIO - 1, full(MEDIUM)),
        "initialized at the min ratio": (MEDIUM, MIN_SQRT_RATIO, full(MEDIUM)),
    }


POOLS = _pools()
AMOUNTS = {"1.0000": E18, "0.0000000000000010000": 1000}
PRICES = {
    "0.50000": encode_price_sqrt(50, 100),
    "2.0000": encode_price_sqrt(200, 100),
    "2.5000": encode_price_sqrt(5, 2),
    "0.40000": encode_price_sqrt(2, 5),
}


def build_pool(fee: int, sqrt_price: int, positions) -> V3PoolState:
    ts = SPACING[fee]
    net: dict[int, int] = {}
    tick = get_tick_at_sqrt_ratio(sqrt_price)
    liquidity = 0
    for lower, upper, liq in positions:
        net[lower] = net.get(lower, 0) + liq
        net[upper] = net.get(upper, 0) - liq
        if lower <= tick < upper:
            liquidity += liq
    net = {k: v for k, v in net.items() if v != 0}
    return V3PoolState(
        address="0x0", token0="t0", token1="t1", fee=fee, tick_spacing=ts,
        sqrt_price_x96=sqrt_price, tick=tick, liquidity=liquidity,
        ticks=TickBitmapSnapshot.from_ticks(ts, net),
    )


def parse_case(desc: str):
    """-> (zero_for_one, amount_specified, sqrt_price_limit or None)."""
    limit = None
    if " to price " in desc:
        desc, p = desc.split(" to price ")
        limit = PRICES[p]
    w = desc.split()
    if w[1] == "exactly":  # swap exactly X tokenA for tokenB
        return w[3] == "token0", AMOUNTS[w[2]], limit
    if w[3] == "exactly":  # swap tokenA for exactly X tokenB
        return w[1] == "token0", -AMOUNTS[w[4]], limit
    # swap tokenA for tokenB to price P  (amountSpecified = type(int256).max)
    return w[1] == "token0", MAX_INT256, limit


def _cases():
    data = json.loads((FIXTURES / "uniswap_v3_pool_swaps_snapshot.json").read_text(encoding="utf-8"))
    out = []
    for c in data["cases"]:
        pool_desc = next(p for p in POOLS if c["name"].startswith(p + " "))
        out.append(pytest.param(pool_desc, c["name"][len(pool_desc) + 1 :], c, id=c["name"]))
    return out


CASES = _cases()


def test_all_snapshot_cases_present():
    assert len(CASES) == 240


@pytest.mark.parametrize("pool_desc,case_desc,expected", CASES)
def test_v3_pool_swap_matches_solidity_snapshot(pool_desc, case_desc, expected):
    pool = build_pool(*POOLS[pool_desc])
    zfo, amount_specified, limit = parse_case(case_desc)
    if "swapError" in expected:
        with pytest.raises(MathRevert):
            swap(pool, zfo, amount_specified, limit)
        return
    res = swap(pool, zfo, amount_specified, limit)
    # pool balance deltas: input positive, output negative
    amount0 = res.amount_in if zfo else -res.amount_out
    amount1 = -res.amount_out if zfo else res.amount_in
    assert pool.tick == int(expected["tickBefore"])
    assert amount0 == int(expected["amount0Delta"])
    assert amount1 == int(expected["amount1Delta"])
    assert res.tick == int(expected["tickAfter"])
    fg_key = "feeGrowthGlobal0X128Delta" if zfo else "feeGrowthGlobal1X128Delta"
    assert res.fee_growth_global_delta_x128 == int(expected[fg_key])


# ---------------------------------------------------------------- additional edge cases
def _sparse_pool(ts=1, fee=100, word_range=None):
    """Liquidity [-1000, 1000] plus ranges far away, to force many word-boundary steps."""
    net = {-1000: 10**20, 1000: -10**20, 50_000: 10**19, 60_000: -10**19}
    return V3PoolState(
        "0x0", "t0", "t1", fee, ts, encode_price_sqrt(1, 1), 0, 10**20,
        TickBitmapSnapshot.from_ticks(ts, net, word_range),
    )


def test_zero_liquidity_region_walks_word_by_word():
    pool = _sparse_pool()
    # enough token1 to push the price through [0, 1000], across the empty gap, into [50000, 60000]
    res = quote_exact_input(pool, False, 10**22)
    assert res.initialized_ticks_crossed >= 2
    # ticks 1000..50000 at spacing 1 span ~192 bitmap words: each is a separate (empty) step
    assert res.steps > 150
    assert res.tick >= 50_000


def test_insufficient_state_raises():
    pool = _sparse_pool(word_range=(-4, 3))
    with pytest.raises(InsufficientState):
        quote_exact_input(pool, False, 10**22)
    # a small swap staying inside the loaded words is fine
    assert quote_exact_input(pool, False, 10**15).amount_out > 0


def test_exact_in_out_roundtrip_multi_tick():
    pool = _sparse_pool()
    for zfo in (True, False):
        for amt in (1, 10**6, 10**18, 10**21):
            r_in = quote_exact_input(pool, zfo, amt)
            if r_in.amount_out == 0:
                continue
            r_out = quote_exact_output(pool, zfo, r_in.amount_out)
            # exact-output for the same output never needs more input
            assert r_out.amount_in <= amt
            assert r_out.amount_out == r_in.amount_out


def test_exact_output_unfillable_reverts_like_quoter():
    pool = _sparse_pool()
    with pytest.raises(MathRevert):
        quote_exact_output(pool, True, 10**40)


def test_no_liquidity_pool_moves_price_without_output():
    pool = V3PoolState("0x0", "t0", "t1", 3000, 60, encode_price_sqrt(1, 1), 0, 0,
                       TickBitmapSnapshot.from_ticks(60, {}))
    res = swap(pool, True, 10**18, encode_price_sqrt(1, 2))
    assert res.amount_in == 0 and res.amount_out == 0
    assert res.sqrt_price_x96 == encode_price_sqrt(1, 2)
