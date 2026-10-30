"""v4quote.routing.best_route on synthetic pools with hand-verified optimal routes.

Reference values (Uniswap v2 getAmountOut, fee 997/1000, floor division):
  out(100; 1000, 1000)        = 100*997*1000 // (1000*1000 + 100*997)        = 99_700_000 // 1_099_700 = 90
  out(100; 1e6, 1e6)          = 100*997*1e6 // (1e6*1000 + 99_700)           = 99
  out(99; 1e6, 1e6)           = 99*997*1e6 // (1e9 + 98_703)                 = 98
  out(1e5; 1e6, 1e6)          = 99_700_000_000_000 // 1_099_700_000          = 90_661
  out(5e4; 1e6, 1e6)          = 49_850_000_000_000 // 1_049_850_000          = 47_482  (x2 = 94_964)
  out(3e5; 3e6, 3e6)          = 271_983,  out(4e5; 3e6, 3e6) = 352_006
For two identical v2 pools the optimal split is 50/50; for depths 3:1 at the same price and fee it is 75/25
(equal marginal price: a/R1 = b/R2). Floor rounding makes the total non-concave at the wei level, so the
bps-resolution optimum can sit a wei above the exact 50/50 value (94_965 at e.g. 4999 bps).
"""

from __future__ import annotations

import time
from itertools import combinations

import pytest

from v4quote import routing as rt
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.liquidity_book import FeeParameters, LBPairState, get_swap_out
from v4quote.amm.uniswap_v2 import V2PairState, get_amount_out
from v4quote.amm.v3 import TickBitmapSnapshot, V3PoolState, quote_exact_input
from v4quote.errors import InsufficientState

A = "0x" + "a1" * 20
B = "0x" + "b2" * 20
C = "0x" + "c3" * 20
D = "0x" + "d4" * 20
E = "0x" + "e5" * 20
Q96 = 1 << 96
ID_ONE = 1 << 23


def v2(addr: str, t0: str, t1: str, r0: int, r1: int, fee=(997, 1000)) -> V2PairState:
    return V2PairState(addr, t0, t1, r0, r1, fee[0], fee[1])


def v3_full_range(addr, t0, t1, liquidity, *, fee=3000, ts=60, word_range=None) -> V3PoolState:
    hi = (887272 // ts) * ts
    ticks = TickBitmapSnapshot.from_ticks(ts, {-hi: liquidity, hi: -liquidity}, word_range)
    return V3PoolState(addr, t0, t1, fee, ts, Q96, 0, liquidity, ticks)


def lb_ladder(addr, tx, ty, *, n=20, x=10**20, y=10**20) -> LBPairState:
    bins = {ID_ONE: (x, y)}
    for k in range(1, n + 1):
        bins[ID_ONE - k] = (0, y)
        bins[ID_ONE + k] = (x, 0)
    params = FeeParameters(
        base_factor=5000, filter_period=30, decay_period=600, reduction_factor=5000, variable_fee_control=40000,
        protocol_share=1000, max_volatility_accumulator=350000, volatility_accumulator=0, volatility_reference=0,
        id_reference=ID_ONE, time_of_last_update=1_000,
    )
    return LBPairState(
        address=addr, token_x=tx, token_y=ty, bin_step=10, active_id=ID_ONE, params=params, bins=bins,
        loaded_range=(ID_ONE - n - 5, ID_ONE + n + 5), exhausted_below=True, exhausted_above=True, chain="avalanche",
    )


def addr(n: int) -> str:
    return "0x" + f"{n:040x}"


# =========================================================================== hand-verified optima
def test_two_hop_beats_direct():
    pools = [
        v2(addr(1), A, B, 1_000, 1_000),  # shallow direct pool
        v2(addr(2), A, C, 10**6, 10**6),
        v2(addr(3), C, B, 10**6, 10**6),
    ]
    r = rt.best_route(A, B, 100, pools, split=1)
    assert r.amount_out == 98
    assert [h.pool_id for h in r.hops] == [addr(2), addr(3)]
    assert [(h.amount_in, h.amount_out) for h in r.hops] == [(100, 99), (99, 98)]
    assert [h.token_in for h in r.hops] == [A, C] and r.hops[-1].token_out == B
    # with splitting allowed the 2-hop path still wins (any share on the direct pool loses output)
    assert rt.best_route(A, B, 100, pools).amount_out == 98
    # the direct-only answer
    d = rt.best_route(A, B, 100, pools, max_hops=1)
    assert d.amount_out == 90 and [h.pool_id for h in d.hops] == [addr(1)]


def test_split_beats_single_identical_pools():
    pools = [v2(addr(1), A, B, 10**6, 10**6), v2(addr(2), B, A, 10**6, 10**6)]  # second pool has reversed order
    single = rt.best_route(A, B, 10**5, pools, split=1)
    assert single.amount_out == 90_661 and not single.is_split
    r = rt.best_route(A, B, 10**5, pools)
    assert r.is_split and len(r.legs) == 2
    assert r.amount_out == 94_965  # 94_964 at exactly 50/50, one wei more at e.g. 4999 bps (floor rounding)
    assert sum(lg.amount_in for lg in r.legs) == 10**5
    assert sum(lg.share_bps for lg in r.legs) == rt.BPS
    assert abs(r.legs[0].share_bps - 5000) <= 60
    for lg in r.legs:  # every leg is exactly the v2 formula on its pool
        (h,) = lg.hops
        assert h.amount_out == get_amount_out(lg.amount_in, 10**6, 10**6)
    assert r.legs[1].hops[0].zero_for_one is False  # B is token0 of the second pool: A -> B is one-for-zero


def test_split_asymmetric_depth_is_75_25():
    pools = [v2(addr(1), A, B, 3 * 10**6, 3 * 10**6), v2(addr(2), A, B, 10**6, 10**6)]
    r = rt.best_route(A, B, 4 * 10**5, pools)
    # 75/25: 271_983 + 90_661 = 362_644 (= brute-force maximum over all shares)
    assert r.amount_out == 362_644
    shares = {lg.hops[0].pool_id: lg.share_bps for lg in r.legs}
    assert abs(shares[addr(1)] - 7500) <= 20
    assert r.amount_out > 352_006  # best single


def _brute_force_v2(pools, t_in, t_out, amount):
    """Exhaustive reference over the same search space (all paths, all pool-disjoint pairs, every bp)."""
    by = {p.address: p for p in pools}

    def out_path(path, amt):
        for pid, zfo in path:
            if amt <= 0:
                return None
            p = by[pid]
            try:
                amt = p.get_amount_out(amt, zfo)
            except Exception:
                return None
            if amt == 0:
                return None
        return amt

    paths = []
    for p in pools:
        if {t_in, t_out} == {p.token0, p.token1}:
            paths.append(((p.address, p.token0 == t_in),))
    for p1 in pools:
        if t_in not in (p1.token0, p1.token1):
            continue
        mid = p1.token1 if p1.token0 == t_in else p1.token0
        if mid == t_out:
            continue
        for p2 in pools:
            if p2 is not p1 and {mid, t_out} == {p2.token0, p2.token1}:
                paths.append(((p1.address, p1.token0 == t_in), (p2.address, p2.token0 == mid)))
    best = max((out_path(pth, amount) or 0) for pth in paths)
    for pa, pb in combinations(paths, 2):
        if {x for x, _ in pa} & {x for x, _ in pb}:
            continue
        for s in range(1, rt.BPS):
            a = amount * s // rt.BPS
            oa, ob = out_path(pa, a), out_path(pb, amount - a)
            if oa is not None and ob is not None:
                best = max(best, oa + ob)
    return best


@pytest.mark.parametrize("case", ["mixed_fees", "two_hop_pair", "three_direct"])
def test_matches_exhaustive_search_on_v2_graphs(case):
    if case == "mixed_fees":
        pools = [
            v2(addr(1), A, B, 5 * 10**6, 4 * 10**6),
            v2(addr(2), A, B, 2 * 10**6, 17 * 10**5, fee=(9975, 10000)),
            v2(addr(3), A, C, 8 * 10**6, 8 * 10**6),
            v2(addr(4), C, B, 6 * 10**6, 5 * 10**6),
        ]
        amount = 7 * 10**5
    elif case == "two_hop_pair":
        pools = [
            v2(addr(1), A, C, 4 * 10**6, 4 * 10**6),
            v2(addr(2), C, B, 4 * 10**6, 4 * 10**6),
            v2(addr(3), A, D, 3 * 10**6, 6 * 10**6),
            v2(addr(4), D, B, 6 * 10**6, 3 * 10**6),
        ]
        amount = 10**6
    else:
        pools = [
            v2(addr(1), A, B, 10**7, 2 * 10**7),
            v2(addr(2), A, B, 3 * 10**6, 6 * 10**6),
            v2(addr(3), A, B, 10**6, 21 * 10**5, fee=(990, 1000)),
        ]
        amount = 2 * 10**6
    r = rt.best_route(A, B, amount, pools, split_candidates=10)
    ref = _brute_force_v2(pools, A, B, amount)
    # bps-resolution optimum; floor rounding can hide a wei-level better share from the ternary refinement
    assert ref - 2 <= r.amount_out <= ref
    # every leg recomputes exactly with the v2 formula
    for lg in r.legs:
        amt = lg.amount_in
        for h in lg.hops:
            p = next(p for p in pools if p.address == h.pool_id)
            assert h.amount_in == amt
            amt = p.get_amount_out(amt, h.zero_for_one)
            assert h.amount_out == amt
        assert lg.amount_out == amt
    if case == "two_hop_pair":  # the two symmetric 2-hop paths are split
        assert r.is_split and all(len(lg.hops) == 2 for lg in r.legs)


def test_split_legs_are_pool_disjoint():
    pools = [
        v2(addr(1), A, C, 10**7, 10**7),  # shared first hop of two 2-hop paths
        v2(addr(2), C, B, 10**6, 10**6),
        v2(addr(3), C, B, 10**6, 10**6),
        v2(addr(4), A, B, 10**5, 10**5),
    ]
    r = rt.best_route(A, B, 10**5, pools, split_candidates=10)
    if r.is_split:
        a, b = ({h.pool_id for h in lg.hops} for lg in r.legs)
        assert not a & b
    # pairs through addr(1) can never be combined: only (A-C-B via 2 or 3) x (direct 4) = 2 pairs
    assert r.stats.split_pairs == 2
    assert r.stats.paths == 3


# =========================================================================== mixed pool kinds
def test_mixed_kinds_two_hop_composition_is_exact():
    ts_ = 2_000
    v3p = v3_full_range(addr(10), A, C, 10**21)
    lbp = lb_ladder(addr(11), C, B)
    direct = v2(addr(12), A, B, 10**18, 10**18)
    amount = 3 * 10**20  # large enough to use several LB bins on the second hop
    r = rt.best_route(A, B, amount, [direct, v3p, lbp], timestamp=ts_, split=1)
    # reference: every candidate computed directly with v4quote.amm
    q = quote_exact_input(v3p, True, amount)
    left, lb_out, lb_fee = get_swap_out(lbp, q.amount_out, True, ts_)
    assert left == 0
    two_hop = lb_out
    one_hop = get_amount_out(amount, 10**18, 10**18)
    assert two_hop > one_hop
    assert r.amount_out == two_hop
    h1, h2 = r.hops
    assert (h1.pool_kind, h2.pool_kind) == ("uniswap_v3", "lfj_lb")
    assert h1.amount_out == q.amount_out and h1.fee_amount == q.fee_amount
    assert h2.amount_in == q.amount_out and h2.fee_amount == lb_fee
    assert h2.crossings >= 1  # LB crossings = bins used beyond the first
    assert h1.gas_units == rt.DEFAULT_GAS.hop("uniswap_v3", h1.crossings)
    assert r.gas_units == h1.gas_units + h2.gas_units


def test_mixed_kinds_split_matches_components():
    ts_ = 2_000
    v3p = v3_full_range(addr(20), A, B, 10**20, fee=500, ts=10)
    lbp = lb_ladder(addr(21), A, B, n=40, x=10**19, y=10**19)
    amount = 3 * 10**19
    r = rt.best_route(A, B, amount, [v3p, lbp], timestamp=ts_)
    singles = []
    for p in (v3p, lbp):
        try:
            one = rt.best_route(A, B, amount, [p], timestamp=ts_, split=1).amount_out
        except rt.NoRoute:
            one = 0
        singles.append(one)
    assert r.amount_out >= max(singles)
    # coarse reference grid (1%) over the only pair: the refined optimum is at least as good
    grid_best = 0
    for s in range(100, rt.BPS, 100):
        a = amount * s // rt.BPS
        oa = quote_exact_input(v3p, True, a)
        left, ob, _ = get_swap_out(lbp, amount - a, True, ts_)
        if oa.fully_filled and left == 0:
            grid_best = max(grid_best, oa.amount_out + ob)
    assert r.amount_out >= grid_best
    assert r.is_split
    for lg in r.legs:
        (h,) = lg.hops
        if h.pool_kind == "uniswap_v3":
            assert h.amount_out == quote_exact_input(v3p, True, lg.amount_in).amount_out
        else:
            assert h.amount_out == get_swap_out(lbp, lg.amount_in, True, ts_)[1]


def test_v4_native_is_wrapped_and_unsupported_hooks_are_skipped():
    hi = (887272 // 10) * 10
    ticks = TickBitmapSnapshot.from_ticks(10, {-hi: 10**21, hi: -(10**21)})
    good = v4.V4PoolState(v4.PoolKey(rt.NATIVE, B, 500, 10), Q96, 0, 0, 500, 10**21, ticks)
    dyn = v4.V4PoolState(v4.PoolKey(rt.NATIVE, B, v4.DYNAMIC_FEE_FLAG, 10), Q96, 0, 0, 100, 10**23, ticks)
    shallow = v2(addr(40), A, B, 10**18, 10**18)
    r = rt.best_route(A, B, 10**18, [good, dyn, shallow], wrapped_native=A, split=1)
    (h,) = r.hops
    assert h.pool_kind == "uniswap_v4" and h.native_in and not h.native_out and h.token_in == A.lower()
    assert h.amount_out == v4.quote_exact_input(good, True, 10**18).amount_out
    assert h.gas_units == rt.DEFAULT_GAS.hop("uniswap_v4", h.crossings) + rt.DEFAULT_GAS.wrap
    assert [pid for pid, _ in r.skipped] == ["0x" + dyn.key.pool_id().hex()]
    assert "dynamic LP fee" in r.skipped[0][1]
    # without wrapped_native the native pools are not connected to A
    r2 = rt.best_route(A, B, 10**18, [good, shallow], split=1)
    assert r2.hops[0].pool_id == addr(40)


# =========================================================================== state problems
def test_insufficient_state_is_raised_or_reported_never_approximated():
    limited = v3_full_range(addr(50), A, B, 10**18, word_range=(-1, 0))  # only 2 bitmap words around tick 0
    fallback = v2(addr(51), A, B, 10**18, 10**18)
    small = rt.best_route(A, B, 10**15, [limited, fallback])
    assert small.complete
    with pytest.raises(rt.RouteInsufficientState) as ei:
        rt.best_route(A, B, 5 * 10**18, [limited, fallback])
    assert isinstance(ei.value, InsufficientState)
    assert ei.value.pool_ids == (addr(50),)
    assert all(i.kind == "insufficient_state" for i in ei.value.issues)
    rep = rt.best_route(A, B, 5 * 10**18, [limited, fallback], on_insufficient="report")
    assert not rep.complete and {i.pool_id for i in rep.issues} == {addr(50)}
    for lg in rep.legs:  # whatever was chosen was priced exactly on known state
        for h in lg.hops:
            if h.pool_id == addr(51):
                assert h.amount_out == get_amount_out(h.amount_in, 10**18, 10**18)


def test_missing_pool_state_is_reported():
    pools = [rt.MissingPool(addr(60), "uniswap_v3", (A, B), reason="load failed"), v2(addr(61), A, B, 10**6, 10**6)]
    with pytest.raises(rt.RouteInsufficientState) as ei:
        rt.best_route(A, B, 10**4, pools)
    assert ei.value.pool_ids == (addr(60),) and ei.value.issues[0].kind == "missing_state"
    rep = rt.best_route(A, B, 10**4, pools, on_insufficient="report")
    assert rep.hops[0].pool_id == addr(61) and not rep.complete
    # a missing pool off every candidate path does not matter
    ok = rt.best_route(A, B, 10**4, [rt.MissingPool(addr(62), "uniswap_v2", (C, D)), v2(addr(61), A, B, 10**6, 10**6)])
    assert ok.complete


def test_only_insufficient_candidates_raise_even_in_report_mode():
    limited = v3_full_range(addr(70), A, B, 10**18, word_range=(-1, 0))
    with pytest.raises(rt.RouteInsufficientState):
        rt.best_route(A, B, 5 * 10**18, [limited], on_insufficient="report")


def test_no_route_and_infeasible():
    with pytest.raises(rt.NoRoute):
        rt.best_route(A, B, 10, [v2(addr(80), A, C, 10**6, 10**6), v2(addr(81), D, B, 10**6, 10**6)])
    with pytest.raises(rt.NoRoute):  # output rounds to 0 on the only pool
        rt.best_route(A, B, 1, [v2(addr(82), A, B, 10**6, 10)])


# =========================================================================== objective, determinism, API
def test_out_per_gas_trades_split_gain_against_gas():
    pools = [v2(addr(1), A, B, 10**6, 10**6), v2(addr(2), A, B, 10**6, 10**6)]
    # split gains 94_965 - 90_661 = 4_304 wei for one more v2 hop (60_000 gas)
    expensive = rt.best_route(A, B, 10**5, pools, out_per_gas=(1, 10))  # 6_000 wei per hop
    assert not expensive.is_split and expensive.net_amount_out == 90_661 - 6_000
    cheap = rt.best_route(A, B, 10**5, pools, out_per_gas=(1, 100))  # 600 wei per hop
    assert cheap.is_split and cheap.net_amount_out == cheap.amount_out - 1_200


def test_deterministic_under_pool_order():
    pools = [
        v2(addr(1), A, B, 5 * 10**6, 4 * 10**6),
        v2(addr(2), A, B, 2 * 10**6, 17 * 10**5),
        v2(addr(3), A, C, 8 * 10**6, 8 * 10**6),
        v2(addr(4), C, B, 6 * 10**6, 5 * 10**6),
    ]
    r1 = rt.best_route(A, B, 7 * 10**5, pools)
    r2 = rt.best_route(A.upper().replace("0X", "0x"), B, 7 * 10**5, list(reversed(pools)))
    assert r1 == r2


def test_lb_requires_timestamp_and_validation():
    lbp = lb_ladder(addr(90), A, B)
    with pytest.raises(ValueError, match="timestamp"):
        rt.best_route(A, B, 10**18, [lbp])
    p = v2(addr(91), A, B, 10**6, 10**6)
    for kw in ({"max_hops": 3}, {"split": 3}, {"on_insufficient": "ignore"}, {"split_grid_bps": 300},
               {"out_per_gas": (1, 0)}):
        with pytest.raises(ValueError):
            rt.best_route(A, B, 10, [p], **kw)
    with pytest.raises(ValueError):
        rt.best_route(A, B, 0, [p])
    with pytest.raises(ValueError):
        rt.best_route(A, A, 10, [p])
    with pytest.raises(ValueError, match="twice"):
        rt.best_route(A, B, 10, [p, p])


def test_legs_json_and_route_pool_label():
    pools = [rt.RoutePool(v2(addr(1), A, B, 10**6, 10**6), dex="lfj_v1"), v2(addr(2), A, B, 10**6, 10**6)]
    r = rt.best_route(A, B, 10**5, pools)
    rows = r.legs_json()
    assert len(rows) == 2 and {row["pool_id"] for row in rows} == {addr(1), addr(2)}
    assert all(isinstance(row["amount_in_raw"], str) and isinstance(row["amount_out_raw"], str) for row in rows)
    assert sum(int(row["amount_out_raw"]) for row in rows) == r.amount_out
    assert {row["dex"] for row in rows} == {"lfj_v1", ""}
    assert sum(row["share_bps"] for row in rows) == rt.BPS


def test_runtime_and_search_space_on_a_realistic_graph():
    """~20 pools, 5 tokens, v3/LB/v2 mixed: one call stays well under a second."""
    ts_ = 2_000
    toks = [A, B, C, D, E]
    pools = []
    k = 100
    for i, j in combinations(range(5), 2):
        pools.append(v3_full_range(addr(k), toks[i], toks[j], 10**21 + k)); k += 1
        pools.append(lb_ladder(addr(k), toks[i], toks[j], n=30)); k += 1
    t0 = time.perf_counter()
    r = rt.best_route(A, B, 5 * 10**20, pools, timestamp=ts_)
    dt = time.perf_counter() - t0
    assert r.stats.paths == 2 + 3 * 4  # 2 direct + 3 intermediates x (2 pools x 2 pools)
    assert r.stats.split_pairs >= 1
    assert dt < 5.0, dt
