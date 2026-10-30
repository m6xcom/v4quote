"""The exact-result fast paths of the amount search: v3 ``want_tick=False`` (same amounts, tick None, refused by
``apply_swap``), the memoised ``get_sqrt_ratio_at_tick`` and lazy Hop records in ``routing`` (same hops / gas as before, built on demand)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from v4quote import routing as rt
from v4quote.amm.liquidity_book.hooks_allowlist import DEFAULT_HOOKS_ALLOWLIST
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.v3 import pool as v3
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import get_sqrt_ratio_at_tick

A = "0x" + "aa" * 20
B = "0x" + "bb" * 20
C = "0x" + "cc" * 20
Q96 = 1 << 96


def v3_state(addr: str = "0x" + "11" * 20) -> v3.V3PoolState:
    net = {-600: 5 * 10**20, -60: 10**21, 60: -10**21, 600: -5 * 10**20}
    return v3.V3PoolState(addr, A, B, 3000, 60, Q96, 0, 10**22, TickBitmapSnapshot.from_ticks(60, net, (-3, 3)))


@pytest.mark.parametrize("amount", [10**15, 10**18, 3 * 10**19, 10**21])
@pytest.mark.parametrize("zfo", [True, False])
def test_want_tick_false_changes_only_the_tick(amount, zfo):
    st = v3_state()
    full = v3.quote_exact_input(st, zfo, amount)
    fast = v3.quote_exact_input(st, zfo, amount, want_tick=False)
    assert (fast.amount_in, fast.amount_out, fast.fee_amount, fast.sqrt_price_x96, fast.liquidity,
            fast.initialized_ticks_crossed, fast.steps, fast.amount_specified_remaining) == \
        (full.amount_in, full.amount_out, full.fee_amount, full.sqrt_price_x96, full.liquidity,
         full.initialized_ticks_crossed, full.steps, full.amount_specified_remaining)
    assert isinstance(full.tick, int) and get_sqrt_ratio_at_tick(full.tick) <= full.sqrt_price_x96
    # the swap ended between initialised ticks (the usual case): the post-swap tick was skipped, everything else equal
    ended_on_tick = full.sqrt_price_x96 in {get_sqrt_ratio_at_tick(t) for t in (-600, -60, 60, 600)}
    assert fast.tick == (full.tick if ended_on_tick else None)
    with pytest.raises(ValueError, match="want_tick"):
        v3.apply_swap(st, replace(fast, tick=None))


def test_sqrt_ratio_memo_is_transparent():
    for t in (-887272, -1, 0, 1, 60, 887272):
        assert get_sqrt_ratio_at_tick(t) == get_sqrt_ratio_at_tick.__wrapped__(t)
    with pytest.raises(Exception):
        get_sqrt_ratio_at_tick(887273)


def test_lazy_hop_records_match_eager_route_output():
    pa = V2PairState("0x" + "31" * 20, A, B, 10**24, 2 * 10**24)
    pb = V2PairState("0x" + "32" * 20, B, C, 10**24, 10**24)
    pc = V2PairState("0x" + "33" * 20, A, C, 10**24, 2 * 10**24)
    route = rt.best_route(A, C, 10**21, [pa, pb, pc], split=2)
    assert route.legs and all(leg.hops for leg in route.legs)
    for leg in route.legs:
        assert leg.gas_units == sum(h.gas_units for h in leg.hops) > 0
        assert leg.amount_out == leg.hops[-1].amount_out and leg.amount_in == leg.hops[0].amount_in
    plist, _skipped = rt._normalise([pa, pb], None, DEFAULT_HOOKS_ALLOWLIST, False)
    s = rt._Search(plist, None, rt.DEFAULT_GAS, DEFAULT_HOOKS_ALLOWLIST, None)
    edges = (rt._Edge(0, 0, 1), rt._Edge(1, 0, 1))
    lo = s.path(0, edges, 10**20)
    assert lo._hops is None and lo._gas is None  # nothing built by the evaluation itself
    assert lo.gas_units == 2 * rt.DEFAULT_GAS.base[rt.KIND_V2] and lo._hops is None  # gas without records
    hops = lo.hops
    assert lo._hops is hops and [h.pool_id for h in hops] == [pa.address, pb.address]
    assert hops[0].amount_in == 10**20 and hops[0].amount_out == hops[1].amount_in and hops[1].amount_out == lo.amount_out
