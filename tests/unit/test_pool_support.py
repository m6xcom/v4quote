"""Pool coverage rules of best_route: ``routing.unsupported_reasons`` and the
per-pool attribution carried by ``UnsupportedPool.skipped``.

Known values: the Avalanche Uniswap v4 USDt/USDC poolIds are the ones listed by the DEX; their PoolKeys (fee 18 / 32 pips, tickSpacing 1, hooks 0) were recovered offline from the ids and confirmed on
chain on 2026-09-29 (StateView.getSlot0 lpFee 18 / 32, protocolFee 0 at block 96388844).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from v4quote import routing as rt
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.liquidity_book import FeeParameters, LBPairState
from v4quote.amm.liquidity_book.pair import HOOK_FLAGS_SHIFT
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.v3 import TickBitmapSnapshot
from v4quote.errors import UnsupportedPool

A = "0x" + "a1" * 20
B = "0x" + "b2" * 20
Q96 = 1 << 96
ZERO = "0x" + "00" * 20
USDT = "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7"
USDC = "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E"


def v2(addr: str) -> V2PairState:
    return V2PairState(addr, A, B, 10**6, 10**6, 997, 1000)


def v4_state(key: v4.PoolKey) -> v4.V4PoolState:
    hi = (887272 // key.tick_spacing) * key.tick_spacing
    ticks = TickBitmapSnapshot.from_ticks(key.tick_spacing, {-hi: 10**18, hi: -10**18})
    lp_fee = 3000 if key.is_dynamic_fee else key.fee
    return v4.V4PoolState(key, Q96, 0, 0, lp_fee, 10**18, ticks)


def lb(addr: str, hooks_parameters: int = 0) -> LBPairState:
    params = FeeParameters(base_factor=5000, filter_period=30, decay_period=600, reduction_factor=5000,
                           variable_fee_control=40000, protocol_share=1000, max_volatility_accumulator=350000,
                           volatility_accumulator=0, volatility_reference=0, id_reference=1 << 23,
                           time_of_last_update=1_000)
    active = 1 << 23
    bins = {active: (10**18, 10**18), active - 1: (0, 10**18), active + 1: (10**18, 0)}
    return LBPairState(addr, A, B, 10, active, params, bins, (active - 10, active + 10), True, True,
                       hooks_parameters=hooks_parameters, chain="avalanche")


def hooked(addr_hook: int, flags: int = 0x151) -> int:
    return (flags << HOOK_FLAGS_SHIFT) | addr_hook


def test_unsupported_reasons_by_kind():
    assert rt.unsupported_reasons(v2("0x" + "01" * 20)) == []
    assert rt.unsupported_reasons(lb("0x" + "02" * 20)) == []  # no hook: modelled
    plain = v4_state(v4.PoolKey(A, B, 500, 10, ZERO))
    assert rt.unsupported_reasons(plain) == []
    dynamic = v4_state(v4.PoolKey(A, B, v4.DYNAMIC_FEE_FLAG, 10, ZERO))
    assert any("dynamic" in r for r in rt.unsupported_reasons(dynamic))
    before_swap = v4_state(v4.PoolKey(A, B, 500, 10, "0x" + "00" * 18 + "0080"))
    assert rt.unsupported_reasons(before_swap)
    assert rt.unsupported_reasons(before_swap, allow_v4_before_swap_hook=True) == []
    # an LB hook that is not in the allowlist, and a RoutePool wrapper is unwrapped
    unk = lb("0x" + "03" * 20, hooked(0xDEAD))
    why = rt.unsupported_reasons(rt.RoutePool(unk, "lfj_lb_v2.2"))
    assert why and "not in allowlist" in why[0]
    # the same rules as best_route: exactly the pools rejected there
    _plist, skipped = rt._normalise([v2("0x" + "04" * 20), unk, dynamic], None, rt.DEFAULT_HOOKS_ALLOWLIST, False)
    assert [p for p, _ in skipped] == [unk.address, "0x" + dynamic.key.pool_id().hex()]


def test_single_pool_raisers_keep_empty_skipped():
    with pytest.raises(UnsupportedPool) as ei:
        v4.check_supported(v4.PoolKey(A, B, v4.DYNAMIC_FEE_FLAG, 10, ZERO))
    assert ei.value.skipped == ()


@pytest.mark.parametrize(("fee", "pool_id"), [
    (18, "0xfe74ff9963652d64086e4467e64ceae7847ebf0139cb4046bcbe02c86e48f256"),
    (32, "0x8dc096ecc5cb7565daa9615d6b6b4e6d1ffb3b16cca4e0971dfaf0ed9cb55c63"),
])
def test_avalanche_v4_usdt_usdc_pools_are_hookless_static_fee(fee, pool_id):
    key = v4.PoolKey(USDT, USDC, fee, 1, ZERO)
    assert "0x" + key.pool_id().hex() == pool_id
    assert not key.is_dynamic_fee and v4.unsupported_reasons(key) == []
    assert rt.unsupported_reasons(v4_state(key)) == []


def test_every_v4_fixture_pool_id_matches_its_key_and_is_modelled():
    """Offline half of test_address_links.py::test_v4_links for every v4 entry of fixtures/pools.yaml."""
    data = yaml.safe_load((Path(__file__).resolve().parents[1] / "fixtures" / "pools.yaml").read_text(encoding="utf-8"))
    seen = 0
    for p in data["pools"]:
        if p["type"] != "uniswap_v4":
            continue
        toks = data["tokens"][p["chain"]]
        k = p["pool_key"]
        key = v4.PoolKey(toks[k["currency0"]], toks[k["currency1"]], k["fee"], k["tick_spacing"], k["hooks"])
        assert "0x" + key.pool_id().hex() == p["pool_id"].lower(), p["id"]
        assert int(key.currency0, 16) < int(key.currency1, 16), p["id"]
        assert v4.unsupported_reasons(key) == [], p["id"]
        seen += 1
    assert seen >= 10


# ---------------------------------------------------------------------------- gas of LB hops through hooked pairs
def _hooked_lb(addr: str, hook: int, proxy_kind: str):
    from dataclasses import replace

    from v4quote.amm.liquidity_book import HookApproval

    h_a, h_b, impl = "0x" + "aa" * 32, "0x" + "bb" * 32, "0x" + "00" * 19 + "ab"
    st = replace(lb(addr, hooked(hook)), hooks_codehash=h_a, hooks_implementation=impl,
                 hooks_implementation_codehash=h_b, hooks_extra_parameters=0, hooks_linked=True, hooks_pair=addr)
    allow = {("avalanche", "0x" + format(hook, "040x")): HookApproval(h_a, impl, h_b, proxy_kind=proxy_kind, pair=addr)}
    return st, allow


@pytest.mark.parametrize(("proxy_kind", "extra"), [("eip1967-beacon", 45_000), ("lfj-immutable-clone", 0)])
def test_hooked_lb_hop_gas_surcharge_by_proxy_kind(proxy_kind, extra):
    """A Pharaoh beacon rewarder with active rewards costs >= 40k over the LB base."""
    from dataclasses import replace

    addr = "0x" + "0a" * 20
    st, allow = _hooked_lb(addr, 0xB00C, proxy_kind)
    assert rt.lb_hook_kind(st, allow) == proxy_kind and rt.lb_hook_kind(lb("0x" + "0b" * 20)) is None
    q = rt.best_route(A, B, 10**15, [st], timestamp=2_000, hooks_allowlist=allow)
    lb_hop = next(h for h in q.hops if h.pool_kind == rt.KIND_LB)
    assert lb_hop.gas_units == rt.DEFAULT_GAS.hop(rt.KIND_LB, lb_hop.crossings) + extra
    assert rt.DEFAULT_GAS.lb_hook["eip1967-beacon"] >= 40_000
    # configurable: a caller's table replaces the default
    table = replace(rt.DEFAULT_GAS, lb_hook={"eip1967-beacon": 60_000, "*": 1})
    q2 = rt.best_route(A, B, 10**15, [st], timestamp=2_000, hooks_allowlist=allow, gas=table)
    lb2 = next(h for h in q2.hops if h.pool_kind == rt.KIND_LB)
    assert lb2.gas_units - rt.DEFAULT_GAS.hop(rt.KIND_LB, lb2.crossings) == (60_000 if extra else 1)


def test_gas_table_hook_kinds():
    g = rt.DEFAULT_GAS
    base = g.hop(rt.KIND_LB, 2)
    assert g.hop(rt.KIND_LB, 2, 0, None) == base                # no hook
    assert g.hop(rt.KIND_LB, 2, 0, "") == base + g.lb_hook["*"]  # hooked, kind unknown
    assert g.hop(rt.KIND_V3, 2, 0, "eip1967-beacon") == g.hop(rt.KIND_V3, 2)  # only LB hops
