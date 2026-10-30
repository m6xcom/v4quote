"""Return-delta v4 hooks (``v4quote.amm.v4_rd_hooks``, ``V4PoolState.rd_hook``): the contracts' arithmetic (PonsV2MemeHook
two floors on the output; PangeaFeeHook feeBps of the quote leg, before or after the pool swap, and its OutOfRange check),
the flag gate, the pool-only paths (``swap``, ``hook_deltas=False``), routing and the native hop / path.
These are offline tests of the contract arithmetic; there are no on-chain (V4Quoter) vectors for these hooks in
this repository yet."""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from v4quote import routing as rt
from v4quote.amm import native
from v4quote.amm import uniswap_v4 as v4m
from v4quote.amm.v3 import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import get_sqrt_ratio_at_tick
from v4quote.amm.v4_rd_hooks import RdHook
from v4quote.errors import MathRevert, UnsupportedPool

PONS = "0xe5e702641ea86f4ae6cc3cdaed2b886f976be044"  # beforeInitialize | afterSwap | afterSwapReturnDelta
PANGEA = "0x15e5e7206c661a1e881f5cd33a94eaf2328fa8cc"  # + beforeAddLiquidity | beforeSwap | beforeSwapReturnDelta
KYBER = "0x4445520306c9c70952bdfec28f3989f53d9f80c4"  # beforeSwap | afterSwap | afterSwapReturnDelta (denied hook)
T0, T1 = "0x" + "0" * 39 + "a", "0x" + "0" * 39 + "b"
PONS_RD = RdHook("output_bps", (100, 200))


def _state(hook: str, rd: RdHook | None, ts: int = 60, tick: int = 0, liq: int = 10**22, fee: int = 0):
    bm = TickBitmapSnapshot.from_ticks(ts, {-600 * ts: liq, 600 * ts: -liq})
    key = v4m.PoolKey(T0, T1, fee, ts, hook)
    return v4m.V4PoolState(key, get_sqrt_ratio_at_tick(tick), tick, 0, fee, liq, bm, rd_hook=rd)


def test_hook_addresses_flags():
    assert PONS_RD.covers(int(PONS, 16) & v4m.ALL_HOOK_MASK)
    pangea = RdHook("quote_bps", (100,), True, 0)
    assert pangea.covers(int(PANGEA, 16) & v4m.ALL_HOOK_MASK)
    assert not PONS_RD.covers(int(PANGEA, 16) & v4m.ALL_HOOK_MASK)  # a beforeSwap delta Pons does not model
    assert not pangea.covers(int(PONS, 16) & v4m.ALL_HOOK_MASK)
    assert not PONS_RD.covers(int(KYBER, 16) & v4m.ALL_HOOK_MASK)  # beforeSwap without a known effect


def test_unsupported_without_model_or_with_a_wrong_one():
    assert _state(PONS, None).unsupported_reasons()  # as before: return-delta hooks rejected
    assert _state(PANGEA, None).unsupported_reasons()
    assert _state(PONS, PONS_RD).unsupported_reasons() == []
    assert _state(PANGEA, RdHook("quote_bps", (100,), True, 10**6)).unsupported_reasons() == []
    bad = _state(KYBER, PONS_RD).unsupported_reasons()
    assert any("does not match" in r for r in bad) and any("RETURNS_DELTA" in r for r in bad)
    with pytest.raises(UnsupportedPool):
        v4m.quote_exact_input(_state(PANGEA, PONS_RD), True, 10**18)
    with pytest.raises(UnsupportedPool):  # exact output not modelled
        v4m.quote_exact_output(_state(PONS, PONS_RD), True, 10**15)


def test_rdhook_validation():
    for bad in (("x", (1, 2)), ("output_bps", (1,)), ("output_bps", (1, -1)), ("output_bps", (1500, 600)),
                ("quote_bps", (100,)), ("quote_bps", (501,), True, 0), ("output_bps", (True, 1))):
        with pytest.raises(ValueError):
            RdHook(*bad)
    r = RdHook("quote_bps", (100,), False, -887220)
    assert RdHook.from_dict(r.as_dict()) == r and RdHook.from_dict(PONS_RD.as_dict()) == PONS_RD


def test_pons_two_floors():
    # 199 * 100 // 1e4 + 199 * 200 // 1e4 = 1 + 3 = 4, one floor of 300 bps would be 5
    assert PONS_RD.output_fee(True, 199) == 4 and PONS_RD.output_fee(False, 199) == 4
    assert RdHook("output_bps", (0, 0)).output_fee(True, 10**18) == 0
    assert PONS_RD.input_fee(True, 10**18) == 0


@pytest.mark.parametrize("zfo", [True, False])
def test_pons_quote_is_pool_minus_output_fee(zfo):
    st = _state(PONS, PONS_RD)
    for a in (1000, 12345, 10**15, 7 * 10**18 + 3):
        pool_r = v4m.quote_exact_input(st, zfo, a, hook_deltas=False)
        assert pool_r == v4m.quote_exact_input(replace(st, rd_hook=None, key=replace(st.key, hooks="0x" + "0" * 40)),
                                              zfo, a)  # the pool swap is the hookless pool
        r = v4m.quote_exact_input(st, zfo, a)
        o = pool_r.amount_out
        assert r.amount_out == o - (o * 100 // 10_000 + o * 200 // 10_000)
        assert r.amount_in == a and r.sqrt_price_x96 == pool_r.sqrt_price_x96 and r.tick == pool_r.tick
        assert v4m.swap(st, zfo, -a) == pool_r  # swap() = the pool (what the Swap log carries)


def test_pangea_quote_leg():
    rd = RdHook("quote_bps", (100,), True, 600 * 60)  # quote = currency0; the range lies below the edge
    st = _state(PANGEA, rd)
    a = 10**18 + 99
    # buy: quote (currency0) in -> beforeSwap takes 1 % of the input, the pool swaps the rest
    fee = a * 100 // 10_000
    pool_r = v4m.quote_exact_input(st, True, a - fee, hook_deltas=False)
    r = v4m.quote_exact_input(st, True, a)
    assert r.amount_out == pool_r.amount_out and r.amount_in == a and r.sqrt_price_x96 == pool_r.sqrt_price_x96
    # sell: quote out -> afterSwap takes 1 % of the pool's output
    pool_r = v4m.quote_exact_input(st, False, a, hook_deltas=False)
    r = v4m.quote_exact_input(st, False, a)
    assert r.amount_out == pool_r.amount_out - pool_r.amount_out * 100 // 10_000
    # quote = currency1: the directions swap roles
    st1 = _state(PANGEA, RdHook("quote_bps", (100,), False, -600 * 60))
    c = v4m.quote_exact_input(st1, True, a, hook_deltas=False)
    assert v4m.quote_exact_input(st1, True, a).amount_out == c.amount_out - c.amount_out // 100
    c = v4m.quote_exact_input(st1, False, a - a // 100, hook_deltas=False)
    assert v4m.quote_exact_input(st1, False, a).amount_out == c.amount_out


def test_pangea_out_of_range():
    # quote = currency0, edge 120: a post-swap tick >= 120 reverts; a sell (oneForZero) moves the tick up
    rd = RdHook("quote_bps", (100,), True, 120)
    st = _state(PANGEA, rd, ts=60)
    small = v4m.quote_exact_input(st, False, 10**15, hook_deltas=False)
    assert small.tick < 120
    v4m.quote_exact_input(st, False, 10**15)
    big = 10**21
    assert v4m.quote_exact_input(st, False, big, hook_deltas=False).tick >= 120
    with pytest.raises(MathRevert, match="OutOfRange"):
        v4m.quote_exact_input(st, False, big)
    # quote = currency1, edge -120: a post-swap tick < -120 reverts; zeroForOne moves the tick down
    st1 = _state(PANGEA, RdHook("quote_bps", (100,), False, -120), ts=60)
    with pytest.raises(MathRevert, match="OutOfRange"):
        v4m.quote_exact_input(st1, True, big)
    v4m.quote_exact_input(st1, True, 10**15)


native_needed = pytest.mark.skipif(not native.ACTIVE, reason="native backend not active")


@native_needed
def test_native_hop_and_path_with_rd_hooks():
    native.enable_v4(True)
    try:
        rnd = random.Random(28)
        for _ in range(200):
            tick = rnd.randrange(-3000, 3000)
            rd = rnd.choice([PONS_RD, RdHook("output_bps", (rnd.randrange(0, 1000), rnd.randrange(0, 1000))),
                             RdHook("quote_bps", (rnd.randrange(30, 501),), rnd.random() < 0.5, 0)])
            hook = PONS if rd.kind == "output_bps" else PANGEA
            if rd.kind == "quote_bps":
                rd = replace(rd, edge_tick=(10**6 if rd.quote_is_currency0 else -10**6))
            st = _state(hook, rd, tick=tick, liq=rnd.choice([10**12, 10**18, 10**22]))
            zfo = rnd.random() < 0.5
            a = rnd.randrange(1, 1 << rnd.randint(10, 80))
            try:
                py = v4m.quote_exact_input(st, zfo, a).amount_out
            except MathRevert:
                py = "revert"
            try:
                r = native.hop(st, 0 if zfo else 1, 1 if zfo else 0, a)
                nat = None if r is None else r[0]
            except MathRevert:
                nat = "revert"
            if rd.kind == "quote_bps":
                assert nat is None  # the range check needs the tick: Python
            else:
                assert nat == py, (rd, zfo, a)
        assert native.path([(_state(PONS, PONS_RD), 0, 1)]) is None  # whole-path native has no per-hop take
    finally:
        native.enable_v4(False)


def test_routing_simulate_uses_trader_amounts():
    st = _state(PONS, PONS_RD)
    plist, skipped = rt._normalise([st], None, rt.DEFAULT_HOOKS_ALLOWLIST, False)
    assert not skipped
    s = rt._Search(plist, None, rt.DEFAULT_GAS, rt.DEFAULT_HOOKS_ALLOWLIST, None)
    e = rt._Edge(0, 0, 1) if hasattr(rt, "_Edge") else None
    if e is None:
        pytest.skip("routing internals changed")
    r = s.hop(e, 10**16)
    assert r is not None and r.amount_out == v4m.quote_exact_input(st, True, 10**16).amount_out
