"""Opt-in native Uniswap v4 quotes (``v4quote.amm.native.enable_v4``): the v3 kernel with the direction's swap fee against
``uniswap_v4.quote_exact_input`` (the Python path of ``routing._Search._simulate``), bit for bit on the amounts and the
crossings; and the routing layer with v4 pools, native v4 on vs the Python implementation.

Skipped when the extension is not installed. Off by default: without ``enable_v4`` a v4 state gets no handle.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

cn = pytest.importorskip("v4quote_native")

from v4quote.amm import native  # noqa: E402
from v4quote.amm import uniswap_v2 as v2m  # noqa: E402
from v4quote.amm import uniswap_v4 as v4m  # noqa: E402
from v4quote.errors import InsufficientState, MathRevert  # noqa: E402

from tests.unit.test_native_equivalence import _amount, _v3_random_pool  # noqa: E402

HOOK_BEFORE_SWAP = "0x" + f"{v4m.BEFORE_SWAP_FLAG:040x}"


def _v4_from_v3(rnd: random.Random, st, t0="0x" + "0" * 39 + "a", t1="0x" + "0" * 39 + "b"):
    """A v4 state with the v3 snapshot's price / liquidity / ticks; static, protocol-fee or hook-fee variants."""
    lp = rnd.choice([0, 1, 100, 500, 3000, 10_000, 100_000, 999_999])
    pf = 0 if rnd.random() < 0.5 else rnd.randrange(0, 1 << 12) | (rnd.randrange(0, 1 << 12) << 12)
    if rnd.random() < 0.3:  # a fee-only hook: dynamic fee resolved per direction
        key = v4m.PoolKey(t0, t1, v4m.DYNAMIC_FEE_FLAG, st.tick_spacing, HOOK_BEFORE_SWAP)
        hf = (rnd.choice([0, 500, 3000, 999_999]), rnd.choice([0, 100, 3000, 30_000]))
        return v4m.V4PoolState(key, st.sqrt_price_x96, st.tick, pf, 0, st.liquidity, st.ticks, hook_fee=hf)
    key = v4m.PoolKey(t0, t1, lp, st.tick_spacing)
    return v4m.V4PoolState(key, st.sqrt_price_x96, st.tick, pf, lp, st.liquidity, st.ticks)


def py_hop(st, zfo: bool, amount: int):
    """routing._Search._simulate for a v4 pool (Python): (amount_out, crossings) / revert / insufficient / other."""
    try:
        r = v4m.quote_exact_input(st, zfo, amount)
        if r.amount_out <= 0:
            return ("revert",)
        return ("ok", r.amount_out, r.initialized_ticks_crossed)
    except InsufficientState as e:
        return ("insufficient", str(e))
    except MathRevert:
        return ("revert",)
    except Exception as e:  # noqa: BLE001
        return ("other", type(e).__name__)


def nat_hop(hd, zfo: bool, amount: int):
    h = hd[0 if zfo else 1]
    i, j = (0, 1) if zfo else (1, 0)
    try:
        out, _fee, crossings = h.hop(i, j, amount)
        return ("ok", out, crossings)
    except cn.Insufficient as e:
        return ("insufficient", str(e))
    except cn.Revert:
        return ("revert",)
    except cn.Fallback:
        return ("fallback",)


def test_v4_off_by_default_gets_no_handle():
    st = _v4_from_v3(random.Random(1), _v3_random_pool(random.Random(1)))
    assert native.build_handle(st, cn, cached_ticks=False, v4=False) is None
    assert not native.v4_enabled()  # nothing in the test session switched it on for good


def test_v4_hop_equals_python_quote():
    rnd = random.Random(20261028)
    n = ok = rev = ins = fb = 0
    for k in range(700):
        v3 = _v3_random_pool(rnd, sparse=k % 5 == 0, word_range=k % 4 == 1, drop_net=k % 6 == 2,
                             zero_liq=k % 17 == 3)
        st = _v4_from_v3(rnd, v3)
        hd = native.build_handle(st, cn, cached_ticks=False, v4=True)
        if max(st.swap_fee(True), st.swap_fee(False)) >= native.MAX_SWAP_FEE:
            assert hd is None
            continue
        assert hd is not None and len(hd) == 2
        for _ in range(4):
            zfo = rnd.random() < 0.5
            amount = _amount(rnd)
            py, nat = py_hop(st, zfo, amount), nat_hop(hd, zfo, amount)
            n += 1
            if nat[0] == "fallback":
                fb += 1
                assert py[0] == "other", (k, zfo, amount, py)
                continue
            assert py[0] != "other" or amount >= 1 << 255, (k, zfo, amount, py, nat)
            if py[0] == "other":  # 2^255+: v4's int256 sign convention; the dispatcher never sends these to native
                continue
            assert nat == py, (k, zfo, amount, py, nat)
            ok += py[0] == "ok"
            rev += py[0] == "revert"
            ins += py[0] == "insufficient"
    print(f"\nv4 via the v3 kernel: {n} quotes, {ok} ok, {rev} reverts, {ins} insufficient, {fb} fallbacks")
    assert n >= 2000 and ok > 500 and rev > 0 and ins > 0


def test_v4_unsupported_or_max_fee_stays_python():
    v3 = _v3_random_pool(random.Random(5))
    ret = "0x" + f"{v4m.BEFORE_SWAP_RETURNS_DELTA_FLAG | v4m.BEFORE_SWAP_FLAG:040x}"
    key = v4m.PoolKey("0x" + "0" * 39 + "a", "0x" + "0" * 39 + "b", v4m.DYNAMIC_FEE_FLAG, v3.tick_spacing, ret)
    st = v4m.V4PoolState(key, v3.sqrt_price_x96, v3.tick, 0, 0, v3.liquidity, v3.ticks, hook_fee=(3000, 3000))
    assert native.build_handle(st, cn, cached_ticks=False, v4=True) is None  # return-delta hook: not modelled
    key = v4m.PoolKey("0x" + "0" * 39 + "a", "0x" + "0" * 39 + "b", 1_000_000, v3.tick_spacing)
    st = v4m.V4PoolState(key, v3.sqrt_price_x96, v3.tick, 0, 1_000_000, v3.liquidity, v3.ticks)
    assert native.build_handle(st, cn, cached_ticks=False, v4=True) is None  # 100 % fee: v3 / v4 steps differ


@pytest.fixture
def v4_backend(monkeypatch):
    if not native.ACTIVE:
        pytest.skip("native backend switched off (V4QUOTE_NATIVE=0)")
    yield
    native.enable_v4(False)


def _v4_cycle(rnd: random.Random):
    """A 3-pool cycle A -> B -> C -> A over v4 / v3 / v2 states, at least one v4 pool."""
    toks = ["0x" + f"{k:040x}" for k in (1, 2, 3)]
    pools = []
    for hop in range(3):
        a, b = toks[hop], toks[(hop + 1) % 3]
        t0, t1 = (a, b) if rnd.random() < 0.5 else (b, a)
        kind = "v4" if hop == 0 else rnd.choice(["v4", "v3", "v2"])
        if kind == "v2":
            r = rnd.randrange(10**20, 10**26)
            pools.append(v2m.V2PairState("0x" + f"{100 + hop:040x}", t0, t1, r, r * rnd.randint(90, 110) // 100))
            continue
        v3 = _v3_random_pool(rnd, word_range=rnd.random() < 0.3)
        if kind == "v3":
            pools.append(replace(v3, address="0x" + f"{100 + hop:040x}", token0=t0, token1=t1))
        else:
            pools.append(_v4_from_v3(rnd, v3, t0, t1))
    return toks, pools


def test_enable_v4_dispatch_and_describe(v4_backend):
    rnd = random.Random(7)
    st = _v4_from_v3(rnd, _v3_random_pool(rnd))
    while max(st.swap_fee(True), st.swap_fee(False)) >= native.MAX_SWAP_FEE or not st.liquidity:
        st = _v4_from_v3(rnd, _v3_random_pool(rnd))
    assert native.handle(st) is None and native.hop(st, 0, 1, 10**6) is None  # off: Python
    assert native.enable_v4(True) and native.describe()["v4"] is True
    h = native.handle(st)
    assert type(h) is tuple and native.quote(st, 0, 1, 10**6) is None  # quote() never serves v4
    for zfo in (True, False):
        i, j = (0, 1) if zfo else (1, 0)
        py = py_hop(st, zfo, 10**6)
        try:
            r = native.hop(st, i, j, 10**6)
            nat = ("ok", r[0], r[2])
        except MathRevert:
            nat = ("revert",)
        except InsufficientState as e:
            nat = ("insufficient", str(e))
        assert nat == py
    assert native.path([(st, 0, 1)]) is not None
    assert not native.enable_v4(False) and native.handle(st) is None  # cache dropped: off again
