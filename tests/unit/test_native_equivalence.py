"""Differential tests: the native backend (``v4quote_native``, crate native/) against the Python implementation, bit for bit.

Skipped when the extension is not installed. Runs whatever ``V4QUOTE_NATIVE`` says (handles are built directly with
``v4quote.amm.native.build_handle``), so the kill switch run still compares the two implementations.

Rule for every comparison (``same``): the native call returns Python's value, or raises Python's ``MathRevert`` /
``InsufficientState`` with the same message, or reports ``Fallback``. A fallback is accepted only where Python itself
raises something else (ZeroDivisionError, IndexError, ...) or where a case is marked as outside the native domain;
a value or a revert where Python raises something else is a failure. Fallback counts are asserted to be zero on the
realistic generators.

Coverage: (a) the core fixtures (Uniswap v3-core swap snapshot pools and TickMath vectors);
(b) the routing hop / path layers on random pool sets, Python vs
native in one process; (c) randomized states, thousands of quotes per kind with edge cases (zero liquidity, tick and
word boundaries, MIN / MAX sqrt price, missing tick data, 1 wei and 2^200+ amounts). The benchmark at the end prints per-hop costs (never asserted).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path

import pytest

cn = pytest.importorskip("v4quote_native")

from v4quote import routing as rt  # noqa: E402
from v4quote.amm import native  # noqa: E402
from v4quote.amm import uniswap_v2 as v2m  # noqa: E402
from v4quote.amm.v3 import pool as v3m  # noqa: E402
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot, compress, position  # noqa: E402
from v4quote.amm.v3.tick_math import (MAX_SQRT_RATIO, MAX_TICK, MIN_SQRT_RATIO, MIN_TICK,  # noqa: E402
                                   get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio)
from v4quote.errors import InsufficientState, MathRevert  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures"
MAX256 = (1 << 256) - 1


# ------------------------------------------------------------------------------------------------ comparison
def py_outcome(f):
    try:
        return ("ok", f())
    except InsufficientState as e:
        return ("insufficient", str(e))
    except MathRevert as e:
        return ("revert", str(e))
    except Exception as e:  # noqa: BLE001
        return ("other", type(e).__name__)


def nat_outcome(f):
    try:
        return ("ok", f())
    except cn.Insufficient as e:
        return ("insufficient", str(e))
    except cn.Revert as e:
        return ("revert", str(e))
    except cn.Fallback as e:
        return ("fallback", str(e))


class Tally:
    def __init__(self) -> None:
        self.n = self.fallbacks = self.reverts = self.insufficient = 0

    def same(self, py, nat, ctx, allow_fallback: bool = False) -> None:
        self.n += 1
        if nat[0] == "fallback":
            self.fallbacks += 1
            # Python carrying on with a negative amount (where the chain would revert on uint underflow) is outside
            # the native domain by design: the dispatcher returns Python's value
            negative = py[0] == "ok" and isinstance(py[1], int) and py[1] < 0
            assert py[0] == "other" or negative or allow_fallback, (ctx, py, nat)
            return
        assert py[0] != "other", ("native answered where Python raises", ctx, py, nat)
        assert nat == py, (ctx, py, nat)
        self.reverts += py[0] == "revert"
        self.insufficient += py[0] == "insufficient"


def h(state):
    hd = native.build_handle(state, cn, cached_ticks=False)
    assert hd is not None, type(state).__name__
    return hd


def v3_fields(r: v3m.SwapResult) -> tuple:
    return (r.amount_in, r.amount_out, r.fee_amount, r.sqrt_price_x96, r.tick, r.liquidity, r.initialized_ticks_crossed,
            r.steps, r.amount_specified_remaining)


def v3_compare(t: Tally, st: v3m.V3PoolState, zfo: bool, amount: int, limit=None, ctx="", allow_fallback=False):
    hd = h(st)
    for want_tick in (True, False):
        py = py_outcome(lambda: v3_fields(v3m.quote_exact_input(st, zfo, amount, limit, want_tick=want_tick)))
        nat = nat_outcome(lambda: tuple(hd.v3_quote_exact_input(zfo, amount, limit, want_tick)))
        t.same(py, nat, (ctx, zfo, amount, limit, want_tick), allow_fallback)


# ------------------------------------------------------------------------------------------------ (a) fixtures
def test_tick_math_vectors_and_sweep():
    data = json.loads((FIX / "uniswap_v3_tickmath_vectors.json").read_text(encoding="utf-8"))
    for tick, want in data["get_sqrt_ratio_at_tick"]:
        assert cn.sqrt_ratio_at_tick(int(tick)) == int(want) == get_sqrt_ratio_at_tick(int(tick))
    for sqrt, want in data["get_tick_at_sqrt_ratio"]:
        assert cn.tick_at_sqrt_ratio(int(sqrt)) == int(want) == get_tick_at_sqrt_ratio(int(sqrt))
    rnd = random.Random(1)
    ticks = [MIN_TICK, MAX_TICK, 0, 1, -1] + [rnd.randint(MIN_TICK, MAX_TICK) for _ in range(3000)]
    for tk in ticks:
        s = get_sqrt_ratio_at_tick(tk)
        assert cn.sqrt_ratio_at_tick(tk) == s
        for x in (s, s + 1, s - 1, s + rnd.randrange(1, 1 << 64)):
            if MIN_SQRT_RATIO <= x < MAX_SQRT_RATIO:
                assert cn.tick_at_sqrt_ratio(x) == get_tick_at_sqrt_ratio(x)
    for bad in (MIN_TICK - 1, MAX_TICK + 1):
        with pytest.raises(cn.Revert, match="^T$"):
            cn.sqrt_ratio_at_tick(bad)
    for bad in (MIN_SQRT_RATIO - 1, MAX_SQRT_RATIO, 0):
        with pytest.raises(cn.Revert, match="^R$"):
            cn.tick_at_sqrt_ratio(bad)


def test_v3_core_snapshot_pools_exact_input():
    """The 15 pools of Uniswap v3-core's swaps.spec snapshot (tests/unit/test_v3_pool.py), every exact-input case
    of the snapshot (amounts, price limits, MAX_INT256 'to price' swaps) plus a sweep of amounts and limits."""
    from tests.unit import test_v3_pool as tvp
    t = Tally()
    for desc, args in tvp.POOLS.items():
        pool = tvp.build_pool(*args)
        for zfo in (True, False):
            for amt in (1, 1000, 10**18, 10**22, tvp.MAX_INT256, 2**255, MAX256):
                v3_compare(t, pool, zfo, amt, None, desc)
            for p in tvp.PRICES.values():
                v3_compare(t, pool, zfo, tvp.MAX_INT256, p, desc)
                v3_compare(t, pool, zfo, 10**18, p, desc)
    for c in tvp.CASES:
        pool_desc, case_desc, _exp = c.values
        zfo, amount, limit = tvp.parse_case(case_desc)
        if amount > 0:
            v3_compare(t, tvp.build_pool(*tvp.POOLS[pool_desc]), zfo, amount, limit, case_desc)
    assert t.fallbacks == 0 and t.n > 400 and t.reverts > 0


# ------------------------------------------------------------------------------------------------ (c) randomized
def _v3_random_pool(rnd: random.Random, *, sparse=False, word_range=False, drop_net=False, zero_liq=False):
    ts = rnd.choice([1, 10, 60, 200])
    lo_t, hi_t = -(MAX_TICK // ts) * ts, (MAX_TICK // ts) * ts
    mode = rnd.random()
    if mode < 0.08:
        sqrt = MIN_SQRT_RATIO + rnd.randrange(0, 1 << 20)
    elif mode < 0.16:
        sqrt = MAX_SQRT_RATIO - 1 - rnd.randrange(0, 1 << 100)
    elif mode < 0.3:
        sqrt = get_sqrt_ratio_at_tick(rnd.randint(-2000, 2000) * ts)  # exactly on an initialisable tick
    else:
        sqrt = get_sqrt_ratio_at_tick(rnd.randint(-200_000, 200_000)) + rnd.randrange(0, 1 << 64)
    sqrt = min(max(sqrt, MIN_SQRT_RATIO), MAX_SQRT_RATIO - 1)
    tick = get_tick_at_sqrt_ratio(sqrt)
    net: dict[int, int] = {}
    liquidity = 0
    npos = 0 if zero_liq else rnd.randint(1, 8)
    span = 50 if not sparse else 20_000
    for _ in range(npos):
        c = (tick // ts) + rnd.randint(-span, span)
        w = rnd.randint(1, span)
        lower = max(lo_t, (c - w) * ts)
        upper = min(hi_t, (c + w) * ts)
        if rnd.random() < 0.15:
            lower, upper = lo_t, hi_t
        if lower >= upper:
            continue
        liq = rnd.choice([1, rnd.randrange(1, 1 << 64), rnd.randrange(1, 1 << 100), rnd.randrange(1, 1 << 120)])
        net[lower] = net.get(lower, 0) + liq
        net[upper] = net.get(upper, 0) - liq
        if lower <= tick < upper:
            liquidity += liq
    net = {k: v for k, v in net.items() if v != 0}
    liquidity = min(liquidity, (1 << 128) - 1)
    wr = None
    if word_range:
        wc = position(compress(tick, ts))[0]
        wr = (wc - rnd.randint(0, 3), wc + rnd.randint(0, 3))
    ticks = TickBitmapSnapshot.from_ticks(ts, net, wr)
    if drop_net and net:
        gone = rnd.choice(sorted(net))
        ticks = TickBitmapSnapshot(ts, ticks.words, {k: v for k, v in net.items() if k != gone}, wr)
    fee = rnd.choice([0, 1, 100, 500, 3000, 10_000, 999_999])
    return v3m.V3PoolState("0xp", "0xa", "0xb", fee, ts, sqrt, tick, liquidity, ticks)


def _amount(rnd: random.Random) -> int:
    k = rnd.random()
    if k < 0.1:
        return rnd.randint(1, 1000)
    if k < 0.75:
        return rnd.randrange(1, 1 << rnd.randint(20, 100))
    if k < 0.95:
        return rnd.randrange(1, 1 << rnd.randint(100, 200))
    return rnd.choice([(1 << 255) - 1, 1 << 255, MAX256, rnd.randrange(1 << 200, 1 << 255)])


def test_v3_randomized():
    rnd = random.Random(20261004)
    t = Tally()
    for k in range(700):
        st = _v3_random_pool(rnd, sparse=k % 5 == 0, word_range=k % 4 == 1, drop_net=k % 6 == 2, zero_liq=k % 17 == 3)
        for _ in range(3):
            zfo = rnd.random() < 0.5
            limit = None
            if rnd.random() < 0.25:
                limit = rnd.randrange(MIN_SQRT_RATIO, MAX_SQRT_RATIO)
            v3_compare(t, st, zfo, _amount(rnd), limit, k)
    print(f"\nv3 randomized: {t.n} quotes, {t.reverts} reverts, {t.insufficient} insufficient, {t.fallbacks} fallbacks")
    assert t.n >= 4000 and t.reverts > 0 and t.insufficient > 0 and t.fallbacks == 0


def test_v3_out_of_domain_states_get_no_handle():
    st = _v3_random_pool(random.Random(3))
    big = TickBitmapSnapshot(st.tick_spacing, {0: 1 << 300}, {}, None)  # a word above 2**256 (Python masks it)
    assert native.build_handle(replace(st, ticks=big), cn, cached_ticks=False) is None
    neg = TickBitmapSnapshot(st.tick_spacing, {}, {0: -(1 << 130)}, None)  # liquidityNet outside int128
    assert native.build_handle(replace(st, ticks=neg), cn, cached_ticks=False) is None


def test_v2_randomized():
    rnd = random.Random(2)
    t = Tally()
    for _ in range(4000):
        r0 = rnd.choice([0, 1, rnd.randrange(1, 1 << 112), (1 << 112) - 1])
        r1 = rnd.choice([0, 1, rnd.randrange(1, 1 << 112), (1 << 112) - 1])
        num, den = rnd.choice([(997, 1000), (9975, 10000), (1, 1), (9999, 10000), (rnd.randint(1, 10**6), 10**6)])
        st = v2m.V2PairState("0xp", "0xa", "0xb", r0, r1, num, den)
        hd = h(st)
        a = rnd.choice([0, 1, _amount(rnd)])
        zfo = rnd.random() < 0.5
        t.same(py_outcome(lambda: st.get_amount_out(a, zfo)), nat_outcome(lambda: hd.quote(0 if zfo else 1, 1 if zfo else 0, a)[0]),
               (r0, r1, num, den, a, zfo))
    assert t.fallbacks == 0 and t.reverts > 0


# ------------------------------------------------------------------------------------------------ (b) routing layers
@pytest.fixture
def backend(monkeypatch):
    """Switch the dispatch layer on / off in-process (the Python run of a comparison)."""
    if not native.ACTIVE:
        pytest.skip("native backend switched off (V4QUOTE_NATIVE=0): routing-layer comparison needs it on")

    def set_(on: bool) -> None:
        monkeypatch.setattr(native, "ACTIVE", on)
    return set_


def _random_ring_states(rnd: random.Random):
    """Three pools A/B, B/C, C/A over random v2 / v3 states with consistent token ids."""
    toks = ["0x" + f"{k:040x}" for k in (1, 2, 3)]
    pools = []
    for hop in range(3):
        a, b = toks[hop], toks[(hop + 1) % 3]
        kind = rnd.choice(["v2", "v3", "v3"])
        addr = "0x" + f"{100 + hop:040x}"
        flip = rnd.random() < 0.5
        t0, t1 = (a, b) if not flip else (b, a)
        if kind == "v2":
            r = rnd.randrange(10**20, 10**26)
            st = v2m.V2PairState(addr, t0, t1, r, r * rnd.randint(90, 110) // 100)
        else:
            p = _v3_random_pool(rnd, word_range=rnd.random() < 0.3)
            st = replace(p, address=addr, token0=t0, token1=t1)
        pools.append(st)
    return toks, pools


def test_best_route_native_equals_python(backend):
    """best_route uses the per-hop native dispatch (SearchStats.hop_evals unchanged): routes and stats identical."""
    rnd = random.Random(12)
    for _ in range(60):
        toks, pools = _random_ring_states(rnd)
        amt = rnd.choice([10**6, 10**18, 10**22])
        out = []
        for on in (False, True):
            backend(on)
            try:
                r = rt.best_route(toks[0], toks[1], amt, pools, on_insufficient="report")
                out.append((r.amount_out, r.legs, r.gas_units, r.issues, r.stats))
            except Exception as e:  # noqa: BLE001
                out.append((type(e).__name__, str(e)))
        assert out[0] == out[1]


def test_dispatch_layer_handles_and_kill_switch(backend):
    st = _v3_random_pool(random.Random(1))
    a = native.handle(st)
    assert a is not None and native.handle(st) is a  # cached per state object
    st2 = replace(st, liquidity=st.liquidity + 1)
    assert native.handle(st2) is not a  # replace -> new state -> new handle
    k = id(st2)
    del st2
    import gc
    gc.collect()
    assert k not in native._by_id  # dropped with the state
    assert native.describe()["math_backend"] == "native" == native.backend()

    class Sub(v3m.V3PoolState):
        pass

    sub = Sub(**{f: getattr(st, f) for f in st.__dataclass_fields__})
    assert native.handle(sub) is None  # subclasses: Python path
    backend(False)
    assert native.handle(st) is None and native.hop(st, 0, 1, 10) is None


def test_kill_switch_env_selects_python(tmp_path):
    import subprocess
    import sys
    code = "from v4quote.amm import native; print(native.backend(), native.ACTIVE)"
    env = dict(__import__("os").environ, V4QUOTE_NATIVE="0")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["python", "False"]
    env["V4QUOTE_NATIVE"] = "1"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["native", "True"]


# ------------------------------------------------------------------------------------------------ benchmark (printed)
def test_benchmark_per_hop_printed():
    """Per-hop cost, Python vs native (call from Python incl. int conversion) vs native without the call. Printed."""
    rows = []

    def bench(f, n=300):
        f()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            f()
        return (time.perf_counter_ns() - t0) / n / 1000

    from tests.unit import test_v3_pool as tvp
    v3_pool = tvp.build_pool(*tvp.POOLS["medium fee, 1:1 price, additional liquidity around current price"])
    cases = [("v3", v3_pool, 10**15), ("v2", v2m.V2PairState("0x", "0xa", "0xb", 10**24, 2 * 10**24), 10**20)]
    for name, st, a in cases:
        hd = h(st)
        if isinstance(st, v3m.V3PoolState):
            pyf = lambda: v3m.quote_exact_input(st, True, a, want_tick=False)  # noqa: E731
        else:
            pyf = lambda: st.get_amount_out(a, True)  # noqa: E731
        try:
            pyf()
            hd.hop(0, 1, a)
        except Exception:  # noqa: BLE001
            continue
        rows.append(f"{name}: python {bench(pyf):6.2f} us   native call {bench(lambda: hd.hop(0, 1, a)):5.2f} us"
                    f"   native math {hd.bench_hop(0, 1, a, 2000) / 1000:5.2f} us")
    print("\n" + "\n".join(rows))
