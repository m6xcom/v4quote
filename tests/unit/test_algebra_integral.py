"""Algebra Integral pools (v4quote.amm.algebra_integral): tick walk, swap equivalence with Algebra's loop, serialisation,
SwapFee decoding, and the chain check: exact-input round trips simulated on Robinhood Chain (eth_simulateV1) equal the model to the wei."""
from __future__ import annotations

import json
import random
from bisect import bisect_left, bisect_right
from dataclasses import replace
from pathlib import Path

import pytest

from v4quote.amm import serialization as ser
from v4quote.amm.algebra_integral import AlgebraTickSnapshot, swap_fee_from_log
from v4quote.amm.v3 import pool as v3m
from v4quote.amm.v3.liquidity_math import add_delta
from v4quote.amm.v3.pool import V3PoolState
from v4quote.amm.v3.swap_math import compute_swap_step_v3
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import MAX_TICK, MIN_TICK, get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio
from v4quote.errors import InsufficientState

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "algebra_integral_robinhood.json"


def _snap(net: dict[int, int], rng: tuple[int, int] | None = None) -> AlgebraTickSnapshot:
    t = TickBitmapSnapshot.from_ticks(1, net, rng)
    return AlgebraTickSnapshot(1, t.words, t.liquidity_net, t.word_range)


def test_walk_matches_sorted_list() -> None:
    r = random.Random(7)
    ticks = sorted(r.sample(range(-200_000, 200_000), 300))
    s = _snap({t: 1 for t in ticks})
    for _ in range(3000):
        x = r.randrange(-210_000, 210_000)
        i = bisect_right(ticks, x) - 1
        assert s.next_initialized_tick_within_one_word(x, True) == ((ticks[i], True) if i >= 0 else (MIN_TICK, False))
        j = bisect_right(ticks, x)
        assert s.next_initialized_tick_within_one_word(x, False) == (
            (ticks[j], True) if j < len(ticks) else (MAX_TICK, False))
    # exact hits: lte includes the tick itself, gt never
    t = ticks[100]
    assert s.next_initialized_tick_within_one_word(t, True) == (t, True)
    assert s.next_initialized_tick_within_one_word(t, False) == (ticks[101], True)
    assert s.next_initialized_tick_within_one_word(-1, True)[0] == ticks[bisect_left(ticks, 0) - 1]


def test_walk_refuses_outside_loaded_words() -> None:
    s = _snap({-300: 5, 600: -5}, rng=(-1, 1))  # ticks -256..511 known; -300 / 600 are beyond
    with pytest.raises(InsufficientState):
        s.next_initialized_tick_within_one_word(0, True)
    with pytest.raises(InsufficientState):
        s.next_initialized_tick_within_one_word(0, False)
    with pytest.raises(ValueError):
        AlgebraTickSnapshot(60, {}, {}, None)


def _algebra_loop(st: V3PoolState, zfo: bool, amount: int) -> tuple[int, int, int, int]:
    """Algebra's exact-input loop written from its description: target = the next initialised tick (linked list),
    no word steps. Returns (amount_out, sqrt_price, liquidity, steps)."""
    ticks = sorted(st.ticks.liquidity_net)
    price, tick, liq, rem, out, steps = st.sqrt_price_x96, st.tick, st.liquidity, amount, 0, 0
    limit = v3m.default_price_limit(zfo)
    while rem and price != limit:
        steps += 1
        if zfo:
            i = bisect_right(ticks, tick) - 1
            nxt = ticks[i] if i >= 0 else MIN_TICK
        else:
            j = bisect_right(ticks, tick)
            nxt = ticks[j] if j < len(ticks) else MAX_TICK
        p_next = get_sqrt_ratio_at_tick(nxt)
        target = max(p_next, limit) if zfo else min(p_next, limit)
        start = price
        price, a_in, a_out, fee = compute_swap_step_v3(price, target, liq, rem, st.fee)
        rem -= a_in + fee
        out += a_out
        if price == p_next and nxt in st.ticks.liquidity_net:
            d = st.ticks.liquidity_net[nxt]
            liq = add_delta(liq, -d if zfo else d)
            tick = nxt - 1 if zfo else nxt
        elif price != start:
            tick = get_tick_at_sqrt_ratio(price)
    return out, price, liq, steps


def _pool(net: dict[int, int], tick: int, liq: int, fee: int = 100) -> V3PoolState:
    return V3PoolState("0x" + "11" * 20, "0x" + "22" * 20, "0x" + "33" * 20, fee, 1, get_sqrt_ratio_at_tick(tick) + 12345,
                       tick, liq, _snap(net))


def test_swap_equals_algebra_loop_and_skips_word_steps() -> None:
    # liquidity over [-2000, 2000) and [-600, 900): empty words between the initialised ticks
    net = {-2000: 10 ** 18, 2000: -10 ** 18, -600: 3 * 10 ** 17, 900: -3 * 10 ** 17}
    st = _pool(net, 10, 13 * 10 ** 17)
    plain = replace(st, ticks=TickBitmapSnapshot.from_ticks(1, net))
    for zfo in (True, False):
        for amt in (10 ** 12, 10 ** 15, 3 * 10 ** 16, 10 ** 17):
            r = v3m.quote_exact_input(st, zfo, amt)
            out, price, liq, steps = _algebra_loop(st, zfo, amt)
            assert (r.amount_out, r.sqrt_price_x96, r.liquidity, r.steps) == (out, price, liq, steps)
            assert r.steps <= v3m.quote_exact_input(plain, zfo, amt).steps  # v3 also stops at word edges
    # a swap that crosses empty words: Algebra takes fewer steps than the v3 word walk
    r = v3m.quote_exact_input(st, False, 10 ** 17)
    assert r.steps < v3m.quote_exact_input(plain, False, 10 ** 17).steps


def test_serialization_round_trip_keeps_the_walk() -> None:
    st = _pool({-2000: 7, 2000: -7}, 0, 7)
    d = ser.to_dict(st)
    assert d["ticks"]["walk"] == "algebra"
    back = ser.from_dict(json.loads(json.dumps(d)))
    assert type(back.ticks) is AlgebraTickSnapshot and back == st
    plain = replace(st, ticks=TickBitmapSnapshot.from_ticks(1, {-2000: 7, 2000: -7}))
    assert "walk" not in ser.to_dict(plain)["ticks"]  # every other snapshot keeps its canonical form
    bad = json.loads(json.dumps(d))
    bad["ticks"]["walk"] = "other"
    with pytest.raises(ValueError):
        ser.from_dict(bad)


def test_swap_fee_from_log() -> None:
    assert swap_fee_from_log(100, 0) == 100
    assert swap_fee_from_log(0, 0) is None  # the pool fell back to lastFee
    assert swap_fee_from_log(100, 5) is None  # plugin fee: not modelled
    assert swap_fee_from_log(1_000_000, 0) is None


def test_onchain_round_trips_equal_the_model() -> None:
    fx = json.loads(FIX.read_text())
    states = {k: ser.from_dict(v) for k, v in fx["states"].items()}
    alg = [s for s in states.values() if isinstance(s.ticks, AlgebraTickSnapshot)]
    assert len(alg) == 1 and alg[0].fee == 100
    exact = refused = 0
    for v in fx["vectors"]:
        st = dict(states)
        a, tok = int(v["amount_in"]), v["start"].lower()
        try:
            for pid in v["hops"]:
                s = st[pid]
                zfo = tok == s.token0.lower()
                r = v3m.quote_exact_input(s, zfo, a)
                st[pid] = v3m.apply_swap(s, r)
                a, tok = r.amount_out, (s.token1 if zfo else s.token0).lower()
        except InsufficientState:
            refused += 1  # walked past the loaded words: the model refuses, never guesses
            continue
        assert a == int(v["chain_out"]), v
        exact += 1
    assert exact >= 18 and refused <= 1


def test_native_kernel_never_serves_an_algebra_snapshot() -> None:
    """The native v3 kernel walks bitmap words (v3 steps): an Algebra state gets no handle, Python quotes it."""
    from v4quote.amm import native

    st = _pool({-2000: 7, 2000: -7}, 0, 7)
    assert native._ticks_handle(st.ticks, object()) is None  # exact-type check before the module is touched
    assert native.build_handle(st, object()) is None
