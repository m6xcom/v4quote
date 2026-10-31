"""v4quote.amm.serialization: property-style lossless round trips for every pool-state type, plus strictness."""

import json
import random
from dataclasses import replace

import pytest

from v4quote.amm import serialization as ser
from v4quote.amm.liquidity_book import FeeParameters, LBPairState, get_swap_out
from v4quote.amm.metric_omm import MAX_POS, MetricOmmPoolState
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.uniswap_v4 import PoolKey, V4PoolState
from v4quote.amm.uniswap_v4 import swap as v4_swap
from v4quote.amm.v4_rd_hooks import RdHook
from v4quote.amm.v3 import TickBitmapSnapshot, V3PoolState, quote_exact_input
from v4quote.amm.v3.tick_math import MAX_TICK, MIN_TICK, get_sqrt_ratio_at_tick
from v4quote.math.uint import MAX_UINT112, MAX_UINT128, MAX_UINT160, MAX_UINT256, MIN_INT128, MAX_INT128

N_CASES = 200
ID_ONE = 1 << 23


def _addr(rng: random.Random) -> str:
    s = "0x" + format(rng.getrandbits(160), "040x")
    return s if rng.random() < 0.5 else s.upper().replace("0X", "0x")  # case must survive as-is


def _edge(rng: random.Random, hi: int, lo: int = 0) -> int:
    return rng.choice([lo, hi, lo + 1, hi - 1, rng.randint(lo, hi), rng.randint(lo, min(hi, lo + 1000))])


def _block(rng):
    return rng.choice([None, 0, rng.randint(1, 10**9), (1 << 53) - 1])


def _bitmap(rng, ts) -> TickBitmapSnapshot:
    if rng.random() < 0.5:
        ticks = {}
        for _ in range(rng.randint(0, 20)):
            t = rng.randint(-(MAX_TICK // ts), MAX_TICK // ts) * ts
            ticks[t] = _edge(rng, MAX_INT128, MIN_INT128)
        wr = rng.choice([None, (rng.randint(-3466, 0), rng.randint(0, 3466))])
        return TickBitmapSnapshot.from_ticks(ts, ticks, wr)
    words = {rng.randint(-3466, 3466): _edge(rng, MAX_UINT256) for _ in range(rng.randint(0, 10))}
    net = {rng.randint(-887272, 887272): _edge(rng, MAX_INT128, MIN_INT128) for _ in range(rng.randint(0, 10))}
    return TickBitmapSnapshot(ts, words, net, rng.choice([None, (-5, 5), (0, 0)]))


def gen_v2(rng):
    den = rng.choice([1000, 10000, 10**6])
    return V2PairState(_addr(rng), _addr(rng), _addr(rng), _edge(rng, MAX_UINT112), _edge(rng, MAX_UINT112),
                       rng.randint(1, den), den, _block(rng))


def gen_v3(rng):
    ts = rng.choice([1, 10, 50, 60, 200])
    return V3PoolState(_addr(rng), _addr(rng), _addr(rng), rng.randint(0, 999_999), ts,
                       _edge(rng, MAX_UINT160), rng.randint(MIN_TICK, MAX_TICK), _edge(rng, MAX_UINT128),
                       _bitmap(rng, ts), _block(rng), rng.randint(0, 255))


def gen_v4(rng):
    ts = rng.choice([1, 10, 60, 200, 32767])
    key = PoolKey(_addr(rng), _addr(rng), rng.choice([0, 500, 3000, 0x800000, 1_000_000]), ts, _addr(rng))
    return V4PoolState(key, _edge(rng, MAX_UINT160), rng.randint(MIN_TICK, MAX_TICK), rng.randint(0, (1 << 24) - 1),
                       rng.randint(0, (1 << 24) - 1), _edge(rng, MAX_UINT128), _bitmap(rng, ts), _block(rng),
                       hook_fee=rng.choice([None, (rng.randint(0, 1_000_000), rng.randint(0, 1_000_000))]),
                       rd_hook=rng.choice([None, RdHook("output_bps", (rng.randint(0, 1000), rng.randint(0, 1000))),
                                           RdHook("quote_bps", (rng.randint(0, 500),), rng.random() < 0.5,
                                                  rng.randint(MIN_TICK, MAX_TICK))]))


def gen_lb(rng):
    params = FeeParameters(
        base_factor=rng.randint(0, 65535), filter_period=rng.randint(0, 4095), decay_period=rng.randint(0, 4095),
        reduction_factor=rng.randint(0, 16383), variable_fee_control=rng.randint(0, (1 << 24) - 1),
        protocol_share=rng.randint(0, 16383), max_volatility_accumulator=rng.randint(0, (1 << 20) - 1),
        volatility_accumulator=rng.randint(0, (1 << 20) - 1), volatility_reference=rng.randint(0, (1 << 20) - 1),
        id_reference=rng.randint(0, (1 << 24) - 1), time_of_last_update=rng.randint(0, (1 << 40) - 1),
    )
    active = rng.randint(0, (1 << 24) - 1)
    bins = {}
    for _ in range(rng.randint(0, 30)):
        bins[rng.randint(max(0, active - 100), min((1 << 24) - 1, active + 100))] = (
            _edge(rng, MAX_UINT128), _edge(rng, MAX_UINT128))  # (0, 0) entries included on purpose
    hooked = rng.random() < 0.5
    return LBPairState(
        _addr(rng), _addr(rng), _addr(rng), rng.randint(1, 250), active, params, bins,
        (max(0, active - 100), min((1 << 24) - 1, active + 100)), rng.random() < 0.5, rng.random() < 0.5,
        rng.choice(["2.1", "2.2"]), _edge(rng, MAX_UINT256) if hooked else 0, _block(rng),
        rng.choice([None, "avalanche", "monad"]),
        ("0x" + format(rng.getrandbits(256), "064x")) if hooked else None,
        rng.choice([None, _addr(rng)]), rng.choice([None, "0x" + format(rng.getrandbits(256), "064x")]),
        rng.choice([None, 0, _edge(rng, MAX_UINT256)]), rng.choice([None, True, False]),
    )


def gen_metric(rng):
    lowest = rng.randint(-40, 0)
    highest = rng.randint(0, 40)
    cur = rng.randint(lowest, highest)
    lo, hi = rng.randint(lowest, cur), rng.randint(cur, highest)
    bins = {i: (_edge(rng, (1 << 104) - 1), _edge(rng, (1 << 104) - 1), rng.randint(1, 65535), rng.randint(0, 65535),
                rng.randint(0, 65535)) for i in range(lo, hi + 1)}
    bid = _edge(rng, MAX_UINT128 - 2, 1)
    return MetricOmmPoolState(
        _addr(rng), _addr(rng), _addr(rng), rng.choice([1, 10**9, 10**12]), rng.choice([1, 10**12]), bid,
        bid + 2, bid + 1, cur, _edge(rng, MAX_POS), rng.randint(-10**6, 10**6), bins, (lo, hi), lowest, highest,
        rng.randint(0, 3), rng.randint(0, 10**6), rng.randint(0, 65535), tuple(_addr(rng) for _ in range(rng.randint(0, 3))),
        _block(rng), rng.choice([None, "robinhood"]))


GENERATORS = {"v2": gen_v2, "v3": gen_v3, "v4": gen_v4, "lb": gen_lb, "metric": gen_metric}


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_round_trip_lossless(kind):
    rng = random.Random(f"serialization-{kind}")
    for _ in range(N_CASES):
        s = GENERATORS[kind](rng)
        j = ser.to_json(s)
        back = ser.from_json(j)
        assert type(back) is type(s)
        assert back == s
        assert ser.to_json(back) == j  # canonical: re-encoding is byte-identical
        assert ser.from_json(j.encode()) == s
        assert ser.pool_kind(s) == kind and json.loads(j)["kind"] == kind
        assert ser.state_hash(back) == ser.state_hash(s)
        _assert_same_types(s, back)


def _assert_same_types(a, b):
    """Container types survive (tuples stay tuples); ints stay ints."""
    for f in a.__dataclass_fields__:
        va, vb = getattr(a, f), getattr(b, f)
        assert type(va) is type(vb) or (hasattr(va, "items") and isinstance(vb, dict)), f
        if isinstance(va, tuple):
            assert all(type(x) is type(y) for x, y in zip(va, vb)), f
    if isinstance(a, LBPairState):
        assert all(type(v) is tuple for v in b.bins.values())
        assert b._sorted_ids == a._sorted_ids


def test_uint256_values_are_decimal_strings():
    s = V3PoolState("0xp", "a", "b", 500, 10, MAX_UINT160, 0, MAX_UINT128,
                    TickBitmapSnapshot(10, {-1: MAX_UINT256}, {-10: MIN_INT128}, (-2, 2)))
    d = json.loads(ser.to_json(s))
    assert d["sqrt_price_x96"] == str(MAX_UINT160) and d["liquidity"] == str(MAX_UINT128)
    assert d["ticks"]["words"] == [[-1, str(MAX_UINT256)]]
    assert d["ticks"]["liquidity_net"] == [[-10, str(MIN_INT128)]]
    assert d["tick"] == 0 and d["fee"] == 500 and d["schema"] == 1


def test_state_hash_changes_with_state():
    s = gen_v2(random.Random(3))
    h = ser.state_hash(s)
    assert h.startswith("0x") and len(h) == 66
    s2 = V2PairState(s.address, s.token0, s.token1, s.reserve0 ^ 1, s.reserve1, s.fee_numerator,
                     s.fee_denominator, s.block_number)
    assert ser.state_hash(s2) != h


def test_quotes_identical_after_round_trip():
    """The math sees the same state: swaps on the deserialised state give identical results."""
    rng = random.Random(5)
    for _ in range(40):
        v2 = V2PairState("p", "a", "b", rng.randint(10**6, 10**30), rng.randint(10**6, 10**30))
        v2b = ser.from_json(ser.to_json(v2))
        amt = rng.randint(1, 10**24)
        assert v2b.get_amount_out(amt, True) == v2.get_amount_out(amt, True)

        ts = 60
        lo, hi = -600 * rng.randint(1, 50), 600 * rng.randint(1, 50)
        liq = rng.randint(10**15, 10**24)
        bm = TickBitmapSnapshot.from_ticks(ts, {lo: liq, hi: -liq})
        tick = rng.randint(lo, hi - 1)
        s3 = V3PoolState("p", "a", "b", 3000, ts, get_sqrt_ratio_at_tick(tick), tick, liq, bm)
        s3b = ser.from_json(ser.to_json(s3))
        zfo = rng.random() < 0.5
        assert quote_exact_input(s3b, zfo, amt) == quote_exact_input(s3, zfo, amt)
        key = PoolKey("0x0000000000000000000000000000000000000000", "0x0000000000000000000000000000000000000001", 3000, ts)
        s4 = V4PoolState(key, get_sqrt_ratio_at_tick(tick), tick, 0, 3000, liq, bm)
        assert v4_swap(ser.from_json(ser.to_json(s4)), zfo, -amt) == v4_swap(s4, zfo, -amt)

    bins = {ID_ONE: (10**20, 10**20)}
    for k in range(1, 10):
        bins[ID_ONE - k], bins[ID_ONE + k] = (0, 10**20), (10**20, 0)
    params = FeeParameters(5000, 30, 600, 5000, 40000, 1000, 350000, 0, 0, ID_ONE, 1_000)
    lb = LBPairState("0xpair", "X", "Y", 10, ID_ONE, params, bins, (ID_ONE - 15, ID_ONE + 15), True, True)
    lbb = ser.from_json(ser.to_json(lb))
    for sfy in (True, False):
        assert get_swap_out(lbb, 10**19, sfy, 2_000) == get_swap_out(lb, 10**19, sfy, 2_000)



# ------------------------------------------------------------------ strictness
def _v2_dict():
    return json.loads(ser.to_json(V2PairState("0xp", "a", "b", 1000, 2000, block_number=7)))


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.update(reserve0="01000"), "canonical"),
    (lambda d: d.update(reserve0="-0"), "canonical"),
    (lambda d: d.update(reserve0="+1000"), "canonical"),
    (lambda d: d.update(reserve0=" 1000"), "canonical"),
    (lambda d: d.update(reserve0=1000), "canonical"),  # JSON number where a string is required
    (lambda d: d.update(fee_numerator="997"), "expected int"),
    (lambda d: d.update(fee_numerator=True), "expected int"),
    (lambda d: d.update(extra=1), "unexpected keys"),
    (lambda d: d.pop("block_number"), "missing keys"),
    (lambda d: d.update(kind="v9"), "unknown pool kind"),
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d.update(schema=True), "schema"),
    (lambda d: d.update(block_number=1 << 53), "JSON-safe"),
])
def test_from_json_rejects_malformed(mutate, msg):
    d = _v2_dict()
    mutate(d)
    with pytest.raises(ValueError, match=msg):
        ser.from_json(json.dumps(d))


def test_from_json_rejects_floats_duplicates_and_unsorted_maps():
    good = ser.to_json(V2PairState("0xp", "a", "b", 1000, 2000))
    with pytest.raises(ValueError, match="floating-point"):
        ser.from_json(good.replace('"fee_denominator":1000', '"fee_denominator":1000.0'))
    with pytest.raises(ValueError, match="NaN"):
        ser.from_json(good.replace('"fee_denominator":1000', '"fee_denominator":NaN'))
    with pytest.raises(ValueError, match="duplicate key"):
        ser.from_json(good[:-1] + ',"reserve0":"1"}')
    v3 = V3PoolState("0xp", "a", "b", 500, 10, 2**96, 0, 10,
                     TickBitmapSnapshot(10, {-1: 5, 2: 7}, {}, None))
    d = json.loads(ser.to_json(v3))
    d["ticks"]["words"] = [[2, "7"], [-1, "5"]]
    with pytest.raises(ValueError, match="strictly increasing"):
        ser.from_json(json.dumps(d))
    d["ticks"]["words"] = [[-1, "5"], [-1, "6"]]
    with pytest.raises(ValueError, match="strictly increasing"):
        ser.from_json(json.dumps(d))
    with pytest.raises(ValueError):
        ser.from_json("[]")


def test_to_json_rejects_unsupported_and_invalid():
    with pytest.raises(ValueError, match="no serialiser"):
        ser.to_json(object())
    s = V3PoolState("0xp", "a", "b", 500, 10, 2**96, 0, 10, TickBitmapSnapshot(10, {}, {}, [-1, 1]))
    with pytest.raises(ValueError, match="tuple"):
        ser.to_json(s)  # a list range would not round-trip to the same value


def test_lb_snapshot_without_hook_link_facts_reads_as_unknown():
    """Snapshots written before 2026-09-28 lack hooks_extra_parameters / hooks_linked: they read as None (unknown),
    which keeps an LFJ-rewarder-hooked pool UnsupportedPool."""
    params = FeeParameters(5000, 30, 600, 5000, 40000, 1000, 350000, 0, 0, ID_ONE, 1_000)
    lb = LBPairState("0xpair", "X", "Y", 10, ID_ONE, params, {ID_ONE: (1, 1)}, (ID_ONE - 1, ID_ONE + 1),
                     hooks_extra_parameters=0, hooks_linked=True)
    d = json.loads(ser.to_json(lb))
    assert d["hooks_extra_parameters"] == "0" and d["hooks_linked"] is True
    del d["hooks_extra_parameters"], d["hooks_linked"]
    old = ser.from_json(json.dumps(d))
    assert old.hooks_extra_parameters is None and old.hooks_linked is None
    d["hooks_linked"] = "yes"
    with pytest.raises(ValueError, match="hooks_linked"):
        ser.from_json(json.dumps(d))


def test_expected_keys_match_encoder():
    rng = random.Random(9)
    for kind, gen in GENERATORS.items():
        st = gen(rng)
        if kind == "v4":  # the optional model keys (written only when set)
            st = replace(st, hook_fee=(1, 2), rd_hook=RdHook("output_bps", (100, 200)))
        d = ser.to_dict(st)
        assert set(d) - {"schema", "kind"} == set(ser._EXPECTED_KEYS[kind])


def test_v4_hook_fee_round_trip_and_old_snapshots():
    """hook_fee (fee-only hook model) survives a round trip and keeps the hooked pool modelled; a state without it
    is encoded exactly as before (no key), and an old snapshot without the key reads as None."""
    ts = 10
    bm = TickBitmapSnapshot.from_ticks(ts, {-1000: 10**20, 1000: -(10**20)})
    hook = "0x06a889870c8f83640d6816319f72e2aa579b6080"  # beforeInitialize | beforeSwap
    key = PoolKey("0x0000000000000000000000000000000000000000", "0x0000000000000000000000000000000000000001",
                  0x800000, ts, hook)
    s = V4PoolState(key, get_sqrt_ratio_at_tick(0), 0, 0, 0, 10**20, bm, 5, hook_fee=(700, 2600))
    b = ser.from_json(ser.to_json(s))
    assert b == s and b.hook_fee == (700, 2600)
    for zfo in (True, False):
        assert v4_swap(b, zfo, -10**15) == v4_swap(s, zfo, -10**15)
    plain = V4PoolState(key, get_sqrt_ratio_at_tick(0), 0, 0, 0, 10**20, bm, 5)
    d = ser.to_dict(plain)
    assert "hook_fee" not in d and ser.from_dict(d).hook_fee is None
    for bad in ([700], [700, 2600, 1], "700", [700, -1], [700, True]):
        with pytest.raises(ValueError):
            ser.from_dict({**ser.to_dict(s), "hook_fee": bad})
    with pytest.raises(ValueError):  # out of the LP fee range: rejected by V4PoolState
        ser.from_dict({**ser.to_dict(s), "hook_fee": [700, 1_000_001]})
