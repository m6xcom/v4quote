"""Metric OMM model: replay of MetricOmmSwapQuoter quotes recorded on Robinhood Chain, plus offline edge cases.

The fixture (``tests/fixtures/metric_robinhood.json``) holds, for 10 pools at one block, the raw on-chain reads
(``read_metric_snapshot``) and ``quoteHypothetical*`` results priced with the provider quote read at the same block.
It was recorded with ``tests/onchain/metric_record.py``. Every quote must either equal the quoter to the unit, revert
like the quoter, or be refused by the model (``UnsupportedPool``); a different number is a failure.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from v4quote.amm import metric_omm as mo
from v4quote.chain.metric import check_api_quotable, metric_state_from_snapshot
from v4quote.errors import InsufficientState, MathRevert, UnsupportedPool

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "metric_robinhood.json"
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
SELECTOR_NAMES = {
    "0x2e5a4599": "FinalAmountIsZero", "0x2c5211c6": "InvalidAmount", "0xd6e666c0": "BidGreaterThanAsk",
    "0x6b95e45f": "BidIsZero", "0xdc5eefea": "ReferencePriceOutOfRange", "0x7f043c76": "BidAskRefAsymmetryOutOfBounds",
}


def _replay(pool: dict) -> dict[str, list[int]]:
    """Per mode: [matched, refused, mismatched, reverted alike]."""
    state = metric_state_from_snapshot(pool["snapshot"])
    stats: dict[str, list[int]] = {}
    for zfo, exact_in, amount, q_in, q_out, err in pool["quotes"]:
        key = ("sell0" if zfo else "buy0") + ("_exact_in" if exact_in else "_exact_out")
        st = stats.setdefault(key, [0, 0, 0, 0])
        fn = mo.quote_exact_input if exact_in else mo.quote_exact_output
        try:
            r = fn(state, zfo, int(amount))
        except UnsupportedPool:
            st[1] += 1
            continue
        except MathRevert as exc:
            st[3 if err is not None and SELECTOR_NAMES.get(err) == str(exc) else 2] += 1
            continue
        st[0 if err is None and (r.amount_in, r.amount_out) == (int(q_in), int(q_out)) else 2] += 1
    return stats


@pytest.mark.parametrize("pool", DATA["pools"], ids=[p["name"] for p in DATA["pools"]])
def test_replay_quoter_to_the_unit(pool):
    stats = _replay(pool)
    assert all(v[2] == 0 for v in stats.values()), stats  # never a different number than the quoter
    for key in ("buy0_exact_out", "sell0_exact_out"):  # exact output: every quote reproduced, none refused
        assert stats[key][1] == 0 and stats[key][0] + stats[key][3] == sum(stats[key]), stats


def test_replay_totals(capsys):
    tot = [0, 0, 0, 0]
    exact_in = [0, 0]
    for pool in DATA["pools"]:
        for key, v in _replay(pool).items():
            tot = [a + b for a, b in zip(tot, v)]
            if key.endswith("exact_in"):
                exact_in = [exact_in[0] + v[0], exact_in[1] + v[1]]
    with capsys.disabled():
        print(f"\nMetric replay: {sum(tot)} quoter calls, {tot[0]} equal to the unit, {tot[3]} revert alike, "
              f"{tot[1]} refused (exact input), {tot[2]} different")
    assert tot[2] == 0
    assert tot[0] >= 1400 and exact_in[0] >= 550  # regression guard on the share the model answers


def test_price_checks_match_pool():
    n = 0
    for pool in DATA["pools"]:
        state = metric_state_from_snapshot(pool["snapshot"])
        for bid, ask, ref, err in pool.get("price_checks", []):
            s = replace(state, bid_x64=int(bid), ask_x64=int(ask), reference_x64=int(ref))
            try:
                mo.check_prices(s)
                got = None
            except MathRevert as exc:
                got = str(exc)
            assert got == (SELECTOR_NAMES[err] if err else None), (pool["name"], bid, ask, ref, err)
            n += 1
    assert n >= 80


def _state(name: str = "WETH/USDG") -> mo.MetricOmmPoolState:
    pool = next(p for p in DATA["pools"] if p["name"] == name)
    return metric_state_from_snapshot(pool["snapshot"])


def test_isqrt():
    rng = random.Random(7)
    for n in [0, 1, 2, 3, 4, 15, 16, 17, (1 << 512) - 1] + [rng.getrandbits(rng.randint(1, 600)) for _ in range(500)]:
        s = mo.isqrt(n)
        assert s * s <= n < (s + 1) * (s + 1)
    with pytest.raises(ValueError):
        mo.isqrt(-1)


def test_decode_bin_word():
    word = 5 | (7 << 104) | (100 << 208) | (3 << 224) | (9 << 240)
    assert mo.decode_bin_word(word) == (5, 7, 100, 3, 9)


def test_bins_outside_loaded_range_raise_insufficient_state():
    st = _state()
    lo, hi = st.cur_bin - 1, st.cur_bin + 1
    trimmed = replace(st, bins={i: st.bins[i] for i in range(lo, hi + 1)}, loaded_range=(lo, hi))
    big = 10 ** 30
    with pytest.raises(InsufficientState):
        mo.quote_exact_output(trimmed, False, big)
    with pytest.raises(InsufficientState):
        mo.quote_exact_output(trimmed, True, big)


def test_partial_fill_flags():
    st = _state()
    r = mo.quote_exact_output(st, False, 10 ** 30)  # more WETH than the pool holds: capped to its liquidity
    assert r.partial and r.amount_out == sum(b[0] for b in st.bins.values()) // st.token0_scale
    r = mo.quote_exact_output(st, False, 10 ** 15)
    assert not r.partial


def test_refusals_and_reverts():
    st = _state()
    with pytest.raises(MathRevert, match="InvalidAmount"):
        mo.quote_exact_input(st, True, 0)
    with pytest.raises(UnsupportedPool, match="paused"):
        mo.quote_exact_output(replace(st, pause_level=1), True, 10 ** 6)
    assert mo.quote_exact_output(replace(st, pause_level=1), True, 10 ** 6, ignore_pause=True).amount_out == 10 ** 6
    with pytest.raises(UnsupportedPool, match="notional"):
        mo.quote_exact_output(replace(st, notional_fee_e8=1), True, 10 ** 6)
    b = st.bins[st.cur_bin]
    fee_bins = dict(st.bins)
    fee_bins[st.cur_bin] = (b[0], b[1], b[2], 0, 5)
    with pytest.raises(UnsupportedPool, match="additional fee"):
        mo.quote_exact_output(replace(st, bins=fee_bins), True, 10 ** 6)
    with pytest.raises(MathRevert, match="BidGreaterThanAsk"):
        mo.quote_exact_output(replace(st, bid_x64=st.ask_x64), True, 10 ** 6)
    with pytest.raises(MathRevert, match="BidIsZero"):  # the API's "unavailable" sentinel
        mo.quote_exact_output(replace(st, bid_x64=0, ask_x64=mo.UNAVAILABLE_ASK), True, 10 ** 6)
    with pytest.raises(MathRevert, match="ReferencePriceOutOfRange"):
        mo.quote_exact_output(replace(st, reference_x64=st.ask_x64 + 1), True, 10 ** 6)


def test_state_validation():
    st = _state()
    with pytest.raises(ValueError):
        replace(st, loaded_range=(st.cur_bin + 1, st.highest_bin))
    with pytest.raises(ValueError):
        replace(st, pos_in_bin=mo.MAX_POS + 1)
    with pytest.raises(ValueError):
        replace(st, token1_scale=0)


def test_api_status_check():
    check_api_quotable({"priceProviderStatus": "healthy", "bidAdj": "5", "askAdj": "6"})
    for bad in ({"priceProviderStatus": "feed_down", "bidAdj": "0", "askAdj": str((1 << 128) - 1)},
                {"priceProviderStatus": "pool_paused", "bidAdj": "5", "askAdj": "6"},
                {"priceProviderStatus": "healthy", "bidAdj": "0", "askAdj": str((1 << 128) - 1)}):
        with pytest.raises(UnsupportedPool):
            check_api_quotable(bad)
