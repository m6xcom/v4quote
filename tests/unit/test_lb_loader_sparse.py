"""load_lb_pair(follow_sparse_bins=...) on a synthetic pair: the getNextNonEmptyBin chain beyond the dense window.

The fake multicall answers like an LBPair whose remaining liquidity sits in a few bins far from the active id
(the case that made test_routing_replay skip lfj_lb21-wavax-usdc-bs20 as insufficient_state after widening).
"""

from __future__ import annotations

import pytest

from v4quote.chain import loaders
from v4quote.chain.loaders import LB_MAX_ID, load_lb_pair
from v4quote.errors import InsufficientState

PAIR = "0x00000000000000000000000000000000000000Ab"
ACTIVE = 1_000
BINS = {998: (0, 7), 1_000: (5, 5), 1_001: (3, 0), 900: (0, 11), 850: (0, 13), 1_100: (17, 0)}


def _next(swap_for_y: bool, bin_id: int) -> int:
    ids = sorted(BINS)
    if swap_for_y:
        below = [i for i in ids if i < bin_id]
        return below[-1] if below else 0
    above = [i for i in ids if i > bin_id]
    return above[0] if above else LB_MAX_ID


def _fake_multicall(calls_log):
    def fake(w3, calls, block, allow_failure=False):
        calls_log.append(len(calls))
        out = []
        for c in calls:
            sig = c.signature
            if sig == "getTokenX()":
                out.append("0x" + "11" * 20)
            elif sig == "getTokenY()":
                out.append("0x" + "22" * 20)
            elif sig == "getBinStep()":
                out.append(20)
            elif sig == "getActiveId()":
                out.append(ACTIVE)
            elif sig == "getStaticFeeParameters()":
                out.append((8_000, 30, 600, 5_000, 40_000, 1_000, 350_000))
            elif sig == "getVariableFeeParameters()":
                out.append((0, 0, ACTIVE, 0))
            elif sig == "getBin(uint24)":
                out.append(BINS.get(c.args[0], (0, 0)))
            elif sig == "getNextNonEmptyBin(bool,uint24)":
                out.append(_next(*c.args))
            else:  # pragma: no cover
                raise AssertionError(sig)
        return out
    return fake


@pytest.fixture
def calls_log(monkeypatch):
    log: list[int] = []
    monkeypatch.setattr(loaders, "multicall", _fake_multicall(log))
    return log


def test_dense_window_only_is_not_exhausted(calls_log):
    s = load_lb_pair(None, PAIR, 1, version="2.1", bins_each_side=2)
    assert s.loaded_range == (998, 1_002)
    assert not s.exhausted_below and not s.exhausted_above
    assert s.next_non_empty_bin(True, 1_000) == 998
    with pytest.raises(InsufficientState):
        s.next_non_empty_bin(True, 998)


def test_follow_reaches_far_bins_and_sentinels(calls_log):
    s = load_lb_pair(None, PAIR, 1, version="2.1", bins_each_side=2, follow_sparse_bins=5)
    assert s.loaded_range == (850, 1_100)
    assert s.exhausted_below and s.exhausted_above
    assert {i: s.bins[i] for i in (900, 850, 1_100)} == {900: (0, 11), 850: (0, 13), 1_100: (17, 0)}
    assert s.next_non_empty_bin(True, 998) == 900
    assert s.next_non_empty_bin(True, 850) is None
    assert s.next_non_empty_bin(False, 1_001) == 1_100
    assert s.next_non_empty_bin(False, 1_100) is None
    # bins between followed ids are known to be empty, not unknown
    assert s.bin(950) == (0, 0)
    # 2 dense multicalls + 2 follow rounds below (900, 850) / 1 above (1100); the 3rd round sees both sentinels
    assert len(calls_log) == 4


def test_follow_budget_limits_steps(calls_log):
    s = load_lb_pair(None, PAIR, 1, version="2.1", bins_each_side=2, follow_sparse_bins=1)
    assert s.loaded_range == (900, 1_100)
    assert not s.exhausted_below and s.exhausted_above
    with pytest.raises(InsufficientState):
        s.next_non_empty_bin(True, 900)
