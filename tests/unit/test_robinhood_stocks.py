"""Robinhood Chain stock-token pools: replay of reference quotes recorded on chain, plus the fee-only hook resolver.

The fixture (``tests/fixtures/robinhood_stocks.json``, recorded with ``tests/onchain/rh_stocks_record.py`` at block
103531492) holds 25 USDG pools of 10 stock tokens: Uniswap v3 (QuoterV2), Ramses V3 and Slipstream CL forks (the
pool's own ``swap()`` run by eth_call), Uniswap v4 without hooks and v4 dynamic-fee pools with reviewed fee-only hooks
(V4Quoter; the state's ``hook_fee`` is the hook's ``beforeSwap`` fee at that block). For every pool: exact input and
exact output, both directions, from dust to swaps crossing up to ~200 initialised ticks. Every quote must equal the
reference to the wei (or revert alike); the model must not refuse any of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import keccak
from web3.exceptions import ContractLogicError

from v4quote.amm import serialization as ser
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.v3 import pool as v3
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import get_sqrt_ratio_at_tick
from v4quote.chain import hook_fees as hf
from v4quote.errors import CoreError, MathRevert, UnsupportedPool

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "robinhood_stocks.json"
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
KINDS = {"uniswap_v3", "ramses_v3", "slipstream", "v4", "v4_fee_hook"}


def model_quote(state, zero_for_one: bool, exact_in: bool, amount: int) -> tuple[int, int]:
    if isinstance(state, v4.V4PoolState):
        r = (v4.quote_exact_input if exact_in else v4.quote_exact_output)(state, zero_for_one, amount)
    else:
        r = (v3.quote_exact_input if exact_in else v3.quote_exact_output)(state, zero_for_one, amount)
    return r.amount_in, r.amount_out


def replay(pool: dict) -> list[int]:
    """[equal to the wei, revert alike, refused by the model, different]."""
    state = ser.from_dict(pool["state"])
    c = [0, 0, 0, 0]
    for zfo, exact_in, amount, q_in, q_out, err in pool["quotes"]:
        try:
            ours = model_quote(state, zfo, exact_in, int(amount))
        except MathRevert:
            c[1 if err is not None else 3] += 1
            continue
        except CoreError:  # UnsupportedPool / InsufficientState: refused, never a number
            c[2] += 1
            continue
        c[0 if err is None and ours == (int(q_in), int(q_out)) else 3] += 1
    return c


@pytest.mark.parametrize("pool", DATA["pools"], ids=[p["name"] for p in DATA["pools"]])
def test_replay_reference_to_the_wei(pool):
    c = replay(pool)
    assert c[3] == 0, c  # never a different number
    assert c[2] == 0, c  # every recorded quote is inside the loaded state: nothing refused
    assert c[0] + c[1] == len(pool["quotes"]) >= 20
    modes = {(zfo, ei) for zfo, ei, *_ in pool["quotes"]}
    assert modes == {(True, True), (True, False), (False, True), (False, False)}


def test_replay_totals(capsys):
    tot = [0, 0, 0, 0]
    by_kind: dict[str, list[int]] = {}
    for pool in DATA["pools"]:
        c = replay(pool)
        tot = [a + b for a, b in zip(tot, c)]
        k = by_kind.setdefault(pool["kind"], [0, 0])
        k[0] += 1
        k[1] += sum(c)
    with capsys.disabled():
        print(f"\nRobinhood stocks replay (block {DATA['block']}): {len(DATA['pools'])} pools, {sum(tot)} quotes, "
              f"{tot[0]} equal to the wei, {tot[1]} revert alike, {tot[2]} refused, {tot[3]} different; "
              f"by kind {by_kind}")
    assert tot[3] == 0 and tot[2] == 0
    assert len(DATA["pools"]) >= 12 and tot[0] >= 600
    assert set(by_kind) == KINDS


def test_fixture_covers_tickers_and_hooks():
    tickers = {p["ticker"] for p in DATA["pools"]}
    assert {"SPCX", "NVDA", "SPY", "CRCL", "META", "GLD", "MSTR", "TSLA", "GOOGL", "AAPL"} <= tickers
    hooked = [p for p in DATA["pools"] if p["kind"] == "v4_fee_hook"]
    assert {p["ticker"] for p in hooked} >= {"SPCX", "GLD"}
    for p in hooked:
        st = ser.from_dict(p["state"])
        assert st.key.is_dynamic_fee and st.hook_fee is not None
        assert hf.reviewed_hook(DATA["chain_id"], p["hooks"]) is not None
        assert "0x" + st.key.pool_id().hex() == p["pool"]
        without = ser.from_dict({**p["state"], "hook_fee": None})
        with pytest.raises(UnsupportedPool):  # the dynamic fee is never guessed
            v4.quote_exact_input(without, True, 10**6)


def test_ui_multiplier_is_display_only():
    """Amounts are raw token units; uiMultiplier (ERC-8056, 1e18 = 1.0) is recorded for display only."""
    seen = 0
    for p in DATA["pools"]:
        for addr, sym, dec, ui in p["tokens"]:
            assert dec in (6, 18)
            if sym == "USDG":
                assert ui is None and dec == 6
            else:
                assert ui is not None and int(ui) >= 10**18
                seen += int(ui) > 10**18
    assert seen > 0  # some stock tokens accrue dividends into the multiplier


# --------------------------------------------------------------------------- hook fee resolver (no network)
HOOK = "0xC15b155a3094934171A15625d1a80Aba267B6080"  # SessionFeeHook, flags 0x2080
PM = "0x8366a39CC670B4001A1121B8F6A443A643e40951"
SENDER = "0x8DC178EFB8111bB0973Dd9D722EBEFF267c98F94"
USDG = "0x5fC5360D0400a0fD4F2Af552Add042D716F1d168"
SPCX = "0x4A0E65a3eccec6dbe60ae065F2e7Bb85fAe35eea"


class _Eth:
    def __init__(self, code: bytes, ret, chain_id: int = 4663) -> None:
        self.code, self.ret, self.chain_id, self.calls = code, ret, chain_id, []

    def get_code(self, addr, block_identifier):
        return self.code

    def call(self, tx, block_identifier):
        data = bytes.fromhex(tx["data"][2:])
        assert data[:4] == hf.BEFORE_SWAP_SELECTOR and tx["from"].lower() == PM.lower()
        sender, key, params, _ = abi_decode(["address", "(address,address,uint24,int24,address)",
                                             "(bool,int256,uint160)", "bytes"], data[4:])
        self.calls.append((sender, params[0]))
        if self.ret is None:
            raise ContractLogicError("execution reverted", data="0x")
        sel, delta, fee = self.ret(params[0])
        return abi_encode(["bytes4", "int256", "uint24"], [sel, delta, fee])


class _W3:
    def __init__(self, *a, **k) -> None:
        self.eth = _Eth(*a, **k)


def _state(hooks: str = HOOK, fee: int = v4.DYNAMIC_FEE_FLAG) -> v4.V4PoolState:
    key = v4.PoolKey(SPCX, USDG, fee, 60, hooks)
    return v4.V4PoolState(key, get_sqrt_ratio_at_tick(0), 0, 0, 0, 10**20, TickBitmapSnapshot(60, {}, {}, (-5, 5)), 7)


def test_with_hook_fee_reads_before_swap_per_direction():
    code = b"\x60\x00"
    pinned = hf.FEE_ONLY_HOOKS[(4663, HOOK.lower())]
    hf.FEE_ONLY_HOOKS[(4663, HOOK.lower())] = hf.FeeOnlyHook(pinned.name, "0x" + keccak(code).hex(), "", True, "")
    try:
        w3 = _W3(code, lambda zfo: (hf.BEFORE_SWAP_SELECTOR, 0, (450 if zfo else 700) | hf.OVERRIDE_FEE_FLAG))
        st = hf.with_hook_fee(w3, PM, _state(), sender=SENDER)
        assert st.hook_fee == (450, 700)
        assert [s.lower() for s, _ in w3.eth.calls] == [SENDER.lower()] * 2  # the swap's sender, both directions
        # no override flag: the pool keeps its slot0 lpFee
        st = hf.with_hook_fee(_W3(code, lambda zfo: (hf.BEFORE_SWAP_SELECTOR, 0, 450)), PM, _state(), sender=SENDER)
        assert st.hook_fee == (0, 0)
        with pytest.raises(UnsupportedPool):  # a delta is not a fee-only hook
            hf.with_hook_fee(_W3(code, lambda zfo: (hf.BEFORE_SWAP_SELECTOR, 5, 450 | hf.OVERRIDE_FEE_FLAG)), PM,
                             _state(), sender=SENDER)
        with pytest.raises(MathRevert):  # paused pool: the swap reverts, so does the resolver
            hf.with_hook_fee(_W3(code, None), PM, _state(), sender=SENDER)
        with pytest.raises(UnsupportedPool):  # code differs from the reviewed build
            hf.with_hook_fee(_W3(b"\x60\x01", lambda zfo: None), PM, _state(), sender=SENDER)
    finally:
        hf.FEE_ONLY_HOOKS[(4663, HOOK.lower())] = pinned


def test_with_hook_fee_refuses_unreviewed_hooks_and_return_deltas():
    unknown = "0x0000000000000000000000000000000000002080"
    with pytest.raises(UnsupportedPool):
        hf.with_hook_fee(_W3(b"", None), PM, _state(unknown), sender=SENDER)
    with pytest.raises(UnsupportedPool):  # same reviewed address on another chain
        hf.with_hook_fee(_W3(b"", None, chain_id=1), PM, _state(), sender=SENDER)
    with pytest.raises(UnsupportedPool):
        hf.check_fee_only_key(v4.PoolKey(SPCX, USDG, v4.DYNAMIC_FEE_FLAG, 60, "0x00000000000000000000000000000000000020C8"))
    with pytest.raises(UnsupportedPool):
        hf.check_fee_only_key(v4.PoolKey(SPCX, USDG, v4.DYNAMIC_FEE_FLAG, 60, "0x0000000000000000000000000000000000002000"))
