"""v4quote.chain.quoters swap simulation helpers (pools without a quoter, e.g. Slipstream CL clones on Arc): request
shape and decoding, offline. Parity with the pools themselves has to be checked on chain."""

from __future__ import annotations

import pytest
from eth_abi import decode, encode
from eth_utils import function_signature_to_4byte_selector, to_checksum_address
from web3.exceptions import ContractLogicError

from v4quote.chain import quoters as qt
from v4quote.chain.multicall import CallFailed

POOL = to_checksum_address("0x" + "12" * 20)
TOKEN = to_checksum_address("0x" + "34" * 20)


class _Eth:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.seen = []

    def call(self, tx, block, overrides=None):
        self.seen.append((tx, block, overrides))
        return self.behaviour(tx, block, overrides)


class _W3:
    def __init__(self, behaviour):
        self.eth = _Eth(behaviour)


def _revert_with(data: bytes):
    def f(tx, block, overrides):
        raise ContractLogicError("execution reverted", data="0x" + data.hex())
    return f


def test_probe_code_is_calldatacopy_revert():
    # PUSH1 0x40 PUSH1 0x04 PUSH1 0x00 CALLDATACOPY PUSH1 0x40 PUSH1 0x00 REVERT
    assert qt.SWAP_PROBE_CODE == "0x" + "6040" "6004" "6000" "37" "6040" "6000" "fd"
    assert qt.MAX_SQRT_RATIO == 1461446703485210103287273052203988822378723970342


@pytest.mark.parametrize("zfo", [True, False])
def test_v3_swap_sim_decodes_callback_deltas(zfo):
    deltas = (1000, -834) if zfo else (-834, 1000)
    w3 = _W3(_revert_with(encode(["int256", "int256"], list(deltas))))
    assert qt.v3_pool_swap_sim(w3, POOL, zfo, 1000, 77) == (1000, 834)
    tx, block, ov = w3.eth.seen[0]
    assert block == 77 and tx["to"] == POOL and tx["from"] == to_checksum_address(qt.SWAP_PROBE)
    assert ov == {to_checksum_address(qt.SWAP_PROBE): {"code": qt.SWAP_PROBE_CODE}}
    data = bytes.fromhex(tx["data"][2:])
    assert data[:4] == function_signature_to_4byte_selector("swap(address,bool,int256,uint160,bytes)")
    recipient, z, amount, limit, _ = decode(["address", "bool", "int256", "uint160", "bytes"], data[4:])
    assert (z, amount) == (zfo, 1000)
    assert limit == (qt.MIN_SQRT_RATIO + 1 if zfo else qt.MAX_SQRT_RATIO - 1)


def test_v3_swap_sim_partial_fill_reported():
    w3 = _W3(_revert_with(encode(["int256", "int256"], [600, -500])))
    assert qt.v3_pool_swap_sim(w3, POOL, True, 1000, 1) == (600, 500)


def test_v3_swap_sim_other_revert_raises():
    w3 = _W3(_revert_with(bytes.fromhex("08c379a0") + encode(["string"], ["SPL"])))
    with pytest.raises(CallFailed) as e:
        qt.v3_pool_swap_sim(w3, POOL, True, 1000, 1)
    assert e.value.reverted


def test_v3_swap_sim_transport_error_is_not_a_revert():
    def boom(tx, block, overrides):
        raise ConnectionError("down")
    with pytest.raises(CallFailed) as e:
        qt.v3_pool_swap_sim(_W3(boom), POOL, True, 1000, 1)
    assert not e.value.reverted


def test_v2_swap_check_out_and_out_plus_one():
    max_out = 500

    def k_check(tx, block, overrides):
        o0, o1, to, _ = decode(["uint256", "uint256", "address", "bytes"], bytes.fromhex(tx["data"][10:]))
        assert to.lower() == qt.V2_SINK.lower() and overrides == {"x": 1}
        if max(o0, o1) > max_out:
            raise ContractLogicError("execution reverted: UniswapV2: K")
        return b""

    w3 = _W3(k_check)
    assert qt.v2_pair_swap_check(w3, POOL, TOKEN, True, 10, 500, 5, balance_override={"x": 1}) == (True, False)
    assert qt.v2_pair_swap_check(w3, POOL, TOKEN, False, 10, 499, 5, balance_override={"x": 1}) == (True, True)
    assert qt.v2_pair_swap_check(w3, POOL, TOKEN, True, 10, 501, 5, balance_override={"x": 1}) == (False, False)
