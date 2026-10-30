"""Offline checks of the chain-layer decoders."""

from eth_abi import encode as abi_encode

from v4quote.amm.uniswap_v4 import PoolKey
from v4quote.chain.loaders import V4_INITIALIZE_TOPIC, decode_v4_initialize
from v4quote.chain.multicall import Call

E18 = 10**18
RATE_6DEC = 10**30  # stored_rates for a 6-decimals coin (10**(36-6))


# ---------------------------------------------------------------- chain-layer decoders (offline)
def test_decode_v4_initialize_log():
    key = PoolKey(
        "0x0000000000000000000000000000000000000000", "0x754704Bc059F8C67012fEd69BC8A327a5aafb603", 500, 10
    )
    pid = key.pool_id()
    log = {
        "topics": [
            V4_INITIALIZE_TOPIC,
            pid,
            bytes(12) + bytes.fromhex(key.currency0[2:]),
            bytes(12) + bytes.fromhex(key.currency1[2:]),
        ],
        "data": abi_encode(
            ["uint24", "int24", "address", "uint160", "int24"], [500, 10, key.hooks, 2**96, 0]
        ),
    }
    got_pid, got_key = decode_v4_initialize(log)
    assert got_pid == pid and got_key.pool_id() == pid and got_key.fee == 500 and got_key.tick_spacing == 10


def test_call_encoding():
    c = Call("0x0000000000000000000000000000000000000001", "getSwapOut(uint128,bool)", (5, True), ("uint128",) * 3)
    data = c.calldata()
    assert data[:4].hex() == "e77366f8"  # keccak("getSwapOut(uint128,bool)")[:4]
    assert len(data) == 4 + 64
