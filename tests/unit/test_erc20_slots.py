"""v4quote.chain.erc20: storage-key helpers for read-only state overrides."""
from __future__ import annotations

from eth_abi import encode
from eth_utils import keccak, to_checksum_address

from v4quote.chain import erc20

HOLDER = "0x" + "ab" * 20


def test_mapping_slot_layouts():
    sol = erc20.mapping_slot(HOLDER, 3)
    assert sol == "0x" + keccak(encode(["address", "uint256"], [to_checksum_address(HOLDER), 3])).hex()
    vy = erc20.mapping_slot(HOLDER, 3, erc20.VYPER)
    assert vy == "0x" + keccak(encode(["uint256", "address"], [3, to_checksum_address(HOLDER)])).hex()
    assert sol != vy


def test_find_balance_slot_with_fake_token():
    token = "0x" + "cd" * 20
    want = erc20.mapping_slot(erc20.PROBE_HOLDER, 7, erc20.VYPER)

    def balance_of(holder, ov):
        diff = ov[to_checksum_address(token)]["stateDiff"]
        v = diff.get(want)
        return int(v, 16) if v else 0

    assert erc20.find_balance_slot(balance_of, token) == (7, erc20.VYPER)
    assert erc20.find_balance_slot(lambda h, o: 0, token, max_slot=3) is None


def test_overrides_merge():
    a = erc20.balance_override("0x" + "01" * 20, HOLDER, 5, 0)
    b = erc20.balance_override("0x" + "01" * 20, "0x" + "02" * 20, 6, 0)
    m = erc20.merge_overrides(a, b)
    (only,) = m.values()
    assert len(only["stateDiff"]) == 2
