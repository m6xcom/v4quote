"""ERC-20 storage helpers for read-only simulations (eth_call / eth_simulateV1 with a state override).

``find_balance_slot`` locates the ``balanceOf`` mapping of a token by overriding candidate storage slots of a probe
address and reading ``balanceOf(probe)`` back (Solidity layout ``keccak(holder . slot)`` and Vyper layout
``keccak(slot . holder)``, slots 0..``max_slot``). Tokens whose balance is not a plain mapping (reflection / rebasing
tokens) are not found -> None, and callers must treat the token as not simulable (never guess).

``balance_override`` / ``allowance_override`` build the ``stateDiff`` entries. Nothing here signs or sends: an
``eth_call`` with a state override never commits state.
"""

from __future__ import annotations

from typing import Any, Callable

from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address

PROBE_HOLDER = "0x00000000000000000000000000000000c0ffee02"  # no code, no balance on any chain
SOLIDITY, VYPER = "solidity", "vyper"


def mapping_slot(key: str, slot: int, layout: str = SOLIDITY) -> str:
    """Storage key of ``mapping[key]`` declared at ``slot``."""
    k = to_checksum_address(key)
    data = abi_encode(["address", "uint256"], [k, slot]) if layout == SOLIDITY else abi_encode(["uint256", "address"], [slot, k])
    return "0x" + keccak(data).hex()


def nested_mapping_slot(key1: str, key2: str, slot: int) -> str:
    """Storage key of ``mapping[key1][key2]`` (Solidity layout), e.g. ``allowance[owner][spender]``."""
    inner = keccak(abi_encode(["address", "uint256"], [to_checksum_address(key1), slot]))
    return "0x" + keccak(abi_encode(["address"], [to_checksum_address(key2)]) + inner).hex()


def word(value: int) -> str:
    return "0x" + value.to_bytes(32, "big").hex()


def balance_override(token: str, holder: str, value: int, slot: int, layout: str = SOLIDITY) -> dict[str, Any]:
    return {to_checksum_address(token): {"stateDiff": {mapping_slot(holder, slot, layout): word(value)}}}


def merge_overrides(*parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in parts:
        for addr, ov in p.items():
            cur = out.setdefault(addr, {})
            for k, v in ov.items():
                if k == "stateDiff":
                    cur.setdefault("stateDiff", {}).update(v)
                else:
                    cur[k] = v
    return out


def find_balance_slot(balance_of: Callable[[str, dict[str, Any]], int | None], token: str, *, max_slot: int = 40,
                      probe: str = PROBE_HOLDER) -> tuple[int, str] | None:
    """``(slot, layout)`` of the token's balance mapping, or None. ``balance_of(probe, overrides)`` must return
    ``token.balanceOf(probe)`` under the state override (an eth_call), None on a call failure."""
    marker = 0x5EED_0000_0000_0000_0000_0000_0000_0001
    for slot in range(max_slot + 1):
        for layout in (SOLIDITY, VYPER):
            v = balance_of(probe, balance_override(token, probe, marker, slot, layout))
            if v == marker:
                return slot, layout
    return None


__all__ = ["PROBE_HOLDER", "SOLIDITY", "VYPER", "balance_override", "find_balance_slot", "mapping_slot",
           "merge_overrides", "nested_mapping_slot", "word"]
