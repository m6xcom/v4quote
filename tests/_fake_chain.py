"""Test-only in-memory chain: answers Multicall3.aggregate3 eth_calls and eth_getStorageAt (no network).

Handlers are registered per (target, signature) and receive the ABI-decoded arguments; they return
``(out_types, values)`` which is ABI-encoded like the real contract would, ``None`` for a revert, or raw
``bytes`` (e.g. b"" for a fallback that answers an unknown selector with empty data). The real
``v4quote.chain.multicall`` encoding/decoding path is therefore exercised end to end.
"""

from __future__ import annotations

from typing import Any, Callable

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import function_signature_to_4byte_selector

from v4quote.chain.multicall import MULTICALL3, _split_types

AGG3_SIG = "aggregate3((address,bool,bytes)[])"
_AGG3_SEL = function_signature_to_4byte_selector(AGG3_SIG)


class FakeEth:
    def __init__(self, block: int, timestamp: int = 0) -> None:
        self.block = block
        self.timestamp = timestamp  # header timestamp returned by get_block(block)
        self.handlers: dict[tuple[str, bytes], tuple[list[str], Callable[..., Any], str]] = {}
        self.storage: dict[tuple[str, int], int] = {}
        self.code: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, tuple]] = []  # (target, signature, args) in call order
        self.allow: list[bool] = []  # aggregate3 allowFailure flag of each call in self.calls
        self.batches: list[list[str]] = []  # signatures of each aggregate3 batch, in order

    # -- registration
    def on(self, target: str, signature: str, fn: Callable[..., Any]) -> None:
        inner = signature[signature.index("(") + 1 : signature.rindex(")")]
        key = (target.lower(), function_signature_to_4byte_selector(signature))
        self.handlers[key] = (_split_types(inner), fn, signature)

    def returns(self, target: str, signature: str, out: tuple[str, ...], *values: Any) -> None:
        self.on(target, signature, lambda *_: (out, values))

    def set_slot(self, target: str, slot: int, value: int) -> None:
        self.storage[(target.lower(), slot)] = value

    # -- web3 surface
    def call(self, tx: dict, block_identifier: int) -> bytes:
        assert block_identifier == self.block, f"call not pinned to block {self.block}"
        data = bytes.fromhex(tx["data"][2:])
        assert tx["to"].lower() == MULTICALL3.lower() and data[:4] == _AGG3_SEL
        (payload,) = abi_decode(["(address,bool,bytes)[]"], data[4:])
        results = []
        self.batches.append([])
        for target, allow, cd in payload:
            in_types, fn, sig = self.handlers[(target.lower(), bytes(cd[:4]))]
            args = tuple(abi_decode(in_types, bytes(cd[4:]))) if in_types else ()
            self.calls.append((target, sig, args))
            self.allow.append(bool(allow))
            self.batches[-1].append(sig)
            res = fn(*args)
            if res is None:
                results.append((False, b""))
            elif isinstance(res, (bytes, bytearray)):
                results.append((True, bytes(res)))
            else:
                out, values = res
                results.append((True, abi_encode(list(out), list(values))))
        return abi_encode(["(bool,bytes)[]"], [results])

    def get_block(self, block_identifier: int) -> dict:
        assert block_identifier == self.block
        return {"number": self.block, "timestamp": self.timestamp}

    def get_storage_at(self, addr: str, slot: int, block_identifier: int) -> bytes:
        assert block_identifier == self.block
        return self.storage.get((addr.lower(), slot), 0).to_bytes(32, "big")

    def get_code(self, addr: str, block_identifier: int) -> bytes:
        assert block_identifier == self.block
        return self.code.get(addr.lower(), b"")


class FakeW3:
    def __init__(self, block: int, timestamp: int = 0) -> None:
        self.eth = FakeEth(block, timestamp)


class FakeScanner:
    """LogScanner stand-in: returns the given logs and records the filter it was called with."""

    def __init__(self, logs) -> None:
        self.logs = list(logs)
        self.requests: list[tuple] = []

    def iter_logs(self, address, topics, from_block, to_block):
        self.requests.append((address, topics, from_block, to_block))
        yield from self.logs


def topic_address(addr: str) -> bytes:
    return bytes(12) + bytes.fromhex(addr[2:])
