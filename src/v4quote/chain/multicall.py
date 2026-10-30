"""Minimal ABI call helpers + Multicall3 batching, always pinned to an explicit block."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import function_signature_to_4byte_selector, to_checksum_address
from web3 import Web3
from web3.exceptions import ContractLogicError

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"  # same address on Monad and Avalanche (checked via eth_getCode)


class CallFailed(RuntimeError):
    """eth_call failed. ``reverted`` distinguishes a contract revert from a transport/RPC failure."""

    def __init__(self, msg: str, reverted: bool) -> None:
        super().__init__(msg)
        self.reverted = reverted


@dataclass(frozen=True)
class Call:
    """A single contract read: ``signature`` like 'getReserves()', ``out`` like ['uint112','uint112','uint32']."""

    target: str
    signature: str
    args: tuple = ()
    out: tuple[str, ...] = ()

    @property
    def in_types(self) -> list[str]:
        inner = self.signature[self.signature.index("(") + 1 : self.signature.rindex(")")]
        return _split_types(inner)

    def calldata(self) -> bytes:
        return function_signature_to_4byte_selector(self.signature) + abi_encode(self.in_types, list(self.args))

    def decode(self, data: bytes) -> Any:
        values = abi_decode(list(self.out), data)
        return values[0] if len(values) == 1 else values


def _split_types(s: str) -> list[str]:
    """Split a comma-separated ABI type list, respecting tuple parentheses."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    if cur:
        out.append(cur)
    return out


class DecodeFailed(RuntimeError):
    """The call succeeded but its return data does not match the expected ABI types (not retried)."""


def _is_revert(exc: Exception) -> bool:
    """A contract revert, decided by exception type only (ContractLogicError and its subclasses:
    ContractCustomError, ContractPanicError, OffchainLookup). web3.py maps JSON-RPC 'execution reverted'
    answers of eth_call to these types."""
    return isinstance(exc, ContractLogicError)


def _raw_call(w3: Web3, tx: dict[str, Any], call: Call, block: int, retries: int) -> bytes:
    for attempt in range(retries + 1):
        try:
            return bytes(w3.eth.call(tx, block_identifier=block))
        except Exception as exc:
            if _is_revert(exc):
                msg = f"{call.signature} on {call.target} reverted at block {block}: {exc}"
                reverted = True
            elif attempt == retries:
                # Transport errors are already URL-masked by MaskedHTTPProvider; still report only the
                # exception type here and never chain the original exception.
                msg = f"{call.signature} on {call.target} failed at block {block}: {type(exc).__name__}"
                reverted = False
            else:
                time.sleep(0.5 * 2**attempt)
                continue
        raise CallFailed(msg, reverted=reverted)
    raise AssertionError("unreachable")


def eth_call(w3: Web3, call: Call, block: int, *, sender: str | None = None, retries: int = 4) -> Any:
    """eth_call pinned to ``block``; retries transport errors (rate limits) with exponential backoff.

    Decoding happens once, outside the retry loop; a decoding problem raises DecodeFailed.
    """
    tx: dict[str, Any] = {"to": to_checksum_address(call.target), "data": "0x" + call.calldata().hex()}
    if sender:
        tx["from"] = to_checksum_address(sender)
    raw = _raw_call(w3, tx, call, block, retries)
    return _decode(call, raw, block)


def _decode(call: Call, data: bytes, block: int) -> Any:
    try:
        return call.decode(data)
    except Exception as exc:
        msg = f"cannot decode {call.signature} on {call.target} at block {block} as {call.out}: {type(exc).__name__}"
    raise DecodeFailed(msg)


_AGG3 = Call(MULTICALL3, "aggregate3((address,bool,bytes)[])", out=("(bool,bytes)[]",))


def multicall(
    w3: Web3,
    calls: Sequence[Call],
    block: int,
    *,
    allow_failure: bool | Sequence[bool] = False,
    chunk: int = 150,
    retries: int = 4,
) -> list[Any]:
    """Execute reads through Multicall3.aggregate3 at ``block``.

    With ``allow_failure`` a reverted call, or a call whose return data does not decode as ``out`` (e.g. a
    fallback function answering an unknown selector with empty data), yields None. ``allow_failure`` is one flag
    for every call or a sequence with one flag per call (optional reads inside an otherwise strict batch).
    ``retries``: transport-error retries of each aggregate3 eth_call (exponential backoff from 0.5 s); callers
    whose transport already retries and fails over (e.g. a failover router) pass 0.
    """
    if isinstance(allow_failure, bool):
        flags = [allow_failure] * len(calls)
    else:
        flags = [bool(f) for f in allow_failure]
        if len(flags) != len(calls):
            raise ValueError(f"allow_failure has {len(flags)} flags for {len(calls)} calls")
    results: list[Any] = []
    for i in range(0, len(calls), chunk):
        batch = calls[i : i + chunk]
        batch_flags = flags[i : i + chunk]
        payload = [(to_checksum_address(c.target), f, c.calldata()) for c, f in zip(batch, batch_flags)]
        agg = Call(MULTICALL3, _AGG3.signature, (payload,), _AGG3.out)
        raw = eth_call(w3, agg, block, retries=retries)
        for c, may_fail, (ok, data) in zip(batch, batch_flags, raw, strict=True):
            if not ok:
                if not may_fail:
                    raise CallFailed(f"multicall: {c.signature} on {c.target} failed", reverted=True)
                results.append(None)
            elif may_fail:
                try:
                    results.append(_decode(c, bytes(data), block))
                except DecodeFailed:
                    results.append(None)
            else:
                results.append(_decode(c, bytes(data), block))
    return results
