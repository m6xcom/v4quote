"""Block / receipt ``logsBloom`` (Yellow Paper M3:2048): what a block header proves about the block's logs.

Every log adds its emitting address and each of its topics: for an item ``x`` with ``h = keccak256(x)`` the three bits
``((h[i] << 8) | h[i + 1]) & 2047`` for i = 0, 2, 4 are set, bit ``k`` counted from the least significant end of the
256-byte big-endian bloom (go-ethereum ``types.Bloom.Add``). Consequences:

* an all-zero bloom proves the block has NO logs (no false negatives);
* ``contains(bloom, x) is False`` proves no log of the block has address / topic ``x``;
* ``contains`` True only means "maybe" (false positives, more likely in blocks with many logs).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from eth_utils import keccak

BLOOM_BITS = 2048
BLOOM_BYTES = BLOOM_BITS // 8


def _bits(item: bytes) -> tuple[int, int, int]:
    h = keccak(item)
    return tuple(((h[i] << 8) | h[i + 1]) & (BLOOM_BITS - 1) for i in (0, 2, 4))  # type: ignore[return-value]


def _bytes(x: str | bytes) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    s = x[2:] if x[:2] in ("0x", "0X") else x
    return bytes.fromhex(s)


def parse_bloom(value: str | bytes | int | None) -> int | None:
    """Header field -> int (None when absent or malformed: the caller must not assume anything)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if 0 <= value < (1 << BLOOM_BITS) else None
    try:
        b = _bytes(value)
    except ValueError:
        return None
    return int.from_bytes(b, "big") if len(b) == BLOOM_BYTES else None


def add(bloom: int, item: str | bytes) -> int:
    for k in _bits(_bytes(item)):
        bloom |= 1 << k
    return bloom


def contains(bloom: int, item: str | bytes) -> bool:
    """False = the item is certainly not in the bloom; True = it may be."""
    return all((bloom >> k) & 1 for k in _bits(_bytes(item)))


def item_mask(item: str | bytes) -> int:
    """The three bloom bits of ``item`` as one int mask: ``contains_mask(bloom, item_mask(x)) == contains(bloom, x)``.
    For hot checks of a fixed set of items (precompute the masks once, then one AND per check)."""
    m = 0
    for k in _bits(_bytes(item)):
        m |= 1 << k
    return m


def contains_mask(bloom: int, mask: int) -> bool:
    """``contains`` with a precomputed ``item_mask``: False = certainly not in the bloom, True = maybe."""
    return bloom & mask == mask


def address_masks(address: str) -> tuple[int, int]:
    """Masks of an address as a log's emitter (20 bytes) and as an indexed topic (left-padded to 32 bytes, e.g. the
    ``from`` / ``to`` of an ERC-20 ``Transfer``). A block whose bloom contains neither has no log emitted by the
    address and no log with the address as a topic."""
    raw = _bytes(address)
    if len(raw) != 20:
        raise ValueError(f"not a 20-byte address: {address!r}")
    return item_mask(raw), item_mask(bytes(12) + raw)


def logs_bloom(logs: Iterable[Mapping[str, Any]]) -> int:
    """Bloom of JSON-RPC log objects (address + topics), as the node builds the header field."""
    b = 0
    for lg in logs:
        b = add(b, lg["address"])
        for t in lg.get("topics") or ():
            b = add(b, t)
    return b


def to_hex(bloom: int) -> str:
    return "0x" + bloom.to_bytes(BLOOM_BYTES, "big").hex()


__all__ = ["BLOOM_BITS", "BLOOM_BYTES", "add", "address_masks", "contains", "contains_mask", "item_mask", "logs_bloom",
           "parse_bloom", "to_hex"]
