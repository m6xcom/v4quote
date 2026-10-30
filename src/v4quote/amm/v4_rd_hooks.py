"""Uniswap v4 hooks with a swap return delta, modelled exactly for exact-input swaps (``V4PoolState.rd_hook``).

A return-delta hook takes part of the swap in the PoolManager's flash accounting, so the trader's amounts differ from
the pool's own swap. Only hooks whose verified source does nothing in the swap path but take a fee of one leg are modelled,
each by its own kind; the arithmetic is the contract's, rounding included (Sourcify sources, chain 4663):

``output_bps`` - PonsV2MemeHook (``contracts/src/v2/hooks/PonsV2MemeHook.sol``, ``_afterSwap``): takes
    ``unspecified * hookFeeBps / 10000 + unspecified * creatorTaxBps / 10000`` (two floors) of the swap's unspecified
    currency, the output of an exact-input swap; the per-pool terms are ``launches(poolId)`` (owner-settable fees:
    the owner's hook-wide default is snapshotted at register; the caller re-reads the terms, never assumes them fixed).
    ``bps = (hookFeeBps, creatorTaxBps)``; an unregistered pool or both 0 = no fee (``bps = (0, 0)``).
``quote_bps`` - PangeaFeeHook (``src/venues/PangeaFeeHook.sol``): ``feeBps`` of the gross quote amount, always in the
    quote currency (``configs(poolId)``: ``quoteIsCurrency0``, ``feeBps``, ``edgeTick``). Exact-input buy (the quote is
    the input): ``beforeSwap`` takes ``amountIn * feeBps / 10000`` and the pool swaps the rest, which must fill in full
    (``PartialFill``). Exact-input sell (the quote is the output): ``afterSwap`` takes ``out * feeBps / 10000`` of the
    pool's output. Every swap reverts ``OutOfRange`` when the post-swap tick is on the empty side of the launch range
    (``quoteIsCurrency0 ? tick >= edgeTick : tick < edgeTick``). The ``devBuyer`` exemption (one fixed launch address)
    is not modelled (a quote for that address would differ).

The pool's own Swap event carries the pool amounts (the beforeSwap fee already deducted, the afterSwap fee not), so a
model check against Swap logs uses the pool amounts (``uniswap_v4.quote_exact_input(..., hook_deltas=False)``).
Exact-output swaps through these hooks are not modelled (``UnsupportedPool``).
"""
from __future__ import annotations

from dataclasses import dataclass

from v4quote.errors import MathRevert

BPS = 10_000
KIND_OUTPUT_BPS = "output_bps"
KIND_QUOTE_BPS = "quote_bps"
KINDS = (KIND_OUTPUT_BPS, KIND_QUOTE_BPS)

# Hooks.sol flags (same values as v4quote.amm.uniswap_v4; repeated here to avoid an import cycle)
_BEFORE_SWAP = 1 << 7
_BEFORE_SWAP_RETURNS_DELTA = 1 << 3
_AFTER_SWAP_RETURNS_DELTA = 1 << 2

# contract bounds: Pons MAX_TOTAL_TRADE_FEE_BPS 2000, Pangea MAX_FEE_BPS 500
_MAX_TOTAL = {KIND_OUTPUT_BPS: 2_000, KIND_QUOTE_BPS: 500}


@dataclass(frozen=True, slots=True)
class RdHook:
    """The per-pool terms of a modelled return-delta hook (see the module doc for the kinds)."""

    kind: str
    bps: tuple[int, ...]  # output_bps: (hookFeeBps, creatorTaxBps); quote_bps: (feeBps,)
    quote_is_currency0: bool = False  # quote_bps only
    edge_tick: int | None = None  # quote_bps only (required there)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"rd hook: unknown kind {self.kind!r}")
        n = 2 if self.kind == KIND_OUTPUT_BPS else 1
        if not isinstance(self.bps, tuple) or len(self.bps) != n or not all(
                isinstance(b, int) and not isinstance(b, bool) and 0 <= b for b in self.bps):
            raise ValueError(f"rd hook {self.kind}: bps must be {n} non-negative ints, got {self.bps!r}")
        if sum(self.bps) > _MAX_TOTAL[self.kind]:
            raise ValueError(f"rd hook {self.kind}: total {sum(self.bps)} bps above the contract bound")
        if self.kind == KIND_QUOTE_BPS and not isinstance(self.edge_tick, int):
            raise ValueError("rd hook quote_bps: edge_tick required")

    # ------------------------------------------------------------------ flags
    def covers(self, hook_flags: int) -> bool:
        """The hook address's swap flags are exactly what this kind models (a return delta it does not know -> no)."""
        f = hook_flags & (_BEFORE_SWAP | _BEFORE_SWAP_RETURNS_DELTA | _AFTER_SWAP_RETURNS_DELTA)
        if self.kind == KIND_OUTPUT_BPS:  # afterSwap delta only, no beforeSwap
            return f == _AFTER_SWAP_RETURNS_DELTA
        return f == (_BEFORE_SWAP | _BEFORE_SWAP_RETURNS_DELTA | _AFTER_SWAP_RETURNS_DELTA)

    @property
    def total_bps(self) -> int:
        return sum(self.bps)

    # ------------------------------------------------------------------ exact input
    def quote_is_input(self, zero_for_one: bool) -> bool:
        return self.kind == KIND_QUOTE_BPS and zero_for_one == self.quote_is_currency0

    def input_fee(self, zero_for_one: bool, amount_in: int) -> int:
        """Taken by beforeSwap from an exact input (the pool swaps ``amount_in - fee``)."""
        if self.quote_is_input(zero_for_one):
            return amount_in * self.bps[0] // BPS
        return 0

    def output_fee(self, zero_for_one: bool, pool_out: int) -> int:
        """Taken by afterSwap from the pool's output of an exact-input swap."""
        if pool_out <= 0:
            return 0
        if self.kind == KIND_OUTPUT_BPS:
            return pool_out * self.bps[0] // BPS + pool_out * self.bps[1] // BPS
        if not self.quote_is_input(zero_for_one):
            return pool_out * self.bps[0] // BPS
        return 0

    def check_post_tick(self, tick: int | None) -> None:
        """PangeaFeeHook.afterSwap OutOfRange (the post-swap tick on the empty side of the launch range)."""
        if self.kind != KIND_QUOTE_BPS:
            return
        if tick is None:
            raise MathRevert("rd hook: post-swap tick needed for the range check")
        if (tick >= self.edge_tick) if self.quote_is_currency0 else (tick < self.edge_tick):
            raise MathRevert(f"OutOfRange({tick}, {self.edge_tick})")

    @property
    def needs_tick(self) -> bool:
        return self.kind == KIND_QUOTE_BPS

    def as_dict(self) -> dict:
        d: dict = {"kind": self.kind, "bps": list(self.bps)}
        if self.kind == KIND_QUOTE_BPS:
            d.update(quote_is_currency0=self.quote_is_currency0, edge_tick=self.edge_tick)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RdHook":
        if not isinstance(d, dict) or set(d) - {"kind", "bps", "quote_is_currency0", "edge_tick"}:
            raise ValueError(f"rd_hook: unexpected value {d!r}")
        q = d.get("quote_is_currency0", False)
        if not isinstance(q, bool):
            raise ValueError("rd_hook.quote_is_currency0: expected a bool")
        return cls(d["kind"], tuple(d["bps"]), q, d.get("edge_tick"))


__all__ = ["BPS", "KINDS", "KIND_OUTPUT_BPS", "KIND_QUOTE_BPS", "RdHook"]
