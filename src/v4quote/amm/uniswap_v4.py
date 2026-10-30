"""Uniswap v4 concentrated-liquidity swaps (PoolManager singleton state).

The price/liquidity math is identical to v3 (shared TickMath / SqrtPriceMath),
with v4's own SwapMath.computeSwapStep, sign convention and fee handling
(LP fee from slot0 combined with the per-direction protocol fee).

Hooks that can change the outcome of a swap are NOT modelled. Such pools are
rejected with ``UnsupportedPool`` rather than priced incorrectly:

* BEFORE_SWAP_RETURNS_DELTA / AFTER_SWAP_RETURNS_DELTA: the hook takes or
  gives tokens, so the user-visible amounts differ from the pool math.
* Dynamic LP fee (fee == 0x800000): the fee may be overridden per swap by
  beforeSwap or changed at any time by the hook.
* BEFORE_SWAP (without delta): the hook runs arbitrary code before the
  swap (it may modify liquidity or the fee); rejected by default, can be
  allowed explicitly with ``allow_before_swap_hook=True`` when the hook is
  known to be passive.

AFTER_SWAP without a return delta cannot alter the current swap and is allowed.

Fee-only hooks (``V4PoolState.hook_fee``): a hook that changes nothing but the LP fee of a swap (a
beforeSwap ``lpFeeOverride`` on a dynamic-fee pool, or a dynamic slot0 lpFee) is modelled when the caller
resolves the fee the hook will apply and sets ``hook_fee = (fee zeroForOne, fee oneForZero)``. Such a state
skips the dynamic-fee and BEFORE_SWAP rejections (the return-delta rejections stay): the caller vouches that
the hook has no other effect on the swap (check it against the pool's Swap logs / V4Quoter).

Return-delta hooks (``V4PoolState.rd_hook``, ``v4quote.amm.v4_rd_hooks``): a hook whose verified source only takes a fee
of one leg is modelled exactly when the caller sets its per-pool terms (``RdHook``) and the hook address's swap flags
are the ones that kind covers. ``swap`` stays the pool's own swap (what the Swap event logs); ``quote_exact_input`` is the
trader's amounts like V4Quoter (``hook_deltas=False``: the pool swap alone, for log checks). Exact output is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address

from v4quote.errors import MathRevert, UnsupportedPool
from v4quote.amm.v3.pool import SwapResult, default_price_limit, swap_loop
from v4quote.amm.v3.swap_math import MAX_SWAP_FEE
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import MAX_SQRT_PRICE, MIN_SQRT_PRICE
from v4quote.amm.v4_rd_hooks import RdHook

# Hooks.sol permission flags (lowest 14 bits of the hook address)
BEFORE_INITIALIZE_FLAG = 1 << 13
AFTER_INITIALIZE_FLAG = 1 << 12
BEFORE_ADD_LIQUIDITY_FLAG = 1 << 11
AFTER_ADD_LIQUIDITY_FLAG = 1 << 10
BEFORE_REMOVE_LIQUIDITY_FLAG = 1 << 9
AFTER_REMOVE_LIQUIDITY_FLAG = 1 << 8
BEFORE_SWAP_FLAG = 1 << 7
AFTER_SWAP_FLAG = 1 << 6
BEFORE_DONATE_FLAG = 1 << 5
AFTER_DONATE_FLAG = 1 << 4
BEFORE_SWAP_RETURNS_DELTA_FLAG = 1 << 3
AFTER_SWAP_RETURNS_DELTA_FLAG = 1 << 2
AFTER_ADD_LIQUIDITY_RETURNS_DELTA_FLAG = 1 << 1
AFTER_REMOVE_LIQUIDITY_RETURNS_DELTA_FLAG = 1 << 0
ALL_HOOK_MASK = (1 << 14) - 1

# LPFeeLibrary
DYNAMIC_FEE_FLAG = 0x800000
MAX_LP_FEE = 1_000_000

# ProtocolFeeLibrary
PIPS_DENOMINATOR = 1_000_000

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class PoolKey:
    currency0: str
    currency1: str
    fee: int
    tick_spacing: int
    hooks: str = ZERO_ADDRESS

    def pool_id(self) -> bytes:
        """PoolIdLibrary.toId: keccak256(abi.encode(key))."""
        return keccak(
            abi_encode(
                ["address", "address", "uint24", "int24", "address"],
                [
                    to_checksum_address(self.currency0),
                    to_checksum_address(self.currency1),
                    self.fee,
                    self.tick_spacing,
                    to_checksum_address(self.hooks),
                ],
            )
        )

    @property
    def hook_flags(self) -> int:
        return int(self.hooks, 16) & ALL_HOOK_MASK

    @property
    def is_dynamic_fee(self) -> bool:
        return self.fee == DYNAMIC_FEE_FLAG

    def as_tuple(self) -> tuple[str, str, int, int, str]:
        return (
            to_checksum_address(self.currency0),
            to_checksum_address(self.currency1),
            self.fee,
            self.tick_spacing,
            to_checksum_address(self.hooks),
        )


def unsupported_reasons(
    key: PoolKey, *, allow_before_swap_hook: bool = False, hook_fee_resolved: bool = False,
    rd_hook: RdHook | None = None,
) -> list[str]:
    """Reasons why a pool's swaps cannot be modelled exactly (empty list = supported).

    ``hook_fee_resolved``: the state carries the fee the hook applies (``V4PoolState.hook_fee``), so a dynamic
    fee and a BEFORE_SWAP hook are accepted; return-delta hooks are rejected unless ``rd_hook`` (the state's
    ``V4PoolState.rd_hook``) models exactly the hook address's swap flags (then its BEFORE_SWAP is accepted too)."""
    reasons: list[str] = []
    flags = key.hook_flags
    rd_ok = rd_hook is not None and rd_hook.covers(flags)
    if rd_hook is not None and not rd_ok:
        reasons.append(f"rd hook model {rd_hook.kind} does not match the hook flags {flags:#06x}")
    if flags & BEFORE_SWAP_RETURNS_DELTA_FLAG and not rd_ok:
        reasons.append("hook has BEFORE_SWAP_RETURNS_DELTA (custom swap accounting)")
    if flags & AFTER_SWAP_RETURNS_DELTA_FLAG and not rd_ok:
        reasons.append("hook has AFTER_SWAP_RETURNS_DELTA (hook takes/gives part of the output)")
    if key.is_dynamic_fee and not hook_fee_resolved:
        reasons.append("dynamic LP fee (hook-controlled, may be overridden per swap)")
    if flags & BEFORE_SWAP_FLAG and not (allow_before_swap_hook or hook_fee_resolved or rd_ok):
        reasons.append("hook has BEFORE_SWAP (arbitrary pre-swap logic)")
    if key.fee != DYNAMIC_FEE_FLAG and key.fee > MAX_LP_FEE:
        reasons.append(f"invalid static fee {key.fee}")
    return reasons


def check_supported(key: PoolKey, *, allow_before_swap_hook: bool = False, hook_fee_resolved: bool = False,
                    rd_hook: RdHook | None = None) -> None:
    reasons = unsupported_reasons(key, allow_before_swap_hook=allow_before_swap_hook,
                                  hook_fee_resolved=hook_fee_resolved, rd_hook=rd_hook)
    if reasons:
        raise UnsupportedPool("; ".join(reasons))


def calculate_swap_fee(protocol_fee: int, lp_fee: int) -> int:
    """ProtocolFeeLibrary.calculateSwapFee: protocolFee + lpFee - protocolFee*lpFee/1e6."""
    protocol_fee &= 0xFFF
    lp_fee &= 0xFFFFFF
    return protocol_fee + lp_fee - (protocol_fee * lp_fee) // PIPS_DENOMINATOR


@dataclass(frozen=True)
class V4PoolState:
    """Snapshot of a v4 pool inside the PoolManager (StateView.getSlot0 + getLiquidity + bitmap)."""

    key: PoolKey
    sqrt_price_x96: int
    tick: int
    protocol_fee: int  # uint24: low 12 bits zeroForOne, high 12 bits oneForZero
    lp_fee: int  # uint24 from slot0
    liquidity: int
    ticks: TickBitmapSnapshot
    block_number: int | None = None
    # LP fee (pips) per direction (zeroForOne, oneForZero) a fee-only hook applies to the next swap, resolved by
    # the caller; None = no hook fee model (dynamic-fee / BEFORE_SWAP pools are then rejected)
    hook_fee: tuple[int, int] | None = None
    # return-delta hook terms (v4quote.amm.v4_rd_hooks), set by the caller for a modelled hook; None = not modelled
    rd_hook: RdHook | None = None

    def __post_init__(self) -> None:
        if self.ticks.tick_spacing != self.key.tick_spacing:
            raise ValueError("tick bitmap spacing mismatch")
        if self.hook_fee is not None and not all(0 <= f <= MAX_LP_FEE for f in self.hook_fee):
            raise ValueError(f"hook_fee out of range: {self.hook_fee}")

    def lp_fee_for(self, zero_for_one: bool) -> int:
        if self.hook_fee is None:
            return self.lp_fee
        return self.hook_fee[0] if zero_for_one else self.hook_fee[1]

    def swap_fee(self, zero_for_one: bool) -> int:
        protocol_fee = (self.protocol_fee & 0xFFF) if zero_for_one else (self.protocol_fee >> 12)
        lp_fee = self.lp_fee_for(zero_for_one)
        return lp_fee if protocol_fee == 0 else calculate_swap_fee(protocol_fee, lp_fee)

    def unsupported_reasons(self, *, allow_before_swap_hook: bool = False) -> list[str]:
        return unsupported_reasons(self.key, allow_before_swap_hook=allow_before_swap_hook,
                                   hook_fee_resolved=self.hook_fee is not None, rd_hook=self.rd_hook)


def swap(
    state: V4PoolState,
    zero_for_one: bool,
    amount_specified: int,
    sqrt_price_limit_x96: int | None = None,
    *,
    allow_before_swap_hook: bool = False,
) -> SwapResult:
    """v4 Pool.swap semantics: amount_specified < 0 exact input, > 0 exact output. The pool swap only: a return-delta
    hook's take is not in it (``quote_exact_input`` adds it)."""
    check_supported(state.key, allow_before_swap_hook=allow_before_swap_hook,
                    hook_fee_resolved=state.hook_fee is not None, rd_hook=state.rd_hook)
    swap_fee = state.swap_fee(zero_for_one)
    if swap_fee >= MAX_SWAP_FEE and amount_specified > 0:
        raise MathRevert("InvalidFeeForExactOut")
    limit = default_price_limit(zero_for_one) if sqrt_price_limit_x96 is None else sqrt_price_limit_x96
    if amount_specified == 0:
        return SwapResult(zero_for_one, 0, 0, 0, state.sqrt_price_x96, state.tick, state.liquidity, 0, 0, 0)
    if zero_for_one:
        if limit >= state.sqrt_price_x96:
            raise MathRevert("PriceLimitAlreadyExceeded")
        if limit <= MIN_SQRT_PRICE:
            raise MathRevert("PriceLimitOutOfBounds")
    else:
        if limit <= state.sqrt_price_x96:
            raise MathRevert("PriceLimitAlreadyExceeded")
        if limit >= MAX_SQRT_PRICE:
            raise MathRevert("PriceLimitOutOfBounds")
    return swap_loop(
        sqrt_price_x96=state.sqrt_price_x96,
        tick=state.tick,
        liquidity=state.liquidity,
        ticks=state.ticks,
        fee_pips=swap_fee,
        zero_for_one=zero_for_one,
        exact_input=amount_specified < 0,
        amount=abs(amount_specified),
        sqrt_price_limit_x96=limit,
        v4=True,
        protocol_fee=(state.protocol_fee & 0xFFF) if zero_for_one else (state.protocol_fee >> 12),
    )


def quote_exact_input(state: V4PoolState, zero_for_one: bool, amount_in: int, *, hook_deltas: bool = True,
                      **kw) -> SwapResult:
    """Equivalent of V4Quoter.quoteExactInputSingle (hookData empty).

    Like BaseV4Quoter._swap, reverts (NotEnoughLiquidity) if the input cannot be fully consumed.
    A modelled return-delta hook (``state.rd_hook``): the trader's amounts - ``amount_in`` in full, ``amount_out``
    after the hook's take; ``fee_amount`` = pool fee + the hook's input-side take; price / tick / liquidity are the
    pool's. ``hook_deltas=False``: the pool swap alone (the Swap event's amounts).
    """
    if amount_in <= 0:
        raise MathRevert("amount_in must be positive")
    rd = state.rd_hook
    if rd is None or not hook_deltas:
        res = swap(state, zero_for_one, -amount_in, **kw)
        if res.amount_in != amount_in:
            raise MathRevert("NotEnoughLiquidity")
        return res
    fee_in = rd.input_fee(zero_for_one, amount_in)
    x = amount_in - fee_in
    if x <= 0:
        raise MathRevert("NothingSwapped")
    res = swap(state, zero_for_one, -x, **kw)
    if res.amount_in != x:
        raise MathRevert("NotEnoughLiquidity")  # Pangea: PartialFill; V4Quoter: NotEnoughLiquidity
    if rd.needs_tick:
        rd.check_post_tick(res.tick)
    out = res.amount_out - rd.output_fee(zero_for_one, res.amount_out)
    return replace(res, amount_in=amount_in, amount_out=out, fee_amount=res.fee_amount + fee_in)


def quote_exact_output(state: V4PoolState, zero_for_one: bool, amount_out: int, **kw) -> SwapResult:
    """Equivalent of V4Quoter.quoteExactOutputSingle (reverts if the output cannot be filled)."""
    if amount_out <= 0:
        raise MathRevert("amount_out must be positive")
    if state.rd_hook is not None:
        raise UnsupportedPool("exact output through a return-delta hook is not modelled")
    res = swap(state, zero_for_one, amount_out, **kw)
    if res.amount_out != amount_out:
        raise MathRevert("NotEnoughLiquidity")
    return res


def apply_swap(state: V4PoolState, result: SwapResult) -> V4PoolState:
    return replace(state, sqrt_price_x96=result.sqrt_price_x96, tick=result.tick, liquidity=result.liquidity)
