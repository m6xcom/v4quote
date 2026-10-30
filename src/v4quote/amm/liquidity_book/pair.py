"""LFJ Liquidity Book v2.1 / v2.2 pair: getSwapOut, getSwapIn and swap over an explicit bin snapshot.

Also covers exact LB v2.2 forks such as Pharaoh DLMM (validated against the
fork's own quoter in the on-chain tests).

The variable fee depends on ``block.timestamp`` (volatility accumulator
decay), so the timestamp of the block being simulated is an explicit input
of every function.

Hooks (v2.2): a pool with a non-zero hooks contract is modelled only if
  * its hook flags are a subset of ALLOWED_HOOK_FLAGS (0x151: BEFORE_SWAP,
    BEFORE_MINT, BEFORE_BURN, BEFORE_BATCH_TRANSFER_FROM); any AFTER_SWAP,
    FLASH_LOAN or other flag is rejected, and
  * (chain, hook address) is in the allowlist (default: hooks_allowlist.DEFAULT_HOOKS_ALLOWLIST),
    the entry is bound to this pair (``entry.pair == state.address``), and the hook's on-chain
    codehash, implementation address and implementation codehash match the entry, and
  * for an LFJ ImmutableClone rewarder hook (``proxy_kind == "lfj-immutable-clone"``) the facts read
    with the pair at the same block show no extra rewarder (``getExtraHooksParameters() == 0``: an extra
    rewarder runs its own ``beforeSwap`` without changing any codehash) and ``isLinked() == true``;
  * for a beacon-proxied rewarder hook (``proxy_kind == "eip1967-beacon"``, Pharaoh DLMM) the hook's
    ``getLBPair()`` is this pair and ``isLinked() == true``.
    Unknown facts (not read, call reverted) count as failures.
Otherwise ``UnsupportedPool`` is raised.
The allowlisted hooks were checked (verified sources / bytecode); the
evidence is a replay of 120 historical swap()
calls (hooks executed) that matched v4quote per bin, plus bytecode/source checks.
Being allowlisted permits SIMULATION only: ``execution_permitted`` returns the
entry's ``execution_allowed`` flag, False for every entry.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from typing import Mapping

from v4quote.errors import InsufficientState, MathRevert, UnsupportedPool
from v4quote.amm.liquidity_book.hooks_allowlist import DEFAULT_HOOKS_ALLOWLIST, HookApproval, HooksAllowlist
from v4quote.math.uint import MAX_UINT128
from v4quote.amm.liquidity_book.math import (
    BASIS_POINT_MAX,
    MAX_LIQUIDITY_PER_BIN,
    SCALE_OFFSET,
    FeeParameters,
    get_fee_amount,
    get_fee_amount_from,
    get_price_from_id,
    mul_shift_round_down,
    mul_shift_round_up,
    safe128,
    shift_div_round_down,
    shift_div_round_up,
)

HOOKS_BEFORE_SWAP_FLAG = 1 << 160
HOOKS_AFTER_SWAP_FLAG = 1 << 161
HOOK_FLAGS_SHIFT = 160
# Hooks.sol flags relative to bit 160: 0x001 BEFORE_SWAP, 0x010 BEFORE_MINT, 0x040 BEFORE_BURN,
# 0x100 BEFORE_BATCH_TRANSFER_FROM. Everything else (AFTER_SWAP, FLASH_LOAN, ...) is rejected.
ALLOWED_HOOK_FLAGS = 0x151


_ADDRESS_MASK = (1 << 160) - 1
MAX_UINT24 = (1 << 24) - 1


@dataclass(frozen=True)
class LBPairState:
    """Snapshot of an LBPair at a block.

    ``bins`` holds (reserveX, reserveY) for every non-empty bin inside
    ``loaded_range`` (inclusive); bins inside the range that are absent are
    empty. Outside the range the state is unknown unless the corresponding
    ``exhausted_*`` flag says there are no more non-empty bins in that
    direction (getNextNonEmptyBin returned the sentinel).
    """

    address: str
    token_x: str
    token_y: str
    bin_step: int
    active_id: int
    params: FeeParameters
    bins: Mapping[int, tuple[int, int]]
    loaded_range: tuple[int, int]
    exhausted_below: bool = False
    exhausted_above: bool = False
    version: str = "2.2"  # "2.1" or "2.2"
    hooks_parameters: int = 0
    block_number: int | None = None
    # facts about the hooks contract read on-chain at block_number (see chain.loaders.resolve_hook)
    chain: str | None = None
    hooks_codehash: str | None = None
    hooks_implementation: str | None = None
    hooks_implementation_codehash: str | None = None
    # LFJ rewarder hooks only (chain.loaders.lfj_hook_link_calls), None = not read / reverted:
    hooks_extra_parameters: int | None = None  # getExtraHooksParameters() as uint256 (0 = no extra rewarder)
    hooks_linked: bool | None = None  # isLinked()
    # beacon-proxied rewarder hooks (Pharaoh DLMMRewarder), None = not read / reverted:
    hooks_pair: str | None = None  # getLBPair() of the hook
    _sorted_ids: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version not in ("2.1", "2.2"):
            raise ValueError("version must be '2.1' or '2.2'")
        ids = tuple(sorted(i for i, (x, y) in self.bins.items() if x or y))
        object.__setattr__(self, "_sorted_ids", ids)

    # -- hooks -----------------------------------------------------------------
    @property
    def hooks_address(self) -> int:
        return self.hooks_parameters & _ADDRESS_MASK

    @property
    def hook_flags(self) -> int:
        return (self.hooks_parameters >> HOOK_FLAGS_SHIFT) & ((1 << 96) - 1)

    def unsupported_reasons(self, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST) -> list[str]:
        if not self.hooks_address:
            return []
        reasons = []
        addr = "0x" + format(self.hooks_address, "040x")
        if self.hook_flags & ~ALLOWED_HOOK_FLAGS:
            reasons.append(f"hook flags 0x{self.hook_flags:x} outside allowed set 0x{ALLOWED_HOOK_FLAGS:x}")
        entry = _approval(self, hooks_allowlist)
        if entry is None:
            reasons.append(f"hook {addr} on {self.chain} not in allowlist")
            return reasons
        if entry.pair is None or entry.pair.lower() != (self.address or "").lower():
            reasons.append(f"hook {addr} is allowlisted for pair {entry.pair}, not for {self.address}")
        if self.hooks_codehash is None or entry.codehash.lower() != self.hooks_codehash.lower():
            reasons.append(f"hook {addr} codehash {self.hooks_codehash} != allowlisted {entry.codehash}")
        if entry.implementation is not None:
            if (self.hooks_implementation or "").lower() != entry.implementation.lower():
                reasons.append(f"hook implementation {self.hooks_implementation} != allowlisted {entry.implementation}")
            if entry.implementation_codehash is not None and (
                (self.hooks_implementation_codehash or "").lower() != entry.implementation_codehash.lower()
            ):
                reasons.append("hook implementation codehash differs from allowlist")
        elif self.hooks_implementation is not None:
            reasons.append(f"hook is a proxy ({self.hooks_implementation}) but allowlist entry pins no implementation")
        if entry.proxy_kind == "lfj-immutable-clone":
            extra = self.hooks_extra_parameters
            if extra is None:
                reasons.append(f"hook {addr}: getExtraHooksParameters() unknown (not read or reverted)")
            elif extra != 0:
                reasons.append(
                    f"hook {addr} has an extra rewarder 0x{extra & _ADDRESS_MASK:040x} "
                    f"(flags 0x{extra >> HOOK_FLAGS_SHIFT:x}); its beforeSwap code is not modelled"
                )
            if self.hooks_linked is None:
                reasons.append(f"hook {addr}: isLinked() unknown (not read or reverted)")
            elif self.hooks_linked is not True:
                reasons.append(f"hook {addr} is not linked to its pair (isLinked() == false)")
        elif entry.proxy_kind == "eip1967-beacon":
            # the proxies of one beacon share a codehash, so only the hook's own view of its pair binds it; a hook
            # whose dlmmPool is another pair (or unlinked) would revert every swap of this one
            if self.hooks_pair is None:
                reasons.append(f"hook {addr}: getLBPair() unknown (not read or reverted)")
            elif self.hooks_pair.lower() != (self.address or "").lower():
                reasons.append(f"hook {addr} serves pair {self.hooks_pair}, not {self.address} (getLBPair())")
            if self.hooks_linked is None:
                reasons.append(f"hook {addr}: isLinked() unknown (not read or reverted)")
            elif self.hooks_linked is not True:
                reasons.append(f"hook {addr} is not linked to its pair (isLinked() == false)")
        return reasons

    # -- bins --------------------------------------------------------------------
    def bin(self, bin_id: int) -> tuple[int, int]:
        lo, hi = self.loaded_range
        if not (lo <= bin_id <= hi):
            raise InsufficientState(f"bin {bin_id} outside loaded range {self.loaded_range}")
        return self.bins.get(bin_id, (0, 0))

    def next_non_empty_bin(self, swap_for_y: bool, bin_id: int) -> int | None:
        """getNextNonEmptyBin: lower ids when swapping X for Y, higher ids otherwise. None = no more bins."""
        ids = self._sorted_ids
        lo, hi = self.loaded_range
        if swap_for_y:
            k = bisect.bisect_left(ids, bin_id) - 1
            if k >= 0 and ids[k] >= lo:
                return ids[k]
            if self.exhausted_below:
                return None
            raise InsufficientState(f"next non-empty bin below {bin_id} is outside loaded range {self.loaded_range}")
        k = bisect.bisect_right(ids, bin_id)
        if k < len(ids) and ids[k] <= hi:
            return ids[k]
        if self.exhausted_above:
            return None
        raise InsufficientState(f"next non-empty bin above {bin_id} is outside loaded range {self.loaded_range}")


def _approval(state: LBPairState, hooks_allowlist: HooksAllowlist) -> HookApproval | None:
    addr = "0x" + format(state.hooks_address, "040x")
    for (chain, hook), approval in hooks_allowlist.items():
        if chain == state.chain and hook.lower() == addr:
            return approval
    return None


def execution_permitted(state: LBPairState, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST) -> bool:
    """Whether transactions may be sent through this pool (policy for callers that send transactions, not the math)."""
    if state.unsupported_reasons(hooks_allowlist):
        return False
    if not state.hooks_address:
        return True
    approval = _approval(state, hooks_allowlist)
    return bool(approval and approval.execution_allowed)


def check_supported(state: LBPairState, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST) -> None:
    reasons = state.unsupported_reasons(hooks_allowlist)
    if reasons:
        raise UnsupportedPool("; ".join(reasons))


def _get_liquidity(x: int, y: int, price: int) -> int:
    liquidity = 0
    if x > 0:
        liquidity = price * x
        if liquidity >= 1 << 256:
            raise MathRevert("BinHelper__LiquidityOverflow")
    if y > 0:
        y <<= SCALE_OFFSET
        liquidity += y
        if liquidity >= 1 << 256 or y >= 1 << 256:
            raise MathRevert("BinHelper__LiquidityOverflow")
    return liquidity


def get_amounts(
    bin_reserves: tuple[int, int],
    params: FeeParameters,
    bin_step: int,
    swap_for_y: bool,
    active_id: int,
    amount_in_left: int,
    version: str,
) -> tuple[int, int, int]:
    """BinHelper.getAmounts for one bin -> (amountInWithFees, amountOutOfBin, totalFees)."""
    price = get_price_from_id(active_id, bin_step)
    reserve_x, reserve_y = bin_reserves
    bin_reserve_out = reserve_y if swap_for_y else reserve_x

    if swap_for_y:
        max_amount_in = safe128(shift_div_round_up(bin_reserve_out, SCALE_OFFSET, price))
    else:
        max_amount_in = safe128(mul_shift_round_up(bin_reserve_out, price, SCALE_OFFSET))

    total_fee = params.total_fee(bin_step)
    max_fee = get_fee_amount(max_amount_in, total_fee)
    max_amount_in += max_fee
    if max_amount_in > MAX_UINT128:
        raise MathRevert("uint128 overflow")

    if amount_in_left >= max_amount_in:
        fee = max_fee
        amount_in = max_amount_in
        amount_out = bin_reserve_out
    else:
        fee = get_fee_amount_from(amount_in_left, total_fee)
        amount_in_no_fee = amount_in_left - fee
        if swap_for_y:
            amount_out = safe128(mul_shift_round_down(amount_in_no_fee, price, SCALE_OFFSET))
        else:
            amount_out = safe128(shift_div_round_down(amount_in_no_fee, SCALE_OFFSET, price))
        if amount_out > bin_reserve_out:
            amount_out = bin_reserve_out
        amount_in = amount_in_left

    if version == "2.2":
        if swap_for_y:
            nx, ny = reserve_x + amount_in, reserve_y - amount_out
        else:
            nx, ny = reserve_x - amount_out, reserve_y + amount_in
        if nx > MAX_UINT128 or ny > MAX_UINT128:
            raise MathRevert("PackedUint128Math__AddOverflow")
        if _get_liquidity(nx, ny, price) > MAX_LIQUIDITY_PER_BIN:
            raise MathRevert("BinHelper__MaxLiquidityPerBinExceeded")

    return amount_in, amount_out, fee


@dataclass(frozen=True)
class BinSwap:
    """One bin of a swap, with the same fields as LBPair's per-bin ``Swap`` event."""

    bin_id: int
    amount_in: int  # amountsInWithFees minus protocol fees (what the event emits)
    amount_out: int
    volatility_accumulator: int
    total_fees: int
    protocol_fees: int


@dataclass(frozen=True)
class LBSwapResult:
    swap_for_y: bool
    amount_in: int  # consumed input including fees
    amount_out: int
    fee: int  # total fees (incl. protocol share)
    amount_in_left: int  # getSwapOut semantics: input that could not be swapped
    bins_crossed: int  # number of bins that contributed liquidity
    active_id: int  # active id after the swap
    params: FeeParameters  # fee parameters after the swap
    bin_swaps: tuple[BinSwap, ...] = ()  # filled by swap() only


def get_swap_out(
    state: LBPairState, amount_in: int, swap_for_y: bool, timestamp: int, *, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST
) -> tuple[int, int, int]:
    """LBPair.getSwapOut -> (amountInLeft, amountOut, fee)."""
    r = _simulate_out(state, amount_in, swap_for_y, timestamp, hooks_allowlist)
    return r.amount_in_left, r.amount_out, r.fee


def _simulate_out(
    state: LBPairState, amount_in: int, swap_for_y: bool, timestamp: int, hooks_allowlist: HooksAllowlist
) -> LBSwapResult:
    check_supported(state, hooks_allowlist)
    if not (0 <= amount_in <= MAX_UINT128):
        raise MathRevert("amount_in must be uint128")
    amounts_in_left = amount_in
    bin_step = state.bin_step
    bin_id = state.active_id
    params = state.params.update_references(state.active_id, timestamp)
    amount_out = 0
    fee = 0
    crossed = 0
    last_id = bin_id

    while True:
        reserves = state.bin(bin_id)
        out_reserve = reserves[1] if swap_for_y else reserves[0]
        if out_reserve != 0:
            params = params.update_volatility_accumulator(bin_id)
            in_with_fees, out_of_bin, fees = get_amounts(
                reserves, params, bin_step, swap_for_y, bin_id, amounts_in_left, state.version
            )
            if in_with_fees > 0:
                amounts_in_left -= in_with_fees
                amount_out += out_of_bin
                fee += fees
                crossed += 1
                last_id = bin_id
        if amounts_in_left == 0:
            break
        next_id = state.next_non_empty_bin(swap_for_y, bin_id)
        if next_id is None or next_id == 0 or next_id == MAX_UINT24:
            break
        bin_id = next_id

    return LBSwapResult(
        swap_for_y=swap_for_y,
        amount_in=amount_in - amounts_in_left,
        amount_out=amount_out,
        fee=fee,
        amount_in_left=amounts_in_left,
        bins_crossed=crossed,
        active_id=bin_id if amounts_in_left == 0 else last_id,
        params=params,
    )


def get_swap_in(
    state: LBPairState, amount_out: int, swap_for_y: bool, timestamp: int, *, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST
) -> tuple[int, int, int]:
    """LBPair.getSwapIn -> (amountIn, amountOutLeft, fee)."""
    check_supported(state, hooks_allowlist)
    if not (0 <= amount_out <= MAX_UINT128):
        raise MathRevert("amount_out must be uint128")
    amount_out_left = amount_out
    amount_in = 0
    fee = 0
    bin_step = state.bin_step
    bin_id = state.active_id
    params = state.params.update_references(state.active_id, timestamp)

    while True:
        reserves = state.bin(bin_id)
        bin_reserves = reserves[1] if swap_for_y else reserves[0]
        if bin_reserves > 0:
            price = get_price_from_id(bin_id, bin_step)
            amount_out_of_bin = amount_out_left if bin_reserves > amount_out_left else bin_reserves
            params = params.update_volatility_accumulator(bin_id)
            if swap_for_y:
                amount_in_without_fee = shift_div_round_up(amount_out_of_bin, SCALE_OFFSET, price) & MAX_UINT128
            else:
                amount_in_without_fee = mul_shift_round_up(amount_out_of_bin, price, SCALE_OFFSET) & MAX_UINT128
            total_fee = params.total_fee(bin_step)
            fee_amount = get_fee_amount(amount_in_without_fee, total_fee)
            amount_in += amount_in_without_fee + fee_amount
            if amount_in > MAX_UINT128:
                raise MathRevert("uint128 overflow")
            amount_out_left -= amount_out_of_bin
            fee += fee_amount
        if amount_out_left == 0:
            break
        next_id = state.next_non_empty_bin(swap_for_y, bin_id)
        if next_id is None or next_id == 0 or next_id == MAX_UINT24:
            break
        bin_id = next_id

    return amount_in, amount_out_left, fee


def swap(
    state: LBPairState, amount_in: int, swap_for_y: bool, timestamp: int, *, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST
) -> tuple[LBSwapResult, LBPairState]:
    """LBPair.swap: returns the result and the pair state after the swap (bins, active id, fee params).

    Reverts (MathRevert) like the contract when liquidity runs out or the output is zero.
    """
    check_supported(state, hooks_allowlist)
    if amount_in == 0:
        raise MathRevert("LBPair__InsufficientAmountIn")
    amounts_left = amount_in
    bin_step = state.bin_step
    active_id = state.active_id
    params = state.params.update_references(state.active_id, timestamp)
    bins = dict(state.bins)
    amount_out = 0
    fee_total = 0
    crossed = 0
    records: list[BinSwap] = []

    while True:
        reserves = state.bin(active_id)
        out_reserve = reserves[1] if swap_for_y else reserves[0]
        if out_reserve != 0:
            params = params.update_volatility_accumulator(active_id)
            in_with_fees, out_of_bin, fees = get_amounts(
                reserves, params, bin_step, swap_for_y, active_id, amounts_left, state.version
            )
            if in_with_fees > 0:
                amounts_left -= in_with_fees
                amount_out += out_of_bin
                fee_total += fees
                crossed += 1
                p_fees = fees * params.protocol_share // BASIS_POINT_MAX
                in_to_bin = in_with_fees - p_fees
                records.append(
                    BinSwap(active_id, in_to_bin, out_of_bin, params.volatility_accumulator, fees, p_fees)
                )
                rx, ry = reserves
                if swap_for_y:
                    bins[active_id] = (rx + in_to_bin, ry - out_of_bin)
                else:
                    bins[active_id] = (rx - out_of_bin, ry + in_to_bin)
        if amounts_left == 0:
            break
        next_id = state.next_non_empty_bin(swap_for_y, active_id)
        if next_id is None or next_id == 0 or next_id == MAX_UINT24:
            raise MathRevert("LBPair__OutOfLiquidity")
        active_id = next_id

    if amount_out == 0:
        raise MathRevert("LBPair__InsufficientAmountOut")

    result = LBSwapResult(
        swap_for_y=swap_for_y,
        amount_in=amount_in,
        amount_out=amount_out,
        fee=fee_total,
        amount_in_left=0,
        bins_crossed=crossed,
        active_id=active_id,
        params=params,
        bin_swaps=tuple(records),
    )
    new_state = replace(state, bins=bins, active_id=active_id, params=params)
    return result, new_state
