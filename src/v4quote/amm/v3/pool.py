"""Concentrated-liquidity swap loop (UniswapV3Pool.swap / v4 Pool.swap) over an explicit state snapshot.

Covers Uniswap v3 and exact forks with parameterised fee / tickSpacing
(PancakeSwap v3, SushiSwap v3, ...). Protocol fees in v3 only change fee
growth accounting, never swap amounts, so they are not needed here.
Uniswap v4 reuses ``swap_loop`` with the v4 step function (see
``v4quote.amm.uniswap_v4``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT128, MAX_UINT160, MAX_UINT256, Q128, to_int256
from v4quote.amm.v3.liquidity_math import add_delta
from v4quote.amm.v3.swap_math import compute_swap_step_v3, compute_swap_step_v4, get_sqrt_price_target
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.tick_math import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
)


@dataclass(frozen=True)
class V3PoolState:
    """Snapshot of a Uniswap-v3-style pool at a given block."""

    address: str
    token0: str
    token1: str
    fee: int  # pips (hundredths of a bip), e.g. 3000 = 0.3%
    tick_spacing: int
    sqrt_price_x96: int
    tick: int
    liquidity: int
    ticks: TickBitmapSnapshot
    block_number: int | None = None
    # slot0.feeProtocol (Uniswap v3 encoding: low 4 bits token0, high 4 bits token1).
    # Only affects fee-growth accounting, never swap amounts.
    fee_protocol: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.fee < 1_000_000):
            raise ValueError("fee out of range")
        if self.tick_spacing <= 0:
            raise ValueError("tick_spacing must be positive")
        if self.ticks.tick_spacing != self.tick_spacing:
            raise ValueError("tick bitmap spacing mismatch")
        if not (0 <= self.liquidity <= MAX_UINT128 and 0 <= self.sqrt_price_x96 <= MAX_UINT160):
            raise ValueError("liquidity/sqrtPrice out of range")


@dataclass(frozen=True)
class SwapResult:
    """Outcome of a simulated swap.

    ``amount_in``/``amount_out`` are absolute token amounts (the input token is
    token0 when ``zero_for_one``). ``amount_in`` includes the fee.
    """

    zero_for_one: bool
    amount_in: int
    amount_out: int
    fee_amount: int
    sqrt_price_x96: int
    tick: int | None  # None only when the caller asked for amounts alone (``want_tick=False``)
    liquidity: int
    initialized_ticks_crossed: int
    steps: int
    amount_specified_remaining: int  # 0 when the requested amount was fully filled
    fee_growth_global_delta_x128: int = 0  # LP fee growth of the input token (mod 2**256)
    protocol_fee_amount: int = 0

    @property
    def fully_filled(self) -> bool:
        return self.amount_specified_remaining == 0


def default_price_limit(zero_for_one: bool) -> int:
    """Limit used by Uniswap quoters when sqrtPriceLimitX96 == 0."""
    return MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1


def swap_loop(
    *,
    sqrt_price_x96: int,
    tick: int,
    liquidity: int,
    ticks: TickBitmapSnapshot,
    fee_pips: int,
    zero_for_one: bool,
    exact_input: bool,
    amount: int,
    sqrt_price_limit_x96: int,
    v4: bool,
    protocol_fee: int = 0,
    want_tick: bool = True,
) -> SwapResult:
    """Shared v3/v4 loop. ``amount`` is the positive specified amount (input if exact_input else output).

    ``protocol_fee`` only affects fee-growth accounting: for v3 it is the
    direction's ``feeProtocol`` denominator (fee/feeProtocol goes to the
    protocol), for v4 the direction's protocol fee in pips.
    Validation of the price limit is done by the callers (it differs between v3 and v4).
    ``want_tick=False``: the post-swap ``tick`` is not computed (``SwapResult.tick`` is None); amounts, price,
    liquidity and crossings are unchanged. The tick after a step that stops between initialised ticks is only ever
    read from the result (the loop ends there: the specified amount is exhausted or the price limit reached), so
    skipping ``get_tick_at_sqrt_ratio`` saves ~1/3 of a one-step quote without touching any amount.
    """
    step_fn = compute_swap_step_v4 if v4 else compute_swap_step_v3
    # Remaining amount in the sign convention of the respective step function.
    if v4:
        remaining = -amount if exact_input else amount
    else:
        remaining = amount if exact_input else -amount

    total_in = 0  # amountIn + fee
    total_out = 0
    total_fee = 0
    crossed = 0
    steps = 0
    fee_growth = 0
    total_protocol = 0

    while remaining != 0 and sqrt_price_x96 != sqrt_price_limit_x96:
        steps += 1
        sqrt_price_start = sqrt_price_x96
        tick_next, initialized = ticks.next_initialized_tick_within_one_word(tick, zero_for_one)
        if tick_next < MIN_TICK:
            tick_next = MIN_TICK
        elif tick_next > MAX_TICK:
            tick_next = MAX_TICK
        sqrt_price_next = get_sqrt_ratio_at_tick(tick_next)
        target = get_sqrt_price_target(zero_for_one, sqrt_price_next, sqrt_price_limit_x96)

        sqrt_price_x96, step_in, step_out, step_fee = step_fn(
            sqrt_price_x96, target, liquidity, remaining, fee_pips
        )

        consumed = to_int256(step_in + step_fee)
        produced = to_int256(step_out)
        if exact_input:
            remaining = remaining + consumed if v4 else remaining - consumed
        else:
            remaining = remaining - produced if v4 else remaining + produced
        total_in += step_in + step_fee
        total_out += step_out
        total_fee += step_fee

        lp_fee = step_fee
        if protocol_fee > 0:
            if v4:
                delta = step_fee if fee_pips == protocol_fee else (step_in + step_fee) * protocol_fee // 1_000_000
            else:
                delta = step_fee // protocol_fee
            lp_fee -= delta
            total_protocol += delta
        if liquidity > 0:
            fee_growth = (fee_growth + (lp_fee * Q128) // liquidity) & MAX_UINT256

        if sqrt_price_x96 == sqrt_price_next:
            if initialized:
                liquidity_net = ticks.get_liquidity_net(tick_next)
                if zero_for_one:
                    liquidity_net = -liquidity_net
                liquidity = add_delta(liquidity, liquidity_net)
                crossed += 1
            tick = tick_next - 1 if zero_for_one else tick_next
        elif sqrt_price_x96 != sqrt_price_start:
            tick = get_tick_at_sqrt_ratio(sqrt_price_x96) if want_tick else None

    return SwapResult(
        zero_for_one=zero_for_one,
        amount_in=total_in,
        amount_out=total_out,
        fee_amount=total_fee,
        sqrt_price_x96=sqrt_price_x96,
        tick=tick,
        liquidity=liquidity,
        initialized_ticks_crossed=crossed,
        steps=steps,
        amount_specified_remaining=abs(remaining),
        fee_growth_global_delta_x128=fee_growth,
        protocol_fee_amount=total_protocol,
    )


def swap(
    state: V3PoolState,
    zero_for_one: bool,
    amount_specified: int,
    sqrt_price_limit_x96: int | None = None,
    *,
    want_tick: bool = True,
) -> SwapResult:
    """UniswapV3Pool.swap semantics: amount_specified > 0 exact input, < 0 exact output. ``want_tick=False``: amounts
    only, ``SwapResult.tick`` is None (see ``swap_loop``); never use such a result with ``apply_swap``."""
    if amount_specified == 0:
        raise MathRevert("AS")
    limit = default_price_limit(zero_for_one) if sqrt_price_limit_x96 is None else sqrt_price_limit_x96
    if zero_for_one:
        if not (limit < state.sqrt_price_x96 and limit > MIN_SQRT_RATIO):
            raise MathRevert("SPL")
    else:
        if not (limit > state.sqrt_price_x96 and limit < MAX_SQRT_RATIO):
            raise MathRevert("SPL")
    return swap_loop(
        sqrt_price_x96=state.sqrt_price_x96,
        tick=state.tick,
        liquidity=state.liquidity,
        ticks=state.ticks,
        fee_pips=state.fee,
        zero_for_one=zero_for_one,
        exact_input=amount_specified > 0,
        amount=abs(amount_specified),
        sqrt_price_limit_x96=limit,
        v4=False,
        protocol_fee=(state.fee_protocol % 16) if zero_for_one else (state.fee_protocol >> 4),
        want_tick=want_tick,
    )


def quote_exact_input(
    state: V3PoolState, zero_for_one: bool, amount_in: int, sqrt_price_limit_x96: int | None = None,
    *, want_tick: bool = True,
) -> SwapResult:
    """Equivalent of QuoterV2.quoteExactInputSingle (amountOut = result.amount_out). ``want_tick=False``: the
    quoter's amounts without the post-swap tick (``tick`` None), for amount searches."""
    if amount_in <= 0:
        raise MathRevert("amount_in must be positive")
    return swap(state, zero_for_one, amount_in, sqrt_price_limit_x96, want_tick=want_tick)


def quote_exact_output(
    state: V3PoolState, zero_for_one: bool, amount_out: int, sqrt_price_limit_x96: int | None = None
) -> SwapResult:
    """Equivalent of QuoterV2.quoteExactOutputSingle (amountIn = result.amount_in).

    Like the quoter, reverts when no price limit is given and the pool cannot
    deliver the full output.
    """
    if amount_out <= 0:
        raise MathRevert("amount_out must be positive")
    res = swap(state, zero_for_one, -amount_out, sqrt_price_limit_x96)
    if sqrt_price_limit_x96 is None and res.amount_out != amount_out:
        raise MathRevert("insufficient liquidity for exact output")
    return res


def apply_swap(state: V3PoolState, result: SwapResult) -> V3PoolState:
    """State after the swap (price, tick, active liquidity). Tick data is unchanged by swaps."""
    if result.tick is None:
        raise ValueError("apply_swap needs a result with the post-swap tick (swap(..., want_tick=True))")
    return replace(state, sqrt_price_x96=result.sqrt_price_x96, tick=result.tick, liquidity=result.liquidity)
