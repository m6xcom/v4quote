"""Uniswap v2 constant-product math (UniswapV2Library.getAmountOut / getAmountIn).

The fee is parameterised as ``fee_numerator / fee_denominator`` of the input
that is kept for the swap (Uniswap v2 and LFJ v1 / Trader Joe v1: 997/1000;
PancakeSwap v2: 9975/10000). Fee-on-transfer tokens are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT112, require

UNISWAP_V2_FEE = (997, 1000)


def get_amount_out(
    amount_in: int,
    reserve_in: int,
    reserve_out: int,
    fee_numerator: int = 997,
    fee_denominator: int = 1000,
) -> int:
    """UniswapV2Library.getAmountOut (rounds down)."""
    require(amount_in > 0, "UniswapV2Library: INSUFFICIENT_INPUT_AMOUNT")
    require(reserve_in > 0 and reserve_out > 0, "UniswapV2Library: INSUFFICIENT_LIQUIDITY")
    amount_in_with_fee = amount_in * fee_numerator
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * fee_denominator + amount_in_with_fee
    return numerator // denominator


def get_amount_in(
    amount_out: int,
    reserve_in: int,
    reserve_out: int,
    fee_numerator: int = 997,
    fee_denominator: int = 1000,
) -> int:
    """UniswapV2Library.getAmountIn (rounds up via +1, exactly as the library does)."""
    require(amount_out > 0, "UniswapV2Library: INSUFFICIENT_OUTPUT_AMOUNT")
    require(reserve_in > 0 and reserve_out > 0, "UniswapV2Library: INSUFFICIENT_LIQUIDITY")
    if amount_out > reserve_out:
        raise MathRevert("ds-math-sub-underflow")  # SafeMath sub in the library
    denominator = (reserve_out - amount_out) * fee_numerator
    if denominator == 0:
        raise MathRevert("division by zero")
    numerator = reserve_in * amount_out * fee_denominator
    return numerator // denominator + 1


@dataclass(frozen=True, slots=True)
class V2PairState:
    """Snapshot of a Uniswap-v2-style pair."""

    address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    fee_numerator: int = 997
    fee_denominator: int = 1000
    block_number: int | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.reserve0 <= MAX_UINT112 and 0 <= self.reserve1 <= MAX_UINT112):
            raise ValueError("reserves must fit uint112")
        if not (0 < self.fee_numerator <= self.fee_denominator):
            raise ValueError("invalid fee")

    def _reserves(self, zero_for_one: bool) -> tuple[int, int]:
        return (self.reserve0, self.reserve1) if zero_for_one else (self.reserve1, self.reserve0)

    def get_amount_out(self, amount_in: int, zero_for_one: bool) -> int:
        r_in, r_out = self._reserves(zero_for_one)
        return get_amount_out(amount_in, r_in, r_out, self.fee_numerator, self.fee_denominator)

    def get_amount_in(self, amount_out: int, zero_for_one: bool) -> int:
        r_in, r_out = self._reserves(zero_for_one)
        return get_amount_in(amount_out, r_in, r_out, self.fee_numerator, self.fee_denominator)

    def after_swap(self, amount_in: int, amount_out: int, zero_for_one: bool) -> "V2PairState":
        """State after a swap that paid ``amount_in`` and received ``amount_out``."""
        if zero_for_one:
            r0, r1 = self.reserve0 + amount_in, self.reserve1 - amount_out
        else:
            r0, r1 = self.reserve0 - amount_out, self.reserve1 + amount_in
        require(r0 > 0 and r1 > 0, "UniswapV2: INSUFFICIENT_LIQUIDITY")
        return V2PairState(
            self.address, self.token0, self.token1, r0, r1,
            self.fee_numerator, self.fee_denominator, self.block_number,
        )
