"""LFJ Liquidity Book math primitives (Uint256x256Math, Uint128x128Math.pow, PriceHelper, FeeHelper,
PairParameterHelper). Identical in LB v2.1 and v2.2 (lfj-gg/joe-v2 tags v2.1.0 / v2.2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT128, MAX_UINT256

SCALE_OFFSET = 128
SCALE = 1 << SCALE_OFFSET
PRECISION = 10**18
BASIS_POINT_MAX = 10_000
MAX_FEE = 10**17  # 10%
MAX_PROTOCOL_SHARE = 2_500
REAL_ID_SHIFT = 1 << 23
MAX_LIQUIDITY_PER_BIN = 65251743116719673010965625540244653191619923014385985379600384103134737  # v2.2 only

MASK_UINT20 = (1 << 20) - 1
MASK_UINT24 = (1 << 24) - 1
MASK_UINT40 = (1 << 40) - 1


# --------------------------------------------------------------------------- Uint256x256Math
def _check256(result: int, what: str) -> int:
    if result > MAX_UINT256:
        raise MathRevert(f"Uint256x256Math__{what}Overflow")
    return result


def mul_div_round_down(x: int, y: int, denominator: int) -> int:
    if denominator == 0:
        raise MathRevert("division by zero")
    return _check256((x * y) // denominator, "MulDiv")


def mul_shift_round_down(x: int, y: int, offset: int) -> int:
    return _check256((x * y) >> offset, "MulShift")


def mul_shift_round_up(x: int, y: int, offset: int) -> int:
    result = mul_shift_round_down(x, y, offset)
    if (x * y) % (1 << offset) != 0:
        result += 1  # checked add in Solidity 0.8
        _check256(result, "MulShift")
    return result


def shift_div_round_down(x: int, offset: int, denominator: int) -> int:
    if denominator == 0:
        raise MathRevert("division by zero")
    return _check256((x << offset) // denominator, "MulDiv")


def shift_div_round_up(x: int, offset: int, denominator: int) -> int:
    result = shift_div_round_down(x, offset, denominator)
    if (x << offset) % denominator != 0:
        result += 1
        _check256(result, "MulDiv")
    return result


def safe128(x: int) -> int:
    if x > MAX_UINT128:
        raise MathRevert("SafeCast__Exceeds128Bits")
    return x


# --------------------------------------------------------------------------- Uint128x128Math.pow
def pow128(x: int, y: int) -> int:
    """Uint128x128Math.pow: x^y for a 128.128 fixed-point x and integer |y| < 2^20."""
    if y == 0:
        return SCALE
    invert = y < 0
    abs_y = -y if y < 0 else y
    result = 0
    if abs_y < 0x100000:
        result = SCALE
        squared = x
        if x > 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:
            squared = MAX_UINT256 // squared
            invert = not invert
        for bit in range(20):
            if abs_y & (1 << bit):
                result = ((result * squared) & MAX_UINT256) >> 128
            if bit < 19:
                squared = ((squared * squared) & MAX_UINT256) >> 128
    if result == 0:
        raise MathRevert("Uint128x128Math__PowUnderflow")
    return MAX_UINT256 // result if invert else result


# --------------------------------------------------------------------------- PriceHelper
def get_base(bin_step: int) -> int:
    return SCALE + (bin_step << SCALE_OFFSET) // BASIS_POINT_MAX


def get_price_from_id(bin_id: int, bin_step: int) -> int:
    """128.128 fixed-point price of token X in token Y for bin ``bin_id``."""
    return pow128(get_base(bin_step), bin_id - REAL_ID_SHIFT)


# --------------------------------------------------------------------------- FeeHelper
def _verify_fee(total_fee: int) -> None:
    if total_fee > MAX_FEE:
        raise MathRevert("FeeHelper__FeeTooLarge")


def get_fee_amount_from(amount_with_fees: int, total_fee: int) -> int:
    """Fee included in ``amount_with_fees`` (rounded up)."""
    _verify_fee(total_fee)
    return ((amount_with_fees * total_fee + PRECISION - 1) // PRECISION) & MAX_UINT128


def get_fee_amount(amount: int, total_fee: int) -> int:
    """Fee to add on top of ``amount`` (rounded up)."""
    _verify_fee(total_fee)
    denominator = PRECISION - total_fee
    return ((amount * total_fee + denominator - 1) // denominator) & MAX_UINT128


# --------------------------------------------------------------------------- PairParameterHelper
@dataclass(frozen=True)
class FeeParameters:
    """Decoded LBPair ``_parameters`` (static + variable parts, minus oracle id / active id)."""

    base_factor: int
    filter_period: int
    decay_period: int
    reduction_factor: int
    variable_fee_control: int
    protocol_share: int
    max_volatility_accumulator: int
    volatility_accumulator: int
    volatility_reference: int
    id_reference: int
    time_of_last_update: int

    def base_fee(self, bin_step: int) -> int:
        return self.base_factor * bin_step * 10**10

    def variable_fee(self, bin_step: int) -> int:
        if self.variable_fee_control == 0:
            return 0
        prod = self.volatility_accumulator * bin_step
        return (prod * prod * self.variable_fee_control + 99) // 100

    def total_fee(self, bin_step: int) -> int:
        return safe128(self.base_fee(bin_step) + self.variable_fee(bin_step))

    def update_references(self, active_id: int, timestamp: int) -> "FeeParameters":
        """PairParameterHelper.updateReferences (called once per swap with block.timestamp)."""
        if timestamp < self.time_of_last_update:
            raise MathRevert("timestamp before timeOfLastUpdate")
        if timestamp > MASK_UINT40:
            raise MathRevert("SafeCast__Exceeds40Bits")
        dt = timestamp - self.time_of_last_update
        p = self
        if dt >= self.filter_period:
            vol_ref = (
                (self.volatility_accumulator * self.reduction_factor // BASIS_POINT_MAX) & MASK_UINT24
                if dt < self.decay_period
                else 0
            )
            if vol_ref > MASK_UINT20:
                raise MathRevert("PairParametersHelper__InvalidParameter")
            p = replace(p, id_reference=active_id, volatility_reference=vol_ref)
        return replace(p, time_of_last_update=timestamp)

    def update_volatility_accumulator(self, active_id: int) -> "FeeParameters":
        delta_id = abs(active_id - self.id_reference)
        vol_acc = self.volatility_reference + delta_id * BASIS_POINT_MAX
        if vol_acc > self.max_volatility_accumulator:
            vol_acc = self.max_volatility_accumulator
        if vol_acc > MASK_UINT20:
            raise MathRevert("PairParametersHelper__InvalidParameter")
        return replace(self, volatility_accumulator=vol_acc)
