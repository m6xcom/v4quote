"""Port of BitMath (most/least significant bit of a non-zero uint256)."""

from v4quote.math.uint import MAX_UINT256, require


def most_significant_bit(x: int) -> int:
    require(0 < x <= MAX_UINT256, "msb: x must be non-zero uint256")
    return x.bit_length() - 1


def least_significant_bit(x: int) -> int:
    require(0 < x <= MAX_UINT256, "lsb: x must be non-zero uint256")
    return (x & -x).bit_length() - 1
