"""Fixed-width integer bounds and checked casts (Solidity SafeCast semantics)."""

from v4quote.errors import MathRevert

MAX_UINT8 = (1 << 8) - 1
MAX_UINT24 = (1 << 24) - 1
MAX_UINT40 = (1 << 40) - 1
MAX_UINT112 = (1 << 112) - 1
MAX_UINT128 = (1 << 128) - 1
MAX_UINT160 = (1 << 160) - 1
MAX_UINT256 = (1 << 256) - 1
MAX_INT128 = (1 << 127) - 1
MIN_INT128 = -(1 << 127)
MAX_INT256 = (1 << 255) - 1
MIN_INT256 = -(1 << 255)

Q96 = 1 << 96
Q128 = 1 << 128


def require(cond: bool, msg: str = "") -> None:
    """Solidity ``require``: raise MathRevert when the condition is false."""
    if not cond:
        raise MathRevert(msg or "require failed")


def wrap256(x: int) -> int:
    """Unchecked uint256 arithmetic result (mod 2**256)."""
    return x & MAX_UINT256


def _check(x: int, lo: int, hi: int, name: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{name}: expected int, got {type(x).__name__}")
    if x < lo or x > hi:
        raise MathRevert(f"{name} overflow: {x}")
    return x


def to_uint128(x: int) -> int:
    return _check(x, 0, MAX_UINT128, "uint128")


def to_uint160(x: int) -> int:
    return _check(x, 0, MAX_UINT160, "uint160")


def to_uint256(x: int) -> int:
    return _check(x, 0, MAX_UINT256, "uint256")


def to_int128(x: int) -> int:
    return _check(x, MIN_INT128, MAX_INT128, "int128")


def to_int256(x: int) -> int:
    return _check(x, MIN_INT256, MAX_INT256, "int256")
