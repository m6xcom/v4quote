"""Human-readable conversions (floats allowed ONLY here, never in the swap path).

Nothing in v4quote.amm / v4quote.math imports this module; tests/unit/test_no_float.py enforces it.
"""

from __future__ import annotations


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Price of token0 in token1 units (human decimals) for display/logging."""
    return (sqrt_price_x96 / 2**96) ** 2 * 10 ** (decimals0 - decimals1)


def lb_price_to_float(price_128x128: int, decimals_x: int, decimals_y: int) -> float:
    """LB 128.128 price of X in Y for display/logging."""
    return price_128x128 / 2**128 * 10 ** (decimals_x - decimals_y)


def format_units(amount: int, decimals: int) -> str:
    """Exact decimal string of a token amount (no float)."""
    sign = "-" if amount < 0 else ""
    q, r = divmod(abs(amount), 10**decimals)
    return f"{sign}{q}.{r:0{decimals}d}" if decimals else f"{sign}{q}"
