"""Port of Uniswap v3/v4 TickMath (identical results in v3-core and v4-core)."""

from functools import lru_cache

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT256

MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342
# v4 naming
MIN_SQRT_PRICE = MIN_SQRT_RATIO
MAX_SQRT_PRICE = MAX_SQRT_RATIO

_RATIO_CONSTANTS = (
    (0x2, 0xFFF97272373D413259A46990580E213A),
    (0x4, 0xFFF2E50F5F656932EF12357CF3C7FDCC),
    (0x8, 0xFFE5CACA7E10E4E61C3624EAA0941CD0),
    (0x10, 0xFFCB9843D60F6159C9DB58835C926644),
    (0x20, 0xFF973B41FA98C081472E6896DFB254C0),
    (0x40, 0xFF2EA16466C96A3843EC78B326B52861),
    (0x80, 0xFE5DEE046A99A2A811C461F1969C3053),
    (0x100, 0xFCBE86C7900A88AEDCFFC83B479AA3A4),
    (0x200, 0xF987A7253AC413176F2B074CF7815E54),
    (0x400, 0xF3392B0822B70005940C7A398E4B70F3),
    (0x800, 0xE7159475A2C29B7443B29C7FA6E889D9),
    (0x1000, 0xD097F3BDFD2022B8845AD8F792AA5825),
    (0x2000, 0xA9F746462D870FDF8A65DC1F90E061E5),
    (0x4000, 0x70D869A156D2A1B890BB3DF62BAF32F7),
    (0x8000, 0x31BE135F97D08FD981231505542FCFA6),
    (0x10000, 0x9AA508B5B7A84E1C677DE54F3E99BC9),
    (0x20000, 0x5D6AF8DEDB81196699C329225EE604),
    (0x40000, 0x2216E584F5FA1EA926041BEDFE98),
    (0x80000, 0x48A170391F7DC42444E8FA2),
)


@lru_cache(maxsize=1 << 16)
def get_sqrt_ratio_at_tick(tick: int) -> int:
    """sqrt(1.0001^tick) * 2^96 as Q64.96, rounded exactly as TickMath.getSqrtRatioAtTick.

    Memoised (pure function of ``tick``): the swap loop asks for the same few ticks of a pool on every quote of an
    amount search, and a dict hit is ~20x cheaper than the 20-step multiplication."""
    abs_tick = -tick if tick < 0 else tick
    if abs_tick > MAX_TICK:
        raise MathRevert("T")

    ratio = 0xFFFCB933BD6FAD37AA2D162D1A594001 if abs_tick & 0x1 else 0x100000000000000000000000000000000
    for bit, constant in _RATIO_CONSTANTS:
        if abs_tick & bit:
            ratio = (ratio * constant) >> 128

    if tick > 0:
        ratio = MAX_UINT256 // ratio

    return (ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1)


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    """Greatest tick such that get_sqrt_ratio_at_tick(tick) <= sqrt_price_x96 (TickMath.getTickAtSqrtRatio)."""
    if not (MIN_SQRT_RATIO <= sqrt_price_x96 < MAX_SQRT_RATIO):
        raise MathRevert("R")
    ratio = sqrt_price_x96 << 32

    msb = ratio.bit_length() - 1
    if msb >= 128:
        r = ratio >> (msb - 127)
    else:
        r = ratio << (127 - msb)

    log_2 = (msb - 128) << 64
    # 14 iterations of the square-and-compare loop (bits 63..50), exactly as the assembly.
    for shift in range(63, 49, -1):
        r = ((r * r) & MAX_UINT256) >> 127
        f = r >> 128
        log_2 |= f << shift
        r >>= f

    log_sqrt10001 = log_2 * 255738958999603826347141  # 128.128 number

    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_hi = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    if tick_low == tick_hi:
        return tick_low
    return tick_hi if get_sqrt_ratio_at_tick(tick_hi) <= sqrt_price_x96 else tick_low


# v4 naming aliases
get_sqrt_price_at_tick = get_sqrt_ratio_at_tick
get_tick_at_sqrt_price = get_tick_at_sqrt_ratio
