"""Metric OMM pools (oracle-priced bin AMM, "prop-AMM"): exact-output and exact-input quotes over a bin snapshot.

A Metric pool takes a bid / ask / reference quote (Q64.64, scaled token1 per scaled token0) from its price provider
and holds liquidity in bins whose edges are fixed distances (``lengthE6``) from the reference price. Inside a bin the
price moves in a straight line with the output the bin pays (docs.metric.xyz, "Liquidity and bins" and "Estimate a fill
from depth").

This model is written from Metric's public documentation, the MIT-licensed interfaces and lens contracts, and
measurements against ``MetricOmmSwapQuoter.quoteHypothetical*`` (eth_call); it contains no code from Metric's
BUSL-licensed pool contracts. What it reproduces:

* Edge prices of every bin: ``floor(ref * (1e6 + dist) / 1e6)`` adjusted by the buy multiplier
  ``ceil(ask * 2^64 / ref)`` (rounded up) or divided by the sell multiplier ``ceil(ref * 2^64 / bid)`` (rounded down).
* The cost of an output ``o`` taken from one bin, per bin and in scaled units: the average of the cost at the opening
  and at the closing price, ``ceil((ceil(o * P0 / 2^64) + ceil(o * P1 / 2^64)) / 2)`` on asks and
  ``ceil((ceil(o * 2^64 / P0) + ceil(o * 2^64 / P1)) / 2)`` on bids, with ``P1`` the price at the position after the
  fill (position = ``uint104`` fraction of the bin, ``type(uint104).max`` = 100 %).
* Native-unit conversion: scaled input rounded up, scaled output rounded down; exact-output requests above the pool's
  liquidity are capped to it and exact-input requests above it are filled partially, as the pool does.

``quote_exact_output`` reproduces the quoter to the unit (verified on Robinhood Chain pools, see the README).

``quote_exact_input`` solves the bin the order ends in exactly (closed-form root). The pool's own solver rounds
inside a bin in a way this model does not reproduce bit for bit, and its result can differ from the exact root by a
small amount. The difference is bounded per bin (price-grid step, one input unit, and a fourth-order term in the bin's
price span, calibrated on quoter measurements with a factor-two margin). When that bound can change the native
amount, the quote raises ``UnsupportedPool`` instead of returning a number that might be off by one unit. On pools whose
output token is scaled up internally (for example a 6-decimal stablecoin) this almost never happens; on 18/18-decimal
pairs priced far below one (``P < 2^64``) most exact-input quotes are refused.

Not modelled (``UnsupportedPool``): per-bin additional fees, a non-zero notional fee, a paused pool (unless
``ignore_pause``, which mirrors the hypothetical quoter). Swap extensions are not evaluated: their interface can only
accept or reject a swap (they return a selector, no amount deltas), so they cannot change a quoted amount, but they can
reject it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from v4quote.errors import InsufficientState, MathRevert, UnsupportedPool

Q64 = 1 << 64
E6 = 1_000_000
MAX_POS = (1 << 104) - 1  # curPosInBin is a uint104 fraction of the bin; type(uint104).max is 100 %
MAX_UINT104 = (1 << 104) - 1
MAX_UINT128 = (1 << 128) - 1
ASYMMETRY_CHECK_DISABLED = 0xFFFF
UNAVAILABLE_ASK = MAX_UINT128  # API sentinel for "not quotable" (bid 0 / ask 2^128-1)

# BinState fields (one storage word): token0BalanceScaled, token1BalanceScaled, lengthE6, addFeeBuyE6, addFeeSellE6
Bin = tuple[int, int, int, int, int]


def _cdiv(a: int, b: int) -> int:
    return -((-a) // b)


def isqrt(n: int) -> int:
    """floor(sqrt(n)) for n >= 0 (integer Newton iteration)."""
    if n < 0:
        raise ValueError("isqrt of a negative number")
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) >> 1)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def _csqrt(n: int) -> int:
    s = isqrt(n)
    return s if s * s == n else s + 1


def decode_bin_word(word: int) -> Bin:
    """Split a raw BinState storage word into (token0Scaled, token1Scaled, lengthE6, addFeeBuyE6, addFeeSellE6)."""
    return (word & MAX_UINT104, (word >> 104) & MAX_UINT104, (word >> 208) & 0xFFFF, (word >> 224) & 0xFFFF,
            (word >> 240) & 0xFFFF)


@dataclass(frozen=True)
class MetricOmmPoolState:
    """Snapshot of a MetricOmmPool plus the provider quote to price it with.

    ``bid_x64`` / ``ask_x64`` / ``reference_x64`` are the provider's raw quote (what ``quoteHypothetical*`` takes), not
    the API's ``bidAdj`` / ``askAdj``. ``bins`` maps bin index -> (token0Scaled, token1Scaled, lengthE6, addFeeBuyE6,
    addFeeSellE6) for every bin in ``loaded_range`` (inclusive); ``lowest_bin`` / ``highest_bin`` are the pool's own
    limits (immutables). Scaled amounts are native amounts times ``token0_scale`` / ``token1_scale``.
    """

    address: str
    token0: str
    token1: str
    token0_scale: int
    token1_scale: int
    bid_x64: int
    ask_x64: int
    reference_x64: int
    cur_bin: int
    pos_in_bin: int
    cur_bin_dist_e6: int
    bins: Mapping[int, Bin]
    loaded_range: tuple[int, int]
    lowest_bin: int
    highest_bin: int
    pause_level: int = 0
    notional_fee_e8: int = 0
    bid_ask_ref_asymmetry_x8: int = ASYMMETRY_CHECK_DISABLED
    extensions: tuple[str, ...] = ()
    block_number: int | None = None
    chain: str | None = None
    _sorted: tuple[int, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lo, hi = self.loaded_range
        if not (self.lowest_bin <= lo <= self.cur_bin <= hi <= self.highest_bin):
            raise ValueError("loaded_range must contain cur_bin and lie inside [lowest_bin, highest_bin]")
        if any(not (lo <= i <= hi) for i in self.bins) or set(range(lo, hi + 1)) - set(self.bins):
            raise ValueError("bins must hold exactly the indices of loaded_range")
        if self.token0_scale < 1 or self.token1_scale < 1:
            raise ValueError("scale multipliers must be >= 1")
        if not 0 <= self.pos_in_bin <= MAX_POS:
            raise ValueError("pos_in_bin out of uint104 range")
        object.__setattr__(self, "_sorted", tuple(sorted(self.bins)))


@dataclass(frozen=True)
class MetricSwapResult:
    """Native amounts of a quote. ``partial`` is True when the pool ran out of liquidity in the swap direction
    (exact input: ``amount_in`` is what the pool takes; exact output: ``amount_out`` is capped to what it holds)."""

    amount_in: int
    amount_out: int
    partial: bool
    end_bin: int


# ----------------------------------------------------------------------------- prices
def check_prices(state: MetricOmmPoolState) -> None:
    """The pool's validation of a provider quote (raises ``MathRevert`` with the pool's error name)."""
    bid, ask, ref = state.bid_x64, state.ask_x64, state.reference_x64
    if bid == 0 and ask > 0:
        raise MathRevert("BidIsZero")
    if bid >= ask:
        raise MathRevert("BidGreaterThanAsk")
    if ref == 0 or not bid <= ref <= ask:
        raise MathRevert("ReferencePriceOutOfRange")
    m = state.bid_ask_ref_asymmetry_x8
    if m != ASYMMETRY_CHECK_DISABLED:
        # r1 = ref/bid - 1, r2 = ask/ref - 1, M = 1 + m/256: 1/M <= r1/r2 <= M (a reference on bid or ask reverts)
        r1n, r1d = ref - bid, bid
        r2n, r2d = ask - ref, ref
        if r1n == 0 or r2n == 0:
            raise MathRevert("BidAskRefAsymmetryOutOfBounds")
        if r1n * r2d * 256 > r2n * r1d * (256 + m) or r2n * r1d * 256 > r1n * r2d * (256 + m):
            raise MathRevert("BidAskRefAsymmetryOutOfBounds")


def _buy_multiplier(state: MetricOmmPoolState) -> int:
    return _cdiv(state.ask_x64 * Q64, state.reference_x64)


def _sell_divisor(state: MetricOmmPoolState) -> int:
    return _cdiv(state.reference_x64 * Q64, state.bid_x64)


def _reference_at(ref: int, dist_e6: int) -> int:
    if dist_e6 < -E6:
        raise MathRevert("InvalidDistance")
    return ref * (E6 + dist_e6) // E6


def ask_edge_price(state: MetricOmmPoolState, dist_e6: int) -> int:
    """Buy price (scaled token1 per scaled token0, Q64.64) at a bin edge ``dist_e6`` from the reference."""
    return _cdiv(_reference_at(state.reference_x64, dist_e6) * _buy_multiplier(state), Q64)


def bid_edge_price(state: MetricOmmPoolState, dist_e6: int) -> int:
    """Sell price at a bin edge ``dist_e6`` from the reference."""
    return _reference_at(state.reference_x64, dist_e6) * Q64 // _sell_divisor(state)


def bin_lower_dist(state: MetricOmmPoolState, i: int) -> int:
    """Distance (E6) of bin ``i``'s lower edge from the reference, walked from the current bin over loaded lengths."""
    lo, hi = state.loaded_range
    if not lo <= i <= hi:
        raise InsufficientState(f"bin {i} outside the loaded range {state.loaded_range}")
    d = state.cur_bin_dist_e6
    if i >= state.cur_bin:
        for j in range(state.cur_bin, i):
            d += state.bins[j][2]
    else:
        for j in range(i, state.cur_bin):
            d -= state.bins[j][2]
    return d


# ----------------------------------------------------------------------------- per-bin math (scaled units)
def _ask_price_at(lo: int, hi: int, pos: int) -> int:
    return lo + _cdiv((hi - lo) * pos, MAX_POS)


def _bid_price_at(lo: int, hi: int, pos: int) -> int:
    return lo + (hi - lo) * pos // MAX_POS


def ask_bin_cost(o: int, bal: int, pos: int, lo: int, hi: int) -> int:
    """Scaled token1 the pool charges for ``o`` scaled token0 out of a bin holding ``bal`` above position ``pos``."""
    p0 = _ask_price_at(lo, hi, pos)
    p1 = _ask_price_at(lo, hi, pos + _cdiv((MAX_POS - pos) * o, bal))
    return _cdiv(_cdiv(o * p0, Q64) + _cdiv(o * p1, Q64), 2)


def bid_bin_cost(o: int, bal: int, pos: int, lo: int, hi: int) -> int:
    """Scaled token0 the pool charges for ``o`` scaled token1 out of a bin holding ``bal`` below position ``pos``."""
    p0 = _bid_price_at(lo, hi, pos)
    p1 = _bid_price_at(lo, hi, pos - _cdiv(pos * o, bal))
    if p0 == 0 or p1 == 0:
        raise MathRevert("zero sell price")
    return _cdiv(_cdiv(o * Q64, p0) + _cdiv(o * Q64, p1), 2)


def _ask_root_x64(r: int, bal: int, pos: int, lo: int, hi: int) -> tuple[int, int]:
    """Bounds (lower, upper) of 2^64 * the exact output for input ``r``: o*P0 + (hi-P0)*o^2/(2*bal) = 2^64 * r."""
    p0 = _ask_price_at(lo, hi, pos)
    s = bal * p0
    disc = s * s + 2 * Q64 * bal * (hi - p0) * r
    num = 2 * Q64 * r * bal * Q64
    return num // (s + _csqrt(disc)), _cdiv(num, s + isqrt(disc))


def _bid_root_x64(r: int, bal: int, pos: int, lo: int, hi: int) -> tuple[int, int]:
    """Bounds of 2^64 * the exact output for input ``r`` on a bid bin (trapezoid in 1/price, price linear in output).

    With P0 the opening price and k = (P0 - lo) / bal: r = o * 2^64 * (2*P0*bal - K*o) / (2*P0*(P0*bal - K*o)),
    K = P0 - lo, i.e. 2^64*K*o^2 - (2*2^64*P0*bal + 2*r*P0*K)*o + 2*r*P0^2*bal = 0 (smaller root)."""
    p0 = _bid_price_at(lo, hi, pos)
    k = p0 - lo
    if k == 0:
        x = r * p0 * Q64
        return x // Q64, _cdiv(x, Q64)
    a = Q64 * k
    b = 2 * Q64 * p0 * bal + 2 * r * p0 * k
    c = 2 * r * p0 * p0 * bal
    disc = b * b - 4 * a * c
    num = 2 * c * Q64
    return num // (b + _csqrt(disc)), _cdiv(num, b + isqrt(disc))


def _band_x64(o_hi_x: int, bal: int, p_lo: int, p_hi: int, input_unit_x: int, solver: bool) -> int:
    """Bound (x 2^64, scaled output units) on the pool's deviation from the exact root inside one bin.

    4 price-grid steps (2*o/P), 2 input units worth of output and, on bids, bal*(span/P)^4/4: the pool's sell-side
    solver was measured to fall short of the exact root by bal*(span/P)^4*f^4*(1-f)/8 on a bin traversed to fraction
    f, taken here with a factor-two margin. The buy side matched the exact root within the grid terms."""
    o = _cdiv(o_hi_x, Q64)
    grid = _cdiv(2 * o * Q64, p_lo)
    term = _cdiv(bal * (p_hi - p_lo) ** 4 * Q64, 4 * p_lo ** 4) if solver else 0
    return grid + 2 * input_unit_x + term


# ----------------------------------------------------------------------------- quotes
def _preflight(state: MetricOmmPoolState, amount: int, ignore_pause: bool) -> None:
    if amount <= 0:
        raise MathRevert("InvalidAmount")
    if state.pause_level and not ignore_pause:
        raise UnsupportedPool(f"pool paused (pauseLevel {state.pause_level})")
    if state.notional_fee_e8:
        raise UnsupportedPool("non-zero notional fee is not modelled")
    check_prices(state)


def _bin_or_raise(state: MetricOmmPoolState, i: int) -> Bin:
    lo, hi = state.loaded_range
    if not lo <= i <= hi:
        raise InsufficientState(f"swap needs bin {i}, loaded range is {state.loaded_range}")
    return state.bins[i]


def quote_exact_output(state: MetricOmmPoolState, zero_for_one: bool, amount_out: int, *,
                       ignore_pause: bool = False) -> MetricSwapResult:
    """Equivalent of ``MetricOmmSwapQuoter.quoteHypotheticalExactOutputSingle`` (open price limit, no extensions
    rejecting). ``zero_for_one``: sell token0 for token1 (walks bids down); otherwise buy token0 (walks asks up)."""
    _preflight(state, amount_out, ignore_pause)
    m_in, m_out = (state.token0_scale, state.token1_scale) if zero_for_one else (state.token1_scale, state.token0_scale)
    rem = amount_out * m_out
    total_in = total_out = 0
    i, pos = state.cur_bin, state.pos_in_bin
    dist = state.cur_bin_dist_e6
    end = state.cur_bin
    while rem > 0:
        b = _bin_or_raise(state, i)
        if (b[4] if zero_for_one else b[3]) != 0:
            raise UnsupportedPool(f"bin {i} has an additional fee (not modelled)")
        bal = b[1] if zero_for_one else b[0]
        if bal > 0:
            end = i
            if zero_for_one:
                lo, hi = bid_edge_price(state, dist), bid_edge_price(state, dist + b[2])
                take = min(rem, bal)
                total_in += bid_bin_cost(take, bal, pos, lo, hi)
            else:
                lo, hi = ask_edge_price(state, dist), ask_edge_price(state, dist + b[2])
                take = min(rem, bal)
                total_in += ask_bin_cost(take, bal, pos, lo, hi)
            total_out += take
            rem -= take
            if rem == 0:
                break
        if zero_for_one:
            if i == state.lowest_bin:
                break
            i -= 1
            dist -= _bin_or_raise(state, i)[2]
            pos = MAX_POS
        else:
            if i == state.highest_bin:
                break
            dist += b[2]
            i += 1
            pos = 0
    if rem == 0 and amount_out * m_out > MAX_UINT128:  # checked after capping to the pool's liquidity
        raise MathRevert("AmountScaledExceedsMax")
    out_native = total_out // m_out
    in_native = _cdiv(total_in, m_in)
    if in_native == 0 or out_native == 0:
        raise MathRevert("FinalAmountIsZero")
    return MetricSwapResult(in_native, out_native, rem > 0, end)


def quote_exact_input(state: MetricOmmPoolState, zero_for_one: bool, amount_in: int, *,
                      ignore_pause: bool = False) -> MetricSwapResult:
    """Equivalent of ``MetricOmmSwapQuoter.quoteHypotheticalExactInputSingle`` (open price limit).

    Raises ``UnsupportedPool`` when the pool's in-bin rounding (not reproduced bit for bit) could change the native
    output; see the module docstring."""
    _preflight(state, amount_in, ignore_pause)
    m_in, m_out = (state.token0_scale, state.token1_scale) if zero_for_one else (state.token1_scale, state.token0_scale)
    if amount_in * m_in > MAX_UINT128:
        raise MathRevert("AmountScaledExceedsMax")
    rem = amount_in * m_in
    total_in = 0
    full_out = 0
    part_lo_x = part_hi_x = band_x = 0
    i, pos = state.cur_bin, state.pos_in_bin
    dist = state.cur_bin_dist_e6
    end = state.cur_bin
    while rem > 0:
        b = _bin_or_raise(state, i)
        if (b[4] if zero_for_one else b[3]) != 0:
            raise UnsupportedPool(f"bin {i} has an additional fee (not modelled)")
        bal = b[1] if zero_for_one else b[0]
        if bal > 0:
            end = i
            if zero_for_one:
                lo, hi = bid_edge_price(state, dist), bid_edge_price(state, dist + b[2])
                full = bid_bin_cost(bal, bal, pos, lo, hi)
            else:
                lo, hi = ask_edge_price(state, dist), ask_edge_price(state, dist + b[2])
                full = ask_bin_cost(bal, bal, pos, lo, hi)
            if rem < full:
                if zero_for_one:
                    part_lo_x, part_hi_x = _bid_root_x64(rem, bal, pos, lo, hi)
                    p0 = _bid_price_at(lo, hi, pos)
                    band_x = _band_x64(part_hi_x, bal, lo, p0, p0, True)  # one scaled token0 unit buys p0/2^64 token1
                else:
                    part_lo_x, part_hi_x = _ask_root_x64(rem, bal, pos, lo, hi)
                    p0 = _ask_price_at(lo, hi, pos)
                    band_x = _band_x64(part_hi_x, bal, p0, hi, _cdiv(Q64 * Q64, p0), False)  # one token1 unit: 2^64/p0
                part_hi_x = min(part_hi_x, bal * Q64)
                if part_hi_x + band_x < Q64:  # the remainder buys nothing here: the pool leaves it unspent
                    part_lo_x = part_hi_x = band_x = 0
                    break
                if part_lo_x - band_x < Q64:
                    if full_out == 0 and part_hi_x + band_x < m_out * Q64:
                        raise MathRevert("FinalAmountIsZero")  # spent or not, nothing comes out in native units
                    raise UnsupportedPool("exact-input remainder is at the edge of buying one scaled unit")
                total_in += rem
                rem = 0
                break
            total_in += full
            full_out += bal
            rem -= full
            if rem == 0:
                break
        if zero_for_one:
            if i == state.lowest_bin:
                break
            i -= 1
            dist -= _bin_or_raise(state, i)[2]
            pos = MAX_POS
        else:
            if i == state.highest_bin:
                break
            dist += b[2]
            i += 1
            pos = 0
    partial = rem > 0
    unit = m_out * Q64
    lo_x = full_out * Q64 + part_lo_x - band_x
    hi_x = full_out * Q64 + part_hi_x + band_x
    out_native = (full_out * Q64 + part_lo_x) // unit
    if max(lo_x, 0) // unit != hi_x // unit:
        raise UnsupportedPool("exact-input result is within the pool's unmodelled in-bin rounding of a native unit")
    in_native = amount_in if not partial else _cdiv(total_in, m_in)
    if in_native == 0 or out_native == 0:
        raise MathRevert("FinalAmountIsZero")
    return MetricSwapResult(in_native, out_native, partial, end)
