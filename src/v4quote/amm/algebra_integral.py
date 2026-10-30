"""Algebra Integral (v1.x) concentrated-liquidity pools: the Uniswap v3 swap math with Algebra's tick walk.

What is the same as Uniswap v3 (so ``v4quote.amm.v3.pool`` quotes these pools; no second copy of the math):
* the swap step (``PriceMovementMath.movePriceTowardsTarget`` = ``SwapMath.computeSwapStep``: fee taken from the
  input as ``mulDiv(amount, 1e6 - fee, 1e6)``, the same rounding of the price and the token deltas), TickMath,
  the crossing (``liquidity += liquidityDelta`` upwards, ``-=`` downwards; tick = nextTick - 1 after crossing down);
* the swap ABI ``swap(recipient, zeroToOne, amountRequired, limitSqrtPrice, data)`` (selector 0x128acb08) and the
  callback semantics (``algebraSwapCallback(int256, int256, bytes)``);
* the events ``Swap`` / ``Mint`` / ``Burn`` (same topics as Uniswap v3: post price, liquidity, tick).

What differs (and is modelled here):
* the tick walk: Algebra jumps from one initialised tick to the next (a linked list, ``prevTickGlobal`` /
  ``nextTickGlobal`` / ``ticks(t).nextTick``), it never stops at a 256-tick bitmap word boundary like v3. Every
  step rounds on its own, so the number of steps matters at the wei level: ``AlgebraTickSnapshot`` answers
  ``next_initialized_tick_within_one_word`` with the next initialised tick itself (across words);
* the bitmap ``tickTable(int16)`` holds UNcompressed ticks (word = tick >> 8, bit = tick & 0xFF, whatever the
  tickSpacing): the snapshot always has ``tick_spacing = 1`` (also the state's ``tick_spacing``, the real spacing
  only matters for minting); ``ticks(int24)`` returns (liquidityTotal uint256, liquidityDelta int128, prevTick,
  nextTick, outerFeeGrowth0, outerFeeGrowth1);
* the fee: ``globalState().lastFee`` is NOT necessarily the fee of the next swap. With the plugin's DYNAMIC_FEE
  flag (pluginConfig bit 7) the plugin's ``beforeSwap`` returns an override fee (adaptive fee from the volatility
  oracle) and ``fee()`` of the pool returns ``plugin.getCurrentFee()``; the applied fee is logged per swap by
  ``SwapFee(address indexed sender, uint24 overrideFee, uint24 pluginFee)`` right BEFORE the ``Swap`` of the same
  call (Integral v1.2). The state's ``fee`` is the fee the next swap pays: ``fee()`` at load, then the logged
  ``overrideFee`` (``swap_fee_from_log``) and ``fee()`` re-reads.

Checked on Robinhood Chain (block 100117724): pool 0x49ffa9b5 WETH/USDG of factory 0xf03875b5, plugin
AlgebraUpgradeablePlugin (Sourcify), adaptive fee alphas 0 -> fee = baseFee 100; exact-input round trips simulated on
chain (``eth_simulateV1``) equal ``v3.pool.quote_exact_input`` on a state of ``load_algebra_integral_pool`` to the wei
(fixture ``tests/fixtures/algebra_integral_robinhood.json``).

Not covered: a plugin fee > 0 (``swap_fee_from_log`` returns None: its effect on the step is not verified on chain,
every logged swap of the pool had pluginFee 0; the book's model check then quarantines the pool), an override fee
of 0 (the pool then charges ``lastFee``), ``swapWithPaymentInAdvance``, a plugin that changes amounts.
"""
from __future__ import annotations

from dataclasses import dataclass

from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot, position
from v4quote.amm.v3.tick_math import MAX_TICK, MIN_TICK
from v4quote.math.bit_math import least_significant_bit, most_significant_bit
from v4quote.math.uint import MAX_UINT256

MAX_FEE = 1_000_000  # pips; Algebra rejects an override fee >= 1e6
DYNAMIC_FEE_FLAG = 128  # Plugins.DYNAMIC_FEE (globalState().pluginConfig)
_MIN_WORD, _MAX_WORD = MIN_TICK >> 8, MAX_TICK >> 8


@dataclass(frozen=True)
class AlgebraTickSnapshot(TickBitmapSnapshot):
    """Initialised ticks of an Algebra Integral pool (``tick_spacing`` must be 1: uncompressed ticks).

    ``next_initialized_tick_within_one_word`` returns the next initialised tick in any word (Algebra's linked-list
    walk); beyond the last one MIN_TICK / MAX_TICK (not initialised, as v3's loop clamps). A word outside
    ``word_range`` on the way raises ``InsufficientState`` (never a guess). A subclass: the native v3 kernel (word
    steps) never gets a handle for it (``v4quote.amm.native`` checks the exact type), the Python loop serves it.
    """

    def __post_init__(self) -> None:
        if self.tick_spacing != 1:
            raise ValueError("AlgebraTickSnapshot holds uncompressed ticks: tick_spacing must be 1")

    def next_initialized_tick_within_one_word(self, tick: int, lte: bool) -> tuple[int, bool]:
        if lte:
            wp, bp = position(tick)
            masked = self.word(wp) & ((1 << (bp + 1)) - 1)
            while not masked:
                wp -= 1
                if wp < _MIN_WORD:
                    return MIN_TICK, False
                masked = self.word(wp)
            return (wp << 8) + most_significant_bit(masked), True
        t = tick + 1
        wp, bp = position(t)
        masked = self.word(wp) & (~((1 << bp) - 1) & MAX_UINT256)
        while not masked:
            wp += 1
            if wp > _MAX_WORD:
                return MAX_TICK, False
            masked = self.word(wp)
        return (wp << 8) + least_significant_bit(masked), True


def swap_fee_from_log(override_fee: int, plugin_fee: int) -> int | None:
    """Fee (pips) the swap step charged, from ``SwapFee(sender, overrideFee, pluginFee)``; None when core cannot
    apply it: overrideFee 0 (the pool fell back to ``lastFee``, not in this log), >= 1e6 (no fee), pluginFee != 0
    (not modelled)."""
    if plugin_fee != 0 or not 0 < override_fee < MAX_FEE:
        return None
    return override_fee
