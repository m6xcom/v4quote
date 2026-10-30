"""Tick bitmap snapshot + port of TickBitmap.nextInitializedTickWithinOneWord.

The swap loop in v3/v4 advances at most one bitmap word (256 * tickSpacing
ticks) per step, even when no tick is initialised in that word. Because
every step rounds independently, the number of steps affects the result at
the wei level, so we reproduce the word-bounded walk exactly instead of
jumping straight to the next initialised tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from v4quote.errors import InsufficientState
from v4quote.math.bit_math import least_significant_bit, most_significant_bit
from v4quote.math.uint import MAX_UINT256
from v4quote.amm.v3.tick_math import MAX_TICK, MIN_TICK


def compress(tick: int, tick_spacing: int) -> int:
    """tick / tickSpacing rounded towards negative infinity."""
    return tick // tick_spacing


def position(compressed: int) -> tuple[int, int]:
    """(wordPos, bitPos) of a compressed tick: int16(compressed >> 8), uint8(compressed % 256)."""
    return compressed >> 8, compressed & 0xFF


@dataclass(frozen=True)
class TickBitmapSnapshot:
    """Initialised ticks of a pool, as loaded from chain.

    ``words`` maps wordPos -> 256-bit word (missing = 0).
    ``liquidity_net`` maps initialised tick -> liquidityNet (int128).
    ``word_range`` is the inclusive (min_word, max_word) range that was actually
    loaded; ``None`` means the snapshot is complete (used for offline tests
    and for pools that were fully scanned).
    """

    tick_spacing: int
    words: Mapping[int, int] = field(default_factory=dict)
    liquidity_net: Mapping[int, int] = field(default_factory=dict)
    word_range: tuple[int, int] | None = None

    @classmethod
    def from_ticks(
        cls,
        tick_spacing: int,
        liquidity_net: Mapping[int, int],
        word_range: tuple[int, int] | None = None,
    ) -> "TickBitmapSnapshot":
        words: dict[int, int] = {}
        for tick in liquidity_net:
            if tick % tick_spacing != 0:
                raise ValueError(f"tick {tick} not a multiple of spacing {tick_spacing}")
            if not (MIN_TICK <= tick <= MAX_TICK):
                raise ValueError(f"tick {tick} out of range")
            w, b = position(compress(tick, tick_spacing))
            words[w] = words.get(w, 0) | (1 << b)
        return cls(tick_spacing, words, dict(liquidity_net), word_range)

    def word(self, word_pos: int) -> int:
        if self.word_range is not None and not (self.word_range[0] <= word_pos <= self.word_range[1]):
            raise InsufficientState(
                f"tick bitmap word {word_pos} outside loaded range {self.word_range}"
            )
        return self.words.get(word_pos, 0)

    def get_liquidity_net(self, tick: int) -> int:
        try:
            return self.liquidity_net[tick]
        except KeyError:
            raise InsufficientState(f"liquidityNet for initialised tick {tick} not loaded") from None

    def next_initialized_tick_within_one_word(self, tick: int, lte: bool) -> tuple[int, bool]:
        ts = self.tick_spacing
        compressed = compress(tick, ts)
        if lte:
            word_pos, bit_pos = position(compressed)
            mask = (1 << bit_pos) - 1 + (1 << bit_pos)
            masked = self.word(word_pos) & mask
            initialized = masked != 0
            if initialized:
                nxt = (compressed - (bit_pos - most_significant_bit(masked))) * ts
            else:
                nxt = (compressed - bit_pos) * ts
        else:
            compressed += 1
            word_pos, bit_pos = position(compressed)
            mask = ~((1 << bit_pos) - 1) & MAX_UINT256
            masked = self.word(word_pos) & mask
            initialized = masked != 0
            if initialized:
                nxt = (compressed + (least_significant_bit(masked) - bit_pos)) * ts
            else:
                nxt = (compressed + (255 - bit_pos)) * ts
        return nxt, initialized
