"""Exceptions raised by v4quote.

Math functions mirror Solidity semantics: wherever the on-chain code would
revert, the Python port raises ``MathRevert`` (or a subclass) instead of
returning a value.
"""


class CoreError(Exception):
    """Base class for all v4quote errors."""


class MathRevert(CoreError):
    """The equivalent on-chain computation would revert (overflow, bad input, ...)."""


class UnsupportedPool(CoreError):
    """The pool uses a feature v4quote deliberately does not model (e.g. swap-altering v4 hooks).

    ``skipped`` lists ``(pool_id, reason)`` of every rejected pool when the error covers several pools
    (e.g. by a multi-pool caller); callers attribute the rejection per pool from it instead of parsing the message.
    Empty when the raiser checked a single pool (the message is the reason)."""

    def __init__(self, *args: object, skipped: tuple[tuple[str, str], ...] = ()) -> None:
        super().__init__(*args)
        self.skipped = tuple(skipped)


class InsufficientState(CoreError):
    """The swap walked outside the tick/bin range that was loaded into the state snapshot."""


class ProtocolRevert(MathRevert):
    """A lending-protocol call would revert for a protocol reason (health factor not below threshold,
    dust rule, cool-off, ...). ``reason`` carries the contract's error name."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class UnsupportedVersion(CoreError):
    """The deployed contract revision / configuration is not modelled (never silently approximated).

    ``reason`` is an optional machine-readable tag (e.g. ``hooked``) that
    callers can store as a skip reason; "" when not set.
    """

    def __init__(self, *args: object, reason: str = "") -> None:
        super().__init__(*args)
        self.reason = reason


class UnreadableState(CoreError, RuntimeError):
    """A required on-chain read failed, did not decode, or disagrees with a cross-check at the same block.

    Loaders raise it (or return an explicit ``None`` that the math checks) instead of substituting 0/False or a
    default: the entity must be skipped by the caller, never priced from made-up data.

    ``reason`` is an optional machine-readable tag (e.g. ``oracle_unreadable``); "" when not set.
    """

    def __init__(self, *args: object, reason: str = "") -> None:
        super().__init__(*args)
        self.reason = reason
