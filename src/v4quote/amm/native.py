"""Optional native (Rust) backend of the hot AMM math: dispatch layer.

``v4quote_native`` (crate ``native/``, an optional install) is a bit-exact port of the exact-input quotes the amount
search uses: Uniswap v2 ``get_amount_out``, Uniswap v3 ``quote_exact_input`` (tick-bitmap walk, ``want_tick=False``)
and the OZ v5 ERC-4626 vault
``deposit_out`` / ``redeem_out`` (``v4quote.amm.erc4626``). It is used when it imports, was built
from the repository's current ``native/`` sources and the environment variable ``V4QUOTE_NATIVE`` is not ``"0"`` (the
kill switch); otherwise every call is the Python code, unchanged. ``backend()`` says which one is active ("native" /
"python"); it is logged once, and ``describe()`` (health records) carries the reason when it is "python".

Why "python" (``reason``, first word = code): ``disabled`` (``V4QUOTE_NATIVE=0``), ``not_installed`` (no ``v4quote_native``
module: the ``native`` extra was not synced), ``broken_build`` (the module exists but its import raised: e.g. a
partial build or an ABI mismatch), ``stale_build`` (the module was built from other sources than
the repository's ``native/`` - after a ``git pull`` without a rebuild it would compute an older version of the math;
``v4quote_native.__source_hash__`` is compared with ``source_hash(<repo>/native)``, the same SHA-256 ``native/build.rs``
embeds). ``backend()`` logs the last three as WARNING with the error text. Without a ``native/`` next to this
package (an installed, non-repository ``core``) the source check is skipped (``build_check`` says so).

Exactness contract (the reason this is safe to switch on): a native call either returns Python's value, raises
Python's error (``MathRevert`` / ``InsufficientState`` with the same message), or reports ``Fallback`` (a value
outside its fixed-width domain, an intermediate Python would carry on with, a Python ``ZeroDivisionError`` ...),
in which case the Python implementation computes that call. Fallbacks are counted (``describe()``), never silent.

Handles: a pool state is converted once into a native handle and reused for every quote on it (that is where the
speedup comes from). The cache is keyed by the identity of the (frozen) state object, verified on every hit (the
cached weak reference must still point at this very object), and dropped when the state is garbage-collected;
``dataclasses.replace`` makes a new state, hence a new handle. The v3 tick data is cached the same way per
``TickBitmapSnapshot`` (states replaced by a swap share it). A state the native code does not cover (other kinds,
subclasses, values outside its domain) gets no handle: Python path (counted: ``no_handle``).

Uniswap v4 (opt-in, ``enable_v4()``; off by default): a ``V4PoolState`` is
quoted by the v3 kernel with the direction's swap fee (LP fee or the fee-only hook's fee, combined with the protocol
fee exactly as ``V4PoolState.swap_fee``), one handle per direction. Exact-input amounts are identical to
``uniswap_v4.quote_exact_input``: both step functions compute the same next price and output for a fee below
1_000_000 pips, and the step consumes the whole remaining input in both conventions. Only the reported ``fee``
of a last, partial step follows the v3 convention (it can be a few wei apart; informational, never used for amounts
or gas). Covered only when ``uniswap_v4.check_supported`` passes for the state (as the Python quote requires) and
both fees are below 1_000_000; anything else stays Python. ``quote()`` never serves v4 (its v3 semantics differ).
A modelled return-delta hook (``V4PoolState.rd_hook``): the handle is the pool's own swap math and ``hop`` applies the hook's
take around it exactly as ``uniswap_v4.quote_exact_input`` (input-side take before, output-side take after); a kind
that needs the post-swap tick (``RdHook.needs_tick``, the range check) is left to Python, and ``path`` (one native
call over the whole path, no per-hop take) returns None for a path with any rd-hooked state (hop by hop instead).

RULE: pool states and their snapshots are never mutated in place. They are frozen dataclasses,
but some fields are plain mappings (``TickBitmapSnapshot.words`` / ``liquidity_net``); writing into such a mapping
after a handle was built would leave the handle with the old values (the identity check cannot see it). Every change
is a new object (``dataclasses.replace``, a new ``TickBitmapSnapshot`` - which is what the loaders, the serializer and
the swap code do); ``tests/unit/test_native_dispatch.py`` keeps the source tree free of in-place writes to them.
"""
from __future__ import annotations

import hashlib
import logging
import os
import weakref
from pathlib import Path
from typing import Any

from v4quote.amm.erc4626 import ERC4626VaultState
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.uniswap_v4 import V4PoolState
from v4quote.amm.v3.pool import V3PoolState
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.errors import InsufficientState, MathRevert

LOG = logging.getLogger("v4quote.amm.native")

ENV = "V4QUOTE_NATIVE"
# <repo>/native for a repository checkout (src/v4quote/amm/native.py -> <repo>)
SOURCE_DIR = Path(__file__).resolve().parents[3].joinpath("native")  # no "/": test_no_float


def source_hash(root: Path) -> str | None:
    """SHA-256 of the crate sources at ``root``, exactly as ``native/build.rs`` computes it (``Cargo.toml``,
    ``Cargo.lock``, ``build.rs``, every ``src/**/*.rs``; sorted ``/``-paths; per file ``path\\nlen\\ncontent\\n`` with
    CRLF -> LF). None when ``root`` is not a complete crate checkout (nothing to compare with)."""
    try:
        src = root.joinpath("src")
        if not src.is_dir():
            return None
        files = ["Cargo.toml", "Cargo.lock", "build.rs"]
        files += [p.relative_to(root).as_posix() for p in src.rglob("*.rs") if p.is_file()]
        h = hashlib.sha256()
        for f in sorted(files):
            data = root.joinpath(f).read_bytes().replace(b"\r\n", b"\n")
            h.update(f.encode() + b"\n" + str(len(data)).encode() + b"\n" + data + b"\n")
        return h.hexdigest()
    except OSError:
        return None


def check_build(mod: Any, root: Path = SOURCE_DIR) -> tuple[str | None, str]:
    """``(reason, build_check)`` for an imported extension module: reason None = usable, else ``"stale_build: ..."``;
    ``build_check`` = "ok" / "skipped (...)" / "mismatch"."""
    built = getattr(mod, "__source_hash__", None)
    if not isinstance(built, str):
        return ("stale_build: v4quote_native has no __source_hash__ (built before the build fingerprint): rebuild "
                "(uv sync --locked --extra native)", "mismatch")
    repo = source_hash(root)
    if repo is None:
        return None, f"skipped (no crate sources at {root})"
    if repo != built:
        return (f"stale_build: v4quote_native was built from sources {built[:12]}, the repository's native is "
                f"{repo[:12]}: rebuild (uv sync --locked --extra native) and restart", "mismatch")
    return None, "ok"


def _load() -> tuple[Any, str, str | None]:
    """``(module or None, reason, build_check)``; reason "" when the native backend is in use."""
    if os.environ.get(ENV, "1").strip() == "0":
        return None, f"disabled: {ENV}=0", None
    try:
        import v4quote_native as mod  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        if e.name == "v4quote_native":
            return None, "not_installed: no module v4quote_native (uv sync --locked --extra native)", None
        return None, f"broken_build: {type(e).__name__}: {str(e)[:200]}", None
    except Exception as e:  # noqa: BLE001  a partial / foreign build (ImportError, OSError ...): Python backend
        return None, f"broken_build: {type(e).__name__}: {str(e)[:200]}", None
    reason, check = check_build(mod)
    if reason is not None:
        return None, reason, check
    return mod, "", check


_cn, _reason, BUILD_CHECK = _load()  # logged by backend() (WARNING unless disabled on purpose)

ACTIVE: bool = _cn is not None
VERSION: str | None = getattr(_cn, "__version__", None)
BUILD: str | None = getattr(_cn, "__build__", None)
STOP_REVERT, STOP_INSUFFICIENT, STOP_FALLBACK = 1, 2, 3

# handles / no_handle: native handles built (cache misses) / states left to Python (not covered or outside the native
# domain); fallbacks / path_fallbacks: native calls / path evaluations that reported Fallback (Python computed them)
_stats = {"handles": 0, "no_handle": 0, "fallbacks": 0, "path_fallbacks": 0, "rd_python": 0, "rd_paths": 0}
_announced = False
_V4 = False  # enable_v4(): V4PoolState quoted by the v3 kernel (see the module doc); off = Python, as before
MAX_SWAP_FEE = 1_000_000


def enable_v4(on: bool = True) -> bool:
    """Opt in (or out) of native v4 quotes for this process; returns whether they are now in use (False when the
    backend is off). Cached v4 entries are dropped so the switch takes effect for every state."""
    global _V4
    _V4 = bool(on)
    for k, e in list(_by_id.items()):
        if type(e[0]()) is V4PoolState:
            _by_id.pop(k, None)
    return _V4 and ACTIVE


def v4_enabled() -> bool:
    return _V4 and ACTIVE


def backend() -> str:
    """"native" when the Rust extension is in use, else "python". Logs the choice once per process (WARNING when the
    native extension is installed but could not be used)."""
    global _announced
    name = "native" if ACTIVE else "python"
    if not _announced:
        _announced = True
        if ACTIVE:
            LOG.info("v4quote math backend: native (v4quote_native %s, build check %s; kill switch %s=0)", BUILD or VERSION,
                     BUILD_CHECK, ENV)
        elif _reason.startswith(("disabled", "not_installed")):
            LOG.info("v4quote math backend: python (%s)", _reason)
        else:
            LOG.warning("v4quote math backend: python (%s)", _reason)
    return name


def describe() -> dict:
    """``math_backend``, the reason when it is "python", the build and the counters, for health records."""
    return {"math_backend": backend(), "v4quote_native": VERSION, "build": BUILD, "build_check": BUILD_CHECK,
            "reason": _reason or None, "v4": v4_enabled(), **_stats}


# ------------------------------------------------------------------------------------------------ handles
# id(obj) -> (weakref to obj, handle or None). The weakref callback removes the entry before the id can be reused.
_by_id: dict[int, tuple[Any, Any]] = {}
_ticks_by_id: dict[int, tuple[Any, Any]] = {}
# V2PairState has __slots__ without __weakref__: cache by value instead (bounded)
_v2_by_value: dict[tuple, Any] = {}
_V2_CACHE_MAX = 4096
_MISS = object()


def _forget(table: dict, key: int, ref: Any) -> None:
    e = table.get(key)
    if e is not None and e[0] is ref:
        del table[key]


def _remember(table: dict, obj: Any, value: Any) -> None:
    k = id(obj)
    ref = weakref.ref(obj, lambda r, k=k, t=table: _forget(t, k, r))
    table[k] = (ref, value)


def _ticks_handle(t: TickBitmapSnapshot, cn: Any) -> Any:
    if type(t) is not TickBitmapSnapshot:
        return None
    try:
        wr = t.word_range
        rng = None if wr is None else (wr[0], wr[1])
        return cn.V3Ticks(t.tick_spacing, list(t.words.items()), list(t.liquidity_net.items()), rng, repr(wr))
    except Exception:  # noqa: BLE001  outside the native domain (huge / negative values, odd types): Python path
        return None


def _ticks(t: TickBitmapSnapshot) -> Any:
    e = _ticks_by_id.get(id(t))
    if e is not None and e[0]() is t:  # the entry is this very object (never a recycled id)
        return e[1]
    h = _ticks_handle(t, _cn)
    _remember(_ticks_by_id, t, h)
    return h


def _v4_handles(state: V4PoolState, cn: Any, cached_ticks: bool) -> tuple[Any, Any] | None:
    """(handle zeroForOne, handle oneForZero): the v3 kernel with each direction's swap fee, or None (not covered)."""
    if state.unsupported_reasons():  # exactly the check uniswap_v4.swap applies (it raises there: Python path)
        return None
    if state.ticks.tick_spacing != state.key.tick_spacing:
        return None
    f0, f1 = state.swap_fee(True), state.swap_fee(False)
    if not (0 <= f0 < MAX_SWAP_FEE and 0 <= f1 < MAX_SWAP_FEE):
        return None
    tk = _ticks(state.ticks) if (cached_ticks and cn is _cn) else _ticks_handle(state.ticks, cn)
    if tk is None:
        return None
    h0 = cn.Pool.v3(tk, state.sqrt_price_x96, state.tick, state.liquidity, f0)
    h1 = h0 if f1 == f0 else cn.Pool.v3(tk, state.sqrt_price_x96, state.tick, state.liquidity, f1)
    return h0, h1


def build_handle(state: Any, cn: Any = None, *, cached_ticks: bool = True, v4: bool | None = None) -> Any:
    """A new native handle for ``state`` with the extension module ``cn`` (default: the active one), uncached; None
    when the state is not covered. Raises (ValueError / OverflowError / TypeError) for values outside the native
    domain; ``handle`` turns that into None. Tests build handles with it even when the backend is switched off.
    A ``V4PoolState`` gives a pair (zeroForOne, oneForZero) when ``v4`` (default: ``enable_v4``) is on."""
    cn = cn or _cn
    v4 = _V4 if v4 is None else v4
    t = type(state)
    if t is V3PoolState:
        tk = _ticks(state.ticks) if (cached_ticks and cn is _cn) else _ticks_handle(state.ticks, cn)
        if tk is None or state.ticks.tick_spacing != state.tick_spacing:
            return None
        return cn.Pool.v3(tk, state.sqrt_price_x96, state.tick, state.liquidity, state.fee)
    if t is V4PoolState:
        return _v4_handles(state, cn, cached_ticks) if v4 else None
    if t is V2PairState:
        return cn.Pool.v2(state.reserve0, state.reserve1, state.fee_numerator, state.fee_denominator)
    if t is ERC4626VaultState:
        make = getattr(cn.Pool, "erc4626", None)  # absent in a build without the vault port: Python path
        if make is None:
            return None
        return make(state.total_assets, state.total_supply, state.decimals_offset)
    return None


def handle(state: Any) -> Any:
    """The native handle of a pool state, or None (backend off, kind not covered, values outside the domain)."""
    if not ACTIVE:
        return None
    if type(state) is V2PairState:
        key = (state.reserve0, state.reserve1, state.fee_numerator, state.fee_denominator)
        h = _v2_by_value.get(key, _MISS)
        if h is _MISS:
            try:
                h = _cn.Pool.v2(*key)
            except Exception:  # noqa: BLE001  outside the native domain: Python path (cached as None)
                h = None
            _stats["handles" if h is not None else "no_handle"] += 1
            if len(_v2_by_value) >= _V2_CACHE_MAX:
                _v2_by_value.clear()
            _v2_by_value[key] = h
        return h
    e = _by_id.get(id(state))
    if e is not None and e[0]() is state:  # the entry is this very object (never a recycled id)
        return e[1]
    try:
        h = build_handle(state)
    except Exception:  # noqa: BLE001  outside the native domain: Python path for this state
        h = None
    _stats["handles" if h is not None else "no_handle"] += 1
    try:
        _remember(_by_id, state, h)
    except TypeError:  # not weak-referenceable: no caching
        pass
    return h


# ------------------------------------------------------------------------------------------------ calls
def hop(state: Any, i: int, j: int, amount: int) -> tuple[int, int | None, int] | None:
    """``v4quote.routing._Search._simulate`` natively: ``(amount_out, fee, crossings)``; raises ``MathRevert`` /
    ``InsufficientState`` exactly like the Python path; None = no handle or fallback (the caller runs Python)."""
    h = handle(state)
    if h is None:
        return None
    if type(h) is tuple:  # v4: the direction's handle (i == 0: zeroForOne)
        if amount >= 1 << 255:  # v4's int256 convention differs there from v3's: Python decides
            return None
        h = h[0 if i == 0 else 1]
        rd = state.rd_hook
        if rd is not None:
            return _rd_hop(h, rd, i, j, amount)
    try:
        return h.hop(i, j, amount)
    except _cn.Revert as e:
        raise MathRevert(str(e)) from None
    except _cn.Insufficient as e:
        raise InsufficientState(str(e)) from None
    except _cn.Fallback:
        _stats["fallbacks"] += 1
        return None


def _rd_hop(h: Any, rd: Any, i: int, j: int, amount: int) -> tuple[int, int | None, int] | None:
    """A v4 hop through a modelled return-delta hook: the pool swap natively, the hook's take as in
    ``uniswap_v4.quote_exact_input``; None = Python (the kind needs the post-swap tick, or a fallback)."""
    if rd.needs_tick:
        _stats["rd_python"] += 1
        return None
    zfo = i == 0
    fee_in = rd.input_fee(zfo, amount)
    x = amount - fee_in
    if x <= 0:
        raise MathRevert("NothingSwapped")
    try:
        out, fee, crossings = h.hop(i, j, x)
    except _cn.Revert as e:
        raise MathRevert(str(e)) from None
    except _cn.Insufficient as e:
        raise InsufficientState(str(e)) from None
    except _cn.Fallback:
        _stats["fallbacks"] += 1
        return None
    return out - rd.output_fee(zfo, out), (None if fee is None else fee + fee_in), crossings


def quote(state: Any, i: int, j: int, amount: int, *, views: bool = False) -> tuple[int, int | None, int, bool] | None:
    """The pool's own exact-input quote, no routing checks: ``(amount_out, fee, crossings, fully_filled)``. v2 / v3:
    ``i == 0`` = token0 -> token1 (v3 is ``quote_exact_input(want_tick=False)``); ERC-4626 ``erc4626.quote`` (0 -> 1 deposit). Errors as in ``hop``; None = run Python."""
    h = handle(state)
    if h is None or type(h) is tuple:  # v4 handles serve hop / path only (v3 quote semantics differ)
        return None
    try:
        return h.quote(i, j, amount, views)
    except _cn.Revert as e:
        raise MathRevert(str(e)) from None
    except _cn.Insufficient as e:
        raise InsufficientState(str(e)) from None
    except _cn.Fallback:
        _stats["fallbacks"] += 1
        return None


def path(hops: list[tuple[Any, int, int]]) -> Any:
    """A native path evaluator over ``[(state, i, j), ...]`` or None when some state has no handle."""
    if not ACTIVE:
        return None
    hs = []
    for st, i, j in hops:
        if type(st) is V4PoolState and st.rd_hook is not None:  # the hook's take is per hop: hop by hop (hop())
            _stats["rd_paths"] += 1
            return None
        h = handle(st)
        if h is None:
            return None
        if type(h) is tuple:  # v4: the direction's handle
            h = h[0 if i == 0 else 1]
        hs.append((h, i, j))
    return _cn.Path(hs)


def note_path_fallback() -> None:
    _stats["path_fallbacks"] += 1


__all__ = ["ACTIVE", "BUILD", "BUILD_CHECK", "ENV", "SOURCE_DIR", "STOP_FALLBACK", "STOP_INSUFFICIENT", "STOP_REVERT",
           "VERSION", "backend", "build_handle", "check_build", "source_hash",
           "describe", "enable_v4", "handle", "hop", "v4_enabled", "note_path_fallback", "path", "quote"]
