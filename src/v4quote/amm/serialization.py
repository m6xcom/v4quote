"""Lossless JSON serialisation of pool-state snapshots (versioned ``schema``; exact integers as decimal strings).

Supported: V2PairState ("v2"), V3PoolState ("v3"), V4PoolState ("v4"), LBPairState ("lb").
``from_json(to_json(s)) == s`` for every valid state, field types included
(tuples stay tuples, mappings come back as dicts).

Format (``schema`` 1): one JSON object ``{"schema": 1, "kind": <pool_kind>, <fields>}``.
* On-chain words (reserves, sqrtPriceX96, liquidity, liquidityNet, bitmap words, bin reserves,
  LB hooksParameters) are canonical decimal strings: no leading zeros, "-" only for
  negatives, never "-0". JSON numbers are never used for them, so no parser can round them through a double.
* Small bounded integers (fee, tick, tickSpacing, binStep, bin ids, word positions, LB fee parameters,
  block_number) are JSON integers; they are checked to be below 2**53.
* Integer-keyed maps are lists of ``[key, value...]`` rows sorted by key.
``to_json`` is canonical (sorted keys, no whitespace), so ``state_hash`` = keccak256 of it identifies a state.
``from_json`` is strict: unknown kind/schema, missing or extra keys, floats, duplicate map keys and
non-canonical integer strings raise ``ValueError``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Union

from eth_utils import keccak

from v4quote.amm.algebra_integral import AlgebraTickSnapshot
from v4quote.amm.liquidity_book.math import FeeParameters
from v4quote.amm.liquidity_book.pair import LBPairState
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.uniswap_v4 import PoolKey, V4PoolState
from v4quote.amm.v3.pool import V3PoolState
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot

SCHEMA = 1
PoolState = Union[V2PairState, V3PoolState, V4PoolState, LBPairState]
POOL_KIND = {
    V2PairState: "v2",
    V3PoolState: "v3",
    V4PoolState: "v4",
    LBPairState: "lb",
}
_SAFE_INT = 1 << 53
_DEC_RE = re.compile(r"-?(0|[1-9][0-9]*)")

_FEE_PARAM_FIELDS = (
    "base_factor", "filter_period", "decay_period", "reduction_factor", "variable_fee_control",
    "protocol_share", "max_volatility_accumulator", "volatility_accumulator", "volatility_reference",
    "id_reference", "time_of_last_update",
)


# ----------------------------------------------------------------------------- scalar codecs
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _big(x: Any, name: str) -> str:
    if not _is_int(x):
        raise ValueError(f"{name}: expected int, got {type(x).__name__}")
    return str(x)


def _big_in(s: Any, name: str) -> int:
    if not isinstance(s, str) or not _DEC_RE.fullmatch(s) or s == "-0":
        raise ValueError(f"{name}: expected canonical decimal string, got {s!r}")
    return int(s)


def _small(x: Any, name: str) -> int:
    if not _is_int(x):
        raise ValueError(f"{name}: expected int, got {type(x).__name__}")
    if not -_SAFE_INT < x < _SAFE_INT:
        raise ValueError(f"{name}: {x} does not fit a JSON-safe integer (|x| < 2**53)")
    return x


_small_in = _small


def _opt_small(x: Any, name: str) -> int | None:
    return None if x is None else _small(x, name)


def _str(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{name}: expected str, got {type(x).__name__}")
    return x


def _opt_str(x: Any, name: str) -> str | None:
    return None if x is None else _str(x, name)


def _bool(x: Any, name: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"{name}: expected bool, got {type(x).__name__}")
    return x


def _list(x: Any, name: str) -> list:
    if not isinstance(x, list):
        raise ValueError(f"{name}: expected list, got {type(x).__name__}")
    return x


def _range_out(r: Any, name: str) -> list[int]:
    if not (isinstance(r, tuple) and len(r) == 2):
        raise ValueError(f"{name}: expected a (lo, hi) tuple")
    return [_small(r[0], name), _small(r[1], name)]


def _range_in(r: Any, name: str) -> tuple[int, int]:
    r = _list(r, name)
    if len(r) != 2:
        raise ValueError(f"{name}: expected [lo, hi]")
    return (_small_in(r[0], name), _small_in(r[1], name))


def _map_out(m: Any, name: str, enc_value: Callable[[Any, str], list]) -> list[list]:
    if not hasattr(m, "items"):
        raise ValueError(f"{name}: expected a mapping")
    return [[_small(k, f"{name} key"), *enc_value(v, f"{name}[{k}]")] for k, v in sorted(m.items())]


def _map_in(rows: Any, name: str, width: int, dec_value: Callable[[list, str], Any]) -> dict:
    out: dict = {}
    prev = None
    for row in _list(rows, name):
        row = _list(row, name)
        if len(row) != width:
            raise ValueError(f"{name}: expected rows of {width} items, got {row!r}")
        k = _small_in(row[0], f"{name} key")
        if prev is not None and k <= prev:
            raise ValueError(f"{name}: keys must be strictly increasing (duplicate or unsorted key {k})")
        prev = k
        out[k] = dec_value(row[1:], f"{name}[{k}]")
    return out


def _check_keys(d: Any, expected: set[str], name: str) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{name}: expected object, got {type(d).__name__}")
    keys = set(d)
    if keys != expected:
        missing, extra = sorted(expected - keys), sorted(keys - expected)
        raise ValueError(f"{name}: missing keys {missing}, unexpected keys {extra}")
    return d


# ----------------------------------------------------------------------------- component codecs
def _ticks_out(t: TickBitmapSnapshot) -> dict:
    if not isinstance(t, TickBitmapSnapshot):
        raise ValueError("ticks: expected TickBitmapSnapshot")
    return {
        "tick_spacing": _small(t.tick_spacing, "ticks.tick_spacing"),
        "words": _map_out(t.words, "ticks.words", lambda v, n: [_big(v, n)]),
        "liquidity_net": _map_out(t.liquidity_net, "ticks.liquidity_net", lambda v, n: [_big(v, n)]),
        "word_range": None if t.word_range is None else _range_out(t.word_range, "ticks.word_range"),
        # Algebra Integral's tick walk (v4quote.amm.algebra_integral); the key is written only for it, so every other
        # snapshot keeps its canonical form (and state_hash)
        **({"walk": "algebra"} if isinstance(t, AlgebraTickSnapshot) else {}),
    }


def _ticks_in(d: Any) -> TickBitmapSnapshot:
    cls = TickBitmapSnapshot
    if isinstance(d, dict) and "walk" in d:
        if d["walk"] != "algebra":
            raise ValueError(f"ticks.walk: unknown {d['walk']!r}")
        d, cls = {k: v for k, v in d.items() if k != "walk"}, AlgebraTickSnapshot
    d = _check_keys(d, {"tick_spacing", "words", "liquidity_net", "word_range"}, "ticks")
    return cls(
        tick_spacing=_small_in(d["tick_spacing"], "ticks.tick_spacing"),
        words=_map_in(d["words"], "ticks.words", 2, lambda v, n: _big_in(v[0], n)),
        liquidity_net=_map_in(d["liquidity_net"], "ticks.liquidity_net", 2, lambda v, n: _big_in(v[0], n)),
        word_range=None if d["word_range"] is None else _range_in(d["word_range"], "ticks.word_range"),
    )


def _fee_params_out(p: FeeParameters) -> dict:
    if not isinstance(p, FeeParameters):
        raise ValueError("params: expected FeeParameters")
    return {f: _small(getattr(p, f), f"params.{f}") for f in _FEE_PARAM_FIELDS}


def _fee_params_in(d: Any) -> FeeParameters:
    d = _check_keys(d, set(_FEE_PARAM_FIELDS), "params")
    return FeeParameters(**{f: _small_in(d[f], f"params.{f}") for f in _FEE_PARAM_FIELDS})


def _bin_out(v: Any, name: str) -> list[str]:
    if not (isinstance(v, tuple) and len(v) == 2):
        raise ValueError(f"{name}: expected (reserveX, reserveY) tuple")
    return [_big(v[0], name), _big(v[1], name)]


def _big_tuple_out(t: Any, name: str) -> list[str]:
    if not isinstance(t, tuple):
        raise ValueError(f"{name}: expected tuple")
    return [_big(x, name) for x in t]


def _str_tuple_out(t: Any, name: str) -> list[str]:
    if not isinstance(t, tuple):
        raise ValueError(f"{name}: expected tuple")
    return [_str(x, name) for x in t]


# ----------------------------------------------------------------------------- per-kind codecs
def _v2_out(s: V2PairState) -> dict:
    return {
        "address": _str(s.address, "address"),
        "token0": _str(s.token0, "token0"),
        "token1": _str(s.token1, "token1"),
        "reserve0": _big(s.reserve0, "reserve0"),
        "reserve1": _big(s.reserve1, "reserve1"),
        "fee_numerator": _small(s.fee_numerator, "fee_numerator"),
        "fee_denominator": _small(s.fee_denominator, "fee_denominator"),
        "block_number": _opt_small(s.block_number, "block_number"),
    }


def _v2_in(d: dict) -> V2PairState:
    return V2PairState(
        address=_str(d["address"], "address"),
        token0=_str(d["token0"], "token0"),
        token1=_str(d["token1"], "token1"),
        reserve0=_big_in(d["reserve0"], "reserve0"),
        reserve1=_big_in(d["reserve1"], "reserve1"),
        fee_numerator=_small_in(d["fee_numerator"], "fee_numerator"),
        fee_denominator=_small_in(d["fee_denominator"], "fee_denominator"),
        block_number=_opt_small(d["block_number"], "block_number"),
    )


def _v3_out(s: V3PoolState) -> dict:
    return {
        "address": _str(s.address, "address"),
        "token0": _str(s.token0, "token0"),
        "token1": _str(s.token1, "token1"),
        "fee": _small(s.fee, "fee"),
        "tick_spacing": _small(s.tick_spacing, "tick_spacing"),
        "sqrt_price_x96": _big(s.sqrt_price_x96, "sqrt_price_x96"),
        "tick": _small(s.tick, "tick"),
        "liquidity": _big(s.liquidity, "liquidity"),
        "ticks": _ticks_out(s.ticks),
        "block_number": _opt_small(s.block_number, "block_number"),
        "fee_protocol": _small(s.fee_protocol, "fee_protocol"),
    }


def _v3_in(d: dict) -> V3PoolState:
    return V3PoolState(
        address=_str(d["address"], "address"),
        token0=_str(d["token0"], "token0"),
        token1=_str(d["token1"], "token1"),
        fee=_small_in(d["fee"], "fee"),
        tick_spacing=_small_in(d["tick_spacing"], "tick_spacing"),
        sqrt_price_x96=_big_in(d["sqrt_price_x96"], "sqrt_price_x96"),
        tick=_small_in(d["tick"], "tick"),
        liquidity=_big_in(d["liquidity"], "liquidity"),
        ticks=_ticks_in(d["ticks"]),
        block_number=_opt_small(d["block_number"], "block_number"),
        fee_protocol=_small_in(d["fee_protocol"], "fee_protocol"),
    )


_KEY_FIELDS = {"currency0", "currency1", "fee", "tick_spacing", "hooks"}


def _v4_out(s: V4PoolState) -> dict:
    k = s.key
    if not isinstance(k, PoolKey):
        raise ValueError("key: expected PoolKey")
    out = {
        "key": {
            "currency0": _str(k.currency0, "key.currency0"),
            "currency1": _str(k.currency1, "key.currency1"),
            "fee": _small(k.fee, "key.fee"),
            "tick_spacing": _small(k.tick_spacing, "key.tick_spacing"),
            "hooks": _str(k.hooks, "key.hooks"),
        },
        "sqrt_price_x96": _big(s.sqrt_price_x96, "sqrt_price_x96"),
        "tick": _small(s.tick, "tick"),
        "protocol_fee": _small(s.protocol_fee, "protocol_fee"),
        "lp_fee": _small(s.lp_fee, "lp_fee"),
        "liquidity": _big(s.liquidity, "liquidity"),
        "ticks": _ticks_out(s.ticks),
        "block_number": _opt_small(s.block_number, "block_number"),
    }
    # written only when set: snapshots of pools without a hook fee model stay byte-identical to before
    if s.hook_fee is not None:
        out["hook_fee"] = [_small(f, "hook_fee") for f in s.hook_fee]
    if s.rd_hook is not None:  # same rule for a return-delta hook model (v4quote.amm.v4_rd_hooks)
        out["rd_hook"] = s.rd_hook.as_dict()
    return out


def _rd_hook_in(x: Any) -> Any:
    if x is None:
        return None
    from v4quote.amm.v4_rd_hooks import RdHook

    return RdHook.from_dict(x)


def _hook_fee_in(x: Any) -> tuple[int, int] | None:
    if x is None:
        return None
    if not isinstance(x, list) or len(x) != 2:
        raise ValueError("hook_fee: expected null or [fee zeroForOne, fee oneForZero]")
    return (_small_in(x[0], "hook_fee"), _small_in(x[1], "hook_fee"))


def _v4_in(d: dict) -> V4PoolState:
    k = _check_keys(d["key"], _KEY_FIELDS, "key")
    key = PoolKey(
        currency0=_str(k["currency0"], "key.currency0"),
        currency1=_str(k["currency1"], "key.currency1"),
        fee=_small_in(k["fee"], "key.fee"),
        tick_spacing=_small_in(k["tick_spacing"], "key.tick_spacing"),
        hooks=_str(k["hooks"], "key.hooks"),
    )
    return V4PoolState(
        key=key,
        sqrt_price_x96=_big_in(d["sqrt_price_x96"], "sqrt_price_x96"),
        tick=_small_in(d["tick"], "tick"),
        protocol_fee=_small_in(d["protocol_fee"], "protocol_fee"),
        lp_fee=_small_in(d["lp_fee"], "lp_fee"),
        liquidity=_big_in(d["liquidity"], "liquidity"),
        ticks=_ticks_in(d["ticks"]),
        block_number=_opt_small(d["block_number"], "block_number"),
        hook_fee=_hook_fee_in(d["hook_fee"]),
        rd_hook=_rd_hook_in(d["rd_hook"]),
    )


def _lb_out(s: LBPairState) -> dict:
    return {
        "address": _str(s.address, "address"),
        "token_x": _str(s.token_x, "token_x"),
        "token_y": _str(s.token_y, "token_y"),
        "bin_step": _small(s.bin_step, "bin_step"),
        "active_id": _small(s.active_id, "active_id"),
        "params": _fee_params_out(s.params),
        "bins": _map_out(s.bins, "bins", _bin_out),
        "loaded_range": _range_out(s.loaded_range, "loaded_range"),
        "exhausted_below": _bool(s.exhausted_below, "exhausted_below"),
        "exhausted_above": _bool(s.exhausted_above, "exhausted_above"),
        "version": _str(s.version, "version"),
        "hooks_parameters": _big(s.hooks_parameters, "hooks_parameters"),
        "block_number": _opt_small(s.block_number, "block_number"),
        "chain": _opt_str(s.chain, "chain"),
        "hooks_codehash": _opt_str(s.hooks_codehash, "hooks_codehash"),
        "hooks_implementation": _opt_str(s.hooks_implementation, "hooks_implementation"),
        "hooks_implementation_codehash": _opt_str(s.hooks_implementation_codehash, "hooks_implementation_codehash"),
        "hooks_extra_parameters": None if s.hooks_extra_parameters is None
        else _big(s.hooks_extra_parameters, "hooks_extra_parameters"),
        "hooks_linked": None if s.hooks_linked is None else _bool(s.hooks_linked, "hooks_linked"),
        "hooks_pair": _opt_str(s.hooks_pair, "hooks_pair"),
    }


def _lb_in(d: dict) -> LBPairState:
    return LBPairState(
        address=_str(d["address"], "address"),
        token_x=_str(d["token_x"], "token_x"),
        token_y=_str(d["token_y"], "token_y"),
        bin_step=_small_in(d["bin_step"], "bin_step"),
        active_id=_small_in(d["active_id"], "active_id"),
        params=_fee_params_in(d["params"]),
        bins=_map_in(d["bins"], "bins", 3, lambda v, n: (_big_in(v[0], n), _big_in(v[1], n))),
        loaded_range=_range_in(d["loaded_range"], "loaded_range"),
        exhausted_below=_bool(d["exhausted_below"], "exhausted_below"),
        exhausted_above=_bool(d["exhausted_above"], "exhausted_above"),
        version=_str(d["version"], "version"),
        hooks_parameters=_big_in(d["hooks_parameters"], "hooks_parameters"),
        block_number=_opt_small(d["block_number"], "block_number"),
        chain=_opt_str(d["chain"], "chain"),
        hooks_codehash=_opt_str(d["hooks_codehash"], "hooks_codehash"),
        hooks_implementation=_opt_str(d["hooks_implementation"], "hooks_implementation"),
        hooks_implementation_codehash=_opt_str(d["hooks_implementation_codehash"], "hooks_implementation_codehash"),
        hooks_extra_parameters=None if d["hooks_extra_parameters"] is None
        else _big_in(d["hooks_extra_parameters"], "hooks_extra_parameters"),
        hooks_linked=None if d["hooks_linked"] is None else _bool(d["hooks_linked"], "hooks_linked"),
        hooks_pair=_opt_str(d["hooks_pair"], "hooks_pair"),
    )


_CODECS: dict[str, tuple[type, Callable[[Any], dict], Callable[[dict], Any]]] = {
    "v2": (V2PairState, _v2_out, _v2_in),
    "v3": (V3PoolState, _v3_out, _v3_in),
    "v4": (V4PoolState, _v4_out, _v4_in),
    "lb": (LBPairState, _lb_out, _lb_in),
}


# ----------------------------------------------------------------------------- public API
def pool_kind(state: PoolState) -> str:
    """'v2' / 'v3' / 'v4' / 'lb' (the stored ``pool_kind``)."""
    try:
        return POOL_KIND[type(state)]
    except KeyError:
        raise ValueError(f"no serialiser for {type(state).__name__}") from None


def to_dict(state: PoolState) -> dict:
    kind = pool_kind(state)
    body = _CODECS[kind][1](state)
    return {"schema": SCHEMA, "kind": kind, **body}


def from_dict(d: Any) -> PoolState:
    if not isinstance(d, dict):
        raise ValueError(f"expected object, got {type(d).__name__}")
    schema, kind = d.get("schema"), d.get("kind")
    if not _is_int(schema) or schema != SCHEMA:
        raise ValueError(f"unsupported schema {schema!r} (expected {SCHEMA})")
    if kind not in _CODECS:
        raise ValueError(f"unknown pool kind {kind!r}")
    fields = {k: v for k, v in d.items() if k not in ("schema", "kind")}
    for k in _ADDED_KEYS.get(kind, ()):  # written since 2026-09-28; older snapshots read them as unknown (None)
        fields.setdefault(k, None)
    _check_keys(fields, set(_EXPECTED_KEYS[kind]), kind)  # exactly the keys the encoder writes
    return _CODECS[kind][2](fields)


_EXPECTED_KEYS = {
    "v2": ("address", "token0", "token1", "reserve0", "reserve1", "fee_numerator", "fee_denominator", "block_number"),
    "v3": ("address", "token0", "token1", "fee", "tick_spacing", "sqrt_price_x96", "tick", "liquidity", "ticks",
           "block_number", "fee_protocol"),
    "v4": ("key", "sqrt_price_x96", "tick", "protocol_fee", "lp_fee", "liquidity", "ticks", "block_number",
           "hook_fee", "rd_hook"),
    "lb": ("address", "token_x", "token_y", "bin_step", "active_id", "params", "bins", "loaded_range",
           "exhausted_below", "exhausted_above", "version", "hooks_parameters", "block_number", "chain",
           "hooks_codehash", "hooks_implementation", "hooks_implementation_codehash", "hooks_extra_parameters",
           "hooks_linked", "hooks_pair"),
}


# keys added to a kind after snapshots were first written (same schema; absent = None = unknown). For an LB pair with
# an LFJ rewarder hook "unknown" keeps the pool UnsupportedPool, never a guess.
_ADDED_KEYS = {"lb": ("hooks_extra_parameters", "hooks_linked", "hooks_pair"),
               # v4: written only when the state carries a hook fee model (2026-10-27) / an rd hook model (10-28)
               "v4": ("hook_fee", "rd_hook")}


def to_json(state: PoolState) -> str:
    """Canonical JSON (sorted keys, no whitespace, ASCII)."""
    return json.dumps(to_dict(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _reject_float(s: str) -> Any:
    raise ValueError(f"floating-point number {s!r} is not allowed in a pool state")


def _reject_constant(s: str) -> Any:
    raise ValueError(f"{s} is not allowed in a pool state")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def from_json(text: str | bytes) -> PoolState:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    d = json.loads(
        text, parse_float=_reject_float, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys
    )
    return from_dict(d)


def state_hash(state: PoolState) -> str:
    """``pool_snapshots.state_hash``: 0x-prefixed keccak256 of ``to_json(state)``."""
    return "0x" + keccak(text=to_json(state)).hex()
