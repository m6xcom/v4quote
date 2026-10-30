"""Route pool sets (YAML) and exact pool-state loading for ``v4quote.routing``.

A route pool set is a YAML file with one section per chain (schema of ``tests/fixtures/route_pools.yaml``)::

    <chain>:
      tokens: {SYMBOL: address, ..., NATIVE: 0x000...}
      wrapped_native: SYMBOL
      pools:
        - {id, dex, type, pool | (pool_key + pool_id), tokens: [SYM, ...], <type-specific fields>}

``type`` is a ``v4quote.routing`` pool kind (uniswap_v2 | uniswap_v3 | uniswap_v4 | lfj_lb).
Type-specific fields: v2 ``fee: [num, den]``; v3 ``fee`` or ``tick_spacing`` + ``key_type`` and
``uniswap_fee_protocol``; v4 ``pool_key`` + ``pool_id`` + ``state_view``; LB ``version`` + ``bin_step``.

``load_pool_state`` reads one pool at a pinned block with the core loaders. Tick / bin windows come in widening
levels (``V3_WORDS``, ``LB_BINS``, ``LB_FOLLOW``): callers start at level 0 and reload a pool one level wider when
``best_route`` raises ``RouteInsufficientState`` for it (never a guess beyond the loaded state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from eth_utils import to_checksum_address
from web3 import Web3

from v4quote import routing as rt
from v4quote.amm import uniswap_v4 as v4
from v4quote.chain.loaders import load_lb_pair, load_v2_pair, load_v3_pool, load_v4_pool

NATIVE = "0x0000000000000000000000000000000000000000"
# widening levels (the last one is the whole tick range / a wide bin window with sparse-bin following)
V3_WORDS: tuple[int | None, ...] = (8, 64, None)
LB_BINS: tuple[int, ...] = (100, 600, 3000)
LB_FOLLOW: tuple[int, ...] = (0, 0, 64)
MAX_WIDTH = len(V3_WORDS) - 1
# the kinds ``load_pool_state`` can load: a route-set YAML naming any other kind -
# e.g. ``erc4626`` - is rejected when parsed
LOADABLE_KINDS = (rt.KIND_V2, rt.KIND_V3, rt.KIND_V4, rt.KIND_LB)


@dataclass(frozen=True)
class RoutePoolSpec:
    """One pool of a route set. ``tokens`` are addresses in pool order (native = ``NATIVE``)."""

    id: str
    dex: str
    kind: str
    pool_id: str  # v4: 0x-poolId (lower case); otherwise the checksum pool address
    tokens: tuple[str, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str | None:
        return None if self.kind == rt.KIND_V4 else self.pool_id

    @property
    def pool_key(self) -> v4.PoolKey:
        k = self.fields["pool_key"]
        return v4.PoolKey(k["currency0"], k["currency1"], k["fee"], k["tick_spacing"], k["hooks"])

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)


@dataclass(frozen=True)
class RoutePoolSet:
    chain: str
    tokens: Mapping[str, str]  # symbol -> checksum address
    wrapped_native: str
    pools: tuple[RoutePoolSpec, ...]

    def route_tokens(self, spec: RoutePoolSpec) -> tuple[str, ...]:
        """Pool tokens as routing sees them: lower case, native mapped to the wrapped token."""
        w = self.wrapped_native.lower()
        return tuple(w if t.lower() == NATIVE else t.lower() for t in spec.tokens)

    def by_pool_id(self, pool_id: str) -> RoutePoolSpec:
        low = pool_id.lower()
        for p in self.pools:
            if p.pool_id.lower() == low:
                return p
        raise KeyError(pool_id)

    def candidates(self, token_in: str, token_out: str, max_hops: int = 2) -> list[RoutePoolSpec]:
        """Pools that lie on some direct or 2-hop path token_in -> token_out (what ``best_route`` can use)."""
        w = self.wrapped_native.lower()
        t_in = w if token_in.lower() == NATIVE else token_in.lower()
        t_out = w if token_out.lower() == NATIVE else token_out.lower()
        toks = {p.id: set(self.route_tokens(p)) for p in self.pools}
        out: dict[str, RoutePoolSpec] = {}
        for p in self.pools:
            if {t_in, t_out} <= toks[p.id]:
                out[p.id] = p
        if max_hops >= 2:
            for p1 in self.pools:
                if t_in not in toks[p1.id]:
                    continue
                for mid in toks[p1.id] - {t_in, t_out}:
                    for p2 in self.pools:
                        if p2.id != p1.id and mid in toks[p2.id] and t_out in toks[p2.id]:
                            out[p1.id] = p1
                            out[p2.id] = p2
        return [p for p in self.pools if p.id in out]

    def touches(self, token: str) -> bool:
        t = self.wrapped_native.lower() if token.lower() == NATIVE else token.lower()
        return any(t in self.route_tokens(p) for p in self.pools)


def _addr(tokens: Mapping[str, str], s: str) -> str:
    if s in tokens:
        return tokens[s]
    if isinstance(s, str) and s.startswith("0x") and len(s) == 42:
        return to_checksum_address(s)
    raise ValueError(f"unknown token symbol {s!r}")


def parse_route_pool_set(data: Mapping[str, Any], chain: str) -> RoutePoolSet:
    """Build a RoutePoolSet from the parsed YAML mapping (strict: unknown kinds / symbols raise ValueError)."""
    if chain not in data:
        raise ValueError(f"route pool set has no section {chain!r}")
    sec = data[chain]
    tokens = {k: to_checksum_address(v) for k, v in (sec.get("tokens") or {}).items()}
    wrapped = _addr(tokens, sec["wrapped_native"])
    pools = []
    seen: set[str] = set()
    for p in sec.get("pools") or []:
        kind = p["type"]
        if kind not in rt.POOL_KINDS:
            raise ValueError(f"pool {p.get('id')}: unknown type {kind!r}")
        if kind not in LOADABLE_KINDS:
            raise ValueError(f"pool {p.get('id')}: type {kind!r} has no route-set loader ({', '.join(LOADABLE_KINDS)})")
        toks = tuple(_addr(tokens, t) for t in p["tokens"])
        fields = dict(p)
        if kind == rt.KIND_V4:
            k = dict(p["pool_key"])
            k["currency0"] = _addr(tokens, k["currency0"])
            k["currency1"] = _addr(tokens, k["currency1"])
            k["hooks"] = to_checksum_address(k["hooks"])
            fields["pool_key"] = k
            key = v4.PoolKey(k["currency0"], k["currency1"], k["fee"], k["tick_spacing"], k["hooks"])
            pid = "0x" + key.pool_id().hex()
            if p.get("pool_id") and p["pool_id"].lower() != pid:
                raise ValueError(f"pool {p['id']}: pool_id does not match keccak(pool_key)")
        else:
            pid = to_checksum_address(p["pool"])
        if pid.lower() in seen:
            raise ValueError(f"pool {pid} listed twice")
        seen.add(pid.lower())
        pools.append(RoutePoolSpec(p["id"], p.get("dex", ""), kind, pid, toks, fields))
    return RoutePoolSet(chain, tokens, wrapped, tuple(pools))


def load_route_pool_set(path: str | Path, chain: str) -> RoutePoolSet:
    return parse_route_pool_set(yaml.safe_load(Path(path).read_text(encoding="utf-8")), chain)


def load_pool_state(
    w3: Web3, spec: RoutePoolSpec, block: int, *, width: int = 0, chain: str | None = None
) -> rt.RoutePool:
    """State of ``spec`` at ``block`` for ``best_route`` (``width`` = widening level, 0 = narrowest).

    Raises whatever the loader raises (CallFailed / DecodeFailed / ValueError for an uninitialised v4 pool);
    callers turn that into ``routing.MissingPool``.
    """
    if not 0 <= width <= MAX_WIDTH:
        raise ValueError(f"width must be in 0..{MAX_WIDTH}")
    if spec.kind == rt.KIND_V4:
        st = load_v4_pool(w3, spec.get("state_view"), spec.pool_key, block, words_each_side=V3_WORDS[width])
    elif spec.kind == rt.KIND_V2:
        num, den = spec.get("fee", [997, 1000])
        st = load_v2_pair(w3, spec.pool_id, block, fee_numerator=num, fee_denominator=den)
    elif spec.kind == rt.KIND_V3:
        st = load_v3_pool(w3, spec.pool_id, block, words_each_side=V3_WORDS[width],
                          uniswap_fee_protocol=spec.get("uniswap_fee_protocol", True))
    elif spec.kind == rt.KIND_LB:
        st = load_lb_pair(w3, spec.pool_id, block, version=str(spec.get("version", "2.2")),
                          bins_each_side=LB_BINS[width], chain=chain, follow_sparse_bins=LB_FOLLOW[width])
    else:  # never a silent fallback to another kind's loader
        raise ValueError(f"pool {spec.id}: no route-set loader for kind {spec.kind!r}")
    return rt.RoutePool(st, spec.dex)


def missing_pool(spec: RoutePoolSpec, reason: str) -> rt.MissingPool:
    return rt.MissingPool(spec.pool_id, spec.kind, spec.tokens, reason, spec.dex)


__all__ = [
    "LB_BINS", "LB_FOLLOW", "LOADABLE_KINDS", "MAX_WIDTH", "NATIVE", "RoutePoolSet", "RoutePoolSpec", "V3_WORDS",
    "load_pool_state", "load_route_pool_set", "missing_pool", "parse_route_pool_set",
]
