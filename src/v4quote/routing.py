"""Best exact-input swap route over v4quote pool states.

``best_route(token_in, token_out, amount_in, pools, ...)`` prices the sale of ``amount_in`` of ``token_in`` into
``token_out`` on explicit pool snapshots, using only the exact AMM math in ``v4quote.amm`` (Uniswap v2 and forks,
Uniswap v3 and exact forks incl. Pharaoh V3, Uniswap v4 without swap-altering hooks, LFJ Liquidity Book v2.1/v2.2
incl. Pharaoh DLMM, OZ v5 ERC-4626 vaults as
asset <-> shares deposit / redeem). No pool math lives here.

Search space (deterministic, integer only):

* paths: every direct pool ``token_in -> token_out`` and, with ``max_hops=2``, every ``token_in -> X -> token_out``
  through two different pools;
* single route: every path with the full amount;
* split (``split=2``): the ``split_candidates`` best paths by output at ``amount_in // 2`` are paired; only
  pool-disjoint pairs are allowed, so each leg is priced on the same pre-state exactly as it would execute (legs
  sharing a pool would see each other's post-state). The share of leg 1 is searched in basis points: a 5% grid,
  then an integer ternary refinement to 1 bp inside +-5% of the best grid point. Leg amounts are
  ``amount_in * share // 10_000`` and the remainder.

The winner maximises ``amount_out`` or, with ``out_per_gas=(num, den)``, ``amount_out - gas_units * num // den``
(gas priced in token_out wei). Ties: fewer legs, fewer hops, less gas, then enumeration order (pool ids sorted).

Every hop amount is what the pool's own swap/quoter returns for that input (v2 ``getAmountOut``, v3 QuoterV2,
v4 V4Quoter, LB ``getSwapOut``, ERC-4626 ``previewDeposit`` / ``previewRedeem``). A hop that would revert, fill only partially, or pay 0 is
infeasible for that amount.

State problems are never approximated:

* ``InsufficientState`` (a swap walks past the loaded ticks/bins) and ``MissingPool`` (a pool of the set whose
  state could not be loaded) are collected as ``RouteIssue``; with ``on_insufficient="raise"`` (default) the
  search finishes and then raises ``RouteInsufficientState`` listing every affected pool and amount, so the caller
  can reload those pools wider and retry; with ``"report"`` the best route over the known states is returned with
  ``complete=False``;
* pools v4quote does not model (``UnsupportedPool``: v4 swap hooks / dynamic fee, LB hooks outside the allowlist)
  are excluded before the search and listed in ``Route.skipped``.

Gas: ``GasTable`` is a per-pool-kind table (base + per tick/bin crossed + wrap/unwrap of the native token) until
receipts calibrate it. The defaults are assumptions, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence, Union

from v4quote.amm import erc4626 as vault4626
from v4quote.amm import native as _native
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.erc4626 import ERC4626VaultState
from v4quote.amm.liquidity_book.hooks_allowlist import DEFAULT_HOOKS_ALLOWLIST, HooksAllowlist
from v4quote.amm.liquidity_book.pair import LBPairState, _simulate_out
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.uniswap_v4 import V4PoolState
from v4quote.amm.v3 import pool as v3
from v4quote.amm.v3.pool import V3PoolState
from v4quote.errors import CoreError, InsufficientState, MathRevert

NATIVE = "0x0000000000000000000000000000000000000000"
BPS = 10_000

KIND_V2 = "uniswap_v2"
KIND_V3 = "uniswap_v3"
KIND_V4 = "uniswap_v4"
KIND_LB = "lfj_lb"
# ERC-4626 vault (OZ v5): tokens (asset, vault), coin 0 -> 1 = deposit, 1 -> 0 = redeem (v4quote.amm.erc4626)
KIND_ERC4626 = "erc4626"
POOL_KINDS = (KIND_V2, KIND_V3, KIND_V4, KIND_LB, KIND_ERC4626)

PoolState = Union[V2PairState, V3PoolState, V4PoolState, LBPairState, ERC4626VaultState]


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class RoutePool:
    """A pool state with an optional DEX label (e.g. ``pharaoh_v3``), carried into the hops."""

    state: PoolState
    dex: str = ""


@dataclass(frozen=True)
class MissingPool:
    """A pool that belongs to the route set but whose state is not available (load failed, not in cache).

    Any path through it is reported as ``missing_state`` (an InsufficientState issue), never skipped silently.
    """

    pool_id: str
    kind: str
    tokens: tuple[str, ...]
    reason: str = ""
    dex: str = ""


PoolInput = Union[PoolState, RoutePool, MissingPool]


@dataclass(frozen=True)
class GasTable:
    """Swap gas units per hop: ``base[kind] + per_crossing[kind] * crossings`` (+ ``wrap`` for a native leg end).

    ``crossings`` = initialised ticks crossed (v3/v4) or bins used beyond the first (LB).
    ``lb_hook``: extra gas of an LB hop through a pair WITH a hooks contract (its ``beforeSwap`` runs inside ``swap``),
    by the hook's proxy kind from the allowlist entry (``HookApproval.proxy_kind``); key ``"*"`` = any other hooked
    pair (hook not in the allowlist or no proxy kind). Estimate:
    the Pharaoh DLMM rewarder behind a BeaconProxy with ACTIVE rewards costs ~30-50k over the LB base (beacon call +
    delegatecall, ~4 pair views, 3-4 SSTOREs, an event) -> 45k by default; LFJ rewarder clones with inactive rewards
    return early (0 until receipts say otherwise).
    Default values are table assumptions until calibrated from receipts.
    """

    base: Mapping[str, int]
    per_crossing: Mapping[str, int]
    wrap: int = 0
    lb_hook: Mapping[str, int] = field(default_factory=dict)

    def hop(self, kind: str, crossings: int, wraps: int = 0, hook_kind: str | None = None) -> int:
        """``hook_kind``: None = the pool has no hook; otherwise the proxy kind of its hook ("" = unknown)."""
        extra = 0
        if kind == KIND_LB and hook_kind is not None:
            extra = self.lb_hook.get(hook_kind, self.lb_hook.get("*", 0)) if hook_kind else self.lb_hook.get("*", 0)
        return self.base[kind] + self.per_crossing.get(kind, 0) * crossings + self.wrap * wraps + extra


DEFAULT_GAS = GasTable(
    base={KIND_V2: 60_000, KIND_V3: 95_000, KIND_V4: 110_000, KIND_LB: 80_000, KIND_ERC4626: 90_000},
    per_crossing={KIND_V3: 22_000, KIND_V4: 15_000, KIND_LB: 12_000},
    wrap=30_000,
    lb_hook={"eip1967-beacon": 45_000, "lfj-immutable-clone": 0, "*": 45_000},
)


def lb_hook_kind(state: object, hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST) -> str | None:
    """Proxy kind of an LB pair's hook for ``GasTable.hop`` (None = no hook; "" = hooked, kind unknown)."""
    if not isinstance(state, LBPairState) or not state.hooks_address:
        return None
    addr = "0x" + format(state.hooks_address, "040x")
    for (chain, hook), approval in hooks_allowlist.items():
        if chain == state.chain and hook.lower() == addr:
            return approval.proxy_kind or ""
    return ""


# --------------------------------------------------------------------------- outputs
@dataclass(frozen=True)
class Hop:
    pool_id: str
    pool_kind: str
    dex: str
    token_in: str  # route token (native currency shown as the wrapped token when wrapped_native is set)
    token_out: str
    amount_in: int
    amount_out: int
    fee_amount: int | None  # in token_in units
    crossings: int
    gas_units: int
    zero_for_one: bool | None  # v2/v3/v4: token0 -> token1; LB: swapForY; ERC-4626: deposit
    native_in: bool = False  # the pool's input currency is the native token (unwrap before the hop)
    native_out: bool = False  # the pool pays the native token (wrap after the hop)


@dataclass(frozen=True)
class Leg:
    share_bps: int  # nominal share of amount_in (the last leg takes the remainder)
    amount_in: int
    amount_out: int
    hops: tuple[Hop, ...]

    @property
    def gas_units(self) -> int:
        return sum(h.gas_units for h in self.hops)

    @property
    def path_id(self) -> str:
        return ">".join(h.pool_id for h in self.hops)


@dataclass(frozen=True)
class RouteIssue:
    kind: Literal["insufficient_state", "missing_state"]
    pool_id: str
    amount_in: int  # hop input that hit the problem
    detail: str


@dataclass(frozen=True)
class SearchStats:
    pools: int  # pools usable in the graph (after unsupported / irrelevant ones are dropped)
    paths: int  # candidate paths (direct + 2-hop)
    split_pairs: int  # pool-disjoint path pairs whose split share was searched
    path_evals: int  # distinct (path, amount) evaluations
    hop_evals: int  # distinct (pool, direction, amount) swap simulations


@dataclass(frozen=True)
class Route:
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    legs: tuple[Leg, ...]
    gas_units: int
    net_amount_out: int | None  # amount_out - gas priced in token_out (only with out_per_gas)
    issues: tuple[RouteIssue, ...]
    skipped: tuple[tuple[str, str], ...]  # (pool_id, reason) excluded as unsupported
    stats: SearchStats

    @property
    def complete(self) -> bool:
        """False if some candidate could not be priced (InsufficientState / missing state)."""
        return not self.issues

    @property
    def is_split(self) -> bool:
        return len(self.legs) > 1

    @property
    def hops(self) -> tuple[Hop, ...]:
        return tuple(h for leg in self.legs for h in leg.hops)

    def legs_json(self) -> list[dict]:
        """JSON-friendly records of the legs; raw amounts as decimal strings."""
        out = []
        for li, leg in enumerate(self.legs):
            for hi, h in enumerate(leg.hops):
                out.append({
                    "leg": li, "hop": hi, "pool_id": h.pool_id, "pool_kind": h.pool_kind, "dex": h.dex,
                    "token_in": h.token_in, "token_out": h.token_out,
                    "amount_in_raw": str(h.amount_in), "amount_out_raw": str(h.amount_out),
                    "fee_raw": None if h.fee_amount is None else str(h.fee_amount),
                    "share_bps": leg.share_bps, "crossings": h.crossings, "gas_units": h.gas_units,
                })
        return out


class NoRoute(CoreError):
    """No feasible path between the tokens for this amount (no pools, or every candidate reverts)."""


class RouteInsufficientState(InsufficientState):
    """Some candidate could not be priced from the loaded state. ``issues`` lists pool ids and hop amounts."""

    def __init__(self, issues: Sequence[RouteIssue]) -> None:
        pools = sorted({i.pool_id for i in issues})
        super().__init__(f"route search hit insufficient/missing state in {len(pools)} pool(s): {', '.join(pools)}")
        self.issues = tuple(issues)
        self.pool_ids = tuple(pools)


# --------------------------------------------------------------------------- internals
@dataclass(frozen=True)
class _Pool:
    pool_id: str
    kind: str
    dex: str
    tokens: tuple[str, ...]  # canonical (lower-case; native mapped to wrapped_native when given)
    native: tuple[bool, ...]  # token at this index is the native currency
    state: PoolState | None
    missing_reason: str = ""


@dataclass(frozen=True)
class _Edge:
    pool: int  # index into the pool list
    i: int  # token index in
    j: int  # token index out


@dataclass(frozen=True)
class _HopOut:
    amount_out: int
    fee: int | None
    crossings: int


class _LegOut:
    """Output of one path for one input. The ``Hop`` records are built on first use: an amount search evaluates a
    path dozens of times and reads only ``amount_out`` (and ``gas_units``); the records of the winning amount are
    built once. ``steps`` = ((edge, hop input, hop result), ...), the data the records are made of; for a native path
    evaluation (``v4quote.amm.native``) they are expanded from the native per-hop results on first use."""

    __slots__ = ("amount_out", "_steps", "_raw", "_search", "_hops", "_gas")

    def __init__(self, amount_out: int, steps: tuple | None, search: "_Search", raw: tuple | None = None) -> None:
        self.amount_out = amount_out
        self._steps = steps
        self._raw = raw  # (edges, amount_in, v4quote_native.PathResult) when steps is None
        self._search = search
        self._hops: tuple[Hop, ...] | None = None
        self._gas: int | None = None

    @property
    def steps(self) -> tuple:
        if self._steps is None:
            edges, amt, res = self._raw
            st = []
            for e, (out, fee, crossings) in zip(edges, res.steps()):
                st.append((e, amt, _HopOut(out, fee, crossings)))
                amt = out
            self._steps = tuple(st)
            self._raw = None
        return self._steps

    @property
    def hops(self) -> tuple[Hop, ...]:
        if self._hops is None:
            self._hops = tuple(self._search._hop_record(e, amt, r) for e, amt, r in self.steps)
        return self._hops

    @property
    def gas_units(self) -> int:
        if self._gas is None:
            self._gas = sum(self._search._hop_gas(e, r) for e, _amt, r in self.steps)
        return self._gas


def _canon(addr: str) -> str:
    return addr.lower()


def _kind_of(state: object) -> str:
    if isinstance(state, V2PairState):
        return KIND_V2
    if isinstance(state, V3PoolState):
        return KIND_V3
    if isinstance(state, V4PoolState):
        return KIND_V4
    if isinstance(state, LBPairState):
        return KIND_LB
    if isinstance(state, ERC4626VaultState):
        return KIND_ERC4626
    raise TypeError(f"unsupported pool state type {type(state).__name__}")


def _raw_tokens(state: PoolState) -> tuple[str, ...]:
    if isinstance(state, (V2PairState, V3PoolState)):
        return (state.token0, state.token1)
    if isinstance(state, V4PoolState):
        return (state.key.currency0, state.key.currency1)
    if isinstance(state, LBPairState):
        return (state.token_x, state.token_y)
    return tuple(state.coins)


def _pool_id(state: PoolState) -> str:
    if isinstance(state, V4PoolState):
        return "0x" + state.key.pool_id().hex()
    return state.address


def unsupported_reasons(
    state: PoolState | RoutePool,
    hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST,
    *,
    allow_v4_before_swap_hook: bool = False,
) -> list[str]:
    """Why ``best_route`` would reject this loaded pool (empty = modelled). The one rule set of
    v4quote: v4 swap hooks / dynamic fee (``uniswap_v4.unsupported_reasons``), LB hooks outside the allowlist or failing
    its pinned facts (``LBPairState.unsupported_reasons``); v2 / v3 / ERC-4626 states are always modelled. Callers
    that report coverage use this instead of re-implementing the rules."""
    if isinstance(state, RoutePool):
        state = state.state
    kind = _kind_of(state)
    if kind == KIND_V4:
        return state.unsupported_reasons(allow_before_swap_hook=allow_v4_before_swap_hook)
    if kind == KIND_LB:
        return state.unsupported_reasons(hooks_allowlist)
    return []


class _Search:
    def __init__(
        self,
        pools: list[_Pool],
        timestamp: int | None,
        gas: GasTable,
        hooks_allowlist: HooksAllowlist,
        wrapped: str | None,
        native_paths: bool = False,
    ) -> None:
        self.pools = pools
        self.timestamp = timestamp
        self.gas = gas
        self.hooks_allowlist = hooks_allowlist
        self.wrapped = wrapped
        self.hop_cache: dict[tuple[int, int, int, int], _HopOut | None] = {}
        self.path_cache: dict[tuple[int, int], _LegOut | None] = {}
        self.issues: dict[tuple[str, int], RouteIssue] = {}
        # whole-path native evaluation (v4quote.amm.native), one native call per path and amount
        # instead of one per hop. Off for best_route, whose SearchStats.hop_evals counts the per-hop cache.
        self.native_paths = native_paths and _native.ACTIVE
        self._npaths: dict[int, object] = {}

    # -- one hop -----------------------------------------------------------------
    def _simulate(self, p: _Pool, e: _Edge, amount: int) -> _HopOut:
        if p.state is None:
            raise InsufficientState(f"state of pool {p.pool_id} not loaded: {p.missing_reason}")
        s = p.state
        zfo = e.i == 0
        if amount <= 0:
            raise MathRevert("zero input")
        if _native.ACTIVE:  # the same quote natively (None: no handle for this state, or a fallback to Python)
            r = _native.hop(s, e.i, e.j, amount)
            if r is not None:
                return _HopOut(*r)
        if p.kind == KIND_V2:
            out = s.get_amount_out(amount, zfo)
            fee, crossings = amount * (s.fee_denominator - s.fee_numerator) // s.fee_denominator, 0
        elif p.kind == KIND_V3:
            r = v3.quote_exact_input(s, zfo, amount, want_tick=False)  # amounts only: the post-swap tick is unused
            if not r.fully_filled:
                raise MathRevert("partial fill: price limit reached")
            out, fee, crossings = r.amount_out, r.fee_amount, r.initialized_ticks_crossed
        elif p.kind == KIND_V4:
            r = v4.quote_exact_input(s, zfo, amount)
            out, fee, crossings = r.amount_out, r.fee_amount, r.initialized_ticks_crossed
        elif p.kind == KIND_LB:
            r = _simulate_out(s, amount, zfo, self.timestamp, self.hooks_allowlist)
            if r.amount_in_left:
                raise MathRevert("LBPair__OutOfLiquidity")
            out, fee, crossings = r.amount_out, r.fee, max(r.bins_crossed - 1, 0)
        else:  # ERC-4626: deposit (0 -> 1) / redeem (1 -> 0), no fee
            out, fee, crossings = vault4626.quote(s, e.i, e.j, amount), 0, 0
        if out <= 0:
            raise MathRevert("zero output")
        return _HopOut(out, fee, crossings)

    def hop(self, e: _Edge, amount: int) -> _HopOut | None:
        key = (e.pool, e.i, e.j, amount)
        if key in self.hop_cache:
            return self.hop_cache[key]
        p = self.pools[e.pool]
        try:
            res: _HopOut | None = self._simulate(p, e, amount)
        except InsufficientState as exc:
            kind = "missing_state" if p.state is None else "insufficient_state"
            self.issues.setdefault((p.pool_id, amount), RouteIssue(kind, p.pool_id, amount, str(exc)))
            res = None
        except MathRevert:
            res = None
        self.hop_cache[key] = res
        return res

    def _hop_gas(self, e: _Edge, r: _HopOut) -> int:
        p = self.pools[e.pool]
        wraps = int(p.native[e.i]) + int(p.native[e.j]) if self.wrapped else 0
        return self.gas.hop(p.kind, r.crossings, wraps, lb_hook_kind(p.state, self.hooks_allowlist))

    def _hop_record(self, e: _Edge, amount: int, r: _HopOut) -> Hop:
        p = self.pools[e.pool]
        n_in, n_out = p.native[e.i], p.native[e.j]
        return Hop(
            pool_id=p.pool_id, pool_kind=p.kind, dex=p.dex,
            token_in=p.tokens[e.i], token_out=p.tokens[e.j],
            amount_in=amount, amount_out=r.amount_out, fee_amount=r.fee, crossings=r.crossings,
            gas_units=self._hop_gas(e, r),
            zero_for_one=e.i == 0,
            native_in=n_in, native_out=n_out,
        )

    # -- one path ------------------------------------------------------------------
    def path(self, idx: int, edges: tuple[_Edge, ...], amount: int) -> _LegOut | None:
        key = (idx, amount)
        if key in self.path_cache:
            return self.path_cache[key]
        if self.native_paths:
            np = self._npaths.get(idx, False)
            if np is False:
                np = self._npaths[idx] = self._native_path(edges)
            if np is not None:
                out, raw = np.quote(amount)
                if out is not None:
                    res = _LegOut(out, None, self, (edges, amount, raw))
                    self.path_cache[key] = res
                    return res
                k, stop, msg, amt_k = raw
                if stop != _native.STOP_FALLBACK:
                    if stop == _native.STOP_INSUFFICIENT:  # exactly what hop() records for an InsufficientState
                        p = self.pools[edges[k].pool]
                        self.issues.setdefault((p.pool_id, amt_k), RouteIssue("insufficient_state", p.pool_id, amt_k, msg))
                    self.path_cache[key] = None
                    return None
                _native.note_path_fallback()  # evaluated hop by hop below (Python where native cannot)
        res: _LegOut | None = None
        amt = amount
        steps = []
        for e in edges:
            r = self.hop(e, amt)
            if r is None:
                break
            steps.append((e, amt, r))
            amt = r.amount_out
        else:
            res = _LegOut(amt, tuple(steps), self)
        self.path_cache[key] = res
        return res

    def _native_path(self, edges: tuple[_Edge, ...]):
        """``v4quote_native.Path`` over the edges, or None if some pool has no state / no native handle."""
        hops = []
        for e in edges:
            p = self.pools[e.pool]
            if p.state is None:
                return None
            hops.append((p.state, e.i, e.j))
        return _native.path(hops)


def _normalise(
    pools: Iterable[PoolInput], wrapped_native: str | None, hooks_allowlist: HooksAllowlist, allow_v4_before_swap: bool,
) -> tuple[list[_Pool], list[tuple[str, str]]]:
    wrapped = _canon(wrapped_native) if wrapped_native else None
    out: list[_Pool] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()

    def canon_tokens(raw: Sequence[str]) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        toks, nat = [], []
        for t in raw:
            c = _canon(t)
            is_native = c == NATIVE
            if is_native and wrapped:
                c = wrapped
            toks.append(c)
            nat.append(is_native)
        return tuple(toks), tuple(nat)

    for item in pools:
        if isinstance(item, MissingPool):
            if item.kind not in POOL_KINDS:
                raise ValueError(f"unknown pool kind {item.kind!r}")
            pid, kind, dex, state = item.pool_id, item.kind, item.dex, None
            toks, nat = canon_tokens(item.tokens)
            reason = item.reason or "state not loaded"
        else:
            dex = ""
            if isinstance(item, RoutePool):
                dex, item = item.dex, item.state
            kind = _kind_of(item)
            pid, state, reason = _pool_id(item), item, ""
            toks, nat = canon_tokens(_raw_tokens(item))
        key = _canon(pid)
        if key in seen:
            raise ValueError(f"pool {pid} listed twice")
        seen.add(key)
        if len(set(toks)) != len(toks):
            skipped.append((pid, "pool has the same token twice after native/wrapped mapping"))
            continue
        if state is not None:
            reasons = unsupported_reasons(state, hooks_allowlist, allow_v4_before_swap_hook=allow_v4_before_swap)
            if reasons:
                skipped.append((pid, "; ".join(reasons)))
                continue
        out.append(_Pool(pid, kind, dex, toks, nat, state, reason))
    out.sort(key=lambda p: _canon(p.pool_id))
    skipped.sort()
    return out, skipped


def _paths(pools: list[_Pool], t_in: str, t_out: str, max_hops: int) -> list[tuple[_Edge, ...]]:
    by_token: dict[str, list[int]] = {}
    for k, p in enumerate(pools):
        for t in p.tokens:
            by_token.setdefault(t, []).append(k)
    paths: list[tuple[_Edge, ...]] = []
    for k in by_token.get(t_in, []):
        p = pools[k]
        if t_out in p.tokens:
            paths.append((_Edge(k, p.tokens.index(t_in), p.tokens.index(t_out)),))
    if max_hops >= 2:
        for k1 in by_token.get(t_in, []):
            p1 = pools[k1]
            i1 = p1.tokens.index(t_in)
            for j1, mid in enumerate(p1.tokens):
                if mid in (t_in, t_out):
                    continue
                for k2 in by_token.get(mid, []):
                    p2 = pools[k2]
                    if k2 == k1 or t_out not in p2.tokens:
                        continue
                    paths.append((_Edge(k1, i1, j1), _Edge(k2, p2.tokens.index(mid), p2.tokens.index(t_out))))
    return paths


def _leg(share: int, amount: int, lo: _LegOut) -> Leg:
    return Leg(share, amount, lo.amount_out, lo.hops)


# --------------------------------------------------------------------------- public API
def best_route(
    token_in: str,
    token_out: str,
    amount_in: int,
    pools: Iterable[PoolInput],
    *,
    max_hops: int = 2,
    split: int = 2,
    timestamp: int | None = None,
    wrapped_native: str | None = None,
    gas: GasTable = DEFAULT_GAS,
    out_per_gas: tuple[int, int] | None = None,
    split_candidates: int = 4,
    split_grid_bps: int = 500,
    hooks_allowlist: HooksAllowlist = DEFAULT_HOOKS_ALLOWLIST,
    allow_v4_before_swap_hook: bool = False,
    on_insufficient: Literal["raise", "report"] = "raise",
) -> Route:
    """Best route selling exactly ``amount_in`` of ``token_in`` for ``token_out`` (see module docstring).

    ``timestamp``: ``block.timestamp`` at which the swap executes (required when LB pools are present; the LB
    variable fee depends on it). ``wrapped_native``: treat the native currency of v4 pools (address 0) as this
    token (1:1 wrap/unwrap, gas ``GasTable.wrap``). ``split=1`` disables splitting (spec ``cons``).
    Raises ``RouteInsufficientState`` (``on_insufficient="raise"``) or ``NoRoute``.
    """
    if not isinstance(amount_in, int) or amount_in <= 0:
        raise ValueError("amount_in must be a positive int")
    if max_hops not in (1, 2):
        raise ValueError("max_hops must be 1 or 2")
    if split not in (1, 2):
        raise ValueError("split must be 1 or 2")
    if on_insufficient not in ("raise", "report"):
        raise ValueError("on_insufficient must be 'raise' or 'report'")
    if not (0 < split_grid_bps < BPS and BPS % split_grid_bps == 0):
        raise ValueError("split_grid_bps must divide 10000")
    if out_per_gas is not None and (out_per_gas[0] < 0 or out_per_gas[1] <= 0):
        raise ValueError("out_per_gas must be (num >= 0, den > 0)")
    wrapped = _canon(wrapped_native) if wrapped_native else None
    t_in = _canon(token_in)
    t_out = _canon(token_out)
    if wrapped:
        t_in = wrapped if t_in == NATIVE else t_in
        t_out = wrapped if t_out == NATIVE else t_out
    if t_in == t_out:
        raise ValueError("token_in == token_out")

    plist, skipped = _normalise(pools, wrapped_native, hooks_allowlist, allow_v4_before_swap_hook)
    paths = _paths(plist, t_in, t_out, max_hops)
    used = sorted({e.pool for pth in paths for e in pth})
    if timestamp is None and any(plist[k].kind == KIND_LB and plist[k].state is not None for k in used):
        raise ValueError("timestamp is required when Liquidity Book pools are on a candidate path")

    s = _Search(plist, timestamp, gas, hooks_allowlist, wrapped)

    def value(out: int, gas_units: int) -> int:
        if out_per_gas is None:
            return out
        return out - gas_units * out_per_gas[0] // out_per_gas[1]

    best: tuple | None = None  # (key, legs)

    def consider(legs: tuple[Leg, ...]) -> None:
        nonlocal best
        out = sum(lg.amount_out for lg in legs)
        g = sum(lg.gas_units for lg in legs)
        key = (value(out, g), out, -len(legs), -sum(len(lg.hops) for lg in legs), -g)
        if best is None or key > best[0]:
            best = (key, legs)

    # single paths, full amount
    for idx, pth in enumerate(paths):
        lo = s.path(idx, pth, amount_in)
        if lo is not None:
            consider((_leg(BPS, amount_in, lo),))

    # two-way split over pool-disjoint pairs of the best candidates at half size
    n_pairs = 0
    if split == 2 and len(paths) >= 2 and amount_in >= 2:
        half = amount_in // 2
        ranked = []
        for idx, pth in enumerate(paths):
            lo = s.path(idx, pth, half)
            if lo is not None:
                ranked.append((-value(lo.amount_out, lo.gas_units), idx))
        ranked.sort()
        top = [idx for _, idx in ranked[: max(split_candidates, 0)]]
        for a_pos in range(len(top)):
            for b_pos in range(a_pos + 1, len(top)):
                a, b = top[a_pos], top[b_pos]
                if {e.pool for e in paths[a]} & {e.pool for e in paths[b]}:
                    continue
                n_pairs += 1
                res = _best_split(s, a, paths[a], b, paths[b], amount_in, split_grid_bps, value)
                if res is not None:
                    consider(res)

    issues = tuple(s.issues[k] for k in sorted(s.issues))
    if issues and on_insufficient == "raise":
        raise RouteInsufficientState(issues)
    if best is None:
        if issues:
            raise RouteInsufficientState(issues)
        raise NoRoute(f"no feasible route {token_in} -> {token_out} for {amount_in} over {len(plist)} pools "
                      f"({len(paths)} paths)")
    legs = best[1]
    out = sum(lg.amount_out for lg in legs)
    g = sum(lg.gas_units for lg in legs)
    stats = SearchStats(len(used), len(paths), n_pairs, len(s.path_cache), len(s.hop_cache))
    return Route(
        token_in=t_in, token_out=t_out, amount_in=amount_in, amount_out=out, legs=legs, gas_units=g,
        net_amount_out=value(out, g) if out_per_gas is not None else None,
        issues=issues, skipped=tuple(skipped), stats=stats,
    )


def _best_split(s: _Search, a: int, pa, b: int, pb, amount: int, grid: int, value) -> tuple[Leg, ...] | None:
    """Best share (bps, 1..9999) of ``amount`` on path ``a``, remainder on ``b``; None if no interior share works."""
    evals: dict[int, tuple | None] = {}

    def f(share: int) -> tuple | None:
        if share in evals:
            return evals[share]
        amt_a = amount * share // BPS
        amt_b = amount - amt_a
        res = None
        if amt_a > 0 and amt_b > 0:
            la = s.path(a, pa, amt_a)
            lb = s.path(b, pb, amt_b) if la is not None else None
            if la is not None and lb is not None:
                out = la.amount_out + lb.amount_out
                res = (value(out, la.gas_units + lb.gas_units), out, share, la, lb, amt_a, amt_b)
        evals[share] = res
        return res

    def score(share: int) -> int:
        r = f(share)
        return -1 if r is None else r[0]

    best_share = None
    for share in range(grid, BPS, grid):
        if f(share) is not None and (best_share is None or score(share) > score(best_share)):
            best_share = share
    if best_share is None:
        return None
    lo, hi = max(best_share - grid, 1), min(best_share + grid, BPS - 1)
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        s1, s2 = score(m1), score(m2)
        if s1 < s2:
            lo = m1 + 1
        elif s1 > s2:
            hi = m2 - 1
        else:  # equal (plateau of a concave function, or both infeasible): keep the middle third
            lo, hi = m1, m2
    for share in range(lo, hi + 1):
        if f(share) is not None and score(share) > score(best_share):
            best_share = share
    _, _, share, la, lb, amt_a, amt_b = f(best_share)
    return (_leg(share, amt_a, la), _leg(BPS - share, amt_b, lb))


__all__ = [
    "BPS", "DEFAULT_GAS", "GasTable", "Hop", "Leg", "MissingPool", "NATIVE", "NoRoute", "POOL_KINDS", "Route",
    "RouteInsufficientState", "RouteIssue", "RoutePool", "SearchStats", "best_route", "lb_hook_kind",
    "unsupported_reasons",
]
