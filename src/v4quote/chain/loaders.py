"""Thin on-chain state loaders: read pool state at a pinned block into v4quote dataclasses.

All reads go through Multicall3 at an explicit block number so that the
state snapshot and any quoter comparison refer to the same chain state.
No math happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eth_abi import decode as abi_decode
from eth_utils import keccak, to_checksum_address
from web3 import Web3

from v4quote.amm import erc4626 as e4626
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.liquidity_book.math import FeeParameters
from v4quote.amm.liquidity_book.pair import LBPairState
from v4quote.amm.uniswap_v4 import PoolKey, V4PoolState
from v4quote.amm.v3.pool import V3PoolState
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot, compress, position
from v4quote.amm.v3.tick_math import MAX_TICK, MIN_TICK
from v4quote.chain.multicall import Call, eth_call, multicall


# --------------------------------------------------------------------------- Uniswap v2
def load_v2_pair(
    w3: Web3, pair: str, block: int, *, fee_numerator: int = 997, fee_denominator: int = 1000
) -> V2PairState:
    t0, t1, reserves = multicall(
        w3,
        [
            Call(pair, "token0()", out=("address",)),
            Call(pair, "token1()", out=("address",)),
            Call(pair, "getReserves()", out=("uint112", "uint112", "uint32")),
        ],
        block,
    )
    return V2PairState(
        address=to_checksum_address(pair),
        token0=t0,
        token1=t1,
        reserve0=reserves[0],
        reserve1=reserves[1],
        fee_numerator=fee_numerator,
        fee_denominator=fee_denominator,
        block_number=block,
    )


# --------------------------------------------------------------------------- tick bitmap helpers
def _word_range(tick: int, tick_spacing: int, words_each_side: int | None) -> tuple[int, int]:
    """Bitmap words to load. ``None`` = every word that can hold a valid tick (complete snapshot)."""
    full_lo, _ = position(compress(MIN_TICK, tick_spacing))
    full_hi, _ = position(compress(MAX_TICK, tick_spacing))
    if words_each_side is None:
        return full_lo, full_hi
    w, _ = position(compress(tick, tick_spacing))
    return max(w - words_each_side, full_lo), min(w + words_each_side, full_hi)


def _initialized_ticks(words: dict[int, int], tick_spacing: int) -> list[int]:
    ticks = []
    for word_pos, word in words.items():
        bit = 0
        while word:
            if word & 1:
                ticks.append(((word_pos << 8) + bit) * tick_spacing)
            word >>= 1
            bit += 1
    return sorted(ticks)


# --------------------------------------------------------------------------- Uniswap v3 and forks
def load_v3_pool(
    w3: Web3,
    pool: str,
    block: int,
    *,
    words_each_side: int | None = 8,
    uniswap_fee_protocol: bool = True,
) -> V3PoolState:
    """Load a Uniswap-v3-style pool (Uniswap v3, PancakeSwap v3, ...) at ``block``.

    Only ``words_each_side`` bitmap words around the current tick are loaded
    (``None`` = the whole tick range); swaps that walk further raise
    InsufficientState instead of guessing.
    ``uniswap_fee_protocol``: decode slot0.feeProtocol with Uniswap's uint8
    layout (fee growth only). PancakeSwap v3 uses a different encoding, pass False.
    """
    t0, t1, fee, ts, slot0, liquidity = multicall(
        w3,
        [
            Call(pool, "token0()", out=("address",)),
            Call(pool, "token1()", out=("address",)),
            Call(pool, "fee()", out=("uint24",)),
            Call(pool, "tickSpacing()", out=("int24",)),
            # decode only the common prefix of slot0 (sqrtPriceX96, tick, 3x uint16, feeProtocol)
            Call(pool, "slot0()", out=("uint160", "int24", "uint16", "uint16", "uint16", "uint256")),
            Call(pool, "liquidity()", out=("uint128",)),
        ],
        block,
    )
    sqrt_price, tick = slot0[0], slot0[1]
    fee_protocol = slot0[5] if uniswap_fee_protocol else 0
    lo, hi = _word_range(tick, ts, words_each_side)
    word_positions = list(range(lo, hi + 1))
    words_raw = multicall(w3, [Call(pool, "tickBitmap(int16)", (wp,), ("uint256",)) for wp in word_positions], block)
    words = {wp: w for wp, w in zip(word_positions, words_raw) if w}
    init_ticks = _initialized_ticks(words, ts)
    tick_infos = multicall(
        w3, [Call(pool, "ticks(int24)", (t,), ("uint128", "int128")) for t in init_ticks], block
    )
    liquidity_net = {t: info[1] for t, info in zip(init_ticks, tick_infos)}
    return V3PoolState(
        address=to_checksum_address(pool),
        token0=t0,
        token1=t1,
        fee=fee,
        tick_spacing=ts,
        sqrt_price_x96=sqrt_price,
        tick=tick,
        liquidity=liquidity,
        ticks=TickBitmapSnapshot(ts, words, liquidity_net, (lo, hi)),
        block_number=block,
        fee_protocol=fee_protocol,
    )


# --------------------------------------------------------------------------- Algebra Integral
def load_algebra_integral_pool(
    w3: Web3,
    pool: str,
    block: int,
    *,
    words_each_side: int | None = 8,
) -> V3PoolState:
    """Load an Algebra Integral pool at ``block`` as a ``V3PoolState`` with an ``AlgebraTickSnapshot``
    (``v4quote.amm.algebra_integral``: v3 math, Algebra's tick walk). ``fee`` = the pool's ``fee()`` (with a dynamic-fee
    plugin: the plugin's current fee, which the next swap pays; NOT ``globalState().lastFee``). Bitmap words
    ``tickTable(int16)`` hold uncompressed ticks: ``words_each_side`` words = 256 ticks each around the current one.
    """
    from v4quote.amm.algebra_integral import AlgebraTickSnapshot

    t0, t1, fee, gs, liquidity = multicall(
        w3,
        [
            Call(pool, "token0()", out=("address",)),
            Call(pool, "token1()", out=("address",)),
            Call(pool, "fee()", out=("uint16",)),
            # globalState: price, tick, lastFee, pluginConfig, communityFee, unlocked
            Call(pool, "globalState()", out=("uint160", "int24", "uint16", "uint8", "uint16", "bool")),
            Call(pool, "liquidity()", out=("uint128",)),
        ],
        block,
    )
    sqrt_price, tick = gs[0], gs[1]
    lo, hi = _word_range(tick, 1, words_each_side)
    word_positions = list(range(lo, hi + 1))
    words_raw = multicall(w3, [Call(pool, "tickTable(int16)", (wp,), ("uint256",)) for wp in word_positions], block)
    words = {wp: w for wp, w in zip(word_positions, words_raw) if w}
    init_ticks = _initialized_ticks(words, 1)
    infos = multicall(
        w3, [Call(pool, "ticks(int24)", (t,), ("uint256", "int128")) for t in init_ticks], block
    )
    return V3PoolState(
        address=to_checksum_address(pool),
        token0=t0,
        token1=t1,
        fee=fee,
        tick_spacing=1,
        sqrt_price_x96=sqrt_price,
        tick=tick,
        liquidity=liquidity,
        ticks=AlgebraTickSnapshot(1, words, {t: i[1] for t, i in zip(init_ticks, infos)}, (lo, hi)),
        block_number=block,
    )


# --------------------------------------------------------------------------- Uniswap v4
def load_v4_pool(
    w3: Web3, state_view: str, key: PoolKey, block: int, *, words_each_side: int | None = 8
) -> V4PoolState:
    """Load a v4 pool through the periphery StateView lens at ``block``."""
    pid = key.pool_id()
    slot0, liquidity = multicall(
        w3,
        [
            Call(state_view, "getSlot0(bytes32)", (pid,), ("uint160", "int24", "uint24", "uint24")),
            Call(state_view, "getLiquidity(bytes32)", (pid,), ("uint128",)),
        ],
        block,
    )
    sqrt_price, tick, protocol_fee, lp_fee = slot0
    if sqrt_price == 0:
        raise ValueError(f"v4 pool {pid.hex()} is not initialised")
    ts = key.tick_spacing
    lo, hi = _word_range(tick, ts, words_each_side)
    word_positions = list(range(lo, hi + 1))
    words_raw = multicall(
        w3,
        [Call(state_view, "getTickBitmap(bytes32,int16)", (pid, wp), ("uint256",)) for wp in word_positions],
        block,
    )
    words = {wp: w for wp, w in zip(word_positions, words_raw) if w}
    init_ticks = _initialized_ticks(words, ts)
    infos = multicall(
        w3,
        [Call(state_view, "getTickLiquidity(bytes32,int24)", (pid, t), ("uint128", "int128")) for t in init_ticks],
        block,
    )
    return V4PoolState(
        key=key,
        sqrt_price_x96=sqrt_price,
        tick=tick,
        protocol_fee=protocol_fee,
        lp_fee=lp_fee,
        liquidity=liquidity,
        ticks=TickBitmapSnapshot(ts, words, {t: i[1] for t, i in zip(init_ticks, infos)}, (lo, hi)),
        block_number=block,
    )


V4_INITIALIZE_TOPIC = keccak(text="Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)")


def decode_v4_initialize(log: dict) -> tuple[bytes, PoolKey]:
    """Decode a PoolManager ``Initialize`` log into (poolId, PoolKey)."""
    topics = [bytes(t) for t in log["topics"]]
    if topics[0] != V4_INITIALIZE_TOPIC:
        raise ValueError("not an Initialize log")
    pool_id = topics[1]
    c0 = to_checksum_address(topics[2][-20:])
    c1 = to_checksum_address(topics[3][-20:])
    fee, tick_spacing, hooks, _sqrt_price, _tick = abi_decode(
        ["uint24", "int24", "address", "uint160", "int24"], bytes(log["data"])
    )
    key = PoolKey(c0, c1, fee, tick_spacing, to_checksum_address(hooks))
    if key.pool_id() != pool_id:
        raise ValueError("Initialize log poolId does not match keccak(PoolKey)")
    return pool_id, key


def find_v4_pool_key(
    w3: Web3, pool_manager: str, pool_id: bytes, from_block: int, to_block: int, *, chunk: int = 2000
) -> PoolKey | None:
    """Scan PoolManager ``Initialize`` logs for ``pool_id`` (topic1) in [from_block, to_block].

    Public RPCs cap eth_getLogs ranges (Avalanche 2048 blocks, Monad ~100), so a
    full-history scan needs an indexer or a paid/own node; pass a narrow range
    when the initialisation block is known.
    """
    start = from_block
    while start <= to_block:
        end = min(start + chunk - 1, to_block)
        logs = w3.eth.get_logs(
            {
                "address": to_checksum_address(pool_manager),
                "topics": ["0x" + V4_INITIALIZE_TOPIC.hex(), "0x" + pool_id.hex()],
                "fromBlock": start,
                "toBlock": end,
            }
        )
        for log in logs:
            return decode_v4_initialize(log)[1]
        start = end + 1
    return None


# --------------------------------------------------------------------------- LFJ Liquidity Book (+ Pharaoh DLMM)
LB_MAX_ID = (1 << 24) - 1
_LB_SENTINELS = (0, LB_MAX_ID)  # getNextNonEmptyBin: no further non-empty bin in that direction


EIP1967_IMPLEMENTATION_SLOT = 0x360894A13BA1A3210667C828492DB98DCA3E2076CC3735A920A3CA505D382BBC
EIP1967_BEACON_SLOT = 0xA3F0AD74E5423AEBFD80D3EF4346578335A9A72AEAEE59FF6CB3582B35133D50
# LFJ ImmutableClone (0age minimal proxy + clones-with-immutable-args) runtime prefix; the 20 bytes that
# follow are the hard-coded implementation address (joe-v2 src/libraries/ImmutableClone.sol).
LFJ_IMMUTABLE_CLONE_PREFIX = bytes.fromhex("363d3d373d3d3d3d61003e806035363936013d73")


PROXY_LFJ_CLONE = "lfj-immutable-clone"
PROXY_EIP1967 = "eip1967"
PROXY_EIP1967_BEACON = "eip1967-beacon"


@dataclass(frozen=True)
class HookCodeInfo:
    codehash: str
    implementation: str | None
    implementation_codehash: str | None
    proxy_kind: str | None = None  # PROXY_* above (same names as HookApproval.proxy_kind); None = not a proxy


def _codehash(code: bytes) -> str:
    return "0x" + keccak(code).hex()


def resolve_hook(w3: Web3, hook: str, block: int) -> HookCodeInfo:
    """Codehash of a hooks contract and, for proxies, its implementation (at ``block``).

    Recognised proxies: LFJ ImmutableClone (implementation embedded in bytecode), EIP-1967
    implementation slot, EIP-1967 beacon slot (implementation = beacon.implementation()).
    """
    hook = to_checksum_address(hook)
    code = bytes(w3.eth.get_code(hook, block_identifier=block))
    impl: str | None = None
    kind: str | None = None
    if code.startswith(LFJ_IMMUTABLE_CLONE_PREFIX):
        start = len(LFJ_IMMUTABLE_CLONE_PREFIX)
        impl = to_checksum_address(code[start : start + 20])
        kind = PROXY_LFJ_CLONE
    else:
        raw_impl = int.from_bytes(bytes(w3.eth.get_storage_at(hook, EIP1967_IMPLEMENTATION_SLOT, block)), "big")
        raw_beacon = int.from_bytes(bytes(w3.eth.get_storage_at(hook, EIP1967_BEACON_SLOT, block)), "big")
        if raw_impl:
            impl = to_checksum_address(raw_impl.to_bytes(20, "big"))
            kind = PROXY_EIP1967
        elif raw_beacon:
            beacon = to_checksum_address(raw_beacon.to_bytes(20, "big"))
            impl = to_checksum_address(eth_call(w3, Call(beacon, "implementation()", (), ("address",)), block))
            kind = PROXY_EIP1967_BEACON
    impl_hash = _codehash(bytes(w3.eth.get_code(impl, block_identifier=block))) if impl else None
    return HookCodeInfo(_codehash(code), impl, impl_hash, kind)


def lfj_hook_link_calls(hook: str) -> list[Call]:
    """Reads that pin the runtime behaviour of an LFJ rewarder hook (ImmutableClone of LBHooksBaseParentRewarder)
    beyond its codehash: ``getExtraHooksParameters()`` (the extra rewarder whose ``beforeSwap`` the hook calls;
    owner / manager can set it without changing any codehash) and ``isLinked()`` (the hook is still the pair's
    hooks contract)."""
    return [Call(hook, "getExtraHooksParameters()", out=("bytes32",)), Call(hook, "isLinked()", out=("bool",))]


def beacon_hook_link_calls(hook: str) -> list[Call]:
    """Reads that bind a beacon-proxied rewarder hook (Pharaoh DLMMRewarder) to its pair: ``getLBPair()`` and
    ``isLinked()``. Every proxy of one beacon has the same codehash, so the codehash cannot tell which pair a hook
    serves."""
    return [Call(hook, "getLBPair()", out=("address",)), Call(hook, "isLinked()", out=("bool",))]


def decode_beacon_hook_link(pair: Any, linked: Any) -> tuple[str | None, bool | None]:
    """Results of ``beacon_hook_link_calls`` (None = reverted / undecodable, i.e. unknown)."""
    return (to_checksum_address(pair) if isinstance(pair, str) else None), (linked if isinstance(linked, bool) else None)


def decode_lfj_hook_link(extra: Any, linked: Any) -> tuple[int | None, bool | None]:
    """Results of ``lfj_hook_link_calls`` (None = reverted / undecodable, i.e. unknown)."""
    extra_int = int.from_bytes(bytes(extra), "big") if isinstance(extra, (bytes, bytearray)) else None
    return extra_int, (linked if isinstance(linked, bool) else None)


def load_lb_pair(
    w3: Web3,
    pair: str,
    block: int,
    *,
    version: str = "2.2",
    bins_each_side: int = 100,
    chain: str | None = None,
    follow_sparse_bins: int = 0,
) -> LBPairState:
    """Load an LB v2.1/v2.2 pair: fee parameters, active id and all bins within +-bins_each_side.

    ``follow_sparse_bins``: beyond the dense window, follow the pair's own ``getNextNonEmptyBin`` chain for up to
    this many further non-empty bins on each side (one multicall round per step). The contract guarantees that
    every bin between two consecutive ids of that chain is empty, so extending ``loaded_range`` to the last
    followed bin keeps the snapshot exact; reaching the sentinel sets ``exhausted_below/above``. This covers
    one-sided pools whose remaining liquidity sits in a few bins far from the active id, which a dense window
    cannot reach at any practical width.
    """
    calls = [
        Call(pair, "getTokenX()", out=("address",)),
        Call(pair, "getTokenY()", out=("address",)),
        Call(pair, "getBinStep()", out=("uint16",)),
        Call(pair, "getActiveId()", out=("uint24",)),
        Call(pair, "getStaticFeeParameters()", out=("uint16",) * 4 + ("uint24", "uint16", "uint24")),
        Call(pair, "getVariableFeeParameters()", out=("uint24", "uint24", "uint24", "uint40")),
    ]
    if version == "2.2":
        calls.append(Call(pair, "getLBHooksParameters()", out=("bytes32",)))
    res = multicall(w3, calls, block)
    tx, ty, bin_step, active_id, st, var = res[:6]
    hooks = int.from_bytes(res[6], "big") if version == "2.2" else 0
    params = FeeParameters(
        base_factor=st[0],
        filter_period=st[1],
        decay_period=st[2],
        reduction_factor=st[3],
        variable_fee_control=st[4],
        protocol_share=st[5],
        max_volatility_accumulator=st[6],
        volatility_accumulator=var[0],
        volatility_reference=var[1],
        id_reference=var[2],
        time_of_last_update=var[3],
    )
    hook_info = None
    hook_calls: list[Call] = []
    if hooks & ((1 << 160) - 1):
        hook_info = resolve_hook(w3, "0x" + format(hooks & ((1 << 160) - 1), "040x"), block)
        if hook_info.proxy_kind == PROXY_LFJ_CLONE:
            hook_calls = lfj_hook_link_calls("0x" + format(hooks & ((1 << 160) - 1), "040x"))
        elif hook_info.proxy_kind == PROXY_EIP1967_BEACON:
            hook_calls = beacon_hook_link_calls("0x" + format(hooks & ((1 << 160) - 1), "040x"))
    lo = max(active_id - bins_each_side, 0)
    hi = min(active_id + bins_each_side, LB_MAX_ID)
    ids = list(range(lo, hi + 1))
    bin_calls = [Call(pair, "getBin(uint24)", (i,), ("uint128", "uint128")) for i in ids]
    edge_calls = [
        Call(pair, "getNextNonEmptyBin(bool,uint24)", (True, lo), ("uint24",)),
        Call(pair, "getNextNonEmptyBin(bool,uint24)", (False, hi), ("uint24",)),
    ]
    strict = bin_calls + edge_calls
    if hook_calls:
        # the hook reads ride in the same Multicall3 as the bins (same block); a revert only makes them unknown
        out = multicall(w3, strict + hook_calls, block, allow_failure=[False] * len(strict) + [True] * len(hook_calls))
        extra_params, hook_pair = None, None
        if hook_info is not None and hook_info.proxy_kind == PROXY_EIP1967_BEACON:
            hook_pair, linked = decode_beacon_hook_link(*out[len(strict):])
        else:
            extra_params, linked = decode_lfj_hook_link(*out[len(strict):])
        out = out[: len(strict)]
    else:
        out = multicall(w3, strict, block)
        extra_params, linked, hook_pair = None, None, None
    bins = {i: (r[0], r[1]) for i, r in zip(ids, out[: len(ids)]) if r[0] or r[1]}
    next_below, next_above = out[-2], out[-1]
    ex_below, ex_above = next_below in _LB_SENTINELS, next_above in _LB_SENTINELS
    for _ in range(max(follow_sparse_bins, 0)):
        # a chain step must move strictly outwards; anything else is treated as "not exhausted, stop here"
        go_below = not ex_below and next_below < lo
        go_above = not ex_above and next_above > hi
        if not (go_below or go_above):
            break
        calls = []
        if go_below:
            calls += [Call(pair, "getBin(uint24)", (next_below,), ("uint128", "uint128")),
                      Call(pair, "getNextNonEmptyBin(bool,uint24)", (True, next_below), ("uint24",))]
        if go_above:
            calls += [Call(pair, "getBin(uint24)", (next_above,), ("uint128", "uint128")),
                      Call(pair, "getNextNonEmptyBin(bool,uint24)", (False, next_above), ("uint24",))]
        res = multicall(w3, calls, block)
        k = 0
        if go_below:
            b, nxt = res[0], res[1]
            if b[0] or b[1]:
                bins[next_below] = (b[0], b[1])
            lo, next_below, ex_below, k = next_below, nxt, nxt in _LB_SENTINELS, 2
        if go_above:
            b, nxt = res[k], res[k + 1]
            if b[0] or b[1]:
                bins[next_above] = (b[0], b[1])
            hi, next_above, ex_above = next_above, nxt, nxt in _LB_SENTINELS
    return LBPairState(
        address=to_checksum_address(pair),
        token_x=tx,
        token_y=ty,
        bin_step=bin_step,
        active_id=active_id,
        params=params,
        bins=bins,
        loaded_range=(lo, hi),
        exhausted_below=ex_below,
        exhausted_above=ex_above,
        version=version,
        hooks_parameters=hooks,
        block_number=block,
        chain=chain,
        hooks_codehash=hook_info.codehash if hook_info else None,
        hooks_implementation=hook_info.implementation if hook_info else None,
        hooks_implementation_codehash=hook_info.implementation_codehash if hook_info else None,
        hooks_extra_parameters=extra_params,
        hooks_linked=linked,
        hooks_pair=hook_pair,
    )


def block_timestamp(w3: Web3, block: int) -> int:
    return int(w3.eth.get_block(block)["timestamp"])


# --------------------------------------------------------------------------- ERC-4626 vault
MAX_UINT256 = (1 << 256) - 1


def erc4626_state_calls(vault: str) -> list[Call]:
    """``totalAssets()`` and ``totalSupply()``: the per-head refresh of an ``ERC4626VaultState``."""
    return [Call(vault, "totalAssets()", out=("uint256",)), Call(vault, "totalSupply()", out=("uint256",))]


def load_erc4626(w3: Web3, vault: str, block: int, *, expected_asset: str | None = None,
                 require_balance_assets: bool = True) -> e4626.ERC4626VaultState:
    """An OZ v5 ERC-4626 vault at ``block``, refused (ValueError) unless it follows ``v4quote.amm.erc4626`` exactly:

    * ``asset()`` equals ``expected_asset`` when given;
    * ``decimals_offset`` = ``decimals() - asset.decimals()`` in 0..18 (OZ: ``decimals = asset decimals + offset``);
    * ``maxDeposit(any)`` is ``type(uint256).max`` (OZ default; another cap is outside the model);
    * ``previewDeposit`` / ``previewRedeem`` of five amounts (1 wei .. 1/10 of the supply / assets) equal the model's;
    * ``require_balance_assets``: ``totalAssets() == asset.balanceOf(vault)`` (the default OZ ``totalAssets``; a
      caller can then track A through asset Transfer logs, which is valid only then).

    One Multicall3 call for the facts, one for the preview probes."""
    res = multicall(w3, [Call(vault, "asset()", out=("address",)), Call(vault, "decimals()", out=("uint8",)),
                         *erc4626_state_calls(vault),
                         Call(vault, "maxDeposit(address)", ("0x000000000000000000000000000000000000dEaD",),
                              ("uint256",))], block)
    asset, dec, total_assets, total_supply, max_dep = res
    asset = asset.lower()
    if expected_asset is not None and asset != expected_asset.lower():
        raise ValueError(f"{vault}: asset() is {asset}, expected {expected_asset.lower()}")
    if max_dep != MAX_UINT256:
        raise ValueError(f"{vault}: maxDeposit is {max_dep}, not type(uint256).max (outside the ERC-4626 model)")
    asset_dec, balance = multicall(w3, [Call(asset, "decimals()", out=("uint8",)),
                                        Call(asset, "balanceOf(address)", (vault,), ("uint256",))], block)
    offset = dec - asset_dec
    if not 0 <= offset <= e4626.MAX_DECIMALS_OFFSET:
        raise ValueError(f"{vault}: decimals {dec} - asset decimals {asset_dec} is not an offset in 0..18")
    if require_balance_assets and balance != total_assets:
        raise ValueError(f"{vault}: totalAssets {total_assets} != asset balance {balance} (not the default OZ model)")
    st = e4626.ERC4626VaultState(address=to_checksum_address(vault), asset=to_checksum_address(asset),
                                 total_assets=total_assets, total_supply=total_supply, decimals_offset=offset,
                                 block_number=block)
    amounts = sorted({1, 10 ** asset_dec, 10 ** dec, max(1, total_assets // 10), max(1, total_supply // 10)})
    probes = ([Call(vault, "previewDeposit(uint256)", (a,), ("uint256",)) for a in amounts]
              + [Call(vault, "previewRedeem(uint256)", (a,), ("uint256",)) for a in amounts])
    got = multicall(w3, probes, block)
    want = [e4626.preview_deposit(st, a) for a in amounts] + [e4626.preview_redeem(st, a) for a in amounts]
    if list(got) != want:
        bad = [(p.signature, p.args[0]) for p, g, w in zip(probes, got, want) if g != w]
        raise ValueError(f"{vault}: previewDeposit / previewRedeem differ from the OZ v5 model at {bad[:3]}")
    return st


__all__ = [
    "load_v2_pair", "load_v3_pool", "load_algebra_integral_pool", "load_v4_pool", "load_lb_pair",
    "load_erc4626", "erc4626_state_calls", "block_timestamp", "eth_call",
]
