"""Re-verify every address in pools.yaml on-chain at the pinned block.

* pool == official factory lookup (getPair / getPool / getAllLBPairs); v4: poolId == keccak(key) and initialised
* quoter/router is wired to the same factory / pool manager / deployer
* CREATE2 derivation with the parameterised deployer reproduces the pool address (where a preset exists)
"""

import pytest
from eth_utils import to_checksum_address

from v4quote.amm.uniswap_v4 import PoolKey
from v4quote.amm.v3.pool_address import PRESETS, compute_pool_address
from v4quote.chain.multicall import Call, eth_call

from tests.onchain.conftest import pools_of_type, token_address

pytestmark = pytest.mark.onchain

Z = "0x0000000000000000000000000000000000000000"


def _eq(a: str, b: str) -> bool:
    return to_checksum_address(a) == to_checksum_address(b)


@pytest.mark.parametrize("p", pools_of_type("uniswap_v2"))
def test_v2_links(p, chain_ctx):
    ctx = chain_ctx(p["chain"])
    a, b = (token_address(p["chain"], t) for t in p["tokens"])
    pair = eth_call(ctx.w3, Call(p["factory"], "getPair(address,address)", (a, b), ("address",)), ctx.block)
    assert _eq(pair, p["pool"])
    router_factory = eth_call(ctx.w3, Call(p["quoter"], "factory()", (), ("address",)), ctx.block)
    assert _eq(router_factory, p["factory"])


@pytest.mark.parametrize("p", pools_of_type("uniswap_v3"))
def test_v3_links(p, chain_ctx):
    ctx = chain_ctx(p["chain"])
    a, b = (token_address(p["chain"], t) for t in p["tokens"])
    key_type = p.get("key_type", "uint24")
    key = p["tick_spacing"] if key_type == "int24" else p["fee"]
    pool = eth_call(
        ctx.w3, Call(p["factory"], f"getPool(address,address,{key_type})", (a, b, key), ("address",)), ctx.block
    )
    assert _eq(pool, p["pool"])
    if key_type == "int24":  # Ramses/Pharaoh QuoterV2 exposes the pool deployer
        dep = eth_call(ctx.w3, Call(p["quoter"], "deployer()", (), ("address",)), ctx.block)
        assert _eq(dep, p["pool_deployer"])
    else:
        qf = eth_call(ctx.w3, Call(p["quoter"], "factory()", (), ("address",)), ctx.block)
        assert _eq(qf, p["factory"])
    preset = PRESETS.get((p["chain"], p["dex"]))
    if preset is not None:
        derived = compute_pool_address(preset.deployer, a, b, key, preset.init_code_hash, preset.key_type)
        assert _eq(derived, p["pool"]), f"CREATE2 derivation mismatch for {p['id']}"


@pytest.mark.parametrize("p", pools_of_type("uniswap_v4"))
def test_v4_links(p, chain_ctx):
    ctx = chain_ctx(p["chain"])
    k = p["pool_key"]
    key = PoolKey(
        token_address(p["chain"], k["currency0"]), token_address(p["chain"], k["currency1"]),
        k["fee"], k["tick_spacing"], k["hooks"],
    )
    assert "0x" + key.pool_id().hex() == p["pool_id"].lower()
    slot0 = eth_call(
        ctx.w3, Call(p["state_view"], "getSlot0(bytes32)", (key.pool_id(),), ("uint160", "int24", "uint24", "uint24")),
        ctx.block,
    )
    assert slot0[0] != 0, "pool not initialised"
    pm = eth_call(ctx.w3, Call(p["quoter"], "poolManager()", (), ("address",)), ctx.block)
    assert _eq(pm, p["pool_manager"])
    pm2 = eth_call(ctx.w3, Call(p["state_view"], "poolManager()", (), ("address",)), ctx.block)
    assert _eq(pm2, p["pool_manager"])


@pytest.mark.parametrize("p", pools_of_type("lfj_lb"))
def test_lb_links(p, chain_ctx):
    ctx = chain_ctx(p["chain"])
    a, b = (token_address(p["chain"], t) for t in p["tokens"])
    pairs = eth_call(
        ctx.w3,
        Call(p["factory"], "getAllLBPairs(address,address)", (a, b), ("(uint16,address,bool,bool)[]",)),
        ctx.block,
    )
    found = {to_checksum_address(addr): bs for bs, addr, _, _ in pairs}
    assert to_checksum_address(p["pool"]) in found
    assert found[to_checksum_address(p["pool"])] == p["bin_step"]
    if p.get("lb_quoter"):
        if p["dex"] == "pharaoh_dlmm":
            f = eth_call(ctx.w3, Call(p["lb_quoter"], "getFactory()", (), ("address",)), ctx.block)
        else:
            sig = "getFactoryV2_2()" if p["version"] == "2.2" else "getFactoryV2_1()"
            f = eth_call(ctx.w3, Call(p["lb_quoter"], sig, (), ("address",)), ctx.block)
        assert _eq(f, p["factory"])


def test_default_hooks_allowlist_matches_chain(chain_ctx):
    """Every DEFAULT_HOOKS_ALLOWLIST entry still matches the chain at the pinned block:

    * hook code: codehash, proxy kind, implementation and implementation codehash;
    * ``entry.pair.getLBHooksParameters()`` points at the hook (and an LFJ clone's ``getLBPair()`` is the pair);
    * an LFJ clone has no extra rewarder (``getExtraHooksParameters() == 0``) and ``isLinked()``;
    * the loader path ``load_lb_pair(entry.pair)`` finds the pool supported (no ``unsupported_reasons``).
    """
    from v4quote.amm.liquidity_book import DEFAULT_HOOKS_ALLOWLIST
    from v4quote.chain.loaders import load_lb_pair, resolve_hook
    from v4quote.chain.multicall import multicall
    from v4quote.chain.rpc import rpc_url

    checked = 0
    for (chain, hook), a in DEFAULT_HOOKS_ALLOWLIST.items():
        if rpc_url(chain) is None:
            continue
        ctx = chain_ctx(chain)
        info = resolve_hook(ctx.w3, hook, ctx.block)
        assert info.codehash.lower() == a.codehash.lower(), hook
        assert info.proxy_kind == a.proxy_kind, hook
        assert (info.implementation or "").lower() == a.implementation.lower(), hook
        assert (info.implementation_codehash or "").lower() == a.implementation_codehash.lower(), hook
        (params,) = multicall(ctx.w3, [Call(a.pair, "getLBHooksParameters()", out=("bytes32",))], ctx.block)
        assert _eq("0x" + params[-20:].hex(), hook), f"pair {a.pair} does not point at hook {hook}"
        if a.proxy_kind == "lfj-immutable-clone":
            lb_pair, extra, linked = multicall(ctx.w3, [
                Call(hook, "getLBPair()", out=("address",)),
                Call(hook, "getExtraHooksParameters()", out=("bytes32",)),
                Call(hook, "isLinked()", out=("bool",)),
            ], ctx.block)
            assert _eq(lb_pair, a.pair), hook
            assert int.from_bytes(extra, "big") == 0, f"hook {hook} has an extra rewarder: 0x{extra.hex()}"
            assert linked is True, hook
        elif a.proxy_kind == "eip1967-beacon":
            lb_pair, linked = multicall(ctx.w3, [
                Call(hook, "getLBPair()", out=("address",)),
                Call(hook, "isLinked()", out=("bool",)),
            ], ctx.block)
            assert _eq(lb_pair, a.pair), hook
            assert linked is True, hook
        st = load_lb_pair(ctx.w3, a.pair, ctx.block, bins_each_side=1, chain=chain)
        assert st.unsupported_reasons() == [], (hook, st.unsupported_reasons())
        checked += 1
    if checked == 0:
        pytest.skip("no RPC configured")
