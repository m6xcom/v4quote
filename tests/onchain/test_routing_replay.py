"""The Avalanche route pool set (fixtures/route_pools.yaml) against the chain (marker ``onchain``).

Every pool of the set must be what its official factory / StateView returns for (tokens, fee / tickSpacing /
binStep) at a pinned block, so ``v4quote.routing_pools.load_route_pool_set`` never routes through an impostor pool.
Needs ``AVALANCHE_RPC_URL`` (skipped when unset).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from eth_utils import to_checksum_address

from v4quote import routing as rt
from v4quote.amm import uniswap_v4 as v4
from v4quote.chain.multicall import Call, eth_call
from v4quote.chain.rpc import CHAINS, RpcTransportError, rpc_url, web3_from_url
from v4quote.routing_pools import RoutePoolSpec, load_pool_state, load_route_pool_set
from tests.onchain.conftest import PIN_LAG

pytestmark = pytest.mark.onchain

CHAIN = "avalanche"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "route_pools.yaml"
MIN_ROUTED = 10
MAX_ROUTED = int(os.environ.get("ROUTING_REPLAY_MAX", "20"))
RS = load_route_pool_set(FIXTURE, CHAIN)


PACE_S = float(os.environ.get("ONCHAIN_PACE_S", "0.05"))


def _paced(w3, min_interval_s: float, retries: int = 8):
    """Rate-limit courtesy for public endpoints: minimum spacing between requests and exponential backoff on
    HTTP 429/5xx/transport errors. Must wrap the provider before its first request (web3 caches the chain)."""
    prov = w3.provider
    orig = prov.make_request
    last = [0.0]

    def make_request(method, params):
        for attempt in range(retries + 1):
            wait = last[0] + min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            try:
                return orig(method, params)
            except RpcTransportError as exc:
                if attempt == retries or exc.status not in (None, 429, 500, 502, 503, 504):
                    raise
                time.sleep(min(20.0, 0.5 * 2**attempt))
        raise AssertionError("unreachable")

    prov.make_request = make_request
    return w3


class Env:
    def __init__(self):
        self.w3 = _paced(web3_from_url(rpc_url(CHAIN)), PACE_S)
        if self.w3.eth.chain_id != CHAINS[CHAIN].chain_id:
            raise RuntimeError(f"{CHAINS[CHAIN].rpc_env} is not Avalanche C-Chain")
        self.head = self.w3.eth.block_number - PIN_LAG


@pytest.fixture(scope="module")
def env() -> Env:
    if rpc_url(CHAIN) is None:
        pytest.skip(f"{CHAINS[CHAIN].rpc_env} not set")
    return Env()


# =========================================================================== pool set verification
def test_route_pool_links(env):
    """Every route pool is what its official factory returns for (tokens, fee / tickSpacing / binStep)."""
    w3, blk = env.w3, env.head
    for p in RS.pools:
        if p.kind == rt.KIND_V4:
            key = p.pool_key
            assert "0x" + key.pool_id().hex() == p.pool_id.lower(), p.id
            slot0 = eth_call(w3, Call(p.get("state_view"), "getSlot0(bytes32)", (key.pool_id(),),
                                      ("uint160", "int24", "uint24", "uint24")), blk)
            assert slot0[0] != 0, p.id
            assert not v4.unsupported_reasons(key), p.id
            continue
        a, b = p.tokens
        if p.kind == rt.KIND_V2:
            got = eth_call(w3, Call(p.get("factory"), "getPair(address,address)", (a, b), ("address",)), blk)
        elif p.kind == rt.KIND_V3:
            key_type = p.get("key_type", "uint24")
            key = p.get("tick_spacing") if key_type == "int24" else p.get("fee")
            got = eth_call(w3, Call(p.get("factory"), f"getPool(address,address,{key_type})", (a, b, key),
                                    ("address",)), blk)
        else:
            pairs = eth_call(w3, Call(p.get("factory"), "getAllLBPairs(address,address)", (a, b),
                                      ("(uint16,address,bool,bool)[]",)), blk)
            found = {to_checksum_address(addr): bs for bs, addr, _, _ in pairs}
            assert found.get(p.pool_id) == p.get("bin_step"), p.id
            continue
        assert to_checksum_address(got) == p.pool_id, p.id


# =========================================================================== loading at B-1
def _load(w3, p: RoutePoolSpec, blk: int, width: int):
    """``v4quote.routing_pools.load_pool_state`` at ``blk``; None if the pool did not exist yet at that block."""
    if p.kind == rt.KIND_V4:
        try:
            return load_pool_state(w3, p, blk, width=width, chain=CHAIN)
        except ValueError:  # not initialised at this block
            return None
    if not bytes(w3.eth.get_code(p.pool_id, block_identifier=blk)):
        return None
    return load_pool_state(w3, p, blk, width=width, chain=CHAIN)
