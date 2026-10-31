"""Metric OMM: on-chain snapshot reader, state loader, quoter wrappers and the public API helper.

Addresses are Metric's CREATE2 deployments, identical on every chain they list (docs.metric.xyz, "Supported chains and
deployments"; checked on Robinhood Chain, chainId 4663).

The exact state is read on chain: ``MetricOmmPoolDataProvider.slot0`` / ``binStates`` (every bin from the pool's
lowest to its highest), the pool's immutables and the factory's notional-fee config, all at one pinned block. The
provider quote (raw bid / ask / reference) comes from ``getLiquidityDepthLive(pool, 1)`` at the same block, sent
without ``from``: restricted providers only answer reads with ``tx.origin == 0``. Pass ``prices`` to price the
snapshot at a quote of your own (what ``quoteHypothetical*`` takes).

The public API (``https://api.metric.xyz``, no key) is used for discovery and status only: its ``bidAdj`` / ``askAdj``
already include the pool's spread and fees and its depth omits the traded part of the current bin, so neither can
reproduce the pool to the unit or be fed back as provider prices (Metric's docs say so explicitly).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from eth_utils import to_checksum_address
from web3 import Web3

from v4quote.amm.metric_omm import MetricOmmPoolState, decode_bin_word
from v4quote.chain.multicall import Call, eth_call
from v4quote.errors import UnreadableState, UnsupportedPool

METRIC_FACTORY = "0x2a53833cc95548cf52c7b159110e22D3a9018f32"
METRIC_DATA_PROVIDER = "0x83d66c5934b1525ee82e6E772c76c7acC778bBB7"
METRIC_QUOTER = "0x3B9ea79f482094323d06064E637a7F4528560fcb"
METRIC_ROUTER = "0x02201E6f13c41506c7Bb635d7543F98826229aB7"
METRIC_API = "https://api.metric.xyz"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

_IMMUTABLES = ("(address,address,address,uint256,uint256,uint256,uint256,uint256,address,int256,int256,address,address,"
               "address,address,address,address,address,uint256,uint256,uint256,uint256,uint256,uint256)")
_DEPTH = "(uint128,uint128,uint128,uint128,uint128,(int16,uint256,uint256,uint256,uint256,uint256,uint256)[]," \
         "(int16,uint256,uint256,uint256,uint256,uint256,uint256)[])"


def read_metric_snapshot(w3: Web3, pool: str, block: int, *, prices: tuple[int, int, int] | None = None,
                         data_provider: str = METRIC_DATA_PROVIDER, factory: str = METRIC_FACTORY,
                         chain: str | None = None) -> dict:
    """Raw on-chain reads of one pool at ``block`` as a JSON-able dict (big integers as decimal strings).

    ``metric_state_from_snapshot`` turns it into a ``MetricOmmPoolState``; tests keep these dicts as fixtures."""
    pool = to_checksum_address(pool)
    imm = eth_call(w3, Call(pool, "getImmutables()", (), (_IMMUTABLES,)), block)
    s0 = eth_call(w3, Call(data_provider, "slot0(address)", (pool,),
                           ("uint8", "int16", "uint104", "int24", "uint24", "uint24", "uint16")), block)
    lowest, highest = int(imm[9]), int(imm[10])
    words = eth_call(w3, Call(data_provider, "binStates(address,int16[])", (pool, list(range(lowest, highest + 1))),
                              ("bytes32[]",)), block)
    fee_cfg = eth_call(w3, Call(factory, "poolFeeConfig(address)", (pool,), ("uint24", "uint24", "uint24", "uint24")),
                       block)
    if prices is None:
        try:
            depth = eth_call(w3, Call(data_provider, "getLiquidityDepthLive(address,uint8)", (pool, 1), (_DEPTH,)), block)
        except Exception as exc:  # provider refused / stale feed / paused: the pool has no live quote at this block
            raise UnreadableState(f"no live provider quote for {pool} at block {block}: {type(exc).__name__}",
                                  reason="provider_unreadable") from None
        prices = (int(depth[0]), int(depth[1]), int(depth[2]))
    return {
        "pool": pool,
        "block": block,
        "chain": chain,
        "immutables": [str(x) if isinstance(x, int) else x for x in imm],
        "slot0": [int(x) if i != 2 else str(x) for i, x in enumerate(s0)],
        "bin_words": [str(int.from_bytes(w, "big")) for w in words],
        "pool_fee_config": [int(x) for x in fee_cfg],
        "prices": [str(int(p)) for p in prices],
    }


def metric_state_from_snapshot(snap: dict) -> MetricOmmPoolState:
    """``MetricOmmPoolState`` from ``read_metric_snapshot`` output (offline: no network)."""
    imm = snap["immutables"]
    s0 = snap["slot0"]
    lowest, highest = int(imm[9]), int(imm[10])
    words = [int(w) for w in snap["bin_words"]]
    if len(words) != highest - lowest + 1:
        raise ValueError("bin_words must cover lowestBin..highestBin")
    bins = {lowest + k: decode_bin_word(w) for k, w in enumerate(words)}
    fee_cfg = snap["pool_fee_config"]
    bid, ask, ref = (int(p) for p in snap["prices"])
    extensions = tuple(to_checksum_address(e) for e in imm[11:18] if int(e, 16) != 0)
    return MetricOmmPoolState(
        address=to_checksum_address(snap["pool"]), token0=to_checksum_address(imm[1]), token1=to_checksum_address(imm[2]),
        token0_scale=int(imm[3]), token1_scale=int(imm[4]), bid_x64=bid, ask_x64=ask, reference_x64=ref,
        cur_bin=int(s0[1]), pos_in_bin=int(s0[2]), cur_bin_dist_e6=int(s0[3]), bins=bins,
        loaded_range=(lowest, highest), lowest_bin=lowest, highest_bin=highest, pause_level=int(s0[0]),
        notional_fee_e8=int(s0[5]) + int(fee_cfg[2]) + int(fee_cfg[3]), bid_ask_ref_asymmetry_x8=int(s0[6]),
        extensions=extensions, block_number=snap.get("block"), chain=snap.get("chain"),
    )


def load_metric_pool(w3: Web3, pool: str, block: int, *, prices: tuple[int, int, int] | None = None,
                     data_provider: str = METRIC_DATA_PROVIDER, factory: str = METRIC_FACTORY,
                     chain: str | None = None) -> MetricOmmPoolState:
    """Load a Metric pool at ``block`` (all bins). ``prices``: raw provider (bid, ask, reference); default = the live
    provider quote at that block (``UnreadableState`` if the provider refuses, e.g. a stale feed or a closed market)."""
    return metric_state_from_snapshot(read_metric_snapshot(w3, pool, block, prices=prices, data_provider=data_provider,
                                                           factory=factory, chain=chain))


def _quote(w3: Web3, quoter: str, fn: str, pool: str, zero_for_one: bool, amount: int, prices: tuple[int, int, int],
           block: int, sender: str = ZERO_ADDRESS) -> tuple[int, int]:
    limit = 0 if zero_for_one else (1 << 128) - 1
    sig = f"{fn}(address,address,bool,uint128,uint128,uint128,uint128,uint128)"
    out = eth_call(w3, Call(quoter, sig, (to_checksum_address(pool), sender, zero_for_one, amount, limit, *prices),
                            ("uint256", "uint256")), block)
    return int(out[0]), int(out[1])


def metric_quoter_exact_input(w3: Web3, pool: str, zero_for_one: bool, amount_in: int,
                              prices: tuple[int, int, int], block: int, *, quoter: str = METRIC_QUOTER
                              ) -> tuple[int, int]:
    """``quoteHypotheticalExactInputSingle`` (open price limit) -> (amountIn, amountOut). Reverts raise CallFailed."""
    return _quote(w3, quoter, "quoteHypotheticalExactInputSingle", pool, zero_for_one, amount_in, prices, block)


def metric_quoter_exact_output(w3: Web3, pool: str, zero_for_one: bool, amount_out: int,
                               prices: tuple[int, int, int], block: int, *, quoter: str = METRIC_QUOTER
                               ) -> tuple[int, int]:
    """``quoteHypotheticalExactOutputSingle`` (open price limit) -> (amountIn, amountOut)."""
    return _quote(w3, quoter, "quoteHypotheticalExactOutputSingle", pool, zero_for_one, amount_out, prices, block)


# ----------------------------------------------------------------------------- public API (discovery / status)
def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "v4quote"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: fixed https endpoint, no credentials sent
        return json.loads(resp.read().decode())


def fetch_metric_api_pools(chain_id: int, *, base_url: str = METRIC_API, timeout: float = 15.0) -> list[dict]:
    """Every pool of ``GET /public/v1/evm/{chain_id}/metadata`` (all pages). No API key is sent."""
    out: list[dict] = []
    offset: int | None = 0
    while offset is not None:
        page = _get_json(f"{base_url}/public/v1/evm/{int(chain_id)}/metadata?offset={offset}", timeout)
        out.extend(page["data"])
        offset = page.get("nextOffset")
    return out


def fetch_metric_api_bid_ask(chain_id: int, pool: str, *, base_url: str = METRIC_API, timeout: float = 15.0) -> dict:
    """``GET /public/v1/evm/{chain_id}/{pool}/bid_ask`` (adjusted prices + depth; status and sentinels)."""
    return _get_json(f"{base_url}/public/v1/evm/{int(chain_id)}/{to_checksum_address(pool)}/bid_ask", timeout)


def check_api_quotable(bid_ask: dict) -> None:
    """Raise ``UnsupportedPool`` when the API marks the pool as not quotable: a ``priceProviderStatus`` other than
    ``healthy`` or the sentinel bid 0 / ask 2^128-1 on both sides."""
    status = bid_ask.get("priceProviderStatus")
    if status != "healthy":
        raise UnsupportedPool(f"Metric API status {status!r}")
    if int(bid_ask.get("bidAdj", 0)) == 0 and int(bid_ask.get("askAdj", 0)) == (1 << 128) - 1:
        raise UnsupportedPool("Metric API reports no price on either side")
