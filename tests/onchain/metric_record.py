"""Record Metric OMM fixtures: pool snapshots + MetricOmmSwapQuoter.quoteHypothetical* results at one block.

    ROBINHOOD_RPC_URL=... uv run python -m tests.onchain.metric_record tests/fixtures/metric_robinhood.json

Every quote uses the provider quote read at the same block (``getLiquidityDepthLive``), so the snapshot and the
quoter see identical state and prices. Sizes come from the snapshot itself: points inside, at the end and just past
the first bins of each side, random sizes up to the side's whole liquidity, dust and more than the pool holds.
Reverts are stored with the decoded error selector (unwrapping the quoter's ``WrappedError``).
"""

from __future__ import annotations

import json
import os
import random
import sys

from eth_abi import decode as abi_decode
from eth_utils import function_signature_to_4byte_selector, to_checksum_address
from web3 import Web3
from web3.exceptions import ContractLogicError

from v4quote.amm import metric_omm as mo
from v4quote.chain.metric import METRIC_QUOTER, metric_state_from_snapshot, read_metric_snapshot

ROBINHOOD_POOLS = {  # name -> pool (Metric API metadata, chain 4663)
    "WETH/USDG": "0x1E8Fc442AD3606719F7016DED6C17B56C3cA7386",
    "META/USDG": "0x90CABdA18279BBB3D935989aF807b08B0e8CeC61",
    "NVDA/USDG": "0xBc6426CdBCE142dfbff33303dc0923509f61eedc",
    "SPCX/USDG": "0xA94E5130a57c48B0b2F266709634A45bE7ae8171",
    "TSLA/USDG": "0x99932C7660dD58cf19f0e82e97F288d88d033480",
    "AAPL/USDG": "0x6a4fc50b14F6D390340bd0BD2fcdB1752852a099",
    "NET/USDG": "0x9735A405CB634f5BCeE2debC0f6326B0d1749661",
    "CASHCAT/USDG": "0x844B36D8B4592F9f387c0a8423e081A720FCCFe6",
    "RAETO/WETH": "0xD9456A05819746E2c160F53427378e2b803646bc",
    "DELTA/WETH": "0xFDB93f70b3347049928c8665d04c131717873600",
}
_WRAPPED = function_signature_to_4byte_selector("WrappedError(address,bytes4,bytes)")


def _revert_selector(exc: Exception) -> str:
    data = getattr(exc, "data", None)
    if isinstance(data, str) and data.startswith("0x"):
        raw = bytes.fromhex(data[2:])
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        return "unknown"
    if raw[:4] == _WRAPPED:
        _, _, inner = abi_decode(["address", "bytes4", "bytes"], raw[4:])
        return "0x" + bytes(inner[:4]).hex() if inner else "0x"
    return "0x" + raw[:4].hex() if raw else "0x"


def raw_quote(w3: Web3, pool: str, zero_for_one: bool, exact_in: bool, amount: int, prices, block: int,
              quoter: str = METRIC_QUOTER) -> list:
    """[amount_in, amount_out, None] or [None, None, error_selector] (hypothetical quote, open price limit)."""
    fn = "quoteHypotheticalExactInputSingle" if exact_in else "quoteHypotheticalExactOutputSingle"
    sig = f"{fn}(address,address,bool,uint128,uint128,uint128,uint128,uint128)"
    from eth_abi import encode
    data = function_signature_to_4byte_selector(sig) + encode(
        ["address", "address", "bool", "uint128", "uint128", "uint128", "uint128", "uint128"],
        [to_checksum_address(pool), "0x" + "00" * 20, zero_for_one, amount, 0 if zero_for_one else (1 << 128) - 1,
         *prices])
    for attempt in range(5):
        try:
            out = w3.eth.call({"to": to_checksum_address(quoter), "data": "0x" + data.hex()}, block_identifier=block)
            a_in, a_out = abi_decode(["uint256", "uint256"], bytes(out))
            return [str(a_in), str(a_out), None]
        except ContractLogicError as exc:
            return [None, None, _revert_selector(exc)]
        except Exception:  # transport (rate limit): back off and retry
            import time
            time.sleep(0.5 * 2 ** attempt)
    raise RuntimeError("quoter call failed repeatedly (transport)")


def side_levels(state: mo.MetricOmmPoolState, zero_for_one: bool, nbins: int) -> list[tuple[int, int]]:
    """Cumulative native (input, output) at the end of the first ``nbins`` non-empty bins of a side (model costs)."""
    m_in, m_out = (state.token0_scale, state.token1_scale) if zero_for_one else (state.token1_scale, state.token0_scale)
    out, cin, cout = [], 0, 0
    i, pos, dist = state.cur_bin, state.pos_in_bin, state.cur_bin_dist_e6
    while len(out) < nbins:
        b = state.bins[i]
        bal = b[1] if zero_for_one else b[0]
        if bal:
            if zero_for_one:
                lo, hi = mo.bid_edge_price(state, dist), mo.bid_edge_price(state, dist + b[2])
                cin += mo.bid_bin_cost(bal, bal, pos, lo, hi)
            else:
                lo, hi = mo.ask_edge_price(state, dist), mo.ask_edge_price(state, dist + b[2])
                cin += mo.ask_bin_cost(bal, bal, pos, lo, hi)
            cout += bal
            out.append((-(-cin // m_in), cout // m_out))
        if zero_for_one:
            if i == state.lowest_bin:
                break
            i -= 1
            dist -= state.bins[i][2]
            pos = mo.MAX_POS
        else:
            if i == state.highest_bin:
                break
            dist += b[2]
            i += 1
            pos = 0
    return out


def side_totals(state: mo.MetricOmmPoolState, zero_for_one: bool) -> tuple[int, int]:
    lv = side_levels(state, zero_for_one, 10_000)
    return lv[-1] if lv else (0, 0)


def quote_sizes(state: mo.MetricOmmPoolState, zero_for_one: bool, exact_in: bool, rng: random.Random,
                n_random: int, nbins: int = 3) -> list[int]:
    levels = side_levels(state, zero_for_one, nbins)
    total_in, total_out = side_totals(state, zero_for_one)
    k = 0 if exact_in else 1
    total = total_in if exact_in else total_out
    sizes = {1, 1000}
    prev = 0
    for lv in levels:
        cur = lv[k]
        if cur > prev:
            for num in (1, 10_000, 300_000, 700_000, 999_999):
                sizes.add(max(1, prev + (cur - prev) * num // 1_000_000))
            sizes.update({cur, cur + 1})
        prev = cur
    if total > 0:
        sizes.update(rng.randint(1, total) for _ in range(n_random))
        sizes.add(total * 2 + 7)
    return sorted(s for s in sizes if 0 < s < (1 << 127))


def asymmetry_vectors(snapshot: dict) -> list[tuple[int, int, int]]:
    """Provider quotes around the pool's bid/ask/reference asymmetry bound (and a reference on bid / on ask)."""
    ref = int(snapshot["prices"][2])
    d2 = ref // 1000
    ask = ref + d2
    m = int(snapshot["slot0"][6])
    big_m = 256 + m
    out = []
    for num, den in ((big_m, 256), (big_m * 999, 256 * 1000), (big_m * 1001, 256 * 1000),
                     (256, big_m), (256 * 999, big_m * 1000), (256 * 1001, big_m * 1000)):
        bid = ref * ref * den // (ref * den + d2 * num)  # r1 = r2 * num / den
        out += [(bid + adj, ask, ref) for adj in (-1, 0, 1)]
    return out + [(ref, ask, ref), (ref - 1000, ref, ref), (ask, ask, ref), (0, ask, ref)]


def record(w3: Web3, pools: dict[str, str], block: int, *, n_random: int = 20, seed: int = 1,
           chain: str = "robinhood") -> dict:
    rng = random.Random(seed)
    out = {"chain": chain, "chain_id": int(w3.eth.chain_id), "block": block, "quoter": METRIC_QUOTER, "pools": []}
    for name, pool in pools.items():
        snap = read_metric_snapshot(w3, pool, block, chain=chain)
        state = metric_state_from_snapshot(snap)
        prices = tuple(int(p) for p in snap["prices"])
        quotes = []
        for zfo in (False, True):
            for exact_in in (True, False):
                for amt in quote_sizes(state, zfo, exact_in, rng, n_random):
                    quotes.append([zfo, exact_in, str(amt), *raw_quote(w3, pool, zfo, exact_in, amt, prices, block)])
        entry = {"name": name, "snapshot": snap, "quotes": quotes}
        if int(snap["slot0"][6]) != 0xFFFF:  # asymmetry bound active: probe it with a small exact-output buy
            entry["price_checks"] = [[str(b), str(a), str(r), raw_quote(w3, pool, False, False, 10 ** 12, (b, a, r), block)[2]]
                                     for b, a, r in asymmetry_vectors(snap)]
        out["pools"].append(entry)
        print(f"{name}: {len(quotes)} quotes", file=sys.stderr)
    return out


def main(argv: list[str]) -> None:
    from v4quote.chain.rpc import get_web3
    w3 = get_web3("robinhood")
    pinned = os.environ.get("METRIC_RECORD_BLOCK")  # e.g. the fork block when recording from a local anvil fork
    block = int(pinned) if pinned else w3.eth.block_number - 2
    data = record(w3, ROBINHOOD_POOLS, block)
    with open(argv[1], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv)
