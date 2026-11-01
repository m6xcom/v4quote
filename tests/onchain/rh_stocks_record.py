"""Record Robinhood Chain stock-token pool fixtures: pool states + on-chain reference quotes at one block.

    ROBINHOOD_RPC_URL=... uv run python -m tests.onchain.rh_stocks_record tests/fixtures/robinhood_stocks.json

(``RH_RECORD_BLOCK`` pins the block, e.g. the fork block when ROBINHOOD_RPC_URL points at a local anvil fork.)

Pools: the stock tokens' highest-volume pools of week 2026-10-19..26 (USDG pairs), by type:
* Uniswap v3 (factory 0x1f7d7550, official deployment) -> reference QuoterV2 0x33e885ed;
* Ramses V3 CL fork (factory 0xe0c4ceb9, Sourcify RamsesV3Pool, owner-settable ``fee()``) and Velodrome Slipstream CL
  clones (factory 0x1ac9db4a, Sourcify CLPool, fee = ``factory.getSwapFee(pool)``): no quoter of their own on this
  chain, so the reference is the pool's own ``swap()`` executed by eth_call (``v3_pool_swap_sim*``);
* Uniswap v4 without hooks and v4 dynamic-fee pools with reviewed fee-only hooks (``v4quote.chain.hook_fees``) ->
  reference V4Quoter 0x8dc178ef. For the hooked pools the state's ``hook_fee`` is the hook's ``beforeSwap`` fee for
  sender = V4Quoter at the block.

Sizes per direction come from the state itself: dust, half of the way to the first initialised tick, just past the
1st / 3rd / 8th initialised tick, and a large swap (price moved ~20 %, or the farthest loaded tick). Exact-output
sizes are the outputs of those exact-input swaps. Reverts are stored as the error text.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from web3 import Web3

from v4quote.amm import serialization as ser
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.v3 import pool as v3
from v4quote.amm.v3.tick_math import get_sqrt_ratio_at_tick
from v4quote.chain.hook_fees import with_hook_fee
from v4quote.chain.loaders import load_v3_pool, load_v4_pool
from v4quote.chain.multicall import Call, CallFailed, eth_call
from v4quote.chain.quoters import (
    v3_pool_swap_sim,
    v3_pool_swap_sim_exact_output,
    v3_quoter_exact_input_single,
    v3_quoter_exact_output_single,
    v4_quoter_exact_input_single,
    v4_quoter_exact_output_single,
)
from v4quote.errors import CoreError, InsufficientState

# Uniswap deployments on Robinhood Chain (docs.uniswap.org; checked on chain: QuoterV2.factory() == V3_FACTORY,
# V4Quoter.poolManager() == StateView.poolManager() == POOL_MANAGER)
V3_FACTORY = "0x1f7d7550b1b028f7571e69a784071f0205fd2efa"
QUOTER_V2 = "0x33e885ed0ec9bf04ecfb19341582aadcb4c8a9e7"
POOL_MANAGER = "0x8366a39cc670b4001a1121b8f6a443a643e40951"
STATE_VIEW = "0xf3334192d15450cdd385c8b70e03f9a6bd9e673b"
V4_QUOTER = "0x8dc178efb8111bb0973dd9d722ebeff267c98f94"
RAMSES_V3_FACTORY = "0xe0c4ceb92d08ca985bb70fe0a22feb121a9854a8"
SLIPSTREAM_FACTORY = "0x1ac9db4a2608ba45d6127b1737949b51bb54b7f3"

USDG = "0x5fc5360d0400a0fd4f2af552add042d716f1d168"
STOCK = {
    "SPCX": "0x4a0e65a3eccec6dbe60ae065f2e7bb85fae35eea", "NVDA": "0xd0601ce157db5bdc3162bbac2a2c8af5320d9eec",
    "SPY": "0x117cc2133c37b721f49de2a7a74833232b3b4c0c", "CRCL": "0xdf0992e440dd0be65bd8439b609d6d4366bf1cb5",
    "META": "0xc0d6457c16cc70d6790dd43521c899c87ce02f35", "GLD": "0xc9a981fee1f9dec688bb123ccdecc63d0debfc4e",
    "MSTR": "0xec262a75e413fafd0df80480274532c79d42da09", "TSLA": "0x322f0929c4625ed5bad873c95208d54e1c003b2d",
    "GOOGL": "0x2e0847e8910a9732eb3fb1bb4b70a580adad4fe3", "AAPL": "0xaf3d76f1834a1d425780943c99ea8a608f8a93f9",
}
KIND_V3, KIND_RAMSES, KIND_SLIP, KIND_V4, KIND_V4_HOOK = "uniswap_v3", "ramses_v3", "slipstream", "v4", "v4_fee_hook"
WORDS = 12  # tick-bitmap words each side (ts 1: ~3000 ticks ~ +-36 % price)


@dataclass(frozen=True)
class Spec:
    name: str
    ticker: str
    kind: str
    pool: str = ""  # v3-style pool address
    fee: int = 0  # v4 key
    ts: int = 0
    hooks: str = v4.ZERO_ADDRESS

    def key(self) -> v4.PoolKey:
        a, b = sorted([STOCK[self.ticker], USDG], key=lambda x: int(x, 16))
        return v4.PoolKey(Web3.to_checksum_address(a), Web3.to_checksum_address(b), self.fee, self.ts,
                          Web3.to_checksum_address(self.hooks))


DYN = v4.DYNAMIC_FEE_FLAG
POOLS = [  # week 10-19..26 volume order inside each ticker (research/robinhood_chain/stocks/pools_by_stock.json)
    Spec("SPCX/USDG v3 500", "SPCX", KIND_V3, "0xc61284332117c3fb23a2a56cceffd07f7af60029"),
    Spec("SPCX/USDG v4 SessionFeeHook", "SPCX", KIND_V4_HOOK, fee=DYN, ts=60,
         hooks="0xc15b155a3094934171a15625d1a80aba267b6080"),
    Spec("SPCX/USDG v4 FablesRWA", "SPCX", KIND_V4_HOOK, fee=DYN, ts=10, hooks="0x5eb87f69be00df39981622fd60a8de4b7837e080"),
    Spec("NVDA/USDG v3 500", "NVDA", KIND_V3, "0xd4eb21209c4d6093f80b5b84f5c45cc093ea14a3"),
    Spec("NVDA/USDG Ramses V3 ts10", "NVDA", KIND_RAMSES, "0xdac1904d823f7d6bcaaf431ceee40c934fed321b"),
    Spec("NVDA/USDG Slipstream ts60", "NVDA", KIND_SLIP, "0x18a5af4e442f8be68968cc1f00d537f8af2d12cd"),
    Spec("NVDA/USDG v4 100", "NVDA", KIND_V4, fee=100, ts=1),
    Spec("SPY/USDG v4 500", "SPY", KIND_V4, fee=500, ts=5),
    Spec("SPY/USDG Ramses V3 ts1", "SPY", KIND_RAMSES, "0x38453c115607463ac284820ce959831042f3df4e"),
    Spec("SPY/USDG v4 100", "SPY", KIND_V4, fee=100, ts=1),
    Spec("CRCL/USDG v3 3000", "CRCL", KIND_V3, "0x654e4143e82a5824445ade0824351c2a9acd95a8"),
    Spec("CRCL/USDG v4 FablesRWA", "CRCL", KIND_V4_HOOK, fee=DYN, ts=60, hooks="0x5eb87f69be00df39981622fd60a8de4b7837e080"),
    Spec("META/USDG v4 3000", "META", KIND_V4, fee=3000, ts=60),
    Spec("META/USDG Ramses V3 ts50", "META", KIND_RAMSES, "0x960f79d8dfec7f2f0c1b24cdac6bb9df1371c6e2"),
    Spec("GLD/USDG v3 500", "GLD", KIND_V3, "0xba2f1ed4ceb2169d538d1e614d847e83c5a55913"),
    Spec("GLD/USDG v4 FablesRWA", "GLD", KIND_V4_HOOK, fee=DYN, ts=10, hooks="0xb608a78761f179f7c56f15e7d13921b92f00a080"),
    Spec("MSTR/USDG v4 2500", "MSTR", KIND_V4, fee=2500, ts=25),
    Spec("TSLA/USDG v4 3000", "TSLA", KIND_V4, fee=3000, ts=60),
    Spec("TSLA/USDG v3 3000", "TSLA", KIND_V3, "0xf4acdaeeb7022862a763c9b1b885e11191c889e3"),
    Spec("TSLA/USDG v4 FablesRWA", "TSLA", KIND_V4_HOOK, fee=DYN, ts=10, hooks="0x67d86050d22d574df046f3d90f722045f714e080"),
    Spec("GOOGL/USDG v3 500", "GOOGL", KIND_V3, "0x34d0dc122cf9a8eb296fc5e0d3a233625d7d19b7"),
    Spec("GOOGL/USDG Ramses V3 ts1", "GOOGL", KIND_RAMSES, "0x5c0eaa89799f7b76606f4ff66c92a2654d178927"),
    Spec("AAPL/USDG v3 500", "AAPL", KIND_V3, "0xaae0d815ee56e4092a5e5c2911e676fea50b2d6d"),
    Spec("AAPL/USDG v4 3000", "AAPL", KIND_V4, fee=3000, ts=60),
    Spec("AAPL/USDG v4 FablesRWA", "AAPL", KIND_V4_HOOK, fee=DYN, ts=10, hooks="0x70a9a88402989226847ec122043ce5e7ff462080"),
]
REFERENCE = {KIND_V3: f"QuoterV2 {QUOTER_V2}", KIND_RAMSES: "pool.swap (eth_call)", KIND_SLIP: "pool.swap (eth_call)",
             KIND_V4: f"V4Quoter {V4_QUOTER}", KIND_V4_HOOK: f"V4Quoter {V4_QUOTER}"}
FACTORY = {KIND_V3: V3_FACTORY, KIND_RAMSES: RAMSES_V3_FACTORY, KIND_SLIP: SLIPSTREAM_FACTORY}


# --------------------------------------------------------------------------- loading
def load_state(w3: Web3, spec: Spec, block: int):
    if spec.kind in (KIND_V3, KIND_RAMSES, KIND_SLIP):
        st = load_v3_pool(w3, spec.pool, block, words_each_side=WORDS, uniswap_fee_protocol=spec.kind == KIND_V3)
        factory = eth_call(w3, Call(spec.pool, "factory()", (), ("address",)), block)
        if factory.lower() != FACTORY[spec.kind]:
            raise AssertionError(f"{spec.name}: factory {factory} is not {FACTORY[spec.kind]}")
        return st
    st = load_v4_pool(w3, STATE_VIEW, spec.key(), block, words_each_side=WORDS)
    if spec.kind == KIND_V4_HOOK:
        st = with_hook_fee(w3, POOL_MANAGER, st, sender=V4_QUOTER, block=block)
    return st


def token_info(w3: Web3, token: str, block: int) -> list:
    sym = eth_call(w3, Call(token, "symbol()", (), ("string",)), block)
    dec = eth_call(w3, Call(token, "decimals()", (), ("uint8",)), block)
    try:  # ERC-8056 display multiplier (1e18 = 1.0); amounts on chain are raw token units regardless
        ui = str(eth_call(w3, Call(token, "uiMultiplier()", (), ("uint256",)), block))
    except CallFailed:
        ui = None
    return [Web3.to_checksum_address(token), sym, int(dec), ui]


# --------------------------------------------------------------------------- model (shared with the replay tests)
def model_quote(state, zero_for_one: bool, exact_in: bool, amount: int) -> tuple[int, int]:
    if isinstance(state, v4.V4PoolState):
        r = (v4.quote_exact_input if exact_in else v4.quote_exact_output)(state, zero_for_one, amount)
    else:
        r = (v3.quote_exact_input if exact_in else v3.quote_exact_output)(state, zero_for_one, amount)
    return r.amount_in, r.amount_out


def _reach(state, zero_for_one: bool, sqrt_limit: int) -> int:
    """Input that moves the price to ``sqrt_limit`` (model)."""
    if isinstance(state, v4.V4PoolState):
        return v4.swap(state, zero_for_one, -(10**30), sqrt_limit).amount_in
    return v3.swap(state, zero_for_one, 10**30, sqrt_limit).amount_in


def sizes(state, zero_for_one: bool) -> list[int]:
    """Exact-input sizes: dust, inside the current range, past the 1st / 3rd / 8th initialised tick, large."""
    ticks = sorted(state.ticks.liquidity_net)
    side = [t for t in reversed(ticks) if t <= state.tick] if zero_for_one else [t for t in ticks if t > state.tick]
    out = {1000}
    reach = []
    for t in side[:8]:
        try:
            reach.append(_reach(state, zero_for_one, get_sqrt_ratio_at_tick(t)))
        except CoreError:
            break
    reach = [r for r in reach if r > 0]
    if reach:
        out.add(max(1, reach[0] // 2))
        for k in (0, 2, 7):
            if k < len(reach):
                out.add(reach[k] + reach[k] // 10_000 + 1)
    p = state.sqrt_price_x96
    target = p * 894_427 // 1_000_000 if zero_for_one else p * 1_118_034 // 1_000_000  # price x0.8 / x1.25
    try:
        out.add(_reach(state, zero_for_one, target))
    except InsufficientState:
        if reach:
            out.add(reach[-1] + 1)
    return sorted(a for a in out if a > 0)


# --------------------------------------------------------------------------- on-chain reference
def chain_quote(w3: Web3, spec: Spec, state, zero_for_one: bool, exact_in: bool, amount: int, block: int) -> list:
    """[amount_in, amount_out, None] or [None, None, error text] from the reference contract at ``block``."""
    try:
        if spec.kind in (KIND_V4, KIND_V4_HOOK):
            key = spec.key()
            if exact_in:
                return [str(amount), str(v4_quoter_exact_input_single(w3, V4_QUOTER, key, zero_for_one, amount, block)),
                        None]
            return [str(v4_quoter_exact_output_single(w3, V4_QUOTER, key, zero_for_one, amount, block)), str(amount),
                    None]
        if spec.kind == KIND_V3:
            t_in, t_out = (state.token0, state.token1) if zero_for_one else (state.token1, state.token0)
            if exact_in:
                q = v3_quoter_exact_input_single(w3, QUOTER_V2, t_in, t_out, state.fee, amount, block)
                return [str(amount), str(q[0]), None]
            q = v3_quoter_exact_output_single(w3, QUOTER_V2, t_in, t_out, state.fee, amount, block)
            return [str(q[0]), str(amount), None]
        if exact_in:
            used, got = v3_pool_swap_sim(w3, spec.pool, zero_for_one, amount, block)
        else:
            used, got = v3_pool_swap_sim_exact_output(w3, spec.pool, zero_for_one, amount, block)
        return [str(used), str(got), None]
    except CallFailed as exc:
        if not exc.reverted:
            raise
        return [None, None, "revert"]


def record_pool(w3: Web3, spec: Spec, block: int) -> dict:
    state = load_state(w3, spec, block)
    t0, t1 = (state.key.currency0, state.key.currency1) if isinstance(state, v4.V4PoolState) else (state.token0,
                                                                                                    state.token1)
    entry = {"name": spec.name, "ticker": spec.ticker, "kind": spec.kind, "reference": REFERENCE[spec.kind],
             "pool": spec.pool or "0x" + spec.key().pool_id().hex(), "hooks": spec.hooks,
             "tokens": [token_info(w3, t0, block), token_info(w3, t1, block)],
             "state": ser.to_dict(state), "quotes": []}
    for zfo in (True, False):
        ins = sizes(state, zfo)
        outs = set()
        for amount in ins:
            q = chain_quote(w3, spec, state, zfo, True, amount, block)
            entry["quotes"].append([zfo, True, str(amount), *q])
            if q[2] is None and int(q[1]) > 0:
                outs.add(int(q[1]))
        outs.add(1000)
        for amount in sorted(outs):
            entry["quotes"].append([zfo, False, str(amount), *chain_quote(w3, spec, state, zfo, False, amount, block)])
    return entry


def record(w3: Web3, block: int, pools: list[Spec] = POOLS) -> dict:
    blk = w3.eth.get_block(block)
    out = {"chain": "robinhood", "chain_id": int(w3.eth.chain_id), "block": block, "timestamp": int(blk["timestamp"]),
           "contracts": {"QuoterV2": QUOTER_V2, "V4Quoter": V4_QUOTER, "StateView": STATE_VIEW,
                         "PoolManager": POOL_MANAGER, "UniswapV3Factory": V3_FACTORY,
                         "RamsesV3Factory": RAMSES_V3_FACTORY, "SlipstreamCLFactory": SLIPSTREAM_FACTORY},
           "pools": []}
    for spec in pools:
        entry = record_pool(w3, spec, block)
        print(f"{spec.name}: {len(entry['quotes'])} quotes", file=sys.stderr)
        out["pools"].append(entry)
    return out


def main(argv: list[str]) -> None:
    from v4quote.chain.rpc import get_web3  # noqa: PLC0415

    w3 = get_web3("robinhood")
    pinned = os.environ.get("RH_RECORD_BLOCK")
    block = int(pinned) if pinned else w3.eth.block_number - 2
    data = record(w3, block)
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"block {block}: {len(data['pools'])} pools, {sum(len(p['quotes']) for p in data['pools'])} quotes",
          file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
