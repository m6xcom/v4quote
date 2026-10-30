"""Replay historical LBPair ``Swap`` events (real swap() calls, hooks included) against v4quote.

For each block that contains a swap on the pair we take the FIRST swap() call of the pair in that
block, rebuild it from its per-bin ``Swap`` events and re-run it with v4quote on the pair state at
``block - 1`` with the swap block's timestamp. Swaps preceded (in the same block) by any other log of
the pair (mint, burn, flash loan, fee-parameter change, LB-token transfers, ...) are skipped and counted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from eth_abi import decode as abi_decode
from eth_utils import keccak, to_checksum_address
from web3 import Web3

from v4quote.amm.liquidity_book import swap as lb_swap
from v4quote.chain.loaders import load_lb_pair
from v4quote.chain.rpc import CHAINS, web3_from_url
from v4quote.errors import InsufficientState

SWAP_TOPIC = keccak(text="Swap(address,address,uint24,bytes32,bytes32,uint24,bytes32,bytes32)")
TOPIC_NAMES = {
    keccak(text=s): s.split("(")[0]
    for s in (
        "DepositedToBins(address,address,uint256[],bytes32[])",
        "WithdrawnFromBins(address,address,uint256[],bytes32[])",
        "CompositionFees(address,uint24,bytes32,bytes32)",
        "CollectedProtocolFees(address,bytes32)",
        "StaticFeeParametersSet(address,uint16,uint16,uint16,uint16,uint24,uint16,uint24)",
        "HooksParametersSet(address,bytes32)",
        "FlashLoan(address,address,uint24,bytes32,bytes32,bytes32)",
        "OracleLengthIncreased(address,uint16)",
        "ForcedDecay(address,uint24,uint24)",
        "TransferBatch(address,address,address,uint256[],uint256[])",
        "ApprovalForAll(address,address,bool)",
    )
}
TOPIC_NAMES[SWAP_TOPIC] = "Swap"
M128 = (1 << 128) - 1

# Public RPCs cap eth_getLogs ranges differently; an optional separate endpoint for log scanning
LOGS_ENV = {"monad": "MONAD_LOGS_RPC_URL", "avalanche": "AVALANCHE_LOGS_RPC_URL"}
DEFAULT_LOG_CHUNK = {"monad": 100, "avalanche": 2048}


def logs_web3(chain: str) -> tuple[Web3, int]:
    url = os.environ.get(LOGS_ENV[chain], "").strip()
    if url:
        return web3_from_url(url), int(os.environ.get(LOGS_ENV[chain] + "_CHUNK", "50000"))
    return web3_from_url(os.environ[CHAINS[chain].rpc_env]), DEFAULT_LOG_CHUNK[chain]


@dataclass
class BinEvent:
    bin_id: int
    amounts_in: int
    amounts_out: int
    vol_acc: int
    total_fees: int
    protocol_fees: int


@dataclass
class ReplayCase:
    block: int
    tx: str
    swap_for_y: bool
    amount_in: int
    events: list[BinEvent]


@dataclass
class ReplayStats:
    pool: str
    considered: int = 0
    matched: int = 0
    by_direction: dict = field(default_factory=lambda: {True: 0, False: 0})
    skipped_prior_activity: int = 0
    skipped_reasons: dict = field(default_factory=dict)
    skipped_other: dict = field(default_factory=dict)
    mismatches: list = field(default_factory=list)
    blocks_scanned: tuple[int, int] | None = None


def _decode_swap(log) -> BinEvent:
    bid, a_in, a_out, vol, tf, pf = abi_decode(
        ["uint24", "bytes32", "bytes32", "uint24", "bytes32", "bytes32"], bytes(log["data"])
    )
    return BinEvent(bid, *(int.from_bytes(x, "big") for x in (a_in, a_out)), vol,
                    int.from_bytes(tf, "big"), int.from_bytes(pf, "big"))


def _side(packed: int, y: bool) -> int:
    return packed >> 128 if y else packed & M128


def collect_cases(chain: str, pair: str, head: int, want_per_direction: int, max_blocks: int):
    """Scan backwards from ``head`` and return (cases, skipped list, (from, to))."""
    w3l, chunk = logs_web3(chain)
    pair = to_checksum_address(pair)
    cases: list[ReplayCase] = []
    skipped: list[tuple[int, str]] = []
    counts = {True: 0, False: 0}
    to_block = head
    lowest = head
    while head - to_block < max_blocks and min(counts.values()) < want_per_direction:
        frm = max(to_block - chunk + 1, 0)
        logs = w3l.eth.get_logs({"address": pair, "fromBlock": frm, "toBlock": to_block})
        by_block: dict[int, list] = {}
        for lg in logs:
            by_block.setdefault(lg["blockNumber"], []).append(lg)
        for bn in sorted(by_block, reverse=True):
            blogs = sorted(by_block[bn], key=lambda x: x["logIndex"])
            first_swap = next((i for i, lg in enumerate(blogs) if bytes(lg["topics"][0]) == SWAP_TOPIC), None)
            if first_swap is None:
                continue
            if first_swap > 0:
                names = sorted({TOPIC_NAMES.get(bytes(lg["topics"][0]), "other") for lg in blogs[:first_swap]})
                skipped.append((bn, "+".join(names)))
                continue
            run = [blogs[first_swap]]
            for lg in blogs[first_swap + 1:]:
                if (bytes(lg["topics"][0]) == SWAP_TOPIC and lg["transactionHash"] == run[-1]["transactionHash"]
                        and lg["logIndex"] == run[-1]["logIndex"] + 1):
                    run.append(lg)
                else:
                    break
            evs = [_decode_swap(lg) for lg in run]
            swap_for_y = _side(evs[0].amounts_in, False) > 0
            if counts[swap_for_y] >= want_per_direction:
                continue
            amount_in = sum(_side(e.amounts_in, not swap_for_y) + _side(e.protocol_fees, not swap_for_y) for e in evs)
            cases.append(ReplayCase(bn, "0x" + bytes(run[0]["transactionHash"]).hex(), swap_for_y, amount_in, evs))
            counts[swap_for_y] += 1
        lowest = frm
        to_block = frm - 1
    return cases, skipped, (lowest, head)


def replay_case(w3: Web3, chain: str, pair: str, version: str, case: ReplayCase, allowlist) -> list[str]:
    """Re-run one swap with v4quote on state(block-1); return a list of human-readable differences."""
    ts = int(w3.eth.get_block(case.block)["timestamp"])
    for side in (40, 200, 1000):
        state = load_lb_pair(w3, pair, case.block - 1, version=version, bins_each_side=side, chain=chain)
        try:
            res, _ = lb_swap(state, case.amount_in, case.swap_for_y, ts, hooks_allowlist=allowlist)
            break
        except InsufficientState:
            continue
    diffs = []
    in_y = not case.swap_for_y  # swapForY: input is X, output is Y
    out_y = case.swap_for_y
    ev_out = sum(_side(e.amounts_out, out_y) for e in case.events)
    if res.amount_out != ev_out:
        diffs.append(f"amountOut core={res.amount_out} chain={ev_out} diff={res.amount_out - ev_out}")
    ours = [(b.bin_id, b.amount_in, b.amount_out, b.volatility_accumulator, b.total_fees, b.protocol_fees)
            for b in res.bin_swaps]
    theirs = [(e.bin_id, _side(e.amounts_in, in_y), _side(e.amounts_out, out_y), e.vol_acc,
               _side(e.total_fees, in_y), _side(e.protocol_fees, in_y)) for e in case.events]
    if ours != theirs:
        diffs.append(f"per-bin events differ: core={ours} chain={theirs}")
    if res.active_id != case.events[-1].bin_id:
        diffs.append(f"activeId core={res.active_id} chain={case.events[-1].bin_id}")
    if res.params.volatility_accumulator != case.events[-1].vol_acc:
        diffs.append(f"volAcc core={res.params.volatility_accumulator} chain={case.events[-1].vol_acc}")
    return diffs
