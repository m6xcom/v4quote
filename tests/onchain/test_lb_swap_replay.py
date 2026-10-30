"""Empirical check of REAL LBPair.swap() calls (hooks executed) against v4quote.

getSwapOut/getSwapIn/LBQuoter are view functions and never run hooks, so the quoter comparison in
test_quotes.py cannot show whether a hook changes swap() results. Here historical Swap events of the
hooked pools are replayed (see lb_replay.py). Requires <CHAIN>_RPC_URL with historical state; for
Monad set MONAD_LOGS_RPC_URL to an endpoint allowing large eth_getLogs ranges.
"""

import os

import pytest

from v4quote.amm.liquidity_book import DEFAULT_HOOKS_ALLOWLIST, REPLAY_PENDING
from tests.onchain import lb_replay
from tests.onchain.conftest import pools_of_type

pytestmark = pytest.mark.onchain

WANT_PER_DIRECTION = int(os.environ.get("LB_REPLAY_PER_DIRECTION", "12"))
MAX_BLOCKS = {"monad": int(os.environ.get("LB_REPLAY_MAX_BLOCKS_MONAD", "600000")),
              "avalanche": int(os.environ.get("LB_REPLAY_MAX_BLOCKS_AVALANCHE", "400000"))}
HOOKED = [p for p in pools_of_type("lfj_lb") if str(p.values[0].get("version")) == "2.2"]


@pytest.mark.parametrize("p", HOOKED)
def test_replay_real_swaps(p, chain_ctx, report):
    ctx = chain_ctx(p["chain"])
    allow = DEFAULT_HOOKS_ALLOWLIST
    cases, skipped, span = lb_replay.collect_cases(
        p["chain"], p["pool"], ctx.block, WANT_PER_DIRECTION, MAX_BLOCKS[p["chain"]]
    )
    stats = lb_replay.ReplayStats(p["id"], blocks_scanned=span)
    for bn, why in skipped:
        stats.skipped_prior_activity += 1
        stats.skipped_reasons[why] = stats.skipped_reasons.get(why, 0) + 1
    for case in cases:
        stats.considered += 1
        diffs = lb_replay.replay_case(ctx.w3, p["chain"], p["pool"], str(p["version"]), case, allow)
        report.rows.append(
            __import__("tests.onchain.conftest", fromlist=["Comparison"]).Comparison(
                p["id"], case.block, f"replay swap() swapForY={case.swap_for_y} in={case.amount_in} bins={len(case.events)}",
                "match" if not diffs else "DIFF", "match", case.tx[:18],
            )
        )
        if diffs:
            stats.mismatches.append((case.block, case.tx, diffs))
        else:
            stats.matched += 1
            stats.by_direction[case.swap_for_y] += 1
    report.notes.append(
        f"replay {p['id']}: blocks {span[0]}..{span[1]}, considered={stats.considered} matched={stats.matched} "
        f"(swapForY: {stats.by_direction[True]}, swapForX: {stats.by_direction[False]}), "
        f"skipped(prior pair activity in block)={stats.skipped_prior_activity} {stats.skipped_reasons}, "
        f"mismatches={len(stats.mismatches)}"
    )
    for m in stats.mismatches:
        report.notes.append(f"replay mismatch {p['id']} block {m[0]} tx {m[1]}: {m[2]}")
    assert not stats.mismatches
    assert stats.considered >= 20, f"only {stats.considered} swaps found in {span}"
    if (p["chain"], p["pool"].lower()) in REPLAY_PENDING:
        # acceptance criterion: >= 24 replayed swaps per pair, 0 wei mismatches
        assert stats.matched >= 24, f"{p['id']}: only {stats.matched} swaps replayed (need 24) in {span}"
    assert stats.by_direction[True] > 0 and stats.by_direction[False] > 0
