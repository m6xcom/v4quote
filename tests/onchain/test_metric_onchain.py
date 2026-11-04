"""Metric OMM pools on Robinhood Chain vs MetricOmmSwapQuoter at one pinned block (marker ``onchain``).

Needs ROBINHOOD_RPC_URL (skipped otherwise). For each pool the state is read with ``load_metric_pool`` (live provider
quote at the block); the same raw bid / ask / reference is passed to ``quoteHypotheticalExact{Input,Output}Single``,
so both sides price identical state. Sizes: dust, inside the first bin, at its end, inside the second bin, across
several bins, random, and more than the pool holds. A quote must equal the quoter to the unit, revert alike, or be
refused by the model (exact input only); a refused exact-input quote is also run with ``bound="lower"``, which must not
exceed the quoter. The counts are printed in the summary table.

A public endpoint keeps recent state for a limited time only; the test reads everything it needs within a few minutes
of the pinned block.
"""

from __future__ import annotations

import random

import pytest

from v4quote.amm import metric_omm as mo
from v4quote.chain.metric import metric_state_from_snapshot, read_metric_snapshot
from v4quote.errors import MathRevert, UnreadableState, UnsupportedPool
from tests.onchain.conftest import Comparison
from tests.onchain.metric_record import ROBINHOOD_POOLS, quote_sizes, raw_quote
from tests.unit.test_metric_omm import SELECTOR_NAMES

pytestmark = pytest.mark.onchain


def _sizes(state: mo.MetricOmmPoolState, zfo: bool, exact_in: bool, rng: random.Random) -> list[int]:
    all_sizes = quote_sizes(state, zfo, exact_in, rng, n_random=2, nbins=3)
    keep = {all_sizes[0], all_sizes[-1]}  # dust and more than the pool holds
    keep.update(rng.sample(all_sizes[1:-1], min(6, len(all_sizes) - 2)))
    return sorted(keep)


@pytest.mark.parametrize("name", list(ROBINHOOD_POOLS))
def test_metric_pool_matches_quoter(name, chain_ctx, report):
    ctx = chain_ctx("robinhood")
    pool = ROBINHOOD_POOLS[name]
    try:
        snap = read_metric_snapshot(ctx.w3, pool, ctx.block, chain="robinhood")
    except UnreadableState as exc:
        report.notes.append(f"{name}: no live provider quote at block {ctx.block} ({exc.reason}); skipped")
        pytest.skip(f"{name}: provider quote unreadable")
    state = metric_state_from_snapshot(snap)
    prices = tuple(int(p) for p in snap["prices"])
    rng = random.Random(f"metric-{name}-{ctx.block}")
    counts = [0, 0, 0, 0]  # equal, refused, different, revert alike
    bounds = [0, 0]  # refused exact-input quotes answered with bound="lower", of them above the quoter
    out_counts = [0, 0]  # exact output: quotes, equal or revert alike
    for zfo in (False, True):
        for exact_in in (True, False):
            for amount in _sizes(state, zfo, exact_in, rng):
                q_in, q_out, err = raw_quote(ctx.w3, pool, zfo, exact_in, amount, prices, ctx.block)
                case = f"{'sell0' if zfo else 'buy0'} {'in' if exact_in else 'out'} {amount}"
                try:
                    r = (mo.quote_exact_input if exact_in else mo.quote_exact_output)(state, zfo, amount)
                    ours = (r.amount_in, r.amount_out)
                except UnsupportedPool:
                    counts[1] += 1
                    report.rows.append(Comparison(name, ctx.block, case, "refused", "refused", "model refused"))
                    if exact_in and not err:
                        try:
                            low = mo.quote_exact_input(state, zfo, amount, bound="lower")
                        except UnsupportedPool:
                            continue
                        bounds = [bounds[0] + 1, bounds[1] + (low.amount_out > int(q_out))]
                    continue
                except MathRevert as exc:
                    ours = str(exc)
                chain = SELECTOR_NAMES.get(err, err) if err else (int(q_in), int(q_out))
                if ours == chain:
                    counts[3 if err else 0] += 1
                else:
                    counts[2] += 1
                if not exact_in:
                    out_counts = [out_counts[0] + 1, out_counts[1] + (ours == chain)]
                report.rows.append(Comparison(name, ctx.block, case, str(ours), str(chain)))
    report.notes.append(f"{name}: {sum(counts)} quotes, {counts[0]} equal to the unit, {counts[3]} revert alike, "
                        f"{counts[1]} refused by the model, {counts[2]} different; exact output {out_counts[1]}/{out_counts[0]}; "
                        f"lower bound on {bounds[0]} refused, {bounds[1]} above the quoter")
    assert counts[2] == 0 and bounds[1] == 0
