"""Robinhood Chain stock-token pools vs QuoterV2 / V4Quoter / the pool's own swap() at one pinned block (marker
``onchain``). Needs ROBINHOOD_RPC_URL (skipped otherwise).

Each pool of ``rh_stocks_record.POOLS`` is loaded and quoted live exactly as the fixture recorder does (exact input and
exact output, both directions, dust to ~20 % price moves), then every reference quote must equal the model to the wei
or revert alike. For fee-only hook pools the FablesRWA ``currentFee(poolId, zeroForOne)`` getter must also agree with
the fee read through ``beforeSwap``.

The public endpoint keeps state for ~6,000 blocks (~10 minutes); the whole module needs about 7 minutes with two
requests in flight at most, so run it on its own (``-k robinhood_stocks``).
"""

from __future__ import annotations

import pytest

from v4quote.amm import serialization as ser
from v4quote.chain.multicall import Call, eth_call
from tests.onchain.conftest import Comparison
from tests.onchain.rh_stocks_record import KIND_V4_HOOK, POOLS, record_pool
from tests.unit.test_robinhood_stocks import replay

pytestmark = pytest.mark.onchain


@pytest.mark.parametrize("spec", POOLS, ids=[s.name for s in POOLS])
def test_stock_pool_matches_reference(spec, chain_ctx, report):
    ctx = chain_ctx("robinhood")
    entry = record_pool(ctx.w3, spec, ctx.block)
    c = replay(entry)
    state = ser.from_dict(entry["state"])
    if spec.kind == KIND_V4_HOOK and "Fables" in spec.name:
        pid = state.key.pool_id()
        got = tuple(eth_call(ctx.w3, Call(spec.hooks, "currentFee(bytes32,bool)", (pid, z), ("uint24",)), ctx.block)
                    for z in (True, False))
        assert got == state.hook_fee
    report.rows.append(Comparison(spec.name, ctx.block, f"{len(entry['quotes'])} quotes ({entry['reference']})",
                                  c[0] + c[1], len(entry["quotes"]), f"refused {c[2]}, different {c[3]}"))
    report.notes.append(f"{spec.name}: {sum(c)} quotes, {c[0]} equal to the wei, {c[1]} revert alike, {c[2]} refused, "
                        f"{c[3]} different")
    assert c[3] == 0, c
