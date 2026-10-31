"""Quote a Metric OMM pool on Robinhood Chain and check it against MetricOmmSwapQuoter (see README)."""

from v4quote.amm import metric_omm as mo
from v4quote.chain.metric import load_metric_pool, metric_quoter_exact_output
from v4quote.chain.rpc import get_web3
from v4quote.errors import UnsupportedPool

POOL = "0x1E8Fc442AD3606719F7016DED6C17B56C3cA7386"  # WETH/USDG on Robinhood Chain (chainId 4663)
w3 = get_web3("robinhood")  # reads ROBINHOOD_RPC_URL
block = w3.eth.block_number - 2
state = load_metric_pool(w3, POOL, block)  # UnreadableState if the provider gives no quote (stale feed, closed market)

ours = mo.quote_exact_output(state, False, 10**17)  # buy 0.1 WETH: USDG in, to the unit
prices = (state.bid_x64, state.ask_x64, state.reference_x64)
assert (ours.amount_in, ours.amount_out) == metric_quoter_exact_output(w3, POOL, False, 10**17, prices, block)

try:
    sold = mo.quote_exact_input(state, True, 10**17)  # sell 0.1 WETH for USDG
except UnsupportedPool:
    sold = None  # the pool's own in-bin rounding could change the last unit: ask the quoter instead
print("exact output (buy 0.1 WETH):", ours)
print("exact input (sell 0.1 WETH):", sold)
