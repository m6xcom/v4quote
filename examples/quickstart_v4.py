"""Quote a Uniswap v4 pool offline from its state at a pinned block, then check the result against V4Quoter.

    AVALANCHE_RPC_URL=https://api.avax.network/ext/bc/C/rpc uv run python examples/quickstart_v4.py

The endpoint must serve state for the pinned block (recent blocks work on public endpoints; older ones need an
archive node).
"""
from v4quote.amm import uniswap_v4 as v4
from v4quote.chain.loaders import load_v4_pool
from v4quote.chain.quoters import v4_quoter_exact_input_single
from v4quote.chain.rpc import get_web3

# Avalanche C-Chain, Uniswap v4: native AVAX / USDC, fee 500 pips, tickSpacing 10, no hooks
STATE_VIEW = "0xc3c9e198c735a4b97e3e683f391ccbdd60b69286"
V4_QUOTER = "0xbe40675bb704506a3c2ccfb762dcfd1e979845c2"
USDC = "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E"
KEY = v4.PoolKey(v4.ZERO_ADDRESS, USDC, 500, 10)

w3 = get_web3("avalanche")  # reads AVALANCHE_RPC_URL; the URL never appears in errors or logs
block = w3.eth.block_number - 2

state = load_v4_pool(w3, STATE_VIEW, KEY, block, words_each_side=16)  # slot0, liquidity, tick bitmap words
amount_in = 25 * 10**18  # 25 AVAX
ours = v4.quote_exact_input(state, True, amount_in).amount_out  # zero_for_one: AVAX -> USDC, pure Python math
chain = v4_quoter_exact_input_single(w3, V4_QUOTER, KEY, True, amount_in, block)

print(f"block {block}: v4quote {ours}  V4Quoter {chain}  diff {ours - chain}")
assert ours == chain
