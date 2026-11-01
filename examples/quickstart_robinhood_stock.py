"""How many USDG do I get for 1 NVDA stock token on Robinhood Chain? Computed locally, checked against QuoterV2.

    ROBINHOOD_RPC_URL=https://rpc.mainnet.chain.robinhood.com uv run python examples/quickstart_robinhood_stock.py

Pool: NVDA/USDG Uniswap v3, fee 500 (the token's highest-volume pool, week 2026-10-19..26). The quote is in raw token
units: 1 NVDA token = 10**18 units. ``uiMultiplier()`` (ERC-8056) only changes how a wallet DISPLAYS a balance
(dividends / splits), never the amounts a pool or a quoter uses.
"""

from v4quote.amm.v3 import pool as v3
from v4quote.chain.loaders import load_v3_pool
from v4quote.chain.multicall import Call, eth_call
from v4quote.chain.quoters import v3_quoter_exact_input_single
from v4quote.chain.rpc import get_web3

POOL = "0xd4eb21209c4d6093f80b5b84f5c45cc093ea14a3"  # NVDA/USDG, Uniswap v3 fee 500 (factory 0x1f7d7550)
QUOTER_V2 = "0x33e885ed0ec9bf04ecfb19341582aadcb4c8a9e7"  # Uniswap QuoterV2 on Robinhood Chain (chainId 4663)
USDG = "0x5fC5360D0400a0fD4F2Af552Add042D716F1d168"  # token0, 6 decimals
NVDA = "0xd0601CE157db5bdC3162BBAc2a2C8AF5320D9EeC"  # token1, 18 decimals

w3 = get_web3("robinhood")  # reads ROBINHOOD_RPC_URL; the URL never appears in errors or logs
block = w3.eth.block_number - 2

state = load_v3_pool(w3, POOL, block, words_each_side=8)
amount_in = 10**18  # 1 NVDA token (raw units)
ours = v3.quote_exact_input(state, False, amount_in).amount_out  # NVDA (token1) -> USDG (token0), computed locally
chain, _, _ = v3_quoter_exact_input_single(w3, QUOTER_V2, NVDA, USDG, state.fee, amount_in, block)
ui = eth_call(w3, Call(NVDA, "uiMultiplier()", (), ("uint256",)), block)

print(f"block {block}: 1 NVDA token -> {ours / 1e6:.6f} USDG (v4quote), {chain / 1e6:.6f} USDG (QuoterV2)")
print(f"uiMultiplier {ui / 1e18:.6f}: a wallet shows 1 token as {ui / 1e18:.6f} NVDA; the swap uses raw units")
assert ours == chain
