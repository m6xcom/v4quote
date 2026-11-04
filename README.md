# v4quote

Exact-integer AMM quote math for Python: Uniswap v2 / v3 / v4 (including some v4 hooks), Algebra Integral,
LFJ Liquidity Book, OpenZeppelin ERC-4626 vaults and Metric OMM pools. Given a
pool's state at a block, `v4quote` returns the same amounts as the protocol's own quoter or `swap()`, to the wei,
including the cases where the contract reverts. It also includes a thin web3.py layer that loads pool state at a
pinned block (Multicall3, StateView, tick-bitmap words) and an optional Rust backend that returns identical results.

* **Exact, not approximate.** Python integers mirror the Solidity arithmetic (rounding, overflow checks,
  `unchecked` blocks). The swap path never uses floats, and `tests/unit/test_no_float.py` enforces this.
* **Refuses instead of guessing.** A pool feature that is not modelled raises `UnsupportedPool` (for example a
  swap-altering v4 hook, or an LB hook outside the allowlist). A swap that walks past the ticks or bins you loaded
  raises `InsufficientState`. A contract revert raises `MathRevert`.
* **Pure functions over frozen state dataclasses.** You can snapshot state, serialise it to JSON losslessly and
  replay quotes offline.

## Supported pools

| Module | Pools |
|---|---|
| `v4quote.amm.uniswap_v2` | Uniswap v2 and x*y=k forks with a parameterised fee (`getAmountOut` / `getAmountIn`) |
| `v4quote.amm.v3` | Uniswap v3 TickMath / SqrtPriceMath / SwapMath / swap loop with the word-by-word tick-bitmap walk. Exact forks: PancakeSwap v3, Pharaoh (Ramses) V3 |
| `v4quote.amm.v3.pool_address` | CREATE2 pool addresses (parameterised deployer, init-code hash and key type) |
| `v4quote.amm.uniswap_v4` | Uniswap v4 pools without swap-altering hooks (v4 SwapMath, LP fee + protocol fee). Optional: fee-only hooks (`hook_fee`) and modelled return-delta hooks (`rd_hook`) |
| `v4quote.amm.v4_rd_hooks` | Return-delta hook models: `output_bps` (PonsV2MemeHook), `quote_bps` (PangeaFeeHook) |
| `v4quote.amm.algebra_integral` | Algebra Integral (v1.2) pools: v3 math over the linked-list tick table, fee = `fee()` / the `SwapFee` log |
| `v4quote.amm.liquidity_book` | LFJ Liquidity Book v2.1 / v2.2 and Pharaoh DLMM (`getSwapOut`, `getSwapIn`, `swap`), with an allowlist of reviewed LB hooks |
| `v4quote.amm.erc4626` | OpenZeppelin v5 ERC-4626 vault treated as an asset <-> shares pool |
| `v4quote.amm.metric_omm` | Metric OMM pools (oracle-priced bins): exact-output quotes; exact-input quotes that are refused (`UnsupportedPool`) when the pool's in-bin rounding could change the result, or with `bound="lower"` return a lower bound of the pool's output instead. Loader and quoter wrappers in `v4quote.chain.metric` |
| `v4quote.routing` | `best_route`: exact-input route over the states above, up to 2 hops and a 2-way split, with gas units per hop |
| `v4quote.chain` | State loaders at a pinned block, Multicall3, quoter wrappers, `eth_getLogs` scanner, RPC from env with URL masking |
| `v4quote.amm.serialization` | Lossless JSON (de)serialisation of every pool state |
| `v4quote.amm.native` | Optional Rust backend (`native/`, module `v4quote_native`) for v2 / v3 / v4 / ERC-4626 exact-input quotes |

## Verification

Each claim below is backed by a test in this repository.

Offline (no network). Run all of them with `uv run pytest -m "not onchain"`.

| Claim | Test |
|---|---|
| v3 TickMath / SqrtPriceMath / SwapMath equal the Uniswap v3-core and v4-core test vectors (incl. 28 + 13 TickMath snapshot vectors) | `tests/unit/test_v3_math.py` |
| v3 swap loop reproduces all 240 cases of v3-core's `UniswapV3Pool.swaps.spec.ts` snapshot | `tests/unit/test_v3_pool.py` |
| Algebra Integral: 20 exact-input swap sequences (1-2 pools, 2 hops) recorded on chain with `eth_simulateV1` (Robinhood Chain, block 100117724): 19 equal the model to the wei, 1 is refused for insufficient loaded state (never a wrong number) | `tests/unit/test_algebra_integral.py` |
| ERC-4626 (OZ v5) previews / deposit / redeem, incl. values read from Gnosis sDAI | `tests/unit/test_erc4626.py` |
| Liquidity Book math, hook rules and fee parameters | `tests/unit/test_liquidity_book.py`, `tests/unit/test_lb_loader_*.py` |
| v4 return-delta hook models follow the hooks' contract arithmetic. These are offline tests only: there are no V4Quoter vectors for these hooks yet | `tests/unit/test_v4_rd_hooks.py` |
| The Rust backend is bit-exact with Python: 4,200 randomized v3 quotes and 2,600 v4 quotes, 0 fallbacks (counts printed with `-s`) | `tests/unit/test_native_equivalence.py`, `tests/unit/test_native_v4.py` |
| Serialisation round-trips every state kind | `tests/unit/test_serialization.py` |
| Robinhood Chain stock tokens: 625 reference quotes on 25 USDG pools of 10 stock tokens recorded at block 103531492 (Uniswap v3 vs QuoterV2; Ramses V3 and Slipstream CL forks vs the pool's own `swap()` run by eth_call; Uniswap v4 without hooks and v4 dynamic-fee pools with fee-only hooks (SessionFeeHook, FablesRWA) vs V4Quoter; exact input and output, both directions, dust to ~20 % price moves crossing up to 220 initialised ticks): 625 equal to the wei, 0 refused, 0 different | `tests/unit/test_robinhood_stocks.py` |
| Metric OMM: 1,760 `quoteHypothetical*` calls on 10 Robinhood Chain pools (eth_call on a fork of block 102703478, stock / WETH / 9- and 18-decimal pairs, both directions, exact input and output): 880/880 exact-output quotes equal to the unit, 589 exact-input quotes equal, 15 revert alike, 276 exact-input quotes refused, 0 different. With `bound="lower"` 270 of the 276 get a lower bound, never above the quoter (median gap 2e-15 of the amount, 0.2 % at most on 1,000-wei dust inputs), 6 stay refused (bound of zero units). Price validation (asymmetry bound) on 88 provider quotes matches the pool | `tests/unit/test_metric_omm.py` |

On-chain (marker `onchain`). These compare the model with the protocol's own contracts at one pinned block. Each
chain's tests run only when `<CHAIN>_RPC_URL` is set and are skipped otherwise. Some of them need an archive
endpoint, and rate-limited public endpoints can fail with HTTP 429.

| Claim | Test | Env |
|---|---|---|
| v2 / v3 / v4 / LB quotes equal Router02, QuoterV2, V4Quoter and `getSwapOut` / `getSwapIn` / LBQuoter (Monad, Avalanche pools in `tests/fixtures/pools.yaml`) | `tests/onchain/test_quotes.py` | `MONAD_RPC_URL`, `AVALANCHE_RPC_URL` |
| Real historical LB `swap()` calls (hooks executed) replay to the wei | `tests/onchain/test_lb_swap_replay.py` | `AVALANCHE_RPC_URL` |
| Fixture pools / hooks are what the official factories and StateView return | `tests/onchain/test_address_links.py`, `tests/onchain/test_routing_replay.py` | `MONAD_RPC_URL`, `AVALANCHE_RPC_URL` |
| ERC-4626 (sDAI) matches the deployed vault's previews on many blocks | `tests/onchain/test_erc4626_onchain.py` | `GNOSIS_RPC_URL` |
| PulseX v1 / v2 pairs and a Uniswap v3 copy match their routers, pair K checks and QuoterV2 | `tests/onchain/test_pulsechain_onchain.py` | `PULSECHAIN_RPC_URL` |
| Robinhood Chain stock-token pools (25: v3, Ramses V3, Slipstream, v4, v4 fee-only hooks) match QuoterV2, V4Quoter or the pool's own `swap()` to the wei at a live pinned block; FablesRWA `currentFee` equals the fee read through `beforeSwap` | `tests/onchain/test_robinhood_stocks_onchain.py` | `ROBINHOOD_RPC_URL` |
| Metric OMM pools (10 on Robinhood Chain) match `MetricOmmSwapQuoter.quoteHypothetical*` priced with the provider quote read at the same block: every quote equal to the unit, reverting alike, or refused by the model | `tests/onchain/test_metric_onchain.py` | `ROBINHOOD_RPC_URL` |

```bash
uv run pytest -m "not onchain"                 # offline suite
AVALANCHE_RPC_URL=... uv run pytest -m onchain  # on-chain comparisons for one chain
```

## Install

Requires Python >= 3.11.

```bash
pip install v4quote
```

The PyPI package is pure Python. The optional Rust backend gives the same results faster; it is built from source
and needs Python >= 3.12 and a Rust toolchain >= 1.90:

```bash
git clone https://github.com/m6xcom/v4quote && cd v4quote
uv sync --extra dev                    # pure Python
uv sync --extra dev --extra native     # + Rust backend (builds ./native with maturin)
```

You can also use pip: `pip install -e ".[dev]"`. To add the Rust backend, also run `pip install ./native`.

The Rust backend is used when it imports, was built from the current `native/` sources (a build fingerprint
is checked), and `V4QUOTE_NATIVE` is not `0`. Otherwise every call runs the Python code, and the results are the
same. `v4quote.amm.native.describe()` tells you which backend is active and why. The native path for v4 pools is
opt-in: call `native.enable_v4()`.

Logging: `v4quote.chain.rpc.get_web3` / `web3_from_url` raise the `web3.providers`, `web3._utils.http_session_manager`,
`urllib3.connectionpool` and `urllib3` loggers to INFO (when unset or lower) and add a redaction filter, because their
DEBUG records contain the endpoint URL, which often embeds an API key. A level you set to INFO or higher is kept.

## Quickstart: quote a v4 pool and check it against V4Quoter

`examples/quickstart_v4.py`:

```python
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

state = load_v4_pool(w3, STATE_VIEW, KEY, block, words_each_side=16)
amount_in = 25 * 10**18  # 25 AVAX
ours = v4.quote_exact_input(state, True, amount_in).amount_out  # AVAX -> USDC, computed locally
chain = v4_quoter_exact_input_single(w3, V4_QUOTER, KEY, True, amount_in, block)
assert ours == chain
```

```bash
AVALANCHE_RPC_URL=https://api.avax.network/ext/bc/C/rpc uv run python examples/quickstart_v4.py
```

When the swap walks past the loaded tick-bitmap words, the quote raises `InsufficientState` instead of returning a
wrong number. Reload with a larger `words_each_side` (or `None` for the full bitmap). After that, `state` is a plain
frozen dataclass: quote it as often as you like, apply a swap with `v4.apply_swap`, or store it with
`v4quote.amm.serialization`.

## Quote a Metric pool

A Metric pool prices from its provider's bid / ask / reference and holds liquidity in bins around the reference.
`load_metric_pool` reads every bin, the pool's position and the provider quote at one block; the same raw quote can be
passed to the pool's hypothetical quoter, so both sides price identical state.

```python
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

# or accept a conservative number: never more than the pool pays (exact=False when it may be lower)
low = mo.quote_exact_input(state, True, 10**17, bound="lower")
print(low.amount_out, low.exact)
```

The lower bound is the low end of a rounding band calibrated on quoter measurements (it is not derived from the pool's
code), the same idea as the safety margin in Metric's "Estimate a fill from depth" guide; read from pool state, the
band is usually a few units wide.

```bash
ROBINHOOD_RPC_URL=https://rpc.mainnet.chain.robinhood.com uv run python examples/quickstart_metric.py
```

Prices in a Metric state are the provider's raw quote. The public API's `bidAdj` / `askAdj` and depth already include
the spread and fees and cannot reproduce the pool to the unit, so the model is loaded from chain state;
`v4quote.chain.metric` also has helpers for the API's pool list and status (`check_api_quotable`). Swap extensions
(StopLoss, velocity guards, allowlists) can only accept or reject a swap, so they never change a quoted amount, but
they can reject it, exactly as they would make the quoter revert.

## Robinhood Chain stock tokens

Robinhood Chain (chainId 4663) stock tokens (NVDA, SPCX, SPY, ...) are plain ERC-20s with 18 decimals, traded mostly
against USDG (6 decimals). Their pools are ordinary concentrated-liquidity pools, so they are quoted with the modules
above:

| Pool type | How | Reference it is checked against |
|---|---|---|
| Uniswap v3 (factory `0x1f7d7550`) | `load_v3_pool` + `v4quote.amm.v3.pool` | QuoterV2 `0x33e885ed` |
| Ramses V3 CL fork (factory `0xe0c4ceb9`, owner-settable fee) | same, `uniswap_fee_protocol=False`; fee = `fee()` at the block | the pool's own `swap()` (`v3_pool_swap_sim`, `v3_pool_swap_sim_exact_output`) |
| Velodrome Slipstream CL clone (factory `0x1ac9db4a`) | same; fee = `fee()` = `factory.getSwapFee(pool)` at the block (dynamic) | the pool's own `swap()` |
| Uniswap v4 without hooks | `load_v4_pool` (StateView `0xf3334192`) + `uniswap_v4` | V4Quoter `0x8dc178ef` |
| Uniswap v4 dynamic fee + fee-only hook | `v4quote.chain.hook_fees.with_hook_fee`: calls the hook's `beforeSwap` from the PoolManager at the block and sets `hook_fee` | V4Quoter |

`hook_fees.FEE_ONLY_HOOKS` lists the reviewed hooks (verified source on Sourcify, pinned code hash): SessionFeeHook
`0xc15b155a` (SPCX; fee by US equity session, 0 for exempt senders, so the fee depends on the `sender` that calls
`PoolManager.swap`: pass your router, or V4Quoter to compare with it) and FablesRWA `0x5eb87f69`, `0xb608a787`,
`0x67d86050`, `0x70a9a884` (SPCX, CRCL, GLD, TSLA, AAPL; session fee curve, poke, cap, per direction). Both only
return an LP-fee override from a `view` `beforeSwap`, with no delta. The fee is the one for a swap in a block with the
read block's timestamp; session boundaries change it, so re-read it per block. Any other hook, or a reviewed address
whose code changed, raises `UnsupportedPool`.

Amounts are raw token units (1 token = 10**18). `uiMultiplier()` (ERC-8056) is display only: a wallet shows a
balance multiplied by it (dividends, splits), but pools and quoters never use it, so do not apply it to quote inputs or
outputs. `examples/quickstart_robinhood_stock.py` quotes 1 NVDA token to USDG and checks it against QuoterV2:

```bash
ROBINHOOD_RPC_URL=https://rpc.mainnet.chain.robinhood.com uv run python examples/quickstart_robinhood_stock.py
```

The public endpoint keeps state for ~6,000 blocks (~10 minutes): load and compare within that window, or record
fixtures as `tests/onchain/rh_stocks_record.py` does.

## Adding a hook model

A v4 pool whose hook changes swap amounts is rejected (`UnsupportedPool`) until the hook has an exact model. To add
one:

1. **Read the verified source.** Model only hooks whose swap path is pure arithmetic over known per-pool terms (for
   example a fee in bps of one leg). Note which callbacks run (`beforeSwap` / `afterSwap`), which leg they take
   from, and the exact rounding.
2. **Describe the model.** Add a kind to `v4quote.amm.v4_rd_hooks.RdHook`:
   * `covers(flags)`: the hook permission flags this kind accounts for;
   * `input_fee` / `output_fee`: the take before and after the pool swap, using the contract's integer
     arithmetic;
   * `check_post_tick`, if the hook reverts on the post-swap tick.

   `uniswap_v4.quote_exact_input` and the native hop apply these around the pool swap. Anything the model cannot
   reproduce must raise `MathRevert` or `UnsupportedPool`, never approximate.
3. **Write offline tests** for the arithmetic: edges, rounding, reverts. Use `tests/unit/test_v4_rd_hooks.py`
   as the pattern.
4. **Prove it against V4Quoter.** Add an `onchain` test that:
   * loads the pool at a pinned block with `load_v4_pool`;
   * sets `rd_hook` from the hook's on-chain per-pool terms;
   * asserts `v4.quote_exact_input(state, zfo, amount).amount_out == v4_quoter_exact_input_single(...)` for both
     directions and several sizes, including one that crosses initialised ticks.

   Pin the hook's code hash in the test so that a redeployment fails loudly.

Fee-only hooks (which only override the LP fee) do not need a new kind. Resolve the fee the hook will apply and pass
`hook_fee=(fee_zero_for_one, fee_one_for_zero)` in the state, then verify it against V4Quoter in the same way.
For a hook whose `beforeSwap` is a `view` fee override, add it to `v4quote.chain.hook_fees.FEE_ONLY_HOOKS` (code hash pinned) and `with_hook_fee` reads the fee from the hook itself.

## Layout

```
src/v4quote/      Python package (amm/, chain/, math/, routing.py, routing_pools.py, errors.py)
native/           optional Rust crate v4quote-native (pyo3 + maturin), bit-exact port of the hot quotes
tests/unit/       offline tests; fixtures in tests/fixtures/
tests/onchain/    comparisons against deployed contracts (marker onchain)
examples/         runnable examples
```

## License

Copyright (C) 2026 m6xcom

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later
version (SPDX: `GPL-2.0-or-later`). It is distributed WITHOUT ANY WARRANTY; see `LICENSE` for the full text.

## Acknowledgements

The swap math is an independent Python / Rust port of the arithmetic of the Uniswap v2-core, v3-core and v4-core
contracts, and of the Algebra Integral, LFJ Liquidity Book and OpenZeppelin ERC-4626 contracts, written to reproduce
their results exactly. The Metric OMM model is written from Metric's public documentation, its MIT-licensed interfaces
and lens contracts, and quoter measurements; it contains no code from Metric's BUSL-licensed pool contracts. Test vectors in `tests/fixtures/uniswap_v3_*.json` and in `tests/unit/test_v3_math.py` come
from the test suites of Uniswap v3-core and v4-core. This project is not affiliated with or endorsed by Uniswap Labs
or the other protocols' teams, including Metric.
