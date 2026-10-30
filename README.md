# v4quote

Exact-integer AMM quote math for Python: Uniswap v2 / v3 / v4 (including some v4 hooks), Algebra Integral,
LFJ Liquidity Book and OpenZeppelin ERC-4626 vaults. Given a
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

```bash
uv run pytest -m "not onchain"                 # offline suite
AVALANCHE_RPC_URL=... uv run pytest -m onchain  # on-chain comparisons for one chain
```

## Install

Requires Python >= 3.11. The native extra needs Python >= 3.12 and a Rust toolchain >= 1.90.

```bash
git clone <this repo> && cd v4quote
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
their results exactly. Test vectors in `tests/fixtures/uniswap_v3_*.json` and in `tests/unit/test_v3_math.py` come
from the test suites of Uniswap v3-core and v4-core. This project is not affiliated with or endorsed by Uniswap Labs
or the other protocols' teams.
