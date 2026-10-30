# Contributing

Thanks for helping. The bar for this library is "equal to the chain, to the wei".

* **Exactness first.** Every formula mirrors the contract's integer arithmetic (rounding direction, overflow
  checks, `unchecked` blocks). No floats in `v4quote.amm` / `v4quote.math`; `tests/unit/test_no_float.py` checks it.
* **Refuse, don't guess.** Unmodelled behaviour raises `UnsupportedPool`; missing loaded state raises
  `InsufficientState`; a contract revert raises `MathRevert`.
* **Every new model needs two kinds of tests:** offline tests (vectors, edges, reverts) and an `onchain` test that
  compares it with the protocol's own quoter or `swap()` via `eth_call` at a pinned block. On-chain tests must skip
  cleanly when their `<CHAIN>_RPC_URL` is unset.
* **Native backend.** If you change math that `native/` ports, keep both in sync. `tests/unit/test_native_equivalence.py`
  must stay green with the extension built (`uv sync --extra dev --extra native`) and without it.
* **No secrets.** RPC URLs come from environment variables only (see `.env.example`). Never commit URLs with keys,
  and never log them. `v4quote.chain.rpc` masks them in errors.

Workflow:

```bash
uv sync --extra dev
uv run pytest -m "not onchain"     # must pass
cd native && cargo test --locked   # if you touched native/
```

Open a pull request with a short description of the change and the evidence (test names, chain and block of the
on-chain comparison).

By contributing you agree that your contribution is licensed under GPL-2.0-or-later, the license of this project.
