"""v4quote ERC-4626 vault (Gnosis sDAI) vs the vault's own ``previewDeposit`` / ``previewRedeem`` (marker ``onchain``;
needs GNOSIS_RPC_URL, an archive endpoint, e.g. the keyless https://rpc.gnosis.gateway.fm).

* ``GNOSIS_ERC4626_BLOCKS`` (default 12) blocks spread over the last ~7 days (120 960 blocks): ``load_erc4626`` (which
  itself refuses a vault whose previews differ from the model at 5 amounts) and 2 x 10 further amounts from 1 wei to
  1/3 of the vault, compared bit for bit with ``previewDeposit`` / ``previewRedeem`` at the same block; the native port
  (``v4quote_native``, when installed) gives the same values;
* the state refresh calls (``totalAssets``, ``totalSupply``) reproduce the loaded state.
"""
from __future__ import annotations

import os

import pytest

from v4quote.amm import erc4626 as m
from v4quote.amm import native
from v4quote.chain.loaders import erc4626_state_calls, load_erc4626
from v4quote.chain.multicall import Call, multicall

pytestmark = pytest.mark.onchain

SDAI = "0xaf204776c7245bf4147c2612bf6e5972ee483701"
WXDAI = "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"


def _row(block, case, ours, chain):
    from tests.onchain.conftest import Comparison  # noqa: PLC0415
    return Comparison(SDAI, block, case, ours, chain)


def test_previews_match_on_many_blocks(chain_ctx, report):
    ctx = chain_ctx("gnosis")
    n = int(os.environ.get("GNOSIS_ERC4626_BLOCKS", "12"))
    step = max(1, 120_960 // max(n, 1))
    cn = None
    try:
        import v4quote_native as cn  # noqa: PLC0415
        if getattr(cn.Pool, "erc4626", None) is None:
            cn = None
    except ImportError:
        pass
    checked = 0
    for k in range(n):
        b = ctx.block - k * step
        st = load_erc4626(ctx.w3, SDAI, b, expected_asset=WXDAI)
        assert st.decimals_offset == 0
        amounts = sorted({1, 2, 3, 999, 10**15, 10**18 + 7, 123_456 * 10**18, st.total_assets // 1000,
                          st.total_supply // 7, st.total_assets // 3})
        calls = ([Call(SDAI, "previewDeposit(uint256)", (a,), ("uint256",)) for a in amounts]
                 + [Call(SDAI, "previewRedeem(uint256)", (a,), ("uint256",)) for a in amounts])
        got = multicall(ctx.w3, calls, b)
        for a, want_d, want_r in zip(amounts, got[: len(amounts)], got[len(amounts):]):
            d, r = m.preview_deposit(st, a), m.preview_redeem(st, a)
            report.rows.append(_row(b, f"previewDeposit({a})", d, want_d))
            report.rows.append(_row(b, f"previewRedeem({a})", r, want_r))
            assert (d, r) == (want_d, want_r), (b, a)
            if cn is not None:
                h = native.build_handle(st, cn)
                assert h.quote(0, 1, a)[0] == d and h.quote(1, 0, a)[0] == r
            checked += 2
        ta, ts = multicall(ctx.w3, erc4626_state_calls(SDAI), b)
        assert (ta, ts) == (st.total_assets, st.total_supply)
    report.notes.append(f"erc4626: {checked} preview comparisons on {n} blocks (native {'on' if cn else 'off'})")
