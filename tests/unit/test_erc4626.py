"""v4quote.amm.erc4626 (OZ v5 ERC-4626 vault as an asset <-> shares pool): formulas, reverts, post-states, routing /
routing integration and, when ``v4quote_native`` is installed, bit-exact equivalence of the Rust port.

On-chain reference values: Gnosis sDAI (0xaf20...3701) at block 48602693 (public RPC, 2026-10-05):
totalAssets 42576090319993786726811503, totalSupply 33782479856640913397697833,
previewDeposit(1e21) = 793461297238384930913, previewRedeem(1e21) = 1260300916352777384584.
``tests/onchain/test_erc4626_onchain.py`` checks the model against previewDeposit / previewRedeem on many blocks.
"""
from __future__ import annotations

import random
import re

import pytest

from v4quote import routing as rt
from v4quote.amm import erc4626 as m
from v4quote.amm import native
from v4quote.amm.erc4626 import ERC4626VaultState
from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT256

WXDAI = "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"
SDAI = "0xaf204776c7245bf4147c2612bf6e5972ee483701"
USDC = "0x2a22f9c3b484c3629090feed35f17ff8f88f76f0"
A0, S0 = 42576090319993786726811503, 33782479856640913397697833
E18 = 10**18


def sdai(a: int = A0, s: int = S0, o: int = 0) -> ERC4626VaultState:
    return ERC4626VaultState(SDAI, WXDAI, a, s, o, 48602693)


def test_onchain_reference_values():
    st = sdai()
    assert m.preview_deposit(st, 1000 * E18) == 793461297238384930913
    assert m.preview_redeem(st, 1000 * E18) == 1260300916352777384584
    assert st.coins == (WXDAI, SDAI)


def test_formulas_floor_and_ceil():
    st = sdai(1000, 800, 0)
    for a in (0, 1, 2, 7, 999, 10**30):
        assert m.preview_deposit(st, a) == a * 801 // 1001
        assert m.preview_withdraw(st, a) == -(-a * 801 // 1001)
        assert m.preview_redeem(st, a) == a * 1001 // 801
        assert m.preview_mint(st, a) == -(-a * 1001 // 801)
    off = sdai(0, 0, 6)  # empty vault with a decimals offset: 10**6 shares per asset unit
    assert m.preview_deposit(off, 5) == 5 * 10**6
    assert m.preview_redeem(off, 5 * 10**6) == 5


def test_round_trip_loses_at_most_two_wei():
    rnd = random.Random(4626)
    for _ in range(2000):
        a = rnd.randrange(10**20, 10**27)
        s = rnd.randrange(a // 2, a)
        st = sdai(a, s)
        x = rnd.choice([1, 2, 3, rnd.randrange(1, 10**6), rnd.randrange(1, 10**24)])
        shares, st2 = m.deposit(st, x)
        back = m.redeem_out(st2, shares)
        assert x - 2 <= back <= x


def test_post_states():
    st = sdai()
    sh, st2 = m.deposit(st, 10 * E18)
    assert (st2.total_assets, st2.total_supply) == (A0 + 10 * E18, S0 + sh)
    got, st3 = m.redeem(st2, sh)
    assert (st3.total_assets, st3.total_supply) == (st2.total_assets - got, S0)
    assert m.apply_flows(st, 5, -3, 1).total_assets == A0 + 5 and m.apply_flows(st, 5, -3).total_supply == S0 - 3
    with pytest.raises(ValueError):
        m.apply_flows(st, -A0 - 1, 0)


def test_reverts():
    st = sdai(100 * E18, 80 * E18)
    with pytest.raises(MathRevert, match=re.escape(m.ERR_MAX_REDEEM)):
        m.redeem_out(st, 80 * E18 + 1)
    with pytest.raises(MathRevert, match=re.escape(m.ERR_MULDIV)):
        m.preview_deposit(sdai(0, MAX_UINT256 - 5), 2)
    with pytest.raises(MathRevert, match=re.escape(m.ERR_SUPPLY_OVERFLOW)):
        m.deposit_out(sdai(E18, MAX_UINT256 - 10), 1)
    with pytest.raises(MathRevert, match=re.escape(m.ERR_ASSETS_OVERFLOW)):
        m.deposit_out(sdai(MAX_UINT256, 1), MAX_UINT256)
    with pytest.raises(ValueError):
        m.quote(st, 0, 0, 1)
    with pytest.raises(ValueError):
        m.preview_deposit(st, -1)
    for bad in ({"total_assets": -1}, {"total_supply": 1 << 256}, {"decimals_offset": 19}, {"asset": SDAI}):
        kw = {"address": SDAI, "asset": WXDAI, "total_assets": 1, "total_supply": 1, **bad}
        with pytest.raises(ValueError):
            ERC4626VaultState(**kw)


def test_quote_directions():
    st = sdai()
    assert m.quote(st, 0, 1, E18) == m.preview_deposit(st, E18)
    assert m.quote(st, 1, 0, E18) == m.preview_redeem(st, E18)


# ------------------------------------------------------------------------------------------------ routing
def test_unsupported_reasons_empty_and_missing_pool_kind():
    assert rt.unsupported_reasons(sdai()) == []
    mp = rt.MissingPool(SDAI, rt.KIND_ERC4626, (WXDAI, SDAI), "not loaded")
    with pytest.raises(rt.RouteInsufficientState) as ei:
        rt.best_route(WXDAI, SDAI, E18, [mp], on_insufficient="report")
    assert ei.value.pool_ids == (SDAI,)


# ------------------------------------------------------------------------------------------------ native port
def _nat(st, i, j, a):
    cn = pytest.importorskip("v4quote_native")
    h = native.build_handle(st, cn)
    try:
        return ("ok", h.quote(i, j, a)[0])
    except cn.Revert as e:
        return ("revert", str(e))
    except cn.Fallback:
        return ("fallback", None)


def _py(st, i, j, a):
    try:
        return ("ok", m.quote(st, i, j, a))
    except MathRevert as e:
        return ("revert", str(e))
    except ValueError:
        return ("other", None)


def test_native_vault_is_bit_exact():
    cn = pytest.importorskip("v4quote_native")
    if getattr(cn.Pool, "erc4626", None) is None:
        pytest.skip("v4quote_native built without the ERC-4626 port")
    rnd = random.Random(26)
    fallbacks = 0
    cases = [(sdai(), 0, 1, 1000 * E18), (sdai(), 1, 0, 1000 * E18), (sdai(0, 0, 3), 0, 1, 7),
             (sdai(0, MAX_UINT256 - 5), 0, 1, 2), (sdai(E18, MAX_UINT256 - 10), 0, 1, 1),
             (sdai(MAX_UINT256, 1), 0, 1, MAX_UINT256), (sdai(100, 80), 1, 0, 81), (sdai(), 0, 1, 0),
             (sdai(), 0, 0, 5)]
    for _ in range(5000):
        a = rnd.choice([0, 1, rnd.randrange(1, 10**9), rnd.randrange(10**18, 10**30), rnd.randrange(1 << 200, 1 << 256)])
        s = rnd.choice([0, 1, rnd.randrange(1, 10**30), rnd.randrange(1 << 200, 1 << 256)])
        st = sdai(a, s, rnd.choice([0, 0, 0, 3, 18]))
        i = rnd.randrange(2)
        amt = rnd.choice([1, rnd.randrange(1, 10**24), rnd.randrange(1, 1 << 256), s, s + 1])
        cases.append((st, i, 1 - i, min(amt, MAX_UINT256)))
    for st, i, j, a in cases:
        p, n = _py(st, i, j, a), _nat(st, i, j, a)
        if n[0] == "fallback":
            fallbacks += 1
            assert p[0] == "other", (st, i, j, a, p)
            continue
        assert p == n, (st, i, j, a, p, n)
    assert fallbacks <= 1  # only the invalid (0, 0) index pair
