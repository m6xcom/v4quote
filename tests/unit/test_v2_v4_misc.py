"""Uniswap v2, v4 specifics and CREATE2 pool address derivation."""

import random
from dataclasses import replace

import pytest

from v4quote.errors import MathRevert, UnsupportedPool
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.uniswap_v2 import V2PairState, get_amount_in, get_amount_out
from v4quote.amm.v3 import TickBitmapSnapshot, V3PoolState
from v4quote.amm.v3 import pool as v3
from v4quote.amm.v3.pool_address import PRESETS, UNISWAP_V3_INIT_CODE_HASH, compute_pool_address
from v4quote.amm.v3.tick_math import get_sqrt_ratio_at_tick

ZERO = "0x0000000000000000000000000000000000000000"


# ---------------------------------------------------------------- Uniswap v2
def test_v2_library_vectors():
    # v2-periphery test/UniswapV2Library.spec.ts
    assert get_amount_out(2, 100, 100) == 1
    assert get_amount_in(1, 100, 100) == 2
    for bad in ((0, 100, 100), (2, 0, 100), (2, 100, 0)):
        with pytest.raises(MathRevert):
            get_amount_out(*bad)
    for bad in ((0, 100, 100), (1, 0, 100), (1, 100, 0)):
        with pytest.raises(MathRevert):
            get_amount_in(*bad)
    with pytest.raises(MathRevert):
        get_amount_in(100, 100, 100)  # cannot take the whole reserve


def test_v2_k_invariant_and_roundtrip():
    rng = random.Random(7)
    for _ in range(500):
        r0, r1 = rng.randint(10**3, 10**30), rng.randint(10**3, 10**30)
        pair = V2PairState("p", "a", "b", r0, r1)
        amt = rng.randint(1, r0)
        out = pair.get_amount_out(amt, True)
        if out == 0:
            continue
        # pair.swap's K check with the 0.3% fee: (r0*1000 + amt*997) * (r1 - out) * 1000 >= r0*r1*1000^2
        assert (r0 * 1000 + amt * 997) * (r1 - out) * 1000 >= r0 * r1 * 1000**2
        assert pair.get_amount_in(out, True) <= amt
        after = pair.after_swap(amt, out, True)
        assert after.reserve0 == r0 + amt and after.reserve1 == r1 - out


def test_v2_fee_parameterisation():
    # PancakeSwap v2 style 0.25% fee
    assert get_amount_out(10**18, 10**21, 10**21, 9975, 10000) > get_amount_out(10**18, 10**21, 10**21)


# ---------------------------------------------------------------- CREATE2
def test_uniswap_v3_mainnet_pool_address():
    # USDC/WETH 0.05% on Ethereum mainnet
    addr = compute_pool_address(
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        500,
        UNISWAP_V3_INIT_CODE_HASH,
    )
    assert addr == "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"


@pytest.mark.parametrize(
    "chain,dex,a,b,key,expected",
    [  # pool addresses confirmed on-chain via factory.getPool (see fixtures/pools.yaml)
        ("monad", "uniswap_v3", "0x3bd359C1119dA7Da1D913D1C4D2B7c461115433A",
         "0x754704Bc059F8C67012fEd69BC8A327a5aafb603", 3000, "0x659bD0BC4167BA25c62E05656F78043E7eD4a9da"),
        ("monad", "pancakeswap_v3", "0x3bd359C1119dA7Da1D913D1C4D2B7c461115433A",
         "0x754704Bc059F8C67012fEd69BC8A327a5aafb603", 500, "0x63e48B725540A3Db24ACF6682a29f877808C53F2"),
        ("avalanche", "pharaoh_v3", "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
         "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E", 10, "0xF01449c0bA930B6e2cacA3dEf3CcBD7a3E589534"),
    ],
)
def test_presets_reproduce_onchain_pools(chain, dex, a, b, key, expected):
    pr = PRESETS[(chain, dex)]
    assert compute_pool_address(pr.deployer, a, b, key, pr.init_code_hash, pr.key_type).lower() == expected.lower()


# ---------------------------------------------------------------- Uniswap v4
def test_v4_pool_ids():
    usdc_monad = "0x754704Bc059F8C67012fEd69BC8A327a5aafb603"
    usdc_avax = "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E"
    assert v4.PoolKey(ZERO, usdc_monad, 500, 10).pool_id().hex() == (
        "18a9fc874581f3ba12b7898f80a683c66fd5877fd74b26a85ba9a3a79c549954"
    )
    assert v4.PoolKey(ZERO, usdc_avax, 500, 10).pool_id().hex() == (
        "6874d3d2b99e6b75ed2fae390907e1d921d4b06b7f48fc6d3fe9c547507e369c"
    )


def test_v4_swap_fee_with_protocol_fee():
    assert v4.calculate_swap_fee(1000, 3000) == 3997
    assert v4.calculate_swap_fee(0, 3000) == 3000
    assert v4.calculate_swap_fee(1000, 0) == 1000


def _hook(flags: int) -> str:
    return "0x" + format((0xABCDEF << 140) | flags, "040x")


def test_v4_hook_policy():
    usdc = "0x754704Bc059F8C67012fEd69BC8A327a5aafb603"
    ok = [v4.PoolKey(ZERO, usdc, 500, 10), v4.PoolKey(ZERO, usdc, 500, 10, _hook(v4.AFTER_SWAP_FLAG))]
    for k in ok:
        v4.check_supported(k)
    bad = [
        v4.PoolKey(ZERO, usdc, v4.DYNAMIC_FEE_FLAG, 10),
        v4.PoolKey(ZERO, usdc, 500, 10, _hook(v4.BEFORE_SWAP_FLAG | v4.BEFORE_SWAP_RETURNS_DELTA_FLAG)),
        v4.PoolKey(ZERO, usdc, 500, 10, _hook(v4.AFTER_SWAP_FLAG | v4.AFTER_SWAP_RETURNS_DELTA_FLAG)),
        v4.PoolKey(ZERO, usdc, 500, 10, _hook(v4.BEFORE_SWAP_FLAG)),
    ]
    for k in bad:
        with pytest.raises(UnsupportedPool):
            v4.check_supported(k)
    v4.check_supported(bad[3], allow_before_swap_hook=True)


def _random_pool(rng, ts):
    lo = -ts * rng.randint(1, max(5000 // ts, 1))
    hi = ts * rng.randint(1, max(5000 // ts, 1))
    lo2, hi2 = lo - ts * rng.randint(1, 300), hi + ts * rng.randint(1, 300)
    L1, L2 = rng.randint(10**15, 10**24), rng.randint(10**15, 10**24)
    net = {lo: L1, hi: -L1, lo2: L2, hi2: -L2}
    tick = rng.randint(lo, hi - 1)
    sqrt_p = get_sqrt_ratio_at_tick(tick) + rng.randint(0, 10**18)
    return tick, sqrt_p, L1 + L2, TickBitmapSnapshot.from_ticks(ts, net)


def test_v4_amounts_equal_v3_without_protocol_fee():
    """v3 and v4 SwapMath split fee/amountIn differently per step, but user-facing amounts must be identical."""
    rng = random.Random(99)
    for _ in range(150):
        ts, fee = rng.choice([(1, 100), (10, 500), (60, 3000), (200, 10000)])
        tick, sqrt_p, liq, bm = _random_pool(rng, ts)
        s3 = V3PoolState("p", "a", "b", fee, ts, sqrt_p, tick, liq, bm)
        key = v4.PoolKey(ZERO, "0x0000000000000000000000000000000000000001", fee, ts)
        s4 = v4.V4PoolState(key, sqrt_p, tick, 0, fee, liq, bm)
        zfo = rng.random() < 0.5
        amt = rng.randint(1, 10**24)
        a = v3.quote_exact_input(s3, zfo, amt)
        try:
            b = v4.quote_exact_input(s4, zfo, amt)
        except MathRevert:  # v4 quoter reverts when the input cannot be fully used
            assert not a.fully_filled
            continue
        assert (a.amount_in, a.amount_out, a.sqrt_price_x96, a.tick) == (b.amount_in, b.amount_out, b.sqrt_price_x96, b.tick)
        if a.amount_out:
            want = max(a.amount_out // 2, 1)
            c, d = v3.quote_exact_output(s3, zfo, want), v4.quote_exact_output(s4, zfo, want)
            assert (c.amount_in, c.amount_out, c.sqrt_price_x96) == (d.amount_in, d.amount_out, d.sqrt_price_x96)


def test_v4_protocol_fee_raises_cost():
    rng = random.Random(3)
    tick, sqrt_p, liq, bm = _random_pool(rng, 10)
    key = v4.PoolKey(ZERO, "0x0000000000000000000000000000000000000001", 500, 10)
    no_pf = v4.V4PoolState(key, sqrt_p, tick, 0, 500, liq, bm)
    pf = v4.V4PoolState(key, sqrt_p, tick, 1000 | (1000 << 12), 500, liq, bm)
    assert pf.swap_fee(True) == v4.calculate_swap_fee(1000, 500)
    a = v4.quote_exact_input(no_pf, True, 10**18)
    b = v4.quote_exact_input(pf, True, 10**18)
    assert b.amount_out < a.amount_out and b.protocol_fee_amount > 0


def test_v4_zero_amount_and_max_fee_exact_out():
    rng = random.Random(5)
    tick, sqrt_p, liq, bm = _random_pool(rng, 10)
    key = v4.PoolKey(ZERO, "0x0000000000000000000000000000000000000001", 500, 10)
    s = v4.V4PoolState(key, sqrt_p, tick, 0, 500, liq, bm)
    assert v4.swap(s, True, 0).amount_out == 0
    s_max = v4.V4PoolState(key, sqrt_p, tick, 0, 1_000_000, liq, bm)
    with pytest.raises(MathRevert):
        v4.swap(s_max, True, 10)  # exact output with 100% fee
    assert v4.swap(s_max, True, -10**6).amount_out == 0  # exact input: everything goes to fees


def test_v4_hook_fee_resolved_models_dynamic_fee_hooks():
    """A fee-only hook (beforeSwap lpFeeOverride on a dynamic-fee pool): with the resolved fee per direction the
    swap equals a static pool with that fee; without it the pool stays rejected; return-delta hooks stay rejected."""
    rng = random.Random(11)
    tick, sqrt_p, liq, bm = _random_pool(rng, 10)
    c1 = "0x0000000000000000000000000000000000000001"
    hook = _hook(v4.BEFORE_INITIALIZE_FLAG | v4.BEFORE_SWAP_FLAG)
    dyn = v4.PoolKey(ZERO, c1, v4.DYNAMIC_FEE_FLAG, 10, hook)
    s = v4.V4PoolState(dyn, sqrt_p, tick, 0, 0, liq, bm)
    assert s.unsupported_reasons()
    with pytest.raises(UnsupportedPool):
        v4.quote_exact_input(s, True, 10**18)
    hf = replace(s, hook_fee=(700, 2600))
    assert hf.unsupported_reasons() == []
    for zfo, fee in ((True, 700), (False, 2600)):
        ref = v4.V4PoolState(v4.PoolKey(ZERO, c1, fee, 10), sqrt_p, tick, 0, fee, liq, bm)
        a, b = v4.quote_exact_input(hf, zfo, 10**18), v4.quote_exact_input(ref, zfo, 10**18)
        assert (a.amount_out, a.sqrt_price_x96, a.fee_amount) == (b.amount_out, b.sqrt_price_x96, b.fee_amount)
    # protocol fee still combines with the hook's LP fee
    pf = replace(hf, protocol_fee=1000 | (1000 << 12))
    assert pf.swap_fee(True) == v4.calculate_swap_fee(1000, 700)
    assert pf.swap_fee(False) == v4.calculate_swap_fee(1000, 2600)
    # return-delta hooks are never modelled, resolved fee or not
    for f in (v4.BEFORE_SWAP_RETURNS_DELTA_FLAG, v4.AFTER_SWAP_RETURNS_DELTA_FLAG):
        k = v4.PoolKey(ZERO, c1, v4.DYNAMIC_FEE_FLAG, 10, _hook(v4.BEFORE_SWAP_FLAG | f))
        st = v4.V4PoolState(k, sqrt_p, tick, 0, 0, liq, bm, hook_fee=(700, 700))
        assert st.unsupported_reasons()
        with pytest.raises(UnsupportedPool):
            v4.quote_exact_input(st, True, 10**18)
    with pytest.raises(ValueError):
        v4.V4PoolState(dyn, sqrt_p, tick, 0, 0, liq, bm, hook_fee=(700, 1_000_001))


def test_v4_hook_fee_through_routing_and_cycle():
    from v4quote import routing as rt

    rng = random.Random(12)
    tick, sqrt_p, liq, bm = _random_pool(rng, 10)
    c1 = "0x0000000000000000000000000000000000000001"
    k = v4.PoolKey(ZERO, c1, v4.DYNAMIC_FEE_FLAG, 10, _hook(v4.BEFORE_SWAP_FLAG))
    s = v4.V4PoolState(k, sqrt_p, tick, 0, 0, liq, bm)
    assert rt.unsupported_reasons(s)
    assert rt.unsupported_reasons(replace(s, hook_fee=(500, 500))) == []
