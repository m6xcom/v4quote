"""v4quote math vs PulseChain (chainId 369) contracts at one pinned block (marker ``onchain``; needs PULSECHAIN_RPC_URL,
e.g. a public endpoint).

* PulseX V1 / V2 pairs (Uniswap v2 forks, fee 29 bp = 9971/10000): ``V2PairState.get_amount_out`` == the PulseX
  Router02 ``getAmountsOut`` of the pair's own factory, both directions, 8 sizes up to 1/2 of the reserve;
* the PAIR's own K check (what any contract calling ``pair.swap`` directly meets): ``swap(out)`` passes and
  ``swap(out + 1)`` reverts under an eth_call state override that credits ``amount_in`` to the pair (PulseX V1, V2,
  9mm V2 ``NineInchPair``, the older "9mm LPs" pairs of the PancakeFactory fork with 9975/10000, the Uniswap V2 copy
  with 997/1000);
* the Uniswap v3 copy (factory 0x1f98431c, QuoterV2 0x61ffe014 from the Ethereum state fork): ``load_v3_pool`` +
  ``quote_exact_input`` == ``QuoterV2.quoteExactInputSingle``.
"""
from __future__ import annotations


import pytest
from eth_utils import to_checksum_address

from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.v3 import pool as v3m
from v4quote.chain import erc20
from v4quote.chain import quoters as qt
from v4quote.chain.loaders import load_v2_pair, load_v3_pool
from v4quote.chain.multicall import Call, eth_call

pytestmark = pytest.mark.onchain

PULSEX_V1_ROUTER = "0x98bf93ebf5c380c0e6ae8e192a7e2ae08edacc02"  # factory 0x1715a3e4 (PulseX V1)
PULSEX_V2_ROUTER = "0x165c3410fc91ef562c50559f7d2289febed552d9"  # factory 0x29ea7545 (PulseX V2)
PULSEX_FEE = (9971, 10000)
V2_PAIRS = [  # (pool, router or None, fee, dex)
    ("0xcbbad671ca3a46a565551335c10144e75554b367", PULSEX_V1_ROUTER, PULSEX_FEE, "pulsex_v1"),
    ("0x322df7921f28f1146cdf62afdac0d6bc0ab80711", PULSEX_V1_ROUTER, PULSEX_FEE, "pulsex_v1"),
    ("0x5ef7aac0de4f2012cb36730da140025b113fada4", PULSEX_V1_ROUTER, PULSEX_FEE, "pulsex_v1"),
    ("0xae8429918fdbf9a5867e3243697637dc56aa76a1", PULSEX_V2_ROUTER, PULSEX_FEE, "pulsex_v2"),
    ("0xe0e1f83a1c64cf65c1a86d7f3445fc4f58f7dcbf", PULSEX_V2_ROUTER, PULSEX_FEE, "pulsex_v2"),
    ("0x14dc128e4dd295aeaf246b1afdce06a623951de5", PULSEX_V2_ROUTER, PULSEX_FEE, "pulsex_v2"),
    ("0xa1186671046c7e19f1083b93b8e72c297e5ba7f7", None, PULSEX_FEE, "9mm_v2"),
    ("0x994e575563f8204d2ccc022e91dd91138b0506c2", None, PULSEX_FEE, "9mm_v2"),
    # "9mm LPs" pairs of the PancakeFactory fork 0x3a0fa788 (unverified code): Pancake fee 9975/10000
    ("0x25f1e190525f8afef9bfbc49f2891f0ecbbd98dc", None, (9975, 10000), "9mm_v2_pancake"),
    ("0x395a9f7d60063c726da704c724811db298f0ada3", None, (9975, 10000), "9mm_v2_pancake"),
    ("0x9bd82673c50acb4a3b883d61e070a3c8d9b08e10", "0x7a250d5630b4cf539739df2c5dacb4c659f2488d", (997, 1000),
     "uniswap_v2_copy"),
]
UNI_V3_POOL = "0x14036bbda496e39dc3db6025fe858431c1ddadb1"
UNI_V3_QUOTER = "0x61ffe014ba17989e743c5f6cb21bf9697530b21e"


def _row(pool, block, case, ours, chain, extra=""):
    from tests.onchain.conftest import Comparison  # noqa: PLC0415
    return Comparison(pool, block, case, ours, chain, extra)


def _sizes(reserve: int) -> list[int]:
    return sorted({max(1, reserve // d) for d in (10**9, 10**7, 10**5, 10**4, 1000, 100, 10, 2)})


@pytest.fixture(scope="module")
def ctx(chain_ctx):
    return chain_ctx("pulsechain")


@pytest.mark.parametrize("pool,router,fee,dex", V2_PAIRS, ids=[f"{p[3]}:{p[0][:10]}" for p in V2_PAIRS])
def test_v2_router_and_k_check(ctx, report, pool, router, fee, dex):
    st = load_v2_pair(ctx.w3, pool, ctx.block, fee_numerator=fee[0], fee_denominator=fee[1])
    slots = {}

    def bal_of(token):
        def f(holder, ov):
            try:
                return int(ctx.w3.eth.call({"to": to_checksum_address(token), "data": "0x70a08231" + "00" * 12
                                            + holder[2:].lower()}, ctx.block, ov).hex(), 16)
            except Exception:  # noqa: BLE001
                return None
        return f

    checked_k = 0
    for zfo in (True, False):
        t_in, t_out = (st.token0, st.token1) if zfo else (st.token1, st.token0)
        r_in = st.reserve0 if zfo else st.reserve1
        for amt in _sizes(r_in):
            ours = st.get_amount_out(amt, zfo)
            if router is not None:
                chain = eth_call(ctx.w3, Call(router, "getAmountsOut(uint256,address[])", (amt, [t_in, t_out]),
                                              ("uint256[]",)), ctx.block)[-1]
                report.rows.append(_row(pool, ctx.block, f"{dex} getAmountsOut({amt},{'0->1' if zfo else '1->0'})",
                                        ours, chain))
                assert ours == chain, (pool, zfo, amt)
            if amt not in (r_in // 10**5, r_in // 10):  # the pair's own K check on two sizes per direction
                continue
            if t_in not in slots:
                slots[t_in] = erc20.find_balance_slot(bal_of(t_in), t_in)
            if slots[t_in] is None:
                report.notes.append(f"{pool}: balance slot of {t_in} not found, K check skipped")
                continue
            have = bal_of(t_in)(pool, {})
            ov = erc20.balance_override(t_in, pool, have + amt, *slots[t_in])
            ok, more = qt.v2_pair_swap_check(ctx.w3, pool, t_in, zfo, amt, ours, ctx.block, balance_override=ov)
            report.rows.append(_row(pool, ctx.block, f"{dex} K swap(out)/swap(out+1) {amt}", str((ok, more)),
                                    str((True, False))))
            assert (ok, more) == (True, False), (pool, zfo, amt)
            checked_k += 1
    if not any(s is not None for s in slots.values()):
        pytest.skip(f"{pool}: no token with a findable balance slot")
    assert checked_k >= 1


def test_uniswap_v3_copy_quoter(ctx, report):
    st = load_v3_pool(ctx.w3, UNI_V3_POOL, ctx.block, words_each_side=None)
    if st.liquidity == 0:
        pytest.skip("Uniswap v3 copy pool without active liquidity at this block")
    for zfo in (True, False):
        t_in, t_out = (st.token0, st.token1) if zfo else (st.token1, st.token0)
        bal = eth_call(ctx.w3, Call(t_in, "balanceOf(address)", (st.address,), ("uint256",)), ctx.block)
        for amt in sorted({max(1, bal // d) for d in (10**8, 10**6, 10**4, 100)}):
            r = v3m.quote_exact_input(st, zfo, amt)
            chain = qt.v3_quoter_exact_input_single(ctx.w3, UNI_V3_QUOTER, t_in, t_out, st.fee, amt, ctx.block)
            chain_out = chain[0] if isinstance(chain, (tuple, list)) else chain
            report.rows.append(_row(UNI_V3_POOL, ctx.block, f"v3 quoteExactInputSingle({amt},{zfo})",
                                    r.amount_out if r.fully_filled else "partial", chain_out))
            if r.fully_filled:
                assert r.amount_out == chain_out, (zfo, amt)


def test_v2_state_from_sync_matches_reserves(ctx):
    """v2 states kept from Sync logs agree with the chain: getReserves at the pinned block == the last Sync before it."""
    pool = V2_PAIRS[3][0]
    r0, r1, _ = eth_call(ctx.w3, Call(pool, "getReserves()", out=("uint112", "uint112", "uint32")), ctx.block)
    st = load_v2_pair(ctx.w3, pool, ctx.block, fee_numerator=9971, fee_denominator=10000)
    assert (st.reserve0, st.reserve1) == (r0, r1)
    assert isinstance(V2PairState(st.address, st.token0, st.token1, r0, r1, 9971, 10000), V2PairState)
