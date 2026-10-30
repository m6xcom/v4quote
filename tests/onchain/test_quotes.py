"""v4quote swap math vs the protocols' own on-chain quoters, on the same pinned block.

For every pool in fixtures/pools.yaml and both swap directions we compare:
  * a tiny, a small and a medium trade (fractions of the input-side reserves), and
  * a trade large enough to cross several initialised ticks / bins (found with v4quote itself),
for exact input and exact output. Every difference must be exactly 0 wei.
"""

from __future__ import annotations

import pytest

from v4quote.errors import InsufficientState, MathRevert
from v4quote.amm import liquidity_book as lb
from v4quote.amm import uniswap_v4 as v4
from v4quote.amm.uniswap_v2 import V2PairState
from v4quote.amm.v3 import pool as v3
from v4quote.chain import quoters
from v4quote.chain.loaders import load_lb_pair, load_v2_pair, load_v3_pool, load_v4_pool
from v4quote.chain.multicall import CallFailed

from v4quote.amm.liquidity_book import DEFAULT_HOOKS_ALLOWLIST
from tests.onchain.conftest import Comparison, pools_of_type, token_address

HOOKS_ALLOWLIST = DEFAULT_HOOKS_ALLOWLIST

pytestmark = pytest.mark.onchain

FRACTIONS = ((1, 10**6, "tiny"), (1, 10**3, "small"), (5, 100, "medium"))
MIN_CROSSINGS = 3


def _q96_virtual_reserves(liquidity: int, sqrt_price: int) -> tuple[int, int]:
    return (liquidity << 96) // sqrt_price, (liquidity * sqrt_price) >> 96


def _record(report, pid, block, case, ours, chain, extra=""):
    report.rows.append(Comparison(pid, block, case, ours, chain, extra))


# =========================================================================== Uniswap v2 / LFJ v1
@pytest.mark.parametrize("p", pools_of_type("uniswap_v2"))
def test_v2_matches_router(p, chain_ctx, report):
    ctx = chain_ctx(p["chain"])
    num, den = p.get("fee", [997, 1000])
    pair: V2PairState = load_v2_pair(ctx.w3, p["pool"], ctx.block, fee_numerator=num, fee_denominator=den)
    failures = []
    for zfo in (True, False):
        t_in, t_out = (pair.token0, pair.token1) if zfo else (pair.token1, pair.token0)
        r_in, r_out = (pair.reserve0, pair.reserve1) if zfo else (pair.reserve1, pair.reserve0)
        for n, d, label in FRACTIONS + ((3, 10, "large"),):
            amt = max(r_in * n // d, 1)
            ours = pair.get_amount_out(amt, zfo)
            chain = quoters.v2_router_amount_out(ctx.w3, p["quoter"], amt, [t_in, t_out], ctx.block)
            _record(report, p["id"], ctx.block, f"exactIn {label} zfo={zfo} in={amt}", ours, chain)
            if ours != chain:
                failures.append((label, zfo, "in", ours, chain))
            out_amt = max(r_out * n // d, 1)
            ours_in = pair.get_amount_in(out_amt, zfo)
            chain_in = quoters.v2_router_amount_in(ctx.w3, p["quoter"], out_amt, [t_in, t_out], ctx.block)
            _record(report, p["id"], ctx.block, f"exactOut {label} zfo={zfo} out={out_amt}", ours_in, chain_in)
            if ours_in != chain_in:
                failures.append((label, zfo, "out", ours_in, chain_in))
    assert not failures, failures


# =========================================================================== Uniswap v3 and forks
def _load_v3(ctx, p, words):
    return load_v3_pool(
        ctx.w3, p["pool"], ctx.block, words_each_side=words,
        uniswap_fee_protocol=p.get("uniswap_fee_protocol", True),
    )


def _v3_sizes(state, zfo):
    x, y = _q96_virtual_reserves(state.liquidity, state.sqrt_price_x96)
    base = x if zfo else y
    sizes = [(label, max(base * n // d, 1)) for n, d, label in FRACTIONS]
    # grow until the swap crosses several initialised ticks (or runs out of loaded state)
    amt = max(base * 5 // 100, 1)
    for _ in range(40):
        try:
            r = v3.quote_exact_input(state, zfo, amt)
        except InsufficientState:
            break
        if r.initialized_ticks_crossed >= MIN_CROSSINGS or not r.fully_filled:
            sizes.append((f"multi-tick({r.initialized_ticks_crossed})", amt))
            break
        amt *= 2
    return sizes


@pytest.mark.parametrize("p", pools_of_type("uniswap_v3"))
def test_v3_matches_quoter_v2(p, chain_ctx, report):
    ctx = chain_ctx(p["chain"])
    state = _load_v3(ctx, p, 16)
    key_type = p.get("key_type", "uint24")
    key = p["tick_spacing"] if key_type == "int24" else p["fee"]
    if key_type == "uint24":
        assert state.fee == p["fee"]
    failures = []
    crossed_max = 0
    for zfo in (True, False):
        t_in, t_out = (state.token0, state.token1) if zfo else (state.token1, state.token0)
        for label, amt in _v3_sizes(state, zfo):
            try:
                ours = v3.quote_exact_input(state, zfo, amt)
            except InsufficientState:
                state = _load_v3(ctx, p, None)  # complete bitmap
                ours = v3.quote_exact_input(state, zfo, amt)
            q_out, q_price, q_crossed = quoters.v3_quoter_exact_input_single(
                ctx.w3, p["quoter"], t_in, t_out, key, amt, ctx.block, key_type=key_type
            )
            crossed_max = max(crossed_max, ours.initialized_ticks_crossed)
            extra = f"fee={state.fee} ticksCrossed={ours.initialized_ticks_crossed} quoterCrossed={q_crossed}"
            _record(report, p["id"], ctx.block, f"exactIn {label} zfo={zfo} in={amt}", ours.amount_out, q_out, extra)
            _record(report, p["id"], ctx.block, f"sqrtPriceAfter {label} zfo={zfo}", ours.sqrt_price_x96, q_price)
            if ours.amount_out != q_out or ours.sqrt_price_x96 != q_price:
                failures.append((label, zfo, "in", ours.amount_out, q_out))
            if ours.amount_out == 0 or not ours.fully_filled:
                continue
            # exact output for the amount the exact-input swap produced
            want = ours.amount_out
            try:
                ours_o = v3.quote_exact_output(state, zfo, want)
                ours_in: int | str = ours_o.amount_in
            except MathRevert:
                ours_in = "revert"
            try:
                q_in, _, _ = quoters.v3_quoter_exact_output_single(
                    ctx.w3, p["quoter"], t_in, t_out, key, want, ctx.block, key_type=key_type
                )
                chain_in: int | str = q_in
            except CallFailed as exc:
                if not exc.reverted:
                    raise
                chain_in = "revert"
            _record(report, p["id"], ctx.block, f"exactOut {label} zfo={zfo} out={want}", ours_in, chain_in)
            if ours_in != chain_in:
                failures.append((label, zfo, "out", ours_in, chain_in))
    assert not failures, failures
    if crossed_max < 2:
        report.notes.append(f"{p['id']}: largest tested swap crossed only {crossed_max} initialised ticks")


# =========================================================================== Uniswap v4
def _v4_key(p):
    k = p["pool_key"]
    return v4.PoolKey(
        token_address(p["chain"], k["currency0"]), token_address(p["chain"], k["currency1"]),
        k["fee"], k["tick_spacing"], k["hooks"],
    )


@pytest.mark.parametrize("p", pools_of_type("uniswap_v4"))
def test_v4_matches_v4_quoter(p, chain_ctx, report):
    ctx = chain_ctx(p["chain"])
    key = _v4_key(p)
    v4.check_supported(key)
    state = load_v4_pool(ctx.w3, p["state_view"], key, ctx.block, words_each_side=16)
    failures = []
    for zfo in (True, False):
        x, y = _q96_virtual_reserves(state.liquidity, state.sqrt_price_x96)
        base = x if zfo else y
        sizes = [(label, max(base * n // d, 1)) for n, d, label in FRACTIONS]
        amt = max(base * 5 // 100, 1)
        for _ in range(40):
            try:
                r = v4.swap(state, zfo, -amt)
            except InsufficientState:
                break
            if r.initialized_ticks_crossed >= MIN_CROSSINGS or not r.fully_filled:
                sizes.append((f"multi-tick({r.initialized_ticks_crossed})", amt))
                break
            amt *= 2
        for label, amt in sizes:
            try:
                ours: int | str = v4.quote_exact_input(state, zfo, amt).amount_out
            except MathRevert:
                ours = "revert"
            except InsufficientState:
                state = load_v4_pool(ctx.w3, p["state_view"], key, ctx.block, words_each_side=None)
                ours = v4.quote_exact_input(state, zfo, amt).amount_out
            try:
                chain: int | str = quoters.v4_quoter_exact_input_single(ctx.w3, p["quoter"], key, zfo, amt, ctx.block)
            except CallFailed as exc:
                if not exc.reverted:
                    raise
                chain = "revert"
            _record(report, p["id"], ctx.block, f"exactIn {label} zfo={zfo} in={amt}", ours, chain,
                    f"lpFee={state.lp_fee} protocolFee={state.protocol_fee}")
            if ours != chain:
                failures.append((label, zfo, "in", ours, chain))
            if not isinstance(ours, int) or ours == 0:
                continue
            try:
                ours_in: int | str = v4.quote_exact_output(state, zfo, ours).amount_in
            except MathRevert:
                ours_in = "revert"
            try:
                chain_in: int | str = quoters.v4_quoter_exact_output_single(
                    ctx.w3, p["quoter"], key, zfo, ours, ctx.block
                )
            except CallFailed as exc:
                if not exc.reverted:
                    raise
                chain_in = "revert"
            _record(report, p["id"], ctx.block, f"exactOut {label} zfo={zfo} out={ours}", ours_in, chain_in)
            if ours_in != chain_in:
                failures.append((label, zfo, "out", ours_in, chain_in))
    assert not failures, failures


# =========================================================================== Liquidity Book (LFJ, Pharaoh DLMM)
@pytest.mark.parametrize("p", pools_of_type("lfj_lb"))
def test_lb_matches_get_swap_out_and_in(p, chain_ctx, report):
    ctx = chain_ctx(p["chain"])
    version = str(p.get("version", "2.2"))
    state = load_lb_pair(ctx.w3, p["pool"], ctx.block, version=version, bins_each_side=150, chain=p["chain"])
    hooks_note = ""
    if state.unsupported_reasons():
        # Hooked pools are simulated only because their hook is in DEFAULT_HOOKS_ALLOWLIST with a matching
        # codehash/implementation; an empty allowlist must reject them.
        from v4quote.errors import UnsupportedPool

        with pytest.raises(UnsupportedPool):
            lb.get_swap_out(state, 1, True, ctx.timestamp, hooks_allowlist={})
        assert not state.unsupported_reasons(HOOKS_ALLOWLIST), state.unsupported_reasons(HOOKS_ALLOWLIST)
        assert lb.execution_permitted(state) is False
        hooks_note = f"hooks=0x{state.hooks_parameters:064x}"
        report.notes.append(f"{p['id']}: hooked pool ({hooks_note}); simulated via DEFAULT_HOOKS_ALLOWLIST, execution not permitted")
    total_x = sum(x for x, _ in state.bins.values())
    total_y = sum(y for _, y in state.bins.values())
    failures = []
    for swap_for_y in (True, False):
        # input is X when swapping for Y; size relative to the *output* side liquidity converted by price
        # is unnecessary: fractions of the loaded input-side liquidity are enough to span several bins.
        base_in = total_x if swap_for_y else total_y
        base_out = total_y if swap_for_y else total_x
        if base_in == 0 or base_out == 0:
            base_in = base_in or 10**18
        sizes = [(label, max(base_in * n // d, 1)) for n, d, label in FRACTIONS]
        amt = max(base_in // 1000, 1)
        for _ in range(40):
            try:
                r = lb.pair._simulate_out(state, amt, swap_for_y, ctx.timestamp, HOOKS_ALLOWLIST)
            except InsufficientState:
                break
            if r.bins_crossed >= MIN_CROSSINGS or r.amount_in_left:
                sizes.append((f"multi-bin({r.bins_crossed})", amt))
                break
            amt *= 2
        for label, amt in sizes:
            try:
                ours = lb.get_swap_out(state, amt, swap_for_y, ctx.timestamp, hooks_allowlist=HOOKS_ALLOWLIST)
            except InsufficientState:
                state = load_lb_pair(ctx.w3, p["pool"], ctx.block, version=version, bins_each_side=600, chain=p["chain"])
                ours = lb.get_swap_out(state, amt, swap_for_y, ctx.timestamp, hooks_allowlist=HOOKS_ALLOWLIST)
            chain = tuple(quoters.lb_get_swap_out(ctx.w3, p["quoter"], amt, swap_for_y, ctx.block))
            bins = lb.pair._simulate_out(state, amt, swap_for_y, ctx.timestamp, HOOKS_ALLOWLIST).bins_crossed
            _record(report, p["id"], ctx.block, f"getSwapOut {label} swapForY={swap_for_y} in={amt}",
                    ours, chain, f"binsCrossed={bins} {hooks_note}".strip())
            if ours != chain:
                failures.append((label, swap_for_y, "out", ours, chain))
            if p.get("lb_quoter") and ours[0] == 0:
                tx, ty = state.token_x, state.token_y
                t_in, t_out = (tx, ty) if swap_for_y else (ty, tx)
                best_pair, q_out = quoters.lb_quoter_best_path_from_amount_in(
                    ctx.w3, p["lb_quoter"], t_in, t_out, amt, ctx.block
                )
                if best_pair.lower() == p["pool"].lower():
                    _record(report, p["id"], ctx.block, f"Quoter.findBestPath {label} swapForY={swap_for_y}",
                            ours[1], q_out)
                    if ours[1] != q_out:
                        failures.append((label, swap_for_y, "quoter", ours[1], q_out))
            # exact output: ask for the output produced above
            want = ours[1]
            if want == 0:
                continue
            ours_in = lb.get_swap_in(state, want, swap_for_y, ctx.timestamp, hooks_allowlist=HOOKS_ALLOWLIST)
            chain_in = tuple(quoters.lb_get_swap_in(ctx.w3, p["quoter"], want, swap_for_y, ctx.block))
            _record(report, p["id"], ctx.block, f"getSwapIn {label} swapForY={swap_for_y} out={want}",
                    ours_in, chain_in)
            if ours_in != chain_in:
                failures.append((label, swap_for_y, "in", ours_in, chain_in))
    assert not failures, failures


