"""On-chain reference quotes via eth_call (used by integration tests to validate v4quote math).

These are the protocols' own quoting paths:
* Uniswap v2 / LFJ v1: Router.getAmountsOut / getAmountsIn
* Uniswap v3 / PancakeSwap v3: QuoterV2.quoteExactInputSingle / quoteExactOutputSingle
* Uniswap v4: V4Quoter.quoteExactInputSingle / quoteExactOutputSingle
* LFJ Liquidity Book: LBPair.getSwapOut / getSwapIn
"""

from __future__ import annotations

from eth_utils import to_checksum_address
from web3 import Web3

from v4quote.amm.uniswap_v4 import PoolKey
from v4quote.chain.multicall import Call, CallFailed, eth_call


def v2_router_amount_out(w3: Web3, router: str, amount_in: int, path: list[str], block: int) -> int:
    amounts = eth_call(w3, Call(router, "getAmountsOut(uint256,address[])", (amount_in, path), ("uint256[]",)), block)
    return amounts[-1]


def v2_router_amount_in(w3: Web3, router: str, amount_out: int, path: list[str], block: int) -> int:
    amounts = eth_call(w3, Call(router, "getAmountsIn(uint256,address[])", (amount_out, path), ("uint256[]",)), block)
    return amounts[0]


def v3_quoter_exact_input_single(
    w3: Web3, quoter: str, token_in: str, token_out: str, fee: int, amount_in: int, block: int,
    sqrt_price_limit_x96: int = 0, *, key_type: str = "uint24",
) -> tuple[int, int, int]:
    """QuoterV2 -> (amountOut, sqrtPriceX96After, initializedTicksCrossed).

    ``fee`` is the pool key parameter: the fee tier for Uniswap/PancakeSwap
    (key_type='uint24'), the tickSpacing for Ramses/Pharaoh V3 (key_type='int24').
    """
    out = eth_call(
        w3,
        Call(
            quoter,
            f"quoteExactInputSingle((address,address,uint256,{key_type},uint160))",
            ((token_in, token_out, amount_in, fee, sqrt_price_limit_x96),),
            ("uint256", "uint160", "uint32", "uint256"),
        ),
        block,
    )
    return out[0], out[1], out[2]


def v3_quoter_exact_output_single(
    w3: Web3, quoter: str, token_in: str, token_out: str, fee: int, amount_out: int, block: int,
    sqrt_price_limit_x96: int = 0, *, key_type: str = "uint24",
) -> tuple[int, int, int]:
    """QuoterV2 -> (amountIn, sqrtPriceX96After, initializedTicksCrossed)."""
    out = eth_call(
        w3,
        Call(
            quoter,
            f"quoteExactOutputSingle((address,address,uint256,{key_type},uint160))",
            ((token_in, token_out, amount_out, fee, sqrt_price_limit_x96),),
            ("uint256", "uint160", "uint32", "uint256"),
        ),
        block,
    )
    return out[0], out[1], out[2]


def lb_quoter_best_path_from_amount_in(
    w3: Web3, quoter: str, token_in: str, token_out: str, amount_in: int, block: int
) -> tuple[str, int]:
    """LBQuoter / Pharaoh DLMMQuoter findBestPathFromAmountIn for a single hop -> (best pair, amountOut).

    Both return a Quote struct whose first two dynamic members are route and pairs,
    followed by binSteps, versions, amounts, virtualAmountsWithoutSlippage, fees.
    """
    out = eth_call(
        w3,
        Call(
            quoter,
            "findBestPathFromAmountIn(address[],uint128)",
            ([token_in, token_out], amount_in),
            ("(address[],address[],uint256[],uint8[],uint128[],uint128[],uint128[])",),
        ),
        block,
    )
    return out[1][0], out[4][1]


def v4_quoter_exact_input_single(
    w3: Web3, quoter: str, key: PoolKey, zero_for_one: bool, amount_in: int, block: int
) -> int:
    out = eth_call(
        w3,
        Call(
            quoter,
            "quoteExactInputSingle(((address,address,uint24,int24,address),bool,uint128,bytes))",
            ((key.as_tuple(), zero_for_one, amount_in, b""),),
            ("uint256", "uint256"),
        ),
        block,
    )
    return out[0]


def v4_quoter_exact_output_single(
    w3: Web3, quoter: str, key: PoolKey, zero_for_one: bool, amount_out: int, block: int
) -> int:
    out = eth_call(
        w3,
        Call(
            quoter,
            "quoteExactOutputSingle(((address,address,uint24,int24,address),bool,uint128,bytes))",
            ((key.as_tuple(), zero_for_one, amount_out, b""),),
            ("uint256", "uint256"),
        ),
        block,
    )
    return out[0]


def lb_get_swap_out(w3: Web3, pair: str, amount_in: int, swap_for_y: bool, block: int) -> tuple[int, int, int]:
    """LBPair.getSwapOut -> (amountInLeft, amountOut, fee)."""
    return eth_call(
        w3, Call(pair, "getSwapOut(uint128,bool)", (amount_in, swap_for_y), ("uint128", "uint128", "uint128")), block
    )


def lb_get_swap_in(w3: Web3, pair: str, amount_out: int, swap_for_y: bool, block: int) -> tuple[int, int, int]:
    """LBPair.getSwapIn -> (amountIn, amountOutLeft, fee)."""
    return eth_call(
        w3, Call(pair, "getSwapIn(uint128,bool)", (amount_out, swap_for_y), ("uint128", "uint128", "uint128")), block
    )


# --------------------------------------------------------------------------- swap simulation (no quoter contract)
# For pools without a usable quoter (QuoterV2 only serves the pools of its own factory: forks such as Velodrome /
# Aerodrome Slipstream CL clones, or v2 forks without a known router), the pool's OWN swap code is executed with
# eth_call and a state override. Read-only: eth_call never commits state and nothing is signed.
#
# v3: ``pool.swap(PROBE, zeroForOne, +amountIn, limit, "")`` from ``PROBE``, whose code is replaced (state override)
# by ``SWAP_PROBE_CODE`` = ``calldatacopy(0, 4, 64); revert(0, 64)``: the pool calls the swap callback on msg.sender
# (``uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes)``), the probe reverts with the two deltas
# and the pool bubbles the revert up - the same trick QuoterV2 uses. The deltas are the pool's own result for the
# exact input, fee included (also a dynamic fee read by the pool at this block).
SWAP_PROBE = "0x00000000000000000000000000000000c0FFEE01"  # no code on chain; given SWAP_PROBE_CODE per call
SWAP_PROBE_CODE = "0x6040600460003760406000fd"
V2_SINK = "0x000000000000000000000000000000000000dEaD"  # receives the v2 swap output inside the eth_call
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342


def _revert_data(exc: Exception) -> bytes | None:
    d = getattr(exc, "data", None)
    if isinstance(d, (bytes, bytearray)):
        return bytes(d)
    if isinstance(d, str) and d.startswith("0x"):
        try:
            return bytes.fromhex(d[2:])
        except ValueError:
            return None
    return None


def v3_pool_swap_sim(w3: Web3, pool: str, zero_for_one: bool, amount_in: int, block: int) -> tuple[int, int]:
    """Exact-input swap of a Uniswap-v3-style pool executed by the pool itself at ``block`` (eth_call with a state
    override, see above) -> ``(amount_in_used, amount_out)``. ``amount_in_used < amount_in`` means the pool ran out of
    price range (partial fill). Raises ``CallFailed`` when the pool reverts for another reason (``reverted=True``) or
    the call fails in transport."""
    from eth_abi import decode as abi_decode  # noqa: PLC0415
    from web3.exceptions import ContractLogicError  # noqa: PLC0415

    probe = to_checksum_address(SWAP_PROBE)
    limit = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
    call = Call(pool, "swap(address,bool,int256,uint160,bytes)", (probe, zero_for_one, amount_in, limit, b""))
    tx = {"from": probe, "to": to_checksum_address(pool), "data": "0x" + call.calldata().hex()}
    try:
        w3.eth.call(tx, block, {probe: {"code": SWAP_PROBE_CODE}})
    except ContractLogicError as exc:
        data = _revert_data(exc)
        if data is None or len(data) != 64:
            raise CallFailed(f"swap sim on {pool} reverted at block {block}: {str(exc)[:120]}", reverted=True) from None
        a0, a1 = abi_decode(["int256", "int256"], data)
        used, got = (a0, -a1) if zero_for_one else (a1, -a0)
        return int(used), int(got)
    except Exception as exc:  # noqa: BLE001  transport: report the type only (URLs are masked upstream)
        raise CallFailed(f"swap sim on {pool} failed at block {block}: {type(exc).__name__}", reverted=False) from None
    raise CallFailed(f"swap sim on {pool} did not reach the callback at block {block}", reverted=True)


def _v2_swap_passes(w3: Web3, pair: str, out0: int, out1: int, overrides: dict, block: int) -> bool:
    from web3.exceptions import ContractLogicError  # noqa: PLC0415

    call = Call(pair, "swap(uint256,uint256,address,bytes)", (out0, out1, to_checksum_address(V2_SINK), b""))
    try:
        w3.eth.call({"to": to_checksum_address(pair), "data": "0x" + call.calldata().hex()}, block, overrides)
    except ContractLogicError:
        return False
    except Exception as exc:  # noqa: BLE001
        raise CallFailed(f"v2 swap check on {pair} failed at block {block}: {type(exc).__name__}",
                         reverted=False) from None
    return True


def v2_pair_swap_check(w3: Web3, pair: str, token_in: str, in_is_token0: bool, amount_in: int, amount_out: int,
                       block: int, *, balance_override: dict) -> tuple[bool, bool]:
    """A Uniswap-v2-style pair executes ``swap`` at ``block`` with ``amount_in`` of ``token_in`` already in the pair
    (state override ``balance_override``, built by the caller: the pair's balance + ``amount_in``): returns
    ``(swap(amount_out) passes, swap(amount_out + 1) passes)``. ``(True, False)`` proves ``amount_out`` is exactly the
    largest output the pair's own K check allows (= what its router would quote)."""
    def outs(x: int) -> tuple[int, int]:
        return (0, x) if in_is_token0 else (x, 0)

    ok = _v2_swap_passes(w3, pair, *outs(amount_out), balance_override, block)
    more = _v2_swap_passes(w3, pair, *outs(amount_out + 1), balance_override, block)
    return ok, more


def v2_pair_amount_out(w3: Web3, pair: str, in_is_token0: bool, amount_in: int, block: int,
                       fee: tuple[int, int] = (997, 1000)) -> int:
    """``getAmountOut`` of a v2 pair from its own ``getReserves`` at ``block`` (what Router02.getAmountsOut computes
    on chain), for pairs whose router is unknown. The fee must have been confirmed for the pair (e.g. with
    ``v2_pair_swap_check``)."""
    r0, r1, _ = eth_call(w3, Call(pair, "getReserves()", out=("uint112", "uint112", "uint32")), block)
    r_in, r_out = (r0, r1) if in_is_token0 else (r1, r0)
    x = amount_in * fee[0]
    return x * r_out // (r_in * fee[1] + x)
