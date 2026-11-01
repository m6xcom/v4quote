"""Fee-only Uniswap v4 hooks: read the LP fee a reviewed hook's ``beforeSwap`` applies, at a pinned block.

A fee-only hook sits on a dynamic-fee pool (``fee == 0x800000``), has the BEFORE_SWAP flag and no return-delta flag,
and its ``beforeSwap`` returns ``(selector, ZERO_DELTA, fee | OVERRIDE_FEE_FLAG)``: it changes the LP fee of the swap
and nothing else. ``v4quote.amm.uniswap_v4`` then quotes the pool exactly once the state carries that fee
(``V4PoolState.hook_fee``).

The fee is read by calling the hook's own ``beforeSwap`` with ``eth_call`` from the PoolManager address (which is
what ``onlyPoolManager`` checks), with the swap's ``sender`` (the contract that calls ``PoolManager.swap``: the router,
or V4Quoter for a quoter comparison), the pool key and the swap direction. The result is the fee for a swap by that
sender in that direction in a block with the read block's timestamp. Nothing is signed and no state is written.

Only hooks in ``FEE_ONLY_HOOKS`` are resolved: each entry was checked against its verified source (Sourcify) and is
pinned to the runtime codehash that was reviewed. Any other hook raises ``UnsupportedPool``; so does a reviewed
address whose code differs from the pinned hash.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from eth_utils import function_signature_to_4byte_selector, keccak, to_checksum_address
from web3 import Web3

from v4quote.amm.uniswap_v4 import (
    AFTER_SWAP_RETURNS_DELTA_FLAG,
    BEFORE_SWAP_FLAG,
    BEFORE_SWAP_RETURNS_DELTA_FLAG,
    MAX_LP_FEE,
    PoolKey,
    V4PoolState,
)
from v4quote.amm.v3.tick_math import MAX_SQRT_PRICE, MIN_SQRT_PRICE
from v4quote.chain.multicall import Call, CallFailed, eth_call
from v4quote.errors import MathRevert, UnreadableState, UnsupportedPool

OVERRIDE_FEE_FLAG = 0x400000  # LPFeeLibrary.OVERRIDE_FEE_FLAG
REMOVE_OVERRIDE_MASK = 0xBFFFFF  # LPFeeLibrary.REMOVE_OVERRIDE_MASK
BEFORE_SWAP_SIG = "beforeSwap(address,(address,address,uint24,int24,address),(bool,int256,uint160),bytes)"
BEFORE_SWAP_SELECTOR = function_signature_to_4byte_selector(BEFORE_SWAP_SIG)


@dataclass(frozen=True)
class FeeOnlyHook:
    """A reviewed fee-only hook. ``sender_dependent``: the fee depends on who calls ``PoolManager.swap``."""

    name: str
    codehash: str
    source: str
    sender_dependent: bool
    note: str


# (chainId, hook address lowercase) -> reviewed hook. Sources: Sourcify full / partial match on chain 4663.
FEE_ONLY_HOOKS: dict[tuple[int, str], FeeOnlyHook] = {
    (4663, "0xc15b155a3094934171a15625d1a80aba267b6080"): FeeOnlyHook(
        "SessionFeeHook", "0x97144f036268c4c71c666361a5870046cbf85eada2e586ddb2ce95c545d7d763",
        "Sourcify exact match, src/sessionhook/SessionFeeHook.sol", True,
        "view beforeSwap: fee = feeFor(poolId, sender, block.timestamp) for both directions (US equity session "
        "class fee, open ramp, timed overrides; 0 for fee-exempt senders); reverts PoolIsPaused when paused",
    ),
    (4663, "0x5eb87f69be00df39981622fd60a8de4b7837e080"): FeeOnlyHook(
        "FablesRWA", "0x577978ad29223268205cb40a5215c46839f5c27a1218beac7d4a0f7cd0aaa341",
        "Sourcify match, src/FablesRWA.sol + src/base/FablesBaseHook.sol", False,
        "view beforeSwap: fee = session floor curve at block.timestamp + directional premium, TTL poke override, "
        "capped at maxFee; the amount is not used (currentFee(poolId, zeroForOne) gives the same fee)",
    ),
    (4663, "0xb608a78761f179f7c56f15e7d13921b92f00a080"): FeeOnlyHook(
        "FablesRWA", "0x514b632f9bb987403ca7a3fb35780b2c092726a437c1efbb4479fd612549d25b",
        "Sourcify match, src/FablesRWA.sol + src/base/FablesBaseHook.sol (earlier build)", False,
        "view beforeSwap: session floor curve + directional premium, TTL poke, cap; amount not used",
    ),
    (4663, "0x67d86050d22d574df046f3d90f722045f714e080"): FeeOnlyHook(
        "FablesRWA", "0xa40a02521b3a20e56851d833baf5baf0ee258e954c0f4e3f96e1bd30c8681db3",
        "Sourcify match, src/FablesRWA.sol + src/base/FablesBaseHook.sol (symmetric build)", False,
        "view beforeSwap: session floor curve, TTL poke (one fee for both directions), cap; amount not used",
    ),
    (4663, "0x70a9a88402989226847ec122043ce5e7ff462080"): FeeOnlyHook(
        "FablesRWA", "0x165ccaced1f37263fd1a36b051c43d1759bab3005c2f7083ab32ee2db3b217e3",
        "Sourcify match, same sources as 0x67d86050 (symmetric build)", False,
        "view beforeSwap: session floor curve, TTL poke (one fee for both directions), cap; amount not used",
    ),
}


def reviewed_hook(chain_id: int, hook: str) -> FeeOnlyHook | None:
    return FEE_ONLY_HOOKS.get((int(chain_id), hook.lower()))


def check_fee_only_key(key: PoolKey) -> None:
    """The key's hook flags allow a fee-only model: BEFORE_SWAP set, no swap return delta."""
    flags = key.hook_flags
    if flags & (BEFORE_SWAP_RETURNS_DELTA_FLAG | AFTER_SWAP_RETURNS_DELTA_FLAG):
        raise UnsupportedPool(f"hook {key.hooks} has a swap return-delta flag ({flags:#06x}): not a fee-only hook")
    if not flags & BEFORE_SWAP_FLAG:
        raise UnsupportedPool(f"hook {key.hooks} has no BEFORE_SWAP flag: nothing to resolve")


def before_swap_fee(w3: Web3, pool_manager: str, key: PoolKey, sender: str, zero_for_one: bool, block: int,
                    amount_specified: int = -1) -> int | None:
    """LP fee the hook's ``beforeSwap`` returns for a swap by ``sender`` at ``block`` (``None``: no override, the pool
    uses its slot0 lpFee). ``amount_specified`` follows v4 (< 0 exact input). Raises ``MathRevert`` when the hook
    reverts (the swap would revert too) and ``UnsupportedPool`` when it returns a delta."""
    limit = MIN_SQRT_PRICE + 1 if zero_for_one else MAX_SQRT_PRICE - 1
    call = Call(key.hooks, BEFORE_SWAP_SIG, (to_checksum_address(sender), key.as_tuple(),
                                             (zero_for_one, amount_specified, limit), b""),
                ("bytes4", "int256", "uint24"))
    try:
        selector, delta, fee = eth_call(w3, call, block, sender=to_checksum_address(pool_manager))
    except CallFailed as exc:
        if exc.reverted:
            raise MathRevert(f"hook {key.hooks} beforeSwap reverts at block {block}") from None
        raise UnreadableState(f"hook {key.hooks} beforeSwap read failed at block {block}") from None
    if bytes(selector) != BEFORE_SWAP_SELECTOR:
        raise UnreadableState(f"hook {key.hooks} beforeSwap returned a wrong selector")
    if delta != 0:
        raise UnsupportedPool(f"hook {key.hooks} beforeSwap returned a non-zero delta")
    if not key.is_dynamic_fee or not fee & OVERRIDE_FEE_FLAG:
        return None  # Hooks.beforeSwap only applies an override on dynamic-fee pools with the flag set
    lp_fee = fee & REMOVE_OVERRIDE_MASK
    if lp_fee > MAX_LP_FEE:
        raise MathRevert(f"LPFeeTooLarge({lp_fee})")
    return lp_fee


def with_hook_fee(w3: Web3, pool_manager: str, state: V4PoolState, *, sender: str, block: int | None = None,
                  chain_id: int | None = None, check_code: bool = True) -> V4PoolState:
    """``state`` with ``hook_fee`` = the fee a reviewed fee-only hook applies to a swap by ``sender`` in each
    direction at ``block`` (default: the state's block). Hooks outside ``FEE_ONLY_HOOKS`` -> ``UnsupportedPool``."""
    key = state.key
    block = state.block_number if block is None else block
    if block is None:
        raise ValueError("block is required (the state has no block_number)")
    cid = int(w3.eth.chain_id) if chain_id is None else chain_id
    hook = reviewed_hook(cid, key.hooks)
    if hook is None:
        raise UnsupportedPool(f"hook {key.hooks} on chain {cid} is not a reviewed fee-only hook")
    check_fee_only_key(key)
    if check_code:
        code = bytes(w3.eth.get_code(to_checksum_address(key.hooks), block_identifier=block))
        if "0x" + keccak(code).hex() != hook.codehash:
            raise UnsupportedPool(f"hook {key.hooks} code differs from the reviewed {hook.name} build")
    fees = []
    for zfo in (True, False):
        f = before_swap_fee(w3, pool_manager, key, sender, zfo, block)
        fees.append(state.lp_fee if f is None else f)
    return replace(state, hook_fee=(fees[0], fees[1]))
