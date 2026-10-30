"""ERC-4626 tokenized vault as a two-token "pool" (asset <-> shares), exact integer math of OpenZeppelin v5 ``ERC4626``.

Model (OpenZeppelin Contracts v5.0, ``token/ERC20/extensions/ERC4626.sol``), with ``A = totalAssets()``,
``S = totalSupply()`` and ``o = _decimalsOffset()``::

    previewDeposit(a) = convertToShares(a, Floor) = mulDiv(a, S + 10**o, A + 1, Floor)
    previewRedeem(s)  = convertToAssets(s, Floor) = mulDiv(s, A + 1, S + 10**o, Floor)
    previewMint(s)    = mulDiv(s, A + 1, S + 10**o, Ceil)
    previewWithdraw(a)= mulDiv(a, S + 10**o, A + 1, Ceil)

``deposit(a)`` pulls ``a`` of the asset and mints ``previewDeposit(a)`` shares; ``redeem(s)`` burns ``s`` shares and pays
``previewRedeem(s)`` of the asset. No fee. A vault that overrides the conversion (fees, other rounding, a non-default
``totalAssets`` that is not the asset balance) is NOT this model: ``v4quote.chain.loaders.load_erc4626`` checks
``previewDeposit`` / ``previewRedeem`` on chain against these formulas at load time and refuses a vault that differs.

Gnosis sDAI (``SavingsXDai`` 0xaf20...3701): OZ v5 ERC4626 over WXDAI,
``totalAssets = WXDAI.balanceOf(vault)``, ``o = 0``, not a proxy; the formulas matched ``previewDeposit`` /
``previewRedeem`` 125/125 times on 25 historical blocks. The price moves only through direct asset transfers into the
vault (``BridgeInterestReceiver.claim()``, every block since 2026-09-28).

Pool view for ``v4quote.routing``: tokens ``(asset, vault)`` (the vault address IS the share token);
coin index 0 -> 1 = ``deposit`` (asset in, shares out), 1 -> 0 = ``redeem`` (shares in, asset out).

Reverts (``MathRevert``): a result above 2**256 - 1 (OZ v5.0 ``Math.mulDiv``: ``MathOverflowedMulDiv``), a mint that
would overflow ``totalSupply`` or an asset balance overflow (checked arithmetic), a redeem of more shares than exist
(``ERC4626ExceededMaxRedeem``: ``maxRedeem(owner) = balanceOf(owner) <= totalSupply``). ``maxDeposit`` is OZ's default
(``type(uint256).max``); a vault with another ``maxDeposit`` is outside the model (the loader checks it).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from v4quote.errors import MathRevert
from v4quote.math.uint import MAX_UINT256

MAX_DECIMALS_OFFSET = 18  # OZ allows uint8, 10**o overflows uint256 for o >= 78; real vaults use 0 .. a few
ERR_MULDIV = "MathOverflowedMulDiv"
ERR_MAX_REDEEM = "ERC4626ExceededMaxRedeem"
ERR_SUPPLY_OVERFLOW = "panic: arithmetic overflow (totalSupply)"
ERR_ASSETS_OVERFLOW = "panic: arithmetic overflow (asset balance)"


@dataclass(frozen=True)
class ERC4626VaultState:
    """Snapshot of an OZ v5 ERC-4626 vault: ``address`` is the vault (= share token), ``asset`` the underlying.

    Not ``slots``: the native backend caches its handle by a weak reference to the (frozen) state object."""

    address: str
    asset: str
    total_assets: int
    total_supply: int
    decimals_offset: int = 0
    block_number: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_assets", "total_supply"):
            v = getattr(self, name)
            if type(v) is not int or not 0 <= v <= MAX_UINT256:
                raise ValueError(f"{name} must be an int in [0, 2**256)")
        if type(self.decimals_offset) is not int or not 0 <= self.decimals_offset <= MAX_DECIMALS_OFFSET:
            raise ValueError(f"decimals_offset must be an int in 0..{MAX_DECIMALS_OFFSET}")
        if self.address.lower() == self.asset.lower():
            raise ValueError("vault and asset must differ")

    @property
    def coins(self) -> tuple[str, str]:
        """``(asset, vault)``: index 0 -> 1 is deposit, 1 -> 0 is redeem."""
        return (self.asset, self.address)

    @property
    def virtual_shares(self) -> int:
        return 10 ** self.decimals_offset


def _mul_div(x: int, y: int, d: int, ceil: bool) -> int:
    q, r = divmod(x * y, d)
    if ceil and r:
        q += 1
    if q > MAX_UINT256:
        raise MathRevert(ERR_MULDIV)
    return q


def convert_to_shares(st: ERC4626VaultState, assets: int, *, ceil: bool = False) -> int:
    return _mul_div(assets, st.total_supply + st.virtual_shares, st.total_assets + 1, ceil)


def convert_to_assets(st: ERC4626VaultState, shares: int, *, ceil: bool = False) -> int:
    return _mul_div(shares, st.total_assets + 1, st.total_supply + st.virtual_shares, ceil)


def preview_deposit(st: ERC4626VaultState, assets: int) -> int:
    """Shares minted by ``deposit(assets)`` (floor)."""
    _check_amount(assets)
    return convert_to_shares(st, assets)


def preview_redeem(st: ERC4626VaultState, shares: int) -> int:
    """Assets paid by ``redeem(shares)`` (floor)."""
    _check_amount(shares)
    return convert_to_assets(st, shares)


def preview_mint(st: ERC4626VaultState, shares: int) -> int:
    """Assets pulled by ``mint(shares)`` (ceil)."""
    _check_amount(shares)
    return convert_to_assets(st, shares, ceil=True)


def preview_withdraw(st: ERC4626VaultState, assets: int) -> int:
    """Shares burnt by ``withdraw(assets)`` (ceil)."""
    _check_amount(assets)
    return convert_to_shares(st, assets, ceil=True)


def _check_amount(x: int) -> None:
    if type(x) is not int or not 0 <= x <= MAX_UINT256:
        raise ValueError("amount must be an int in [0, 2**256)")


def deposit_out(st: ERC4626VaultState, assets: int) -> int:
    """``deposit(assets)`` as executed: the shares minted, with the reverts of the transfer-in and the mint."""
    shares = preview_deposit(st, assets)
    if st.total_assets + assets > MAX_UINT256:
        raise MathRevert(ERR_ASSETS_OVERFLOW)
    if st.total_supply + shares > MAX_UINT256:
        raise MathRevert(ERR_SUPPLY_OVERFLOW)
    return shares


def redeem_out(st: ERC4626VaultState, shares: int) -> int:
    """``redeem(shares)`` as executed by an owner holding ``shares``: the assets paid (``shares <= totalSupply``)."""
    _check_amount(shares)
    if shares > st.total_supply:
        raise MathRevert(ERR_MAX_REDEEM)
    return convert_to_assets(st, shares)  # <= A whenever shares <= S (floor of s (A+1) / (S + 10**o))


def deposit(st: ERC4626VaultState, assets: int) -> tuple[int, ERC4626VaultState]:
    """Shares and the post-state (A + assets, S + shares)."""
    shares = deposit_out(st, assets)
    return shares, replace(st, total_assets=st.total_assets + assets, total_supply=st.total_supply + shares)


def redeem(st: ERC4626VaultState, shares: int) -> tuple[int, ERC4626VaultState]:
    """Assets and the post-state (A - assets, S - shares)."""
    assets = redeem_out(st, shares)
    return assets, replace(st, total_assets=st.total_assets - assets, total_supply=st.total_supply - shares)


def quote(st: ERC4626VaultState, i: int, j: int, amount: int) -> int:
    """Exact-input quote in pool-coin terms: ``(0, 1)`` = deposit, ``(1, 0)`` = redeem."""
    if (i, j) == (0, 1):
        return deposit_out(st, amount)
    if (i, j) == (1, 0):
        return redeem_out(st, amount)
    raise ValueError(f"bad coin indices ({i}, {j}) for an ERC-4626 vault (0 = asset, 1 = shares)")


def apply_flows(st: ERC4626VaultState, d_assets: int, d_supply: int, block_number: int | None = None) -> ERC4626VaultState:
    """The state after net asset-balance / supply changes (e.g. summed from simulated ``Transfer`` logs: asset
    transfers to / from the vault, share mints / burns). Raises ValueError when a result leaves [0, 2**256)."""
    return replace(st, total_assets=st.total_assets + d_assets, total_supply=st.total_supply + d_supply,
                   block_number=st.block_number if block_number is None else block_number)


__all__ = [
    "ERC4626VaultState", "ERR_ASSETS_OVERFLOW", "ERR_MAX_REDEEM", "ERR_MULDIV", "ERR_SUPPLY_OVERFLOW",
    "MAX_DECIMALS_OFFSET", "apply_flows", "convert_to_assets", "convert_to_shares", "deposit", "deposit_out",
    "preview_deposit", "preview_mint", "preview_redeem", "preview_withdraw", "quote", "redeem", "redeem_out",
]
