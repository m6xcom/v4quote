//! OZ v5 ERC-4626 vault as an asset <-> shares pool (src/v4quote/amm/erc4626.py): ``deposit_out`` / ``redeem_out``.
//!
//! previewDeposit(a) = floor(a (S + 10^o) / (A + 1)); previewRedeem(s) = floor(s (A + 1) / (S + 10^o)); 512-bit
//! intermediates, the same reverts (and messages) as the Python module.

use crate::num::*;

pub const ERR_MULDIV: &str = "MathOverflowedMulDiv";
pub const ERR_MAX_REDEEM: &str = "ERC4626ExceededMaxRedeem";
pub const ERR_SUPPLY_OVERFLOW: &str = "panic: arithmetic overflow (totalSupply)";
pub const ERR_ASSETS_OVERFLOW: &str = "panic: arithmetic overflow (asset balance)";
pub const MAX_DECIMALS_OFFSET: u32 = 18;

#[derive(Debug)]
pub struct Vault {
    pub total_assets: U256,
    pub total_supply: U256,
    /// 10 ** decimals_offset
    pub virtual_shares: U256,
}

/// floor(x * y / d) with the OZ v5.0 ``Math.mulDiv`` overflow revert; ``d`` > 0.
fn mul_div(x: U256, y: U512, d: U512) -> R<U256> {
    let p = widen(x).checked_mul(y).ok_or(MathErr::Fallback("u512 overflow"))?;
    if d.is_zero() {
        return fallback("division by zero");
    }
    match narrow(p.wrapping_div(d)) {
        Some(q) => Ok(q),
        None => revert(ERR_MULDIV),
    }
}

impl Vault {
    pub fn new(total_assets: U256, total_supply: U256, decimals_offset: u32) -> Result<Self, String> {
        if decimals_offset > MAX_DECIMALS_OFFSET {
            return Err("decimals_offset out of range".into());
        }
        Ok(Vault { total_assets, total_supply, virtual_shares: u(10u64.pow(decimals_offset)) })
    }

    fn supply_plus_virtual(&self) -> U512 {
        widen(self.total_supply) + widen(self.virtual_shares)
    }

    fn assets_plus_one(&self) -> U512 {
        widen(self.total_assets) + widen(ONE)
    }

    pub fn preview_deposit(&self, assets: U256) -> R<U256> {
        mul_div(assets, self.supply_plus_virtual(), self.assets_plus_one())
    }

    pub fn preview_redeem(&self, shares: U256) -> R<U256> {
        mul_div(shares, self.assets_plus_one(), self.supply_plus_virtual())
    }

    /// ``deposit(assets)``: shares minted; reverts on the asset-balance / supply overflow like Python.
    pub fn deposit_out(&self, assets: U256) -> R<U256> {
        let shares = self.preview_deposit(assets)?;
        if self.total_assets.checked_add(assets).is_none() {
            return revert(ERR_ASSETS_OVERFLOW);
        }
        if self.total_supply.checked_add(shares).is_none() {
            return revert(ERR_SUPPLY_OVERFLOW);
        }
        Ok(shares)
    }

    /// ``redeem(shares)`` by an owner holding them: assets paid; more shares than the supply revert.
    pub fn redeem_out(&self, shares: U256) -> R<U256> {
        if shares > self.total_supply {
            return revert(ERR_MAX_REDEEM);
        }
        self.preview_redeem(shares)
    }

    /// Coin indices: (0, 1) deposit, (1, 0) redeem; anything else is a Python ValueError (Fallback here).
    pub fn quote(&self, i: usize, j: usize, amount: U256) -> R<U256> {
        match (i, j) {
            (0, 1) => self.deposit_out(amount),
            (1, 0) => self.redeem_out(amount),
            _ => fallback("bad coin indices for an ERC-4626 vault"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(x: u64) -> U256 {
        u(x) * u(1_000_000_000_000_000_000)
    }

    #[test]
    fn sdai_block_48602693() {
        // totalAssets / totalSupply read on Gnosis at block 48602693; previewDeposit(1e21) / previewRedeem(1e21) there
        let a = U256::from_str_radix("42576090319993786726811503", 10).unwrap();
        let s = U256::from_str_radix("33782479856640913397697833", 10).unwrap();
        let v = Vault::new(a, s, 0).unwrap();
        assert_eq!(v.preview_deposit(e(1000)).unwrap(), U256::from_str_radix("793461297238384930913", 10).unwrap());
        assert_eq!(v.preview_redeem(e(1000)).unwrap(), U256::from_str_radix("1260300916352777384584", 10).unwrap());
    }

    #[test]
    fn reverts_and_edges() {
        let v = Vault::new(e(100), e(80), 0).unwrap();
        assert_eq!(v.redeem_out(e(80) + ONE), Err(MathErr::Revert(ERR_MAX_REDEEM)));
        assert_eq!(v.redeem_out(e(80)).unwrap(), (e(80) * (e(100) + ONE)) / (e(80) + ONE));
        assert!(v.quote(0, 0, ONE).is_err());
        // empty vault: 1 share per asset (+ virtual offset)
        let z = Vault::new(ZERO, ZERO, 3).unwrap();
        assert_eq!(z.preview_deposit(u(7)).unwrap(), u(7000));
        // result above 2^256 - 1
        let big = Vault::new(ZERO, MAX_U256 - u(5), 0).unwrap();
        assert_eq!(big.preview_deposit(u(2)), Err(MathErr::Revert(ERR_MULDIV)));
        // supply overflow on mint
        let full = Vault::new(e(1), MAX_U256 - u(10), 0).unwrap();
        assert_eq!(full.deposit_out(u(1)), Err(MathErr::Revert(ERR_SUPPLY_OVERFLOW)));
        let fat = Vault::new(MAX_U256, ONE, 0).unwrap();
        assert_eq!(fat.deposit_out(MAX_U256), Err(MathErr::Revert(ERR_ASSETS_OVERFLOW)));
        assert!(Vault::new(ONE, ONE, 19).is_err());
    }
}
