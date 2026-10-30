//! Uniswap v2 constant product: ``get_amount_out`` (src/v4quote/amm/uniswap_v2.py), 512-bit intermediates.

use crate::num::*;

#[derive(Debug)]
pub struct V2 {
    pub reserve0: U256,
    pub reserve1: U256,
    pub fee_numerator: u64,
    pub fee_denominator: u64,
}

pub fn get_amount_out(amount_in: U256, reserve_in: U256, reserve_out: U256, num: u64, den: u64) -> R<U256> {
    if amount_in.is_zero() {
        return revert("UniswapV2Library: INSUFFICIENT_INPUT_AMOUNT");
    }
    if reserve_in.is_zero() || reserve_out.is_zero() {
        return revert("UniswapV2Library: INSUFFICIENT_LIQUIDITY");
    }
    // amount <= 2^256, num, den < 2^64, reserves <= 2^256 (the state keeps them <= 2^112): every product < 2^512
    // except amount_in_with_fee * reserve_out with both near 2^256, which checked_mul turns into a Fallback.
    let aif: U512 = widen(amount_in).checked_mul(widen(u(num))).ok_or(MathErr::Fallback("u512 overflow"))?;
    let numerator = aif.checked_mul(widen(reserve_out)).ok_or(MathErr::Fallback("u512 overflow"))?;
    let denominator = widen(reserve_in)
        .checked_mul(widen(u(den)))
        .and_then(|v| v.checked_add(aif))
        .ok_or(MathErr::Fallback("u512 overflow"))?;
    if denominator.is_zero() {
        return fallback("division by zero");
    }
    narrow(numerator.wrapping_div(denominator)).ok_or(MathErr::Fallback("u256 overflow"))
}

impl V2 {
    pub fn new(reserve0: U256, reserve1: U256, fee_numerator: u64, fee_denominator: u64) -> Result<Self, String> {
        if !(0 < fee_numerator && fee_numerator <= fee_denominator) {
            return Err("invalid fee".into());
        }
        Ok(V2 { reserve0, reserve1, fee_numerator, fee_denominator })
    }

    pub fn amount_out(&self, amount_in: U256, zero_for_one: bool) -> R<U256> {
        let (r_in, r_out) = if zero_for_one { (self.reserve0, self.reserve1) } else { (self.reserve1, self.reserve0) };
        get_amount_out(amount_in, r_in, r_out, self.fee_numerator, self.fee_denominator)
    }

    /// The routing hop's fee: amount * (den - num) // den.
    pub fn fee(&self, amount_in: U256) -> R<U256> {
        let p: U512 = amount_in.widening_mul(u(self.fee_denominator - self.fee_numerator));
        narrow(p.wrapping_div(widen(u(self.fee_denominator)))).ok_or(MathErr::Fallback("u256 overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniswap_v2_get_amount_out() {
        // 1e18 in, 100e18 / 200e18 reserves, 0.3 %
        let e18 = u(1_000_000_000_000_000_000);
        let out = get_amount_out(e18, e18 * u(100), e18 * u(200), 997, 1000).unwrap();
        // 997e18 * 200e18 / (100e18 * 1000 + 997e18)
        let want = (e18 * u(997) * e18 * u(200)) / (e18 * u(100_000) + e18 * u(997));
        assert_eq!(out, want);
        assert!(matches!(get_amount_out(ZERO, e18, e18, 997, 1000), Err(MathErr::Revert(_))));
        assert!(matches!(get_amount_out(e18, ZERO, e18, 997, 1000), Err(MathErr::Revert(_))));
    }
}
