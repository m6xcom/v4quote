//! Fixed-width integers and the error model shared by every AMM port.
//!
//! The Python implementation computes with unbounded ints and raises ``MathRevert`` / ``InsufficientState`` where the chain would
//! revert. The native port must return exactly what Python returns, so every operation is one of:
//!
//! * a value that is provably identical to Python's (exact integer arithmetic in U256 / U512),
//! * ``Revert`` / ``Insufficient`` where Python raises ``MathRevert`` / ``InsufficientState`` (same message text),
//! * ``Fallback`` where the fixed width cannot represent what Python would compute (an intermediate above 2^256, a
//!   negative intermediate Python would carry on with) or where Python would raise some other exception
//!   (``ZeroDivisionError``, ``IndexError``, ``TypeError``). The Python dispatcher then runs the Python code for that
//!   call, so a fallback can cost time but never changes a result.

pub use ruint::aliases::{U256, U512};

/// ``Copy`` and heap-free on purpose: the hot loops propagate it with ``?`` at every step (a ``String`` inside made
/// them ~2x slower). Messages with data are formatted at the Python boundary (``MathErr::message``).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathErr {
    /// Python raises ``MathRevert(msg)``.
    Revert(&'static str),
    /// Python raises ``InsufficientState("tick bitmap word {pos} outside loaded range {word_range}")``.
    WordOutOfRange(i64),
    /// Python raises ``InsufficientState("liquidityNet for initialised tick {tick} not loaded")``.
    TickNotLoaded(i64),
    /// The native port cannot reproduce Python here: run the Python code instead.
    Fallback(&'static str),
}

impl MathErr {
    /// The Python exception message; ``word_range_repr`` is the v3 snapshot's ``repr(word_range)``.
    pub fn message(&self, word_range_repr: &str) -> String {
        match self {
            MathErr::Revert(m) | MathErr::Fallback(m) => (*m).to_string(),
            MathErr::WordOutOfRange(p) => format!("tick bitmap word {p} outside loaded range {word_range_repr}"),
            MathErr::TickNotLoaded(t) => format!("liquidityNet for initialised tick {t} not loaded"),
        }
    }

    pub fn is_insufficient(&self) -> bool {
        matches!(self, MathErr::WordOutOfRange(_) | MathErr::TickNotLoaded(_))
    }
}

pub type R<T> = Result<T, MathErr>;

#[inline]
pub fn revert<T>(msg: &'static str) -> R<T> {
    Err(MathErr::Revert(msg))
}

#[inline]
pub fn fallback<T>(why: &'static str) -> R<T> {
    Err(MathErr::Fallback(why))
}

pub const ZERO: U256 = U256::ZERO;
pub const ONE: U256 = U256::from_limbs([1, 0, 0, 0]);
pub const MAX_U256: U256 = U256::MAX;

#[inline]
pub const fn u(x: u64) -> U256 {
    U256::from_limbs([x, 0, 0, 0])
}

#[inline]
pub const fn u128_(x: u128) -> U256 {
    U256::from_limbs([x as u64, (x >> 64) as u64, 0, 0])
}

#[inline]
pub fn widen(x: U256) -> U512 {
    let l = x.as_limbs();
    U512::from_limbs([l[0], l[1], l[2], l[3], 0, 0, 0, 0])
}

#[inline]
pub fn narrow(x: U512) -> Option<U256> {
    let l = x.as_limbs();
    if (l[4] | l[5] | l[6] | l[7]) != 0 {
        None
    } else {
        Some(U256::from_limbs([l[0], l[1], l[2], l[3]]))
    }
}

/// U256 value as u128 when it fits.
#[inline]
pub fn to_u128(x: U256) -> Option<u128> {
    let l = x.as_limbs();
    if (l[2] | l[3]) != 0 {
        None
    } else {
        Some((l[0] as u128) | ((l[1] as u128) << 64))
    }
}

// ------------------------------------------------------------------ checked arithmetic (overflow / zero => Fallback)
#[inline(always)]
pub fn add(a: U256, b: U256) -> R<U256> {
    match a.checked_add(b) {
        Some(v) => Ok(v),
        None => Err(MathErr::Fallback("u256 add overflow")),
    }
}

#[inline(always)]
pub fn sub(a: U256, b: U256) -> R<U256> {
    match a.checked_sub(b) {
        Some(v) => Ok(v),
        None => Err(MathErr::Fallback("negative intermediate")),
    }
}

#[inline(always)]
pub fn mul(a: U256, b: U256) -> R<U256> {
    // bit_len(a * b) <= bit_len(a) + bit_len(b): no overflow check needed below 257 bits
    if a.bit_len() + b.bit_len() <= 256 {
        return Ok(a.wrapping_mul(b));
    }
    match a.checked_mul(b) {
        Some(v) => Ok(v),
        None => Err(MathErr::Fallback("u256 mul overflow")),
    }
}

#[inline(always)]
pub fn div(a: U256, b: U256) -> R<U256> {
    if b.is_zero() {
        return Err(MathErr::Fallback("division by zero"));
    }
    Ok(a.wrapping_div(b))
}

/// |a - b| <= 1
#[inline]
pub fn within_one(a: U256, b: U256) -> bool {
    let d = if a >= b { a.wrapping_sub(b) } else { b.wrapping_sub(a) };
    d <= ONE
}

// ------------------------------------------------------------------ FullMath / UnsafeMath (v4quote.math.full_math)
/// floor(a * b / d); reverts like FullMath.mulDiv (``mulDiv: denominator is zero`` / ``mulDiv: overflow``).
#[inline(always)]
pub fn mul_div(a: U256, b: U256, d: U256) -> R<U256> {
    if d.is_zero() {
        return revert("mulDiv: denominator is zero");
    }
    if a.bit_len() + b.bit_len() <= 256 {
        // the product fits 256 bits (so does the quotient): one 256-bit division
        return Ok(a.wrapping_mul(b).wrapping_div(d));
    }
    let p: U512 = a.widening_mul(b);
    let q = p.wrapping_div(widen(d));
    match narrow(q) {
        Some(v) => Ok(v),
        None => revert("mulDiv: overflow"),
    }
}

/// ceil(a * b / d); reverts like FullMath.mulDivRoundingUp.
#[inline(always)]
pub fn mul_div_rounding_up(a: U256, b: U256, d: U256) -> R<U256> {
    if d.is_zero() {
        return revert("mulDiv: denominator is zero");
    }
    if a.bit_len() + b.bit_len() <= 256 {
        let (q, r) = a.wrapping_mul(b).div_rem(d);
        // q <= a * b < 2^256 - 1 unless a * b == 2^256 - 1 exactly with d == 1 (then r == 0)
        return Ok(if r.is_zero() { q } else { q.wrapping_add(ONE) });
    }
    let p: U512 = a.widening_mul(b);
    let (q, r) = p.div_rem(widen(d));
    let q = match narrow(q) {
        Some(v) => v,
        None => return revert("mulDiv: overflow"),
    };
    if !r.is_zero() {
        if q == MAX_U256 {
            return revert("mulDivRoundingUp: overflow");
        }
        return Ok(q.wrapping_add(ONE));
    }
    Ok(q)
}

/// UnsafeMath.divRoundingUp: ceil(x / y), 0 when y == 0.
#[inline(always)]
pub fn div_rounding_up(x: U256, y: U256) -> U256 {
    if y.is_zero() {
        return ZERO;
    }
    let (q, r) = x.div_rem(y);
    if r.is_zero() {
        q
    } else {
        q.wrapping_add(ONE)
    }
}

// ------------------------------------------------------------------ int256 as two's complement U256 (TickMath only)
#[inline]
pub fn i256_from_i128(x: i128) -> U256 {
    if x >= 0 {
        u128_(x as u128)
    } else {
        // two's complement of |x|
        (!u128_(x.unsigned_abs())).wrapping_add(ONE)
    }
}

/// Arithmetic shift right of a two's complement int256.
#[inline]
pub fn sar(x: U256, s: usize) -> U256 {
    if x.bit(255) {
        !((!x) >> s)
    } else {
        x >> s
    }
}

/// A two's complement int256 known to be small, as i64 (None when it is not in the i64 range).
#[inline]
pub fn i256_to_i64(x: U256) -> Option<i64> {
    if x.bit(255) {
        let mag = (!x).wrapping_add(ONE);
        let m = to_u128(mag)?;
        if m <= (i64::MAX as u128) + 1 {
            Some((m as i128).wrapping_neg() as i64)
        } else {
            None
        }
    } else {
        let m = to_u128(x)?;
        if m <= i64::MAX as u128 {
            Some(m as i64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_div_basics() {
        assert_eq!(mul_div(u(7), u(3), u(2)).unwrap(), u(10));
        assert_eq!(mul_div_rounding_up(u(7), u(3), u(2)).unwrap(), u(11));
        assert_eq!(mul_div_rounding_up(u(8), u(3), u(2)).unwrap(), u(12));
        assert!(matches!(mul_div(u(1), u(1), ZERO), Err(MathErr::Revert(_))));
        // full 512-bit intermediate
        assert_eq!(mul_div(MAX_U256, MAX_U256, MAX_U256).unwrap(), MAX_U256);
        assert_eq!(mul_div(MAX_U256, u(2), u(1)), Err(MathErr::Revert("mulDiv: overflow")));
        assert_eq!(mul_div_rounding_up(MAX_U256, MAX_U256, MAX_U256 - ONE), Err(MathErr::Revert("mulDiv: overflow")));
        assert_eq!(div_rounding_up(u(7), ZERO), ZERO);
        assert_eq!(div_rounding_up(u(7), u(2)), u(4));
    }

    #[test]
    fn signed_helpers() {
        for v in [-5i128, -1, 0, 1, 77, -(1i128 << 100), (1i128 << 100)] {
            let x = i256_from_i128(v);
            assert_eq!(i256_to_i64(sar(x, 0)), i64::try_from(v).ok());
        }
        // floor semantics of the arithmetic shift
        assert_eq!(i256_to_i64(sar(i256_from_i128(-1), 4)), Some(-1));
        assert_eq!(i256_to_i64(sar(i256_from_i128(-17), 4)), Some(-2));
        assert_eq!(i256_to_i64(sar(i256_from_i128(17), 4)), Some(1));
    }
}
