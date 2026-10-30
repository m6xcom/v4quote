//! Uniswap v3 (and exact forks): TickMath, SqrtPriceMath, SwapMath.computeSwapStep (v3, exact input) and the swap
//! loop over a tick-bitmap snapshot. Port of src/v4quote/amm/v3/{tick_math,sqrt_price_math,swap_math,tick_bitmap,
//! liquidity_math,pool}.py restricted to what ``quote_exact_input`` uses (exact input, v3 step, protocol fee and fee
//! growth not computed: they never change an amount and never raise).

use std::collections::HashMap;
use std::sync::Arc;

use ruint::uint;

use crate::num::*;

pub const MIN_TICK: i64 = -887272;
pub const MAX_TICK: i64 = 887272;
pub const MIN_SQRT_RATIO: U256 = uint!(4295128739_U256);
pub const MAX_SQRT_RATIO: U256 = uint!(1461446703485210103287273052203988822378723970342_U256);
const MAX_UINT160: U256 = uint!(0xffffffffffffffffffffffffffffffffffffffff_U256);
const MAX_INT256: U256 = uint!(0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256);
const Q96: U256 = uint!(0x1000000000000000000000000_U256);
const MAX_SWAP_FEE: u64 = 1_000_000;

const RATIO_CONSTANTS: [(i64, U256); 19] = [
    (0x2, uint!(0xFFF97272373D413259A46990580E213A_U256)),
    (0x4, uint!(0xFFF2E50F5F656932EF12357CF3C7FDCC_U256)),
    (0x8, uint!(0xFFE5CACA7E10E4E61C3624EAA0941CD0_U256)),
    (0x10, uint!(0xFFCB9843D60F6159C9DB58835C926644_U256)),
    (0x20, uint!(0xFF973B41FA98C081472E6896DFB254C0_U256)),
    (0x40, uint!(0xFF2EA16466C96A3843EC78B326B52861_U256)),
    (0x80, uint!(0xFE5DEE046A99A2A811C461F1969C3053_U256)),
    (0x100, uint!(0xFCBE86C7900A88AEDCFFC83B479AA3A4_U256)),
    (0x200, uint!(0xF987A7253AC413176F2B074CF7815E54_U256)),
    (0x400, uint!(0xF3392B0822B70005940C7A398E4B70F3_U256)),
    (0x800, uint!(0xE7159475A2C29B7443B29C7FA6E889D9_U256)),
    (0x1000, uint!(0xD097F3BDFD2022B8845AD8F792AA5825_U256)),
    (0x2000, uint!(0xA9F746462D870FDF8A65DC1F90E061E5_U256)),
    (0x4000, uint!(0x70D869A156D2A1B890BB3DF62BAF32F7_U256)),
    (0x8000, uint!(0x31BE135F97D08FD981231505542FCFA6_U256)),
    (0x10000, uint!(0x9AA508B5B7A84E1C677DE54F3E99BC9_U256)),
    (0x20000, uint!(0x5D6AF8DEDB81196699C329225EE604_U256)),
    (0x40000, uint!(0x2216E584F5FA1EA926041BEDFE98_U256)),
    (0x80000, uint!(0x48A170391F7DC42444E8FA2_U256)),
];

// ------------------------------------------------------------------------------------------------ TickMath
/// sqrt(1.0001^tick) * 2^96 (TickMath.getSqrtRatioAtTick); ``T`` revert outside [MIN_TICK, MAX_TICK].
pub fn get_sqrt_ratio_at_tick(tick: i64) -> R<U256> {
    let abs_tick = tick.unsigned_abs() as i64;
    if tick.unsigned_abs() > MAX_TICK as u64 {
        return revert("T");
    }
    let mut ratio = if abs_tick & 1 != 0 {
        uint!(0xFFFCB933BD6FAD37AA2D162D1A594001_U256)
    } else {
        uint!(0x100000000000000000000000000000000_U256)
    };
    for (bit, c) in RATIO_CONSTANTS.iter() {
        if abs_tick & bit != 0 {
            // ratio <= 2^128 and c < 2^128: the product fits 256 bits
            ratio = ratio.wrapping_mul(*c) >> 128usize;
        }
    }
    if tick > 0 {
        ratio = MAX_U256.wrapping_div(ratio);
    }
    let low32 = ratio.as_limbs()[0] & 0xFFFF_FFFF;
    Ok((ratio >> 32usize).wrapping_add(if low32 == 0 { ZERO } else { ONE }))
}

/// Greatest tick with get_sqrt_ratio_at_tick(tick) <= sqrt_price (TickMath.getTickAtSqrtRatio); ``R`` revert.
pub fn get_tick_at_sqrt_ratio(sqrt_price_x96: U256) -> R<i64> {
    if !(MIN_SQRT_RATIO <= sqrt_price_x96 && sqrt_price_x96 < MAX_SQRT_RATIO) {
        return revert("R");
    }
    let ratio = sqrt_price_x96 << 32usize; // < 2^193
    let msb = ratio.bit_len() as i64 - 1;
    let mut r = if msb >= 128 { ratio >> ((msb - 127) as usize) } else { ratio << ((127 - msb) as usize) };
    let mut log_2: i128 = (msb as i128 - 128) << 64;
    let mut shift = 63i32;
    while shift > 49 {
        r = r.wrapping_mul(r) >> 127usize;
        let f = r >> 128usize; // 0 or 1
        let fb = if f.is_zero() { 0i128 } else { 1i128 };
        log_2 |= fb << shift;
        r >>= if fb == 1 { 1usize } else { 0usize };
        shift -= 1;
    }
    // log_sqrt10001 = log_2 * 255738958999603826347141 (|.| < 2^150: int256 two's complement arithmetic is exact)
    let log_sqrt10001 = i256_from_i128(log_2).wrapping_mul(uint!(255738958999603826347141_U256));
    let low = sar(log_sqrt10001.wrapping_sub(uint!(3402992956809132418596140100660247210_U256)), 128);
    let hi = sar(log_sqrt10001.wrapping_add(uint!(291339464771989622907027621153398088495_U256)), 128);
    let tick_low = i256_to_i64(low).ok_or(MathErr::Fallback("tick out of range"))?;
    let tick_hi = i256_to_i64(hi).ok_or(MathErr::Fallback("tick out of range"))?;
    if tick_low == tick_hi {
        return Ok(tick_low);
    }
    Ok(if get_sqrt_ratio_at_tick(tick_hi)? <= sqrt_price_x96 { tick_hi } else { tick_low })
}

// ------------------------------------------------------------------------------------------------ SqrtPriceMath
/// SafeCast to uint160. Python raises ``MathRevert(f"uint160 overflow: {x}")``: reported as Fallback, so the Python
/// code raises its own message (the case needs a price above the uint160 range; never on a real pool).
fn to_uint160(x: U256) -> R<U256> {
    if x > MAX_UINT160 {
        return fallback("uint160 overflow");
    }
    Ok(x)
}

/// get_next_sqrt_price_from_amount0_rounding_up(..., add=True)
fn next_from_amount0_up_add(sqrt_px96: U256, liquidity: U256, amount: U256) -> R<U256> {
    if amount.is_zero() {
        return Ok(sqrt_px96);
    }
    let numerator1 = liquidity << 96usize; // liquidity <= 2^128 - 1
    let product = amount.wrapping_mul(sqrt_px96);
    if product.wrapping_div(amount) == sqrt_px96 {
        let denominator = numerator1.wrapping_add(product);
        if denominator >= numerator1 {
            return mul_div_rounding_up(numerator1, sqrt_px96, denominator);
        }
    }
    let denom = match numerator1.wrapping_div(sqrt_px96).checked_add(amount) {
        Some(v) => v,
        None => return revert("add overflow"),
    };
    Ok(div_rounding_up(numerator1, denom))
}

/// get_next_sqrt_price_from_amount1_rounding_down(..., add=True)
fn next_from_amount1_down_add(sqrt_px96: U256, liquidity: U256, amount: U256) -> R<U256> {
    let quotient = if amount <= MAX_UINT160 {
        (amount << 96usize).wrapping_div(liquidity)
    } else {
        mul_div(amount, Q96, liquidity)?
    };
    match sqrt_px96.checked_add(quotient) {
        Some(v) => to_uint160(v),
        None => fallback("uint160 overflow"),
    }
}

fn next_sqrt_price_from_input(sqrt_px96: U256, liquidity: U256, amount_in: U256, zero_for_one: bool) -> R<U256> {
    if sqrt_px96.is_zero() || liquidity.is_zero() {
        return revert("InvalidPriceOrLiquidity");
    }
    if zero_for_one {
        next_from_amount0_up_add(sqrt_px96, liquidity, amount_in)
    } else {
        next_from_amount1_down_add(sqrt_px96, liquidity, amount_in)
    }
}

pub fn get_amount0_delta(mut a: U256, mut b: U256, liquidity: U256, round_up: bool) -> R<U256> {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if a.is_zero() {
        return revert("InvalidPrice");
    }
    let numerator1 = liquidity << 96usize;
    let numerator2 = b.wrapping_sub(a);
    if round_up {
        Ok(div_rounding_up(mul_div_rounding_up(numerator1, numerator2, b)?, a))
    } else {
        Ok(mul_div(numerator1, numerator2, b)?.wrapping_div(a))
    }
}

pub fn get_amount1_delta(mut a: U256, mut b: U256, liquidity: U256, round_up: bool) -> R<U256> {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if round_up {
        mul_div_rounding_up(liquidity, b.wrapping_sub(a), Q96)
    } else {
        mul_div(liquidity, b.wrapping_sub(a), Q96)
    }
}

// ------------------------------------------------------------------------------------------------ SwapMath (v3)
/// compute_swap_step_v3 for exact input (``amount_remaining`` > 0): (sqrt_next, amount_in, amount_out, fee_amount).
pub fn compute_swap_step_v3_exact_in(
    current: U256,
    target: U256,
    liquidity: U256,
    amount_remaining: U256,
    fee_pips: u64,
) -> R<(U256, U256, U256, U256)> {
    let zero_for_one = current >= target;
    let less_fee = mul_div(amount_remaining, u(MAX_SWAP_FEE - fee_pips), u(MAX_SWAP_FEE))?;
    let mut amount_in = if zero_for_one {
        get_amount0_delta(target, current, liquidity, true)?
    } else {
        get_amount1_delta(current, target, liquidity, true)?
    };
    let next = if less_fee >= amount_in {
        target
    } else {
        next_sqrt_price_from_input(current, liquidity, less_fee, zero_for_one)?
    };
    let is_max = target == next;
    let amount_out;
    if zero_for_one {
        if !is_max {
            amount_in = get_amount0_delta(next, current, liquidity, true)?;
        }
        amount_out = get_amount1_delta(next, current, liquidity, false)?;
    } else {
        if !is_max {
            amount_in = get_amount1_delta(current, next, liquidity, true)?;
        }
        amount_out = get_amount0_delta(current, next, liquidity, false)?;
    }
    let fee_amount = if next != target {
        // Python: amount_remaining - amount_in (a negative value would flow on in Python: not reproduced)
        sub(amount_remaining, amount_in)?
    } else {
        mul_div_rounding_up(amount_in, u(fee_pips), u(MAX_SWAP_FEE - fee_pips))?
    };
    Ok((next, amount_in, amount_out, fee_amount))
}

// ------------------------------------------------------------------------------------------------ tick bitmap
/// TickBitmapSnapshot: words (wordPos -> 256-bit word), liquidityNet by initialised tick, the loaded word range.
#[derive(Debug)]
pub struct Ticks {
    pub spacing: i64,
    words: HashMap<i64, U256>,
    liquidity_net: HashMap<i64, i128>,
    word_range: Option<(i64, i64)>,
    /// Python ``repr`` of ``word_range`` (the InsufficientState message embeds it).
    pub word_range_repr: String,
}

impl Ticks {
    pub fn new(
        spacing: i64,
        words: Vec<(i64, U256)>,
        liquidity_net: Vec<(i64, i128)>,
        word_range: Option<(i64, i64)>,
        word_range_repr: String,
    ) -> Result<Self, String> {
        if spacing <= 0 || spacing > (1 << 23) {
            return Err("tick_spacing out of the native range".into());
        }
        let mut w = HashMap::with_capacity(words.len());
        for (k, v) in words {
            // keys far outside int16 are not something the pool can produce; keep the arithmetic in i64 safe
            if k.unsigned_abs() > (1 << 40) {
                return Err("word position out of the native range".into());
            }
            w.insert(k, v);
        }
        let mut n = HashMap::with_capacity(liquidity_net.len());
        for (k, v) in liquidity_net {
            n.insert(k, v);
        }
        Ok(Ticks { spacing, words: w, liquidity_net: n, word_range, word_range_repr })
    }

    fn word(&self, pos: i64) -> R<U256> {
        if let Some((lo, hi)) = self.word_range {
            if !(lo <= pos && pos <= hi) {
                return Err(MathErr::WordOutOfRange(pos));
            }
        }
        Ok(self.words.get(&pos).copied().unwrap_or(ZERO))
    }

    fn get_liquidity_net(&self, tick: i64) -> R<i128> {
        match self.liquidity_net.get(&tick) {
            Some(v) => Ok(*v),
            None => Err(MathErr::TickNotLoaded(tick)),
        }
    }

    /// TickBitmap.nextInitializedTickWithinOneWord
    pub fn next_initialized_tick_within_one_word(&self, tick: i64, lte: bool) -> R<(i64, bool)> {
        let ts = self.spacing;
        let mut compressed = tick.div_euclid(ts); // floor (ts > 0)
        if lte {
            let word_pos = compressed >> 8;
            let bit_pos = (compressed & 0xFF) as usize;
            let mask = if bit_pos == 255 { MAX_U256 } else { (ONE << (bit_pos + 1)).wrapping_sub(ONE) };
            let masked = self.word(word_pos)? & mask;
            if !masked.is_zero() {
                let msb = masked.bit_len() as i64 - 1;
                Ok(((compressed - (bit_pos as i64 - msb)) * ts, true))
            } else {
                Ok(((compressed - bit_pos as i64) * ts, false))
            }
        } else {
            compressed += 1;
            let word_pos = compressed >> 8;
            let bit_pos = (compressed & 0xFF) as usize;
            let mask = !((ONE << bit_pos).wrapping_sub(ONE));
            let masked = self.word(word_pos)? & mask;
            if !masked.is_zero() {
                let lsb = masked.trailing_zeros() as i64;
                Ok(((compressed + (lsb - bit_pos as i64)) * ts, true))
            } else {
                Ok(((compressed + (255 - bit_pos as i64)) * ts, false))
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ pool + swap loop
#[derive(Debug)]
pub struct Pool {
    pub ticks: Arc<Ticks>,
    pub sqrt_price_x96: U256,
    pub tick: i64,
    pub liquidity: u128,
    pub fee: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapOut {
    pub amount_in: U256,
    pub amount_out: U256,
    pub fee_amount: U256,
    pub sqrt_price_x96: U256,
    pub tick: Option<i64>,
    pub liquidity: u128,
    pub crossed: u64,
    pub steps: u64,
    pub remaining: U256,
}

/// SafeCast to int256 (Python: ``MathRevert(f"int256 overflow: {x}")``, reported as Fallback like to_uint160).
fn to_int256_check(x: U512) -> R<()> {
    if x > widen(MAX_INT256) {
        return fallback("int256 overflow");
    }
    Ok(())
}

/// LiquidityMath.addDelta with y = (negated ? -net : net); ``LS`` / ``LA``.
fn add_delta(x: u128, net: i128, negate: bool) -> R<u128> {
    let mag = net.unsigned_abs();
    let neg = if mag == 0 { false } else { (net < 0) != negate };
    if neg {
        if mag > x {
            return revert("LS");
        }
        Ok(x - mag)
    } else {
        match x.checked_add(mag) {
            Some(z) => Ok(z),
            None => revert("LA"),
        }
    }
}

impl Pool {
    pub fn new(ticks: Arc<Ticks>, sqrt_price_x96: U256, tick: i64, liquidity: u128, fee: u64) -> Result<Self, String> {
        if fee >= MAX_SWAP_FEE {
            return Err("fee out of range".into());
        }
        if sqrt_price_x96 > MAX_UINT160 {
            return Err("sqrtPrice out of range".into());
        }
        if tick.unsigned_abs() > (1u64 << 40) {
            return Err("tick out of the native range".into());
        }
        Ok(Pool { ticks, sqrt_price_x96, tick, liquidity, fee })
    }

    /// v3 ``quote_exact_input(state, zero_for_one, amount_in, sqrt_price_limit_x96, want_tick=...)``.
    pub fn quote_exact_input(
        &self,
        zero_for_one: bool,
        amount_in: U256,
        sqrt_price_limit_x96: Option<U256>,
        want_tick: bool,
    ) -> R<SwapOut> {
        if amount_in.is_zero() {
            return revert("amount_in must be positive");
        }
        let limit = match sqrt_price_limit_x96 {
            Some(l) => l,
            None => {
                if zero_for_one {
                    MIN_SQRT_RATIO.wrapping_add(ONE)
                } else {
                    MAX_SQRT_RATIO.wrapping_sub(ONE)
                }
            }
        };
        if zero_for_one {
            if !(limit < self.sqrt_price_x96 && limit > MIN_SQRT_RATIO) {
                return revert("SPL");
            }
        } else if !(limit > self.sqrt_price_x96 && limit < MAX_SQRT_RATIO) {
            return revert("SPL");
        }
        self.swap_loop_exact_in(zero_for_one, amount_in, limit, want_tick)
    }

    fn swap_loop_exact_in(&self, zero_for_one: bool, amount: U256, limit: U256, want_tick: bool) -> R<SwapOut> {
        let ticks = &*self.ticks;
        let mut sqrt_price = self.sqrt_price_x96;
        let mut tick: Option<i64> = Some(self.tick);
        let mut liquidity = self.liquidity;
        let mut remaining = amount;
        let mut total_in = ZERO;
        let mut total_out = ZERO;
        let mut total_fee = ZERO;
        let mut crossed = 0u64;
        let mut steps = 0u64;

        while !remaining.is_zero() && sqrt_price != limit {
            steps += 1;
            let start = sqrt_price;
            let t = match tick {
                Some(t) => t,
                None => return fallback("tick unknown after a partial step (Python would raise TypeError)"),
            };
            let (mut tick_next, initialized) = ticks.next_initialized_tick_within_one_word(t, zero_for_one)?;
            tick_next = tick_next.clamp(MIN_TICK, MAX_TICK);
            let sqrt_next = get_sqrt_ratio_at_tick(tick_next)?;
            let target = if zero_for_one {
                if sqrt_next < limit { limit } else { sqrt_next }
            } else if sqrt_next > limit {
                limit
            } else {
                sqrt_next
            };
            let (p, step_in, step_out, step_fee) =
                compute_swap_step_v3_exact_in(sqrt_price, target, u128_(liquidity), remaining, self.fee)?;
            sqrt_price = p;
            let consumed_w = widen(step_in) + widen(step_fee);
            to_int256_check(consumed_w)?;
            to_int256_check(widen(step_out))?;
            let consumed = narrow(consumed_w).ok_or(MathErr::Fallback("u256 add overflow"))?;
            // Python carries on with a negative remaining (exact-output semantics): not reproduced
            remaining = sub(remaining, consumed)?;
            total_in = add(total_in, consumed)?;
            total_out = add(total_out, step_out)?;
            total_fee = add(total_fee, step_fee)?;

            if sqrt_price == sqrt_next {
                if initialized {
                    let net = ticks.get_liquidity_net(tick_next)?;
                    liquidity = add_delta(liquidity, net, zero_for_one)?;
                    crossed += 1;
                }
                tick = Some(if zero_for_one { tick_next - 1 } else { tick_next });
            } else if sqrt_price != start {
                tick = if want_tick { Some(get_tick_at_sqrt_ratio(sqrt_price)?) } else { None };
            }
        }
        Ok(SwapOut {
            amount_in: total_in,
            amount_out: total_out,
            fee_amount: total_fee,
            sqrt_price_x96: sqrt_price,
            tick,
            liquidity,
            crossed,
            steps,
            remaining,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_math_known_values() {
        assert_eq!(get_sqrt_ratio_at_tick(MIN_TICK).unwrap(), MIN_SQRT_RATIO);
        assert_eq!(get_sqrt_ratio_at_tick(MAX_TICK).unwrap(), MAX_SQRT_RATIO);
        assert_eq!(get_sqrt_ratio_at_tick(0).unwrap(), Q96);
        assert!(matches!(get_sqrt_ratio_at_tick(MAX_TICK + 1), Err(MathErr::Revert(m)) if m == "T"));
        assert!(matches!(get_sqrt_ratio_at_tick(MIN_TICK - 1), Err(MathErr::Revert(m)) if m == "T"));
        assert_eq!(get_tick_at_sqrt_ratio(MIN_SQRT_RATIO).unwrap(), MIN_TICK);
        assert_eq!(get_tick_at_sqrt_ratio(MAX_SQRT_RATIO - ONE).unwrap(), MAX_TICK - 1);
        assert!(matches!(get_tick_at_sqrt_ratio(MAX_SQRT_RATIO), Err(MathErr::Revert(m)) if m == "R"));
        for t in [-887272i64, -500_000, -1, 0, 1, 7, 60, 123_456, 887_271] {
            let s = get_sqrt_ratio_at_tick(t).unwrap();
            assert_eq!(get_tick_at_sqrt_ratio(s).unwrap(), t);
            if t > MIN_TICK {
                assert_eq!(get_tick_at_sqrt_ratio(s - ONE).unwrap(), t - 1);
            }
        }
    }

    #[test]
    fn bitmap_walk_matches_the_definition() {
        // ticks at spacing 10: -200, 70, 2600 (word 1)
        let net = vec![(-200i64, 5i128), (70, -3), (2600, 1)];
        let mut words: HashMap<i64, U256> = HashMap::new();
        for (t, _) in &net {
            let c = t.div_euclid(10);
            let e = words.entry(c >> 8).or_insert(ZERO);
            *e |= ONE << ((c & 0xFF) as usize);
        }
        let tk = Ticks::new(10, words.into_iter().collect(), net, None, "None".into()).unwrap();
        assert_eq!(tk.next_initialized_tick_within_one_word(0, true).unwrap(), (0, false)); // -200 is in word -1
        assert_eq!(tk.next_initialized_tick_within_one_word(-1, true).unwrap(), (-200, true));
        assert_eq!(tk.next_initialized_tick_within_one_word(0, false).unwrap(), (70, true));
        assert_eq!(tk.next_initialized_tick_within_one_word(70, false).unwrap(), (2550, false));
        assert_eq!(tk.next_initialized_tick_within_one_word(2550, false).unwrap(), (2600, true));
        assert_eq!(tk.next_initialized_tick_within_one_word(-201, true).unwrap(), (-2560, false));
    }

    #[test]
    fn insufficient_state_message() {
        let tk = Ticks::new(1, vec![], vec![], Some((-4, 3)), "(-4, 3)".into()).unwrap();
        let e = tk.next_initialized_tick_within_one_word(5000, false).unwrap_err();
        assert_eq!(e, MathErr::WordOutOfRange(19));
        assert_eq!(e.message(&tk.word_range_repr), "tick bitmap word 19 outside loaded range (-4, 3)");
    }

    #[test]
    fn add_delta_semantics() {
        assert_eq!(add_delta(10, 5, false).unwrap(), 15);
        assert_eq!(add_delta(10, 5, true).unwrap(), 5);
        assert_eq!(add_delta(10, -5, true).unwrap(), 15);
        assert!(matches!(add_delta(4, 5, true), Err(MathErr::Revert(m)) if m == "LS"));
        assert!(matches!(add_delta(u128::MAX, 1, false), Err(MathErr::Revert(m)) if m == "LA"));
        assert_eq!(add_delta(1, i128::MIN, true).unwrap(), 1 + (1u128 << 127usize));
    }
}
