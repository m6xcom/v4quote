//! Micro-benchmark of the integer primitives (cargo run --release --example ops).
//! Printed, not asserted: machines (and their load) differ.
use std::hint::black_box;
use std::time::Instant;

use v4quote_native::num::*;
use ruint::uint;

fn bench(name: &str, n: u32, mut f: impl FnMut(u32) -> U256) {
    let t = Instant::now();
    let mut acc = ZERO;
    for k in 0..n {
        acc ^= f(k);
    }
    black_box(acc);
    println!("{name:<40} {:7.1} ns", t.elapsed().as_nanos() as f64 / f64::from(n));
}

fn main() {
    let n = 2_000_000;
    let a = uint!(1234567890123456789012345678_U256); // ~2^90
    let b = uint!(987654321098765432109876543210987654321098765432_U256); // ~2^160
    let d = uint!(1000000000000000000_U256);
    bench("u256 mul (90b x 90b, checked)", n, |k| mul(black_box(a + u(k as u64)), black_box(a)).unwrap());
    bench("u256 div 180b / 60b", n, |k| black_box((a + u(k as u64)) * a) / black_box(d));
    bench("u256 div 250b / 160b", n, |k| black_box((b + u(k as u64)) * a) / black_box(b));
    bench("mul_div 160x160/160 (512-bit path)", n, |k| mul_div(black_box(b + u(k as u64)), black_box(b), black_box(b)).unwrap());
    bench("mul_div 90x90/60 (256-bit path)", n, |k| mul_div(black_box(a + u(k as u64)), black_box(a), black_box(d)).unwrap());
}
