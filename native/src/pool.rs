//! One pool of any supported kind and the per-hop semantics of ``v4quote.routing._Search._simulate``.

use std::sync::Arc;

use crate::num::*;
use crate::v2::V2;
use crate::v3;
use crate::vault::Vault;

#[derive(Debug, Clone)]
pub enum PoolRef {
    V2(Arc<V2>),
    V3(Arc<v3::Pool>),
    Vault(Arc<Vault>),
}

/// Raw exact-input quote: amount out, fee, initialised ticks crossed, fully filled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quote {
    pub out: U256,
    pub fee: Option<U256>,
    pub crossings: u64,
    pub filled: bool,
}

impl PoolRef {
    /// ``repr(word_range)`` of a v3 snapshot (embedded in the InsufficientState message), "" for other kinds.
    pub fn word_range_repr(&self) -> &str {
        match self {
            PoolRef::V3(p) => &p.ticks.word_range_repr,
            _ => "",
        }
    }

    /// The Python exception message of an error raised by this pool.
    pub fn message(&self, e: MathErr) -> String {
        e.message(self.word_range_repr())
    }

    pub fn kind(&self) -> &'static str {
        match self {
            PoolRef::V2(_) => "uniswap_v2",
            PoolRef::V3(_) => "uniswap_v3",
            PoolRef::Vault(_) => "erc4626",
        }
    }

    /// The pool's own exact-input quote, without routing checks:
    /// v2 ``get_amount_out``; v3 ``quote_exact_input(want_tick=False)``; ERC-4626 ``deposit_out`` (0 -> 1) / ``redeem_out`` (1 -> 0), fee 0. For v2 / v3, ``i == 0``
    /// means token0 -> token1.
    pub fn quote(&self, i: usize, j: usize, amount: U256, _views: bool) -> R<Quote> {
        match self {
            PoolRef::V2(p) => {
                if i > 1 || j > 1 {
                    return fallback("token index out of range");
                }
                let out = p.amount_out(amount, i == 0)?;
                Ok(Quote { out, fee: Some(p.fee(amount)?), crossings: 0, filled: true })
            }
            PoolRef::V3(p) => {
                if i > 1 || j > 1 {
                    return fallback("token index out of range");
                }
                let r = p.quote_exact_input(i == 0, amount, None, false)?;
                Ok(Quote { out: r.amount_out, fee: Some(r.fee_amount), crossings: r.crossed, filled: r.remaining.is_zero() })
            }
            PoolRef::Vault(p) => Ok(Quote { out: p.quote(i, j, amount)?, fee: Some(ZERO), crossings: 0, filled: true }),
        }
    }

    /// ``_Search._simulate`` for this pool: amount > 0, v3 partial fill and a zero output revert. The fee of a v2 hop is ``amount * (den - num) // den``.
    pub fn hop(&self, i: usize, j: usize, amount: U256) -> R<Quote> {
        if amount.is_zero() {
            return revert("zero input");
        }
        let q = self.quote(i, j, amount, false)?;
        if !q.filled {
            return revert("partial fill: price limit reached");
        }
        if q.out.is_zero() {
            return revert("zero output");
        }
        Ok(q)
    }
}

/// Where a path evaluation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathStop {
    pub hop: usize,
    pub amount: U256,
    pub err: MathErr,
}

/// Exact output of a path for ``amount`` (each hop fed with the previous hop's output), like ``_Search.path``.
pub fn path_quote(hops: &[(PoolRef, usize, usize)], amount: U256) -> Result<Vec<Quote>, PathStop> {
    let mut out = Vec::with_capacity(hops.len());
    let mut amt = amount;
    for (k, (p, i, j)) in hops.iter().enumerate() {
        match p.hop(*i, *j, amt) {
            Ok(q) => {
                amt = q.out;
                out.push(q);
            }
            Err(err) => return Err(PathStop { hop: k, amount: amt, err }),
        }
    }
    Ok(out)
}
