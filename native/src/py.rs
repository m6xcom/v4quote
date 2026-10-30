//! Python bindings (module ``v4quote_native``). Thin: conversion of Python ints, pool handles, the path evaluator and
//! the three error classes. Every math call runs under ``catch_unwind``: a panic (a bug) becomes ``Fallback`` and the
//! Python dispatcher computes that call in Python, it never takes the worker down.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyTuple};

use crate::num::*;
use crate::pool::{path_quote, PoolRef, Quote};
use crate::v2::V2;
use crate::v3;
use crate::vault::Vault;

create_exception!(v4quote_native, Revert, PyException, "The Python implementation raises MathRevert(message) here.");
create_exception!(v4quote_native, Insufficient, PyException, "The Python implementation raises InsufficientState(message) here.");
create_exception!(v4quote_native, Fallback, PyException, "Not reproducible natively: compute this call in Python.");

// ------------------------------------------------------------------------------------------------ ints
/// Python int -> U256; None when it is not an int in [0, 2^256).
fn to_u256(ob: &Bound<'_, PyAny>) -> Option<U256> {
    if let Ok(v) = ob.extract::<u128>() {
        return Some(u128_(v));
    }
    if !ob.is_instance_of::<PyInt>() {
        return None;
    }
    let b = ob.call_method1("to_bytes", (32, "little")).ok()?; // OverflowError: negative or >= 2^256
    let b = b.downcast::<PyBytes>().ok()?;
    U256::try_from_le_slice(b.as_bytes())
}

fn req_u256(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<U256> {
    to_u256(ob).ok_or_else(|| PyValueError::new_err(format!("{what}: not an int in [0, 2**256)")))
}

fn arg_u256(ob: &Bound<'_, PyAny>) -> PyResult<U256> {
    to_u256(ob).ok_or_else(|| Fallback::new_err("argument is not an int in [0, 2**256)"))
}

fn arg_idx(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    match ob.extract::<u64>() {
        Ok(v) if v < 64 => Ok(v as usize),
        _ => Err(Fallback::new_err("index is not a small non-negative int")),
    }
}

fn py_u256(py: Python<'_>, x: U256) -> PyResult<PyObject> {
    if let Some(v) = to_u128(x) {
        return Ok(v.into_pyobject(py)?.into_any().unbind());
    }
    let bytes = x.to_le_bytes::<32>();
    let o = py.get_type::<PyInt>().call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))?;
    Ok(o.unbind())
}

fn py_opt(py: Python<'_>, x: Option<U256>) -> PyResult<PyObject> {
    match x {
        Some(v) => py_u256(py, v),
        None => Ok(py.None()),
    }
}

/// ``word_range_repr``: the pool's (``PoolRef::word_range_repr``), "" where no InsufficientState can arise.
fn err_to_py(e: MathErr, word_range_repr: &str) -> PyErr {
    match e {
        MathErr::Revert(m) => Revert::new_err(m),
        MathErr::Fallback(w) => Fallback::new_err(w),
        _ => Insufficient::new_err(e.message(word_range_repr)),
    }
}

/// Runs native math; a panic becomes ``Fallback``.
fn guarded<T>(f: impl FnOnce() -> R<T>) -> R<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(MathErr::Fallback("native panic")),
    }
}

// ------------------------------------------------------------------------------------------------ v3 tick data
/// TickBitmapSnapshot, built once per snapshot and shared by every pool state that reuses it.
#[pyclass(frozen, module = "v4quote_native")]
pub struct V3Ticks {
    inner: Arc<v3::Ticks>,
}

#[pymethods]
impl V3Ticks {
    #[new]
    #[pyo3(signature = (tick_spacing, words, liquidity_net, word_range, word_range_repr))]
    fn new(
        tick_spacing: i64,
        words: Vec<(i64, Bound<'_, PyAny>)>,
        liquidity_net: Vec<(i64, i128)>,
        word_range: Option<(i64, i64)>,
        word_range_repr: String,
    ) -> PyResult<Self> {
        let mut w = Vec::with_capacity(words.len());
        for (k, v) in words {
            w.push((k, req_u256(&v, "bitmap word")?));
        }
        let t = v3::Ticks::new(tick_spacing, w, liquidity_net, word_range, word_range_repr).map_err(PyValueError::new_err)?;
        Ok(V3Ticks { inner: Arc::new(t) })
    }
}

// ------------------------------------------------------------------------------------------------ pool handle
/// One pool state prepared for quotes (build once per state, reuse for every quote on it).
#[pyclass(frozen, module = "v4quote_native")]
pub struct Pool {
    inner: PoolRef,
}

fn quote_tuple(py: Python<'_>, q: &Quote) -> PyResult<(PyObject, PyObject, u64)> {
    Ok((py_u256(py, q.out)?, py_opt(py, q.fee)?, q.crossings))
}

#[pymethods]
impl Pool {
    /// Uniswap v2 pair: reserves, fee numerator / denominator.
    #[staticmethod]
    fn v2(reserve0: &Bound<'_, PyAny>, reserve1: &Bound<'_, PyAny>, fee_numerator: u64, fee_denominator: u64) -> PyResult<Self> {
        let p = V2::new(req_u256(reserve0, "reserve0")?, req_u256(reserve1, "reserve1")?, fee_numerator, fee_denominator)
            .map_err(PyValueError::new_err)?;
        Ok(Pool { inner: PoolRef::V2(Arc::new(p)) })
    }

    /// Uniswap v3 pool: tick data, sqrtPriceX96, tick, liquidity, fee (pips).
    #[staticmethod]
    fn v3(ticks: &V3Ticks, sqrt_price_x96: &Bound<'_, PyAny>, tick: i64, liquidity: u128, fee: u64) -> PyResult<Self> {
        let p = v3::Pool::new(ticks.inner.clone(), req_u256(sqrt_price_x96, "sqrt_price_x96")?, tick, liquidity, fee)
            .map_err(PyValueError::new_err)?;
        Ok(Pool { inner: PoolRef::V3(Arc::new(p)) })
    }

    /// OZ v5 ERC-4626 vault: totalAssets, totalSupply, decimals offset (coins: 0 = asset, 1 = shares).
    #[staticmethod]
    fn erc4626(total_assets: &Bound<'_, PyAny>, total_supply: &Bound<'_, PyAny>, decimals_offset: u32) -> PyResult<Self> {
        let p = Vault::new(req_u256(total_assets, "total_assets")?, req_u256(total_supply, "total_supply")?, decimals_offset)
            .map_err(PyValueError::new_err)?;
        Ok(Pool { inner: PoolRef::Vault(Arc::new(p)) })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// ``_Search._simulate`` semantics: (amount_out, fee | None, crossings); raises Revert / Insufficient / Fallback.
    fn hop(&self, py: Python<'_>, i: &Bound<'_, PyAny>, j: &Bound<'_, PyAny>, amount: &Bound<'_, PyAny>)
        -> PyResult<(PyObject, PyObject, u64)> {
        let (i, j, a) = (arg_idx(i)?, arg_idx(j)?, arg_u256(amount)?);
        match guarded(|| self.inner.hop(i, j, a)) {
            Ok(q) => quote_tuple(py, &q),
            Err(e) => Err(err_to_py(e, self.inner.word_range_repr())),
        }
    }

    /// The pool's own exact-input quote (no routing checks): (amount_out, fee | None, crossings, fully_filled).
    /// ``views``: kept for API compatibility (no supported kind distinguishes it).
    #[pyo3(signature = (i, j, amount, views=false))]
    fn quote(&self, py: Python<'_>, i: &Bound<'_, PyAny>, j: &Bound<'_, PyAny>, amount: &Bound<'_, PyAny>, views: bool)
        -> PyResult<(PyObject, PyObject, u64, bool)> {
        let (i, j, a) = (arg_idx(i)?, arg_idx(j)?, arg_u256(amount)?);
        match guarded(|| self.inner.quote(i, j, a, views)) {
            Ok(q) => Ok((py_u256(py, q.out)?, py_opt(py, q.fee)?, q.crossings, q.filled)),
            Err(e) => Err(err_to_py(e, self.inner.word_range_repr())),
        }
    }

    /// Benchmark helper: ``n`` native ``hop`` evaluations without the Python call overhead; mean ns per hop.
    fn bench_hop(&self, i: usize, j: usize, amount: &Bound<'_, PyAny>, n: u32) -> PyResult<f64> {
        let a = arg_u256(amount)?;
        let t = std::time::Instant::now();
        let mut ok = 0u32;
        for _ in 0..n {
            if std::hint::black_box(self.inner.hop(i, j, std::hint::black_box(a))).is_ok() {
                ok += 1;
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / f64::from(n.max(1));
        std::hint::black_box(ok);
        Ok(ns)
    }

    /// v3 only: the full ``quote_exact_input`` result (amount_in, amount_out, fee_amount, sqrt_price_x96,
    /// tick | None, liquidity, initialized_ticks_crossed, steps, amount_specified_remaining).
    #[pyo3(signature = (zero_for_one, amount_in, sqrt_price_limit_x96=None, want_tick=true))]
    fn v3_quote_exact_input<'py>(
        &self,
        py: Python<'py>,
        zero_for_one: bool,
        amount_in: &Bound<'py, PyAny>,
        sqrt_price_limit_x96: Option<&Bound<'py, PyAny>>,
        want_tick: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let p = match &self.inner {
            PoolRef::V3(p) => p.clone(),
            _ => return Err(PyValueError::new_err("not a v3 pool")),
        };
        let a = arg_u256(amount_in)?;
        let lim = match sqrt_price_limit_x96 {
            Some(o) => Some(arg_u256(o)?),
            None => None,
        };
        let r = guarded(|| p.quote_exact_input(zero_for_one, a, lim, want_tick))
            .map_err(|e| err_to_py(e, &p.ticks.word_range_repr))?;
        let items: Vec<PyObject> = vec![
            py_u256(py, r.amount_in)?,
            py_u256(py, r.amount_out)?,
            py_u256(py, r.fee_amount)?,
            py_u256(py, r.sqrt_price_x96)?,
            match r.tick {
                Some(t) => t.into_pyobject(py)?.into_any().unbind(),
                None => py.None(),
            },
            r.liquidity.into_pyobject(py)?.into_any().unbind(),
            r.crossed.into_pyobject(py)?.into_any().unbind(),
            r.steps.into_pyobject(py)?.into_any().unbind(),
            py_u256(py, r.remaining)?,
        ];
        PyTuple::new(py, items)
    }
}

// ------------------------------------------------------------------------------------------------ path
/// Per-hop results of one successful path evaluation; converted to Python only when asked for.
#[pyclass(frozen, module = "v4quote_native")]
pub struct PathResult {
    hops: Vec<Quote>,
}

#[pymethods]
impl PathResult {
    /// ((amount_out, fee | None, crossings), ...) per hop.
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut v: Vec<Bound<'py, PyTuple>> = Vec::with_capacity(self.hops.len());
        for q in &self.hops {
            let (a, b, c) = quote_tuple(py, q)?;
            v.push(PyTuple::new(py, [a, b, c.into_pyobject(py)?.into_any().unbind()])?);
        }
        PyTuple::new(py, v)
    }
}

/// A fixed sequence of hops (pool, i, j), each fed with the previous hop's output (``_Search.path``).
#[pyclass(frozen, module = "v4quote_native")]
pub struct Path {
    hops: Vec<(PoolRef, usize, usize)>,
}

pub const STOP_REVERT: u8 = 1;
pub const STOP_INSUFFICIENT: u8 = 2;
pub const STOP_FALLBACK: u8 = 3;

#[pymethods]
impl Path {
    #[new]
    fn new(hops: Vec<(PyRef<'_, Pool>, usize, usize)>) -> PyResult<Self> {
        if hops.is_empty() {
            return Err(PyValueError::new_err("empty path"));
        }
        Ok(Path { hops: hops.iter().map(|(p, i, j)| (p.inner.clone(), *i, *j)).collect() })
    }

    /// ``(amount_out, PathResult)`` or ``(None, (hop, stop, message, hop_amount_in))`` with stop 1 = revert,
    /// 2 = insufficient state, 3 = fallback (evaluate the path in Python).
    fn quote(&self, py: Python<'_>, amount: &Bound<'_, PyAny>) -> PyResult<(PyObject, PyObject)> {
        let a = match to_u256(amount) {
            Some(a) => a,
            None => {
                let t = (0usize, STOP_FALLBACK, "amount not an int in [0, 2**256)", amount.clone().unbind());
                return Ok((py.None(), t.into_pyobject(py)?.into_any().unbind()));
            }
        };
        let r = match catch_unwind(AssertUnwindSafe(|| path_quote(&self.hops, a))) {
            Ok(r) => r,
            Err(_) => {
                let t = (0usize, STOP_FALLBACK, "native panic", amount.clone().unbind());
                return Ok((py.None(), t.into_pyobject(py)?.into_any().unbind()));
            }
        };
        match r {
            Ok(hops) => {
                let out = py_u256(py, hops.last().map(|q| q.out).unwrap_or(a))?;
                let res = Py::new(py, PathResult { hops })?;
                Ok((out, res.into_any()))
            }
            Err(stop) => {
                let kind = match stop.err {
                    MathErr::Revert(_) => STOP_REVERT,
                    MathErr::Fallback(_) => STOP_FALLBACK,
                    _ => STOP_INSUFFICIENT,
                };
                let msg = self.hops[stop.hop].0.message(stop.err);
                let t = (stop.hop, kind, msg, py_u256(py, stop.amount)?);
                Ok((py.None(), t.into_pyobject(py)?.into_any().unbind()))
            }
        }
    }

    fn __len__(&self) -> usize {
        self.hops.len()
    }
}

// ------------------------------------------------------------------------------------------------ test helpers
fn res_u256(py: Python<'_>, r: R<U256>) -> PyResult<PyObject> {
    match r {
        Ok(v) => py_u256(py, v),
        Err(e) => Err(err_to_py(e, "")),
    }
}

/// TickMath.getSqrtRatioAtTick
#[pyfunction]
fn sqrt_ratio_at_tick(py: Python<'_>, tick: i64) -> PyResult<PyObject> {
    res_u256(py, guarded(|| v3::get_sqrt_ratio_at_tick(tick)))
}

/// TickMath.getTickAtSqrtRatio
#[pyfunction]
fn tick_at_sqrt_ratio(sqrt_price_x96: &Bound<'_, PyAny>) -> PyResult<i64> {
    let s = arg_u256(sqrt_price_x96)?;
    guarded(|| v3::get_tick_at_sqrt_ratio(s)).map_err(|e| err_to_py(e, ""))
}

#[pymodule]
fn v4quote_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // build fingerprint: v4quote.amm.native refuses a build whose sources differ from the repository's
    m.add("__source_hash__", crate::SOURCE_HASH)?;
    m.add("__build__", crate::BUILD)?;
    m.add("Revert", py.get_type::<Revert>())?;
    m.add("Insufficient", py.get_type::<Insufficient>())?;
    m.add("Fallback", py.get_type::<Fallback>())?;
    m.add("STOP_REVERT", STOP_REVERT)?;
    m.add("STOP_INSUFFICIENT", STOP_INSUFFICIENT)?;
    m.add("STOP_FALLBACK", STOP_FALLBACK)?;
    m.add_class::<V3Ticks>()?;
    m.add_class::<Pool>()?;
    m.add_class::<Path>()?;
    m.add_class::<PathResult>()?;
    m.add_function(wrap_pyfunction!(sqrt_ratio_at_tick, m)?)?;
    m.add_function(wrap_pyfunction!(tick_at_sqrt_ratio, m)?)?;
    Ok(())
}
