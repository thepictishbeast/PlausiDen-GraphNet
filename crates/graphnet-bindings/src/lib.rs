//! GraphNet Python bindings (PyO3 0.22 / abi3-py310).
//!
//! BUG ASSUMPTION: skeleton only. `version()` + `banner()` are exposed so the
//! Python side can verify the FFI bridge works end-to-end before Phase 1
//! implements anything.
//!
//! Full API surface lands in Phase 2 (see `docs/PLAN.md` §7).

#![forbid(unsafe_code)]

use pyo3::prelude::*;

/// Returns the engine version string.
#[pyfunction]
fn version() -> &'static str {
    graphnet_engine::VERSION
}

/// Returns the engine banner string.
#[pyfunction]
fn banner() -> &'static str {
    graphnet_engine::banner()
}

/// The `graphnet` Python module.
#[pymodule]
fn graphnet(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(banner, m)?)?;
    Ok(())
}
