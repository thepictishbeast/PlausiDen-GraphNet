//! GraphNet Python bindings (PyO3 0.24 / abi3-py310).
//!
//! Exposes the graphnet-engine surface to Python as the `_graphnet_native`
//! native extension. The user-facing `graphnet` Python package re-exports
//! these as a friendlier facade.
//!
//! Phase 2 surface:
//!
//! - `PyHypervector`        — wraps `plausiden_hdc::Hypervector`
//! - `PyOperation`          — wraps `Operation` (Identity/Dense/HrrBind)
//! - `PyStack`              — wraps `Stack` (new + add_op + forward + trace)
//! - `PyForwardTrace`       — wraps `ForwardTrace` for live introspection
//! - module-level `bind`/`bundle`/`unbind`/`cos_sim`/`hamming` HDC functions
//! - module-level `snapshot`/`restore` round-trip
//!
//! Phase 4+ adds rich 3D/viz-targeted methods (PCA projections, etc.); for
//! now the bindings expose enough surface for Jupyter REPL + tutorial use.

#![forbid(unsafe_code)]
#![allow(clippy::needless_pass_by_value)] // PyO3 idioms

use graphnet_engine::{
    apply_intervention, snapshot as eng_snapshot, stack_from_yaml as eng_stack_from_yaml,
    stack_to_yaml as eng_stack_to_yaml, undo as eng_undo, Intervention, Operation, Stack,
};
use plausiden_hdc::{
    bind as hdc_bind, bundle as hdc_bundle, cos_sim as hdc_cos_sim, hamming as hdc_hamming,
    permute as hdc_permute, unbind as hdc_unbind, Hypervector,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

/// A bipolar hypervector (PyO3 wrapper around `plausiden_hdc::Hypervector`).
#[pyclass(name = "Hypervector", module = "graphnet._graphnet_native")]
#[derive(Clone)]
struct PyHypervector {
    inner: Hypervector,
}

#[pymethods]
impl PyHypervector {
    /// Construct a fresh random hypervector with the given dimensionality + seed.
    #[staticmethod]
    fn random(dim: usize, seed: u64) -> Self {
        Self {
            inner: Hypervector::random_seeded(dim, seed),
        }
    }

    /// Dimensionality.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return the underlying bipolar data as a Python list of i8s.
    fn as_list(&self) -> Vec<i8> {
        self.inner.as_slice().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("Hypervector(dim={})", self.inner.dim())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A Stack operation: Identity, Dense (HDC bind), or HrrBind (FFT-based binding).
#[pyclass(name = "Operation", module = "graphnet._graphnet_native")]
#[derive(Clone)]
struct PyOperation {
    inner: Operation,
}

#[pymethods]
impl PyOperation {
    /// Construct an Identity (pass-through) operation.
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: Operation::Identity,
        }
    }

    /// Construct a Dense binding operation with the given key hypervector.
    #[staticmethod]
    fn dense(key: PyHypervector) -> Self {
        Self {
            inner: Operation::Dense { key: key.inner },
        }
    }

    /// Construct an HRR / FFT binding operation with the given key hypervector.
    #[staticmethod]
    fn hrr_bind(key: PyHypervector) -> Self {
        Self {
            inner: Operation::HrrBind { key: key.inner },
        }
    }

    /// Short tag for tracing / logging.
    fn tag(&self) -> &'static str {
        self.inner.tag()
    }

    fn __repr__(&self) -> String {
        format!("Operation({})", self.inner.tag())
    }
}

/// One operation's intermediate output captured during forward_with_trace.
#[pyclass(name = "OperationOutput", module = "graphnet._graphnet_native")]
#[derive(Clone)]
struct PyOperationOutput {
    #[pyo3(get)]
    tag: String,
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    output: PyHypervector,
}

/// Per-operation activation capture from one forward pass.
#[pyclass(name = "ForwardTrace", module = "graphnet._graphnet_native")]
#[derive(Clone)]
struct PyForwardTrace {
    #[pyo3(get)]
    input: PyHypervector,
    #[pyo3(get)]
    per_op: Vec<PyOperationOutput>,
    #[pyo3(get)]
    bundled: PyHypervector,
}

/// A Stack: heterogeneous bundle of operations sharing a hypervector input.
#[pyclass(name = "Stack", module = "graphnet._graphnet_native")]
struct PyStack {
    inner: Stack,
}

#[pymethods]
impl PyStack {
    /// Construct a new empty Stack at the given dimensionality.
    #[new]
    fn new(dim: usize) -> Self {
        Self {
            inner: Stack::new(dim),
        }
    }

    /// Append an operation.
    fn add_operation(&mut self, op: PyOperation) {
        self.inner.add_operation(op.inner);
    }

    /// Stack dimensionality.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Number of operations in the Stack.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Run a forward pass; returns the bundled output hypervector.
    fn forward(&self, input: PyHypervector) -> PyResult<PyHypervector> {
        self.inner
            .forward(&input.inner)
            .map(|inner| PyHypervector { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Run a forward pass and capture per-operation outputs in a ForwardTrace.
    fn forward_with_trace(&self, input: PyHypervector) -> PyResult<PyForwardTrace> {
        let trace = self
            .inner
            .forward_with_trace(&input.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyForwardTrace {
            input: PyHypervector { inner: trace.input },
            per_op: trace
                .per_op
                .into_iter()
                .map(|o| PyOperationOutput {
                    tag: o.tag,
                    index: o.index,
                    output: PyHypervector { inner: o.output },
                })
                .collect(),
            bundled: PyHypervector {
                inner: trace.bundled,
            },
        })
    }

    /// Apply an intervention. Returns an opaque undo-handle (an integer id
    /// you can pass to [`undo_intervention`] to reverse it).
    ///
    /// `kind` is `"add"`, `"remove"`, or `"replace"`.
    #[pyo3(signature = (kind, op=None, index=None))]
    fn apply_intervention(
        &mut self,
        kind: &str,
        op: Option<PyOperation>,
        index: Option<usize>,
    ) -> PyResult<Vec<u8>> {
        let intervention = match kind {
            "add" => {
                let op = op.ok_or_else(|| PyValueError::new_err("add requires `op`"))?;
                Intervention::AddOperation {
                    op: op.inner,
                    at: index,
                }
            }
            "remove" => {
                let index =
                    index.ok_or_else(|| PyValueError::new_err("remove requires `index`"))?;
                Intervention::RemoveOperation { index }
            }
            "replace" => {
                let op = op.ok_or_else(|| PyValueError::new_err("replace requires `op`"))?;
                let index =
                    index.ok_or_else(|| PyValueError::new_err("replace requires `index`"))?;
                Intervention::ReplaceOperation {
                    index,
                    op: op.inner,
                }
            }
            other => return Err(PyValueError::new_err(format!("unknown kind: {other}"))),
        };
        let token = apply_intervention(&mut self.inner, intervention)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        bincode::serde::encode_to_vec(&token, bincode::config::standard())
            .map_err(|e| PyRuntimeError::new_err(format!("encode token: {e}")))
    }

    /// Undo an intervention given the opaque handle from [`apply_intervention`].
    fn undo_intervention(&mut self, token_bytes: Vec<u8>) -> PyResult<()> {
        let (token, _) = bincode::serde::decode_from_slice::<graphnet_engine::UndoToken, _>(
            &token_bytes,
            bincode::config::standard(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("decode token: {e}")))?;
        eng_undo(&mut self.inner, token).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Inspect: list of operation tags in the Stack.
    fn op_tags(&self) -> Vec<String> {
        self.inner
            .operations()
            .iter()
            .map(|o| o.tag().to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Stack(dim={}, len={})", self.inner.dim(), self.inner.len())
    }
}

// --- module-level HDC functions ---

/// HDC bind (elementwise multiplication; self-inverse for bipolar).
#[pyfunction]
fn bind(a: PyHypervector, b: PyHypervector) -> PyResult<PyHypervector> {
    hdc_bind(&a.inner, &b.inner)
        .map(|inner| PyHypervector { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// HDC unbind (= bind, kept for code-reading clarity).
#[pyfunction]
fn unbind(c: PyHypervector, k: PyHypervector) -> PyResult<PyHypervector> {
    hdc_unbind(&c.inner, &k.inner)
        .map(|inner| PyHypervector { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// HDC bundle (additive superposition with majority threshold).
#[pyfunction]
fn bundle(vectors: Vec<PyHypervector>) -> PyResult<PyHypervector> {
    let refs: Vec<&Hypervector> = vectors.iter().map(|v| &v.inner).collect();
    hdc_bundle(&refs)
        .map(|inner| PyHypervector { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Cosine similarity in [-1, 1].
#[pyfunction]
fn cos_sim(a: PyHypervector, b: PyHypervector) -> PyResult<f64> {
    hdc_cos_sim(&a.inner, &b.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Hamming distance in [0, 1].
#[pyfunction]
fn hamming(a: PyHypervector, b: PyHypervector) -> PyResult<f64> {
    hdc_hamming(&a.inner, &b.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Circular permutation (shift) of a hypervector by `shift` positions.
#[pyfunction]
fn permute(v: PyHypervector, shift: usize) -> PyHypervector {
    PyHypervector {
        inner: hdc_permute(&v.inner, shift),
    }
}

/// Bipolar element-wise negate (multiply every entry by -1).
#[pyfunction]
fn negate(v: PyHypervector) -> PyResult<PyHypervector> {
    let data: Vec<i8> = v.inner.as_slice().iter().map(|x| -x).collect();
    Hypervector::from_bipolar(data)
        .map(|inner| PyHypervector { inner })
        .ok_or_else(|| PyRuntimeError::new_err("negate produced non-bipolar"))
}

// --- YAML architecture spec ---

/// Serialise a Stack to a YAML architecture spec.
#[pyfunction]
fn stack_to_yaml(stack: &PyStack) -> PyResult<String> {
    eng_stack_to_yaml(&stack.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Parse a Stack from a YAML architecture spec.
#[pyfunction]
fn stack_from_yaml(yaml: &str) -> PyResult<PyStack> {
    eng_stack_from_yaml(yaml)
        .map(|inner| PyStack { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// --- snapshot / restore ---

/// Snapshot a Stack to bytes.
#[pyfunction]
fn snapshot<'py>(py: Python<'py>, stack: &PyStack) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = eng_snapshot(&stack.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Restore a Stack from bytes previously produced by `snapshot`.
#[pyfunction]
fn restore(bytes: Vec<u8>) -> PyResult<PyStack> {
    graphnet_engine::restore(&bytes)
        .map(|inner| PyStack { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// The `_graphnet_native` Python module.
#[pymodule]
fn _graphnet_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(banner, m)?)?;
    m.add_function(wrap_pyfunction!(bind, m)?)?;
    m.add_function(wrap_pyfunction!(unbind, m)?)?;
    m.add_function(wrap_pyfunction!(bundle, m)?)?;
    m.add_function(wrap_pyfunction!(cos_sim, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(permute, m)?)?;
    m.add_function(wrap_pyfunction!(negate, m)?)?;
    m.add_function(wrap_pyfunction!(stack_to_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(stack_from_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(restore, m)?)?;
    m.add_class::<PyHypervector>()?;
    m.add_class::<PyOperation>()?;
    m.add_class::<PyStack>()?;
    m.add_class::<PyForwardTrace>()?;
    m.add_class::<PyOperationOutput>()?;
    Ok(())
}
