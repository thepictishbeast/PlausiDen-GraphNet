//! Stack operations — the primitive units a Stack composes.
//!
//! Each operation consumes a hypervector and produces a hypervector. Stacks
//! apply multiple operations in parallel and bundle the results (see
//! [`crate::stack::Stack::forward`]).
//!
//! Phase 1 ships three operations:
//!
//! - [`Operation::Identity`] — pass-through (skip-connection equivalent)
//! - [`Operation::Dense`] — HDC dense binding with a learned key vector
//! - [`Operation::HrrBind`] — HRR / FFT-based binding (circular convolution)
//!
//! Phase 2 adds `GatedRoute`; Phase 3 adds `Aggregate` for stack-of-stacks
//! aggregation across substructures.

use plausiden_hdc::{bind, permute as hdc_permute, Hypervector};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors during operation execution.
#[derive(Debug, Error)]
pub enum OperationError {
    /// HDC primitive error (dim mismatch).
    #[error("hdc: {0}")]
    Hdc(#[from] plausiden_hdc::HdcError),

    /// The key vector required by the operation is missing.
    #[error("operation `{0}` requires a key vector but none was set")]
    MissingKey(&'static str),
}

/// One operation mode within a Stack.
///
/// Each variant carries the parameters that make it executable (e.g. a key
/// hypervector for `Dense` and `HrrBind`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Operation {
    /// Pass-through. Returns the input unchanged.
    Identity,

    /// HDC dense binding: `bind(input, key)`.
    Dense {
        /// The key hypervector that input is bound against.
        key: Hypervector,
    },

    /// HRR / FFT binding: circular convolution of input and key in
    /// frequency-domain. Slower than [`Operation::Dense`] but preserves
    /// spectral structure (useful for periodic / harmonic signals).
    HrrBind {
        /// The key hypervector that input is convolved with.
        key: Hypervector,
    },

    /// Circular permutation (rotation): shifts the hypervector by `shift`
    /// positions. Useful for positional encoding in HDC.
    Permute {
        /// Number of positions to rotate by.
        shift: usize,
    },

    /// Elementwise negate: every bipolar component flips sign.
    /// Self-inverse.
    Negate,
}

impl Operation {
    /// Apply the operation to the given input hypervector.
    ///
    /// BUG ASSUMPTION: `Dense` and `HrrBind` require operand and key to have
    /// equal dimensionality; mismatches surface as `OperationError::Hdc`.
    pub fn apply(&self, input: &Hypervector) -> Result<Hypervector, OperationError> {
        match self {
            Operation::Identity => Ok(input.clone()),
            Operation::Dense { key } => Ok(bind(input, key)?),
            Operation::HrrBind { key } => hrr_bind(input, key),
            Operation::Permute { shift } => Ok(hdc_permute(input, *shift)),
            Operation::Negate => {
                let data: Vec<i8> = input.as_slice().iter().map(|x| -x).collect();
                Hypervector::from_bipolar(data).ok_or(OperationError::Hdc(
                    plausiden_hdc::HdcError::DimMismatch {
                        a: input.dim(),
                        b: input.dim(),
                    },
                ))
            }
        }
    }

    /// A short tag for tracing / logging.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Operation::Identity => "identity",
            Operation::Dense { .. } => "dense",
            Operation::HrrBind { .. } => "hrr_bind",
            Operation::Permute { .. } => "permute",
            Operation::Negate => "negate",
        }
    }
}

/// HRR binding via circular convolution.
///
/// Algorithm:
/// 1. Lift bipolar `i8` inputs to `f64` Complex pairs.
/// 2. FFT both inputs.
/// 3. Pointwise multiply in frequency domain.
/// 4. iFFT.
/// 5. Re-quantize back to bipolar via sign.
///
/// O(D log D) instead of O(D) for `Dense`, but spectral-preserving.
fn hrr_bind(a: &Hypervector, b: &Hypervector) -> Result<Hypervector, OperationError> {
    if a.dim() != b.dim() {
        return Err(OperationError::Hdc(plausiden_hdc::HdcError::DimMismatch {
            a: a.dim(),
            b: b.dim(),
        }));
    }
    let dim = a.dim();

    let mut a_c: Vec<Complex<f64>> = a
        .as_slice()
        .iter()
        .map(|&x| Complex::new(f64::from(x), 0.0))
        .collect();
    let mut b_c: Vec<Complex<f64>> = b
        .as_slice()
        .iter()
        .map(|&x| Complex::new(f64::from(x), 0.0))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(dim);
    let ifft = planner.plan_fft_inverse(dim);

    fft.process(&mut a_c);
    fft.process(&mut b_c);

    let mut prod: Vec<Complex<f64>> = a_c.iter().zip(&b_c).map(|(x, y)| x * y).collect();
    ifft.process(&mut prod);

    // Re-quantize real-part back to bipolar via sign. (iFFT is unnormalised in
    // rustfft so we don't care about magnitude — we only need the sign.)
    let data: Vec<i8> = prod
        .iter()
        .map(|c| if c.re >= 0.0 { 1i8 } else { -1 })
        .collect();
    Hypervector::from_bipolar(data).ok_or(OperationError::Hdc(
        plausiden_hdc::HdcError::DimMismatch { a: dim, b: dim },
    ))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use plausiden_hdc::cos_sim;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn identity_returns_input_unchanged() {
        let v = hv(1);
        let op = Operation::Identity;
        let out = op.apply(&v).expect("ok");
        assert_eq!(v, out);
    }

    #[test]
    fn dense_binds_to_key() {
        let v = hv(1);
        let k = hv(2);
        let op = Operation::Dense { key: k.clone() };
        let out = op.apply(&v).expect("ok");
        // Binding decorrelates: cos_sim of (v ⊗ k) and v ≈ 0 for random k.
        let s = cos_sim(&out, &v).expect("ok");
        assert!(s.abs() < 0.1, "dense bind self-sim = {s}, expected ~0");
    }

    #[test]
    fn dense_self_inverse_via_double_bind() {
        let v = hv(1);
        let k = hv(2);
        let op = Operation::Dense { key: k.clone() };
        let once = op.apply(&v).expect("ok");
        let twice = op.apply(&once).expect("ok");
        // For bipolar bind, double-bind = identity.
        assert_eq!(v, twice);
    }

    #[test]
    fn hrr_bind_runs_at_canonical_dim() {
        let v = Hypervector::random_seeded(1_024, 1);
        let k = Hypervector::random_seeded(1_024, 2);
        let op = Operation::HrrBind { key: k };
        let out = op.apply(&v).expect("ok");
        assert_eq!(out.dim(), 1_024);
    }

    #[test]
    fn dim_mismatch_in_dense_errors() {
        let v = Hypervector::random_seeded(100, 1);
        let k = Hypervector::random_seeded(200, 2);
        let op = Operation::Dense { key: k };
        let err = op.apply(&v).expect_err("mismatch should error");
        assert!(matches!(err, OperationError::Hdc(_)));
    }

    #[test]
    fn tag_is_stable() {
        assert_eq!(Operation::Identity.tag(), "identity");
        assert_eq!(Operation::Dense { key: hv(1) }.tag(), "dense");
        assert_eq!(Operation::HrrBind { key: hv(1) }.tag(), "hrr_bind");
        assert_eq!(Operation::Permute { shift: 3 }.tag(), "permute");
        assert_eq!(Operation::Negate.tag(), "negate");
    }

    #[test]
    fn permute_shifts_then_unshifts() {
        let v = hv(1);
        let op = Operation::Permute { shift: 17 };
        let out = op.apply(&v).expect("ok");
        // Permuted vector should not equal input (very small probability of collision).
        assert_ne!(out, v);
        // Permuting by dim is identity.
        let dim = v.dim();
        let id_op = Operation::Permute { shift: dim };
        let out_id = id_op.apply(&v).expect("ok");
        assert_eq!(out_id, v);
    }

    #[test]
    fn negate_is_self_inverse() {
        let v = hv(1);
        let op = Operation::Negate;
        let once = op.apply(&v).expect("ok");
        let twice = op.apply(&once).expect("ok");
        assert_eq!(twice, v);
        // Cosine similarity of v and -v is exactly -1.
        let sim = cos_sim(&v, &once).expect("ok");
        assert!((sim + 1.0).abs() < 1e-9, "negate cos_sim = {sim}, expected -1");
    }
}
