//! Hardware backend abstraction (plan §18.1).
//!
//! GraphNet runs HDC primitives on different compute substrates. The
//! [`Backend`] trait lets the engine choose a backend per Stack (or per
//! tensor in a future refinement) without leaking that choice into the
//! consumer's view.
//!
//! v1 ships [`CpuBackend`] only; CUDA / Metal / Wgpu / Tpu / quantum land
//! as additional implementors of the same trait. The engine doesn't
//! depend on any of them transitively — they're opt-in feature flags
//! when those backends ship.
//!
//! BUG ASSUMPTION: the trait is intentionally minimal. Anything richer
//! (broadcast, batching, mixed precision) lands when a non-CPU backend
//! makes it worth the cost.

use plausiden_hdc::{bind, bundle, cos_sim, hamming, unbind, HdcError, Hypervector};

/// A compute backend GraphNet can run HDC operations on.
///
/// Every backend exposes the same surface as `plausiden-hdc`'s free
/// functions but routed through the implementation's native execution
/// engine. CPU uses `plausiden-hdc` directly; GPU backends will use
/// CUDA / Metal / wgpu kernels.
pub trait Backend: Send + Sync {
    /// Short tag for telemetry / arch summaries.
    fn tag(&self) -> &'static str;

    /// True if this backend can be constructed in the current environment.
    /// (CPU = always; CUDA = nvidia-smi visible; etc.)
    fn available(&self) -> bool;

    /// HDC bind on this backend.
    fn bind(&self, a: &Hypervector, b: &Hypervector) -> Result<Hypervector, HdcError>;

    /// HDC unbind on this backend.
    fn unbind(&self, c: &Hypervector, k: &Hypervector) -> Result<Hypervector, HdcError>;

    /// HDC bundle on this backend.
    fn bundle(&self, vectors: &[&Hypervector]) -> Result<Hypervector, HdcError>;

    /// Cosine similarity on this backend.
    fn cos_sim(&self, a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError>;

    /// Hamming distance on this backend.
    fn hamming(&self, a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError>;
}

/// CPU backend — always available; routes directly to `plausiden-hdc`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn tag(&self) -> &'static str {
        "cpu"
    }

    fn available(&self) -> bool {
        true
    }

    fn bind(&self, a: &Hypervector, b: &Hypervector) -> Result<Hypervector, HdcError> {
        bind(a, b)
    }

    fn unbind(&self, c: &Hypervector, k: &Hypervector) -> Result<Hypervector, HdcError> {
        unbind(c, k)
    }

    fn bundle(&self, vectors: &[&Hypervector]) -> Result<Hypervector, HdcError> {
        bundle(vectors)
    }

    fn cos_sim(&self, a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError> {
        cos_sim(a, b)
    }

    fn hamming(&self, a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError> {
        hamming(a, b)
    }
}

/// Pick the best available backend (currently always [`CpuBackend`]).
///
/// Future implementations will probe for CUDA / Metal / wgpu via a
/// feature-flag cascade and prefer GPU when present.
#[must_use]
pub fn pick_backend() -> Box<dyn Backend> {
    Box::new(CpuBackend)
}

/// Native adapter trait — the Rust mirror of `graphnet.adapters.ModelAdapter`.
///
/// Plug-in adapters for new model families (Mamba native, transformer
/// kernels, etc.) implement this without touching the engine.
pub trait NativeAdapter: Send + Sync {
    /// Family tag for the arch summary.
    fn family(&self) -> &'static str;

    /// Forward pass over a hypervector input.
    fn forward(&self, input: &Hypervector) -> Result<Hypervector, HdcError>;
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn cpu_backend_tag_is_cpu() {
        assert_eq!(CpuBackend.tag(), "cpu");
    }

    #[test]
    fn cpu_backend_is_always_available() {
        assert!(CpuBackend.available());
    }

    #[test]
    fn cpu_backend_matches_plausiden_hdc_bind() {
        let a = hv(1);
        let b = hv(2);
        let direct = bind(&a, &b).expect("ok");
        let through_backend = CpuBackend.bind(&a, &b).expect("ok");
        assert_eq!(direct, through_backend);
    }

    #[test]
    fn cpu_backend_matches_plausiden_hdc_unbind() {
        let a = hv(1);
        let k = hv(2);
        let bound = bind(&a, &k).expect("ok");
        let recovered = CpuBackend.unbind(&bound, &k).expect("ok");
        assert_eq!(a, recovered);
    }

    #[test]
    fn cpu_backend_matches_plausiden_hdc_bundle() {
        let a = hv(1);
        let b = hv(2);
        let c = hv(3);
        let direct = bundle(&[&a, &b, &c]).expect("ok");
        let through_backend = CpuBackend.bundle(&[&a, &b, &c]).expect("ok");
        assert_eq!(direct, through_backend);
    }

    #[test]
    fn cpu_backend_cos_sim_self_is_one() {
        let v = hv(1);
        let s = CpuBackend.cos_sim(&v, &v).expect("ok");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cpu_backend_hamming_self_is_zero() {
        let v = hv(1);
        let h = CpuBackend.hamming(&v, &v).expect("ok");
        assert_eq!(h, 0.0);
    }

    #[test]
    fn pick_backend_returns_cpu_by_default() {
        let b = pick_backend();
        assert_eq!(b.tag(), "cpu");
        assert!(b.available());
    }

    /// Dynamic-dispatch sanity: backend can be stored as Box<dyn Backend>
    /// and used polymorphically.
    #[test]
    fn dyn_dispatch_compiles_and_runs() {
        let backends: Vec<Box<dyn Backend>> = vec![Box::new(CpuBackend), pick_backend()];
        for b in &backends {
            assert!(b.available());
            let v = hv(1);
            let s = b.cos_sim(&v, &v).expect("ok");
            assert!((s - 1.0).abs() < 1e-9);
        }
    }
}
