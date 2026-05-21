//! The Model trait — anything GraphNet can introspect, intervene on, and run.
//!
//! Phase 1 ships with [`crate::stack::Stack`] as the only implementor.
//! Phase 5 adds Transformer + Mamba + arbitrary `nn.Module` adapters.
//!
//! BUG ASSUMPTION: the trait is intentionally minimal in Phase 1 — just
//! `forward()` + introspection (`dim`, `arch_summary`). Intervention and
//! snapshot APIs land in Phase 7 + Phase 9 respectively.

use plausiden_hdc::Hypervector;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::stack::{Stack, StackError};

/// Errors specific to Model trait implementations.
#[derive(Debug, Error)]
pub enum ModelError {
    /// The model's underlying Stack execution failed.
    #[error("stack: {0}")]
    Stack(#[from] StackError),

    /// The adapter has no implementation for this method yet
    /// (Phase 5+ work in progress).
    #[error("adapter `{0}`: not implemented yet")]
    NotImplemented(&'static str),

    /// A bring-your-own-model adapter's forward closure failed.
    #[error("external model `{family}`: {msg}")]
    External {
        /// User-supplied adapter family name.
        family: String,
        /// Free-form error message from the user-supplied forward closure.
        msg: String,
    },
}

/// A descriptive summary of a model's architecture, for visualisation +
/// audit + REST API responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchSummary {
    /// Adapter family ("stack", "transformer", "mamba", ...).
    pub family: String,
    /// Input dimensionality (for HDC models, hypervector D).
    pub input_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Number of internal substructures (operations for a Stack,
    /// layers for a transformer, ...).
    pub substructures: usize,
    /// Free-form annotations (human-readable).
    pub notes: Vec<String>,
}

/// The Model trait — uniform interface GraphNet uses to drive any AI family.
///
/// Phase 1: forward + arch summary + dim introspection.
/// Phase 7: intervention API (`apply_intervention`, `undo`).
/// Phase 9: snapshot/restore (`snapshot` / `restore_from`).
pub trait Model: Send + Sync {
    /// Run a forward pass.
    ///
    /// BUG ASSUMPTION: input dimensionality must match `input_dim`.
    fn forward(&self, input: &Hypervector) -> Result<Hypervector, ModelError>;

    /// Architectural summary for the viz layer.
    fn arch_summary(&self) -> ArchSummary;

    /// Input hypervector dimensionality.
    fn input_dim(&self) -> usize;

    /// Output hypervector dimensionality.
    fn output_dim(&self) -> usize;
}

/// Bring-your-own-model adapter.
///
/// Wrap **any** forward function (LFI, a custom RNN, a wired-up transformer,
/// an HTTP-backed inference server, anything that maps `Hypervector → Hypervector`)
/// as a `Model` for GraphNet to consume.
///
/// # Example
///
/// ```
/// use graphnet_engine::{ExternalModel, Model};
/// use plausiden_hdc::Hypervector;
///
/// let my_ai = ExternalModel::new(
///     "my-custom-lfi",
///     10_000,  // input_dim
///     10_000,  // output_dim
///     |input: &Hypervector| {
///         // call into your AI here; return Hypervector
///         Ok(input.clone())  // (stub: identity)
///     },
/// );
///
/// let v = Hypervector::random_seeded(10_000, 42);
/// let out = my_ai.forward(&v).expect("ok");
/// assert_eq!(out.dim(), 10_000);
/// ```
///
/// The closure must be `Send + Sync + 'static` so the model can be shared
/// across threads (continuous-execution mode in Phase 6).
/// Type alias for the closure GraphNet invokes on every forward pass.
pub type ForwardFn = Box<dyn Fn(&Hypervector) -> Result<Hypervector, String> + Send + Sync>;

/// Bring-your-own-model adapter — concrete struct backing [`ExternalModel::new`].
pub struct ExternalModel {
    family: String,
    input_dim: usize,
    output_dim: usize,
    notes: Vec<String>,
    forward_fn: ForwardFn,
}

impl ExternalModel {
    /// Construct a new external adapter.
    ///
    /// - `family` is shown in arch summary + viz; pick a descriptive name.
    /// - `input_dim` / `output_dim` are the hypervector dimensionalities your
    ///   forward function expects + produces.
    /// - `forward_fn` is the closure invoked on every `forward()` call.
    pub fn new<F>(
        family: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        forward_fn: F,
    ) -> Self
    where
        F: Fn(&Hypervector) -> Result<Hypervector, String> + Send + Sync + 'static,
    {
        Self {
            family: family.into(),
            input_dim,
            output_dim,
            notes: Vec::new(),
            forward_fn: Box::new(forward_fn),
        }
    }

    /// Builder: attach a free-form note for the arch summary panel.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl std::fmt::Debug for ExternalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalModel")
            .field("family", &self.family)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("notes", &self.notes)
            .field("forward_fn", &"<closure>")
            .finish()
    }
}

impl Model for ExternalModel {
    fn forward(&self, input: &Hypervector) -> Result<Hypervector, ModelError> {
        (self.forward_fn)(input).map_err(|msg| ModelError::External {
            family: self.family.clone(),
            msg,
        })
    }

    fn arch_summary(&self) -> ArchSummary {
        ArchSummary {
            family: self.family.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            substructures: 0,
            notes: self.notes.clone(),
        }
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl Model for Stack {
    fn forward(&self, input: &Hypervector) -> Result<Hypervector, ModelError> {
        Ok(Stack::forward(self, input)?)
    }

    fn arch_summary(&self) -> ArchSummary {
        let op_tags: Vec<String> = self
            .operations()
            .iter()
            .map(|o| o.tag().to_string())
            .collect();
        ArchSummary {
            family: "stack".to_string(),
            input_dim: self.dim(),
            output_dim: self.dim(),
            substructures: self.len(),
            notes: vec![format!("ops: [{}]", op_tags.join(", "))],
        }
    }

    fn input_dim(&self) -> usize {
        Stack::dim(self)
    }

    fn output_dim(&self) -> usize {
        Stack::dim(self)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::op::Operation;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn stack_implements_model_forward() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let v = hv(1);
        let out = Model::forward(&s, &v).expect("ok");
        assert_eq!(v, out);
    }

    #[test]
    fn stack_arch_summary_describes_ops() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(2) });
        let summary = s.arch_summary();
        assert_eq!(summary.family, "stack");
        assert_eq!(summary.input_dim, 1_000);
        assert_eq!(summary.output_dim, 1_000);
        assert_eq!(summary.substructures, 2);
        assert!(summary.notes[0].contains("identity"));
        assert!(summary.notes[0].contains("dense"));
    }

    #[test]
    fn stack_input_output_dims_match_construction() {
        let s = Stack::new(2_048);
        assert_eq!(s.input_dim(), 2_048);
        assert_eq!(s.output_dim(), 2_048);
    }

    #[test]
    fn model_trait_is_send_sync() {
        fn requires_send_sync<T: Send + Sync>() {}
        requires_send_sync::<Stack>();
        requires_send_sync::<ExternalModel>();
    }

    #[test]
    fn external_model_wraps_identity_closure() {
        let m = ExternalModel::new("test-id", 1_000, 1_000, |v: &Hypervector| Ok(v.clone()));
        let v = hv(1);
        let out = m.forward(&v).expect("ok");
        assert_eq!(v, out);
    }

    #[test]
    fn external_model_arch_summary_reflects_construction() {
        let m = ExternalModel::new("my-custom-lfi", 1_024, 512, |v: &Hypervector| Ok(v.clone()))
            .with_note("trained 2026-05-17");
        let summary = m.arch_summary();
        assert_eq!(summary.family, "my-custom-lfi");
        assert_eq!(summary.input_dim, 1_024);
        assert_eq!(summary.output_dim, 512);
        assert_eq!(summary.substructures, 0);
        assert_eq!(summary.notes, vec!["trained 2026-05-17".to_string()]);
    }

    #[test]
    fn external_model_propagates_closure_error() {
        let m = ExternalModel::new("flaky", 1_000, 1_000, |_v: &Hypervector| {
            Err("OOM on inference".to_string())
        });
        let v = hv(1);
        let err = m.forward(&v).expect_err("should error");
        if let ModelError::External { family, msg } = err {
            assert_eq!(family, "flaky");
            assert!(msg.contains("OOM"));
        } else {
            unreachable!("wrong variant");
        }
    }

    #[test]
    fn external_model_dyn_dispatch_works() {
        let stack = Stack::new(1_000).with_operation(Operation::Identity);
        let external = ExternalModel::new("ext", 1_000, 1_000, |v: &Hypervector| Ok(v.clone()));
        let models: Vec<Box<dyn Model>> = vec![Box::new(stack), Box::new(external)];
        let v = hv(1);
        for m in &models {
            let out = m.forward(&v).expect("ok");
            assert_eq!(out.dim(), 1_000);
        }
    }
}
