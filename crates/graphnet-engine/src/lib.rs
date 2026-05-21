//! GraphNet core engine.
//!
//! Phase 1 lands here: `Model` trait + `Stack` execution + intervention API
//! foundation, depending on `plausiden-hdc` for the HDC substrate primitives.
//!
//! BUG ASSUMPTION: the public surface in Phase 1 is intentionally narrow —
//! `Stack` supports three operations (`Dense`, `HrrBind`, `Identity`); more
//! land in subsequent phases (`GatedRoute` Phase 2, multi-arch adapters
//! Phase 5, intervention API Phase 7).
//!
//! Module layout:
//!
//! - `op` — `Operation` enum + per-op execution
//! - `stack` — `Stack` struct + `forward()` execution
//! - `model` — `Model` trait (`Stack` is one impl; transformer/Mamba in Phase 5)

#![forbid(unsafe_code)]

pub mod backend;
pub mod continuous;
pub mod history;
pub mod intervene;
pub mod logging;
pub mod model;
pub mod monitor;
pub mod op;
pub mod snapshot;
pub mod stack;
pub mod trace;
pub mod yaml_spec;

pub use backend::{pick_backend, Backend, CpuBackend, NativeAdapter};
pub use continuous::{ContinuousError, ContinuousRunner, ForwardEvent};
pub use history::{HistoryError, InterventionHistory, DEFAULT_HISTORY_CAPACITY};
pub use intervene::{apply_intervention, undo, Intervention, InterventionError, UndoToken};
pub use logging::{
    forward_span, init as init_logging, intervene_span, is_graphnet_log_dir, monitor_span,
    new_session_id, session_id, LoggingError, LoggingHandle,
};
pub use model::{ArchSummary, ExternalModel, Model, ModelError};
pub use monitor::{estimate_cost, flop_estimate, CostEstimate, ResourceMonitor, ResourceSample};
pub use op::{Operation, OperationError};
pub use snapshot::{
    restore, signed_snapshot, snapshot, verify_and_restore, SignedSnapshot, SnapshotError,
};
pub use stack::{Stack, StackError};
pub use trace::{ForwardTrace, OperationOutput};
pub use yaml_spec::{
    stack_from_yaml, stack_to_yaml, ArchitectureSpec, OperationSpec, SpecError, SPEC_VERSION,
};

use thiserror::Error;

/// Top-level engine error wrapping per-subsystem errors.
#[derive(Debug, Error)]
pub enum Error {
    /// HDC primitive error (dim mismatch, empty bundle).
    #[error("hdc: {0}")]
    Hdc(#[from] plausiden_hdc::HdcError),

    /// Stack execution error.
    #[error("stack: {0}")]
    Stack(#[from] StackError),

    /// Model adapter error.
    #[error("model: {0}")]
    Model(#[from] ModelError),

    /// Operation error.
    #[error("op: {0}")]
    Op(#[from] OperationError),

    /// Intervention API error.
    #[error("intervene: {0}")]
    Intervene(#[from] InterventionError),

    /// Snapshot / restore error.
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapshotError),

    /// Continuous-mode runner error.
    #[error("continuous: {0}")]
    Continuous(#[from] ContinuousError),

    /// History (undo/redo) error.
    #[error("history: {0}")]
    History(#[from] HistoryError),

    /// YAML spec export/import error.
    #[error("yaml: {0}")]
    Yaml(#[from] SpecError),

    /// Underlying I/O failure (snapshot/restore, log read).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience result alias for engine APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Crate version, exposed for plugin / adapter version negotiation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns a banner describing the engine state.
#[must_use]
pub fn banner() -> &'static str {
    concat!(
        "graphnet-engine v",
        env!("CARGO_PKG_VERSION"),
        " (Phase 1 — Model + Stack + 3 ops; intervention API + snapshot in Phase 7+ / 9)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_mentions_version() {
        assert!(banner().contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn version_constant_matches_cargo() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
