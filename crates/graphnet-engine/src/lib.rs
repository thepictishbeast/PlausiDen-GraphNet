//! GraphNet core engine.
//!
//! BUG ASSUMPTION: this crate is a v0 skeleton. The public surface is reserved
//! but unimplemented; calling any API will return `Error::NotImplemented`
//! until the corresponding Phase task lands (`GRAPHNET_BUILD_PLAN.md` §7).
//!
//! Architectural plan: `crates/graphnet-engine/src/` will grow these modules:
//!
//! - `model` — `Model` trait, snapshot/restore, subscribe channels
//! - `stack` — Stack + StackOfStacks implementations
//! - `intervene` — typed Intervention API (weight edits, op add/remove)
//! - `monitor` — RAM/CPU/GPU/FLOPs/wall-time/energy sampling
//! - `record` — session recording, replay, log-driven reconstruction
//! - `adapters` — Adapter trait + per-family adapter wrappers
//!
//! See `docs/PLAN.md` (project root) for the full design.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Top-level engine error.
#[derive(Debug, Error)]
pub enum Error {
    /// The requested operation is not yet implemented (Phase still pending).
    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),

    /// Underlying I/O failure (snapshot/restore, log read).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience result alias for engine APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Crate version, exposed for plugin / adapter version negotiation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns a banner describing the engine state.
///
/// BUG ASSUMPTION: returns a static string today; once Phase 1 lands this will
/// include compile-time feature flags + backend list.
#[must_use]
pub fn banner() -> &'static str {
    concat!(
        "graphnet-engine v",
        env!("CARGO_PKG_VERSION"),
        " (Phase 0 scaffold — APIs reserved, not implemented)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_mentions_version() {
        let b = banner();
        assert!(b.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn version_constant_matches_cargo() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn error_not_implemented_displays() {
        let e = Error::NotImplemented("model.forward");
        let msg = format!("{e}");
        assert!(msg.contains("not implemented"));
        assert!(msg.contains("model.forward"));
    }
}
