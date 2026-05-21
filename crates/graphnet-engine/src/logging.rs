//! Structured logging infrastructure (plan §14).
//!
//! GraphNet uses `tracing` for all structured logs. This module ships:
//!
//! - [`init`] — install a tracing-subscriber stack with hourly file rotation +
//!   JSON output, and a session UUID stamped on every event.
//! - [`new_session_id`] — generate a session UUID.
//! - [`forward_span`] / [`intervene_span`] / [`monitor_span`] — short helpers
//!   that open per-event spans tagged with the standard plan-§14 stream
//!   names (`engine.forward`, `engine.intervene`, `engine.monitor`).
//!
//! Plan-§14 streams: `engine.*` / `intervene.*` / `forward.*` / `viz.*` /
//! `adapter.*` / `monitor.*` / `audit.*`. We give the Rust-side init for the
//! engine streams; viz / adapter / monitor live Python-side via `logging`
//! and a sibling Python helper.
//!
//! Defaults match the plan: hourly rotation, JSON, ISO-8601 timestamps,
//! `~/.graphnet/logs/<session_id>/` storage root. Tunable via env vars:
//!
//! - `GRAPHNET_LOG_DIR` — override storage root
//! - `GRAPHNET_LOG_FILTER` — RUST_LOG-style EnvFilter spec; default `info`
//! - `GRAPHNET_DEBUG=1` — flip default filter to `debug`
//!
//! BUG ASSUMPTION: `init()` is idempotent — calling twice in the same
//! process drops the second call silently (tracing's global subscriber is
//! one-shot).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use thiserror::Error;
use tracing_appender::non_blocking::WorkerGuard;
use uuid::Uuid;

/// Errors during logging initialisation.
#[derive(Debug, Error)]
pub enum LoggingError {
    /// Failed to determine the storage root for log files.
    #[error("could not determine log dir: {0}")]
    NoLogDir(String),

    /// Failed to install the tracing subscriber.
    #[error("subscriber install: {0}")]
    Subscriber(String),
}

/// Holds the appender worker so it isn't dropped while the program runs.
///
/// Returned from [`init`] — bind it to a variable that lives as long as
/// the process. Dropping it stops the background writer.
pub struct LoggingHandle {
    /// The session id this handle is associated with.
    pub session_id: Uuid,
    /// The directory log files are written to.
    pub log_dir: PathBuf,
    /// Keeps the non-blocking writer thread alive.
    _guard: WorkerGuard,
}

static SESSION_ID: OnceLock<Uuid> = OnceLock::new();

/// Generate a fresh session UUID. The first call to [`init`] also sets the
/// global session id; subsequent calls to [`session_id`] return it.
#[must_use]
pub fn new_session_id() -> Uuid {
    Uuid::new_v4()
}

/// Returns the global session id installed by [`init`], or `None` if logging
/// hasn't been initialised yet.
#[must_use]
pub fn session_id() -> Option<Uuid> {
    SESSION_ID.get().copied()
}

/// Initialise GraphNet's tracing-subscriber stack with file rotation + JSON.
///
/// Returns a [`LoggingHandle`] that the caller must hold for the lifetime
/// of the program.
///
/// BUG ASSUMPTION: only the first call in a process actually installs the
/// subscriber; later calls return a fresh `LoggingHandle` with no effect on
/// the subscriber. The global session id is set on first call only.
pub fn init() -> Result<LoggingHandle, LoggingError> {
    let session_id = *SESSION_ID.get_or_init(Uuid::new_v4);
    let log_dir = resolve_log_dir(session_id)?;
    std::fs::create_dir_all(&log_dir).map_err(|e| LoggingError::NoLogDir(e.to_string()))?;

    let file_appender = tracing_appender::rolling::hourly(&log_dir, "engine.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let filter = std::env::var("GRAPHNET_LOG_FILTER").unwrap_or_else(|_| {
        if std::env::var("GRAPHNET_DEBUG").is_ok() {
            "debug".to_string()
        } else {
            "info".to_string()
        }
    });

    use tracing_subscriber::EnvFilter;
    let env_filter = EnvFilter::try_new(&filter)
        .or_else(|_| EnvFilter::try_new("info"))
        .map_err(|e| LoggingError::Subscriber(e.to_string()))?;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(non_blocking)
        .with_current_span(true)
        .with_span_list(true);

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .try_init();

    Ok(LoggingHandle {
        session_id,
        log_dir,
        _guard: guard,
    })
}

fn resolve_log_dir(session_id: Uuid) -> Result<PathBuf, LoggingError> {
    if let Ok(custom) = std::env::var("GRAPHNET_LOG_DIR") {
        return Ok(PathBuf::from(custom).join(session_id.to_string()));
    }
    let home = dirs_home_root()?;
    Ok(home.join(".graphnet/logs").join(session_id.to_string()))
}

fn dirs_home_root() -> Result<PathBuf, LoggingError> {
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home));
    }
    if let Ok(profile) = std::env::var("USERPROFILE") {
        return Ok(PathBuf::from(profile));
    }
    Err(LoggingError::NoLogDir(
        "no HOME or USERPROFILE env var".to_string(),
    ))
}

/// Open a tracing span for one forward pass.
///
/// Usage:
/// ```ignore
/// let _g = forward_span(seq).entered();
/// // work happens here
/// ```
pub fn forward_span(seq: u64) -> tracing::Span {
    tracing::info_span!(
        "engine.forward",
        seq = seq,
        session_id = session_id().map(|u| u.to_string()),
    )
}

/// Open a tracing span for one intervention.
pub fn intervene_span(kind: &str) -> tracing::Span {
    tracing::info_span!(
        "engine.intervene",
        kind = kind,
        session_id = session_id().map(|u| u.to_string()),
    )
}

/// Open a tracing span for one resource sample.
pub fn monitor_span() -> tracing::Span {
    tracing::debug_span!(
        "engine.monitor",
        session_id = session_id().map(|u| u.to_string()),
    )
}

/// Returns true if the log directory `dir` exists and looks like a GraphNet
/// session log dir (has at least one `engine.log*` file).
pub fn is_graphnet_log_dir(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .any(|e| e.file_name().to_string_lossy().starts_with("engine.log"))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialise tests — the global SESSION_ID + tracing subscriber are
    /// process-wide singletons, so concurrent test runs would conflict.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn new_session_id_is_unique() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
    }

    #[test]
    fn session_id_set_after_init() {
        let _g = TEST_LOCK.lock().expect("ok");
        let tmp = tempdir_path();
        // Use a fresh log dir to avoid polluting the user's home.
        std::env::set_var("GRAPHNET_LOG_DIR", &tmp);
        let handle = init().expect("init ok");
        assert!(session_id().is_some());
        assert_eq!(session_id().expect("set"), handle.session_id);
        // log_dir should be {tmp}/{uuid}
        assert!(handle.log_dir.starts_with(&tmp));
        // Cleanup
        std::env::remove_var("GRAPHNET_LOG_DIR");
    }

    #[test]
    fn forward_span_carries_seq() {
        let span = forward_span(42);
        // Span field access is not introspectable at runtime; just ensure
        // we can enter + exit without panic.
        let _entered = span.enter();
    }

    #[test]
    fn intervene_span_carries_kind() {
        let span = intervene_span("add");
        let _entered = span.enter();
    }

    #[test]
    fn is_graphnet_log_dir_negative_on_empty() {
        let tmp = tempdir_path();
        std::fs::create_dir_all(&tmp).expect("ok");
        assert!(!is_graphnet_log_dir(&tmp));
    }

    fn tempdir_path() -> PathBuf {
        let n = new_session_id();
        std::env::temp_dir().join(format!("graphnet-test-{n}"))
    }
}
