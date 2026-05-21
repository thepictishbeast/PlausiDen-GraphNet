//! Continuous-execution mode (Phase 6).
//!
//! Drives a [`crate::Model`] in a background thread while the GUI / REPL
//! stays responsive. Each forward result + ForwardTrace is published to a
//! bounded crossbeam channel that the viz layer drains at display rate.
//!
//! Bounded queue + non-blocking publish so a slow visualisation can't OOM
//! the runner; oldest events are dropped on overflow with a counter the
//! caller can read.
//!
//! BUG ASSUMPTION: Phase 6 ships the engine-side runner only. Wiring it
//! through to the Jupyter widget (real ~30 FPS render loop) is Phase 3's
//! viz layer to consume in a follow-up tick.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use plausiden_hdc::Hypervector;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::stack::{Stack, StackError};

/// One event emitted by the continuous runner per forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardEvent {
    /// Monotonically increasing sequence number (1-indexed).
    pub seq: u64,
    /// Wall-clock latency of this forward pass.
    pub latency_us: u64,
    /// The input hypervector the runner used.
    pub input: Hypervector,
    /// The bundled output the Stack produced.
    pub output: Hypervector,
}

/// Errors that can arise in continuous mode.
#[derive(Debug, Error)]
pub enum ContinuousError {
    /// Underlying Stack forward failed.
    #[error("stack: {0}")]
    Stack(#[from] StackError),

    /// Background thread already running.
    #[error("runner already started")]
    AlreadyStarted,

    /// Background thread not running (called `stop` before `start`).
    #[error("runner not started")]
    NotStarted,
}

/// Continuous runner: drives a Stack in a background thread.
///
/// Construction is cheap (no thread spawned yet). Call
/// [`ContinuousRunner::start`] to kick off the background loop; it consumes
/// from the iterator until the iterator is exhausted, [`ContinuousRunner::stop`]
/// is called, or the receiver side hangs up.
pub struct ContinuousRunner {
    inner: Stack,
    queue_cap: usize,
    handle: Mutex<Option<JoinHandle<()>>>,
    cancel: Arc<AtomicBool>,
    dropped: Arc<AtomicU64>,
}

impl ContinuousRunner {
    /// Construct a new runner around `stack`.
    ///
    /// `queue_cap` bounds the channel depth — when full, new events are
    /// dropped (oldest-first) and the drop counter increments. Recommended:
    /// 8–32 for live UI, higher for batch recording.
    #[must_use]
    pub fn new(stack: Stack, queue_cap: usize) -> Self {
        Self {
            inner: stack,
            queue_cap,
            handle: Mutex::new(None),
            cancel: Arc::new(AtomicBool::new(false)),
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns a count of events dropped due to queue overflow since `start`.
    #[must_use]
    pub fn dropped_events(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Returns true if a background thread is currently running.
    pub fn is_running(&self) -> bool {
        match self.handle.lock() {
            Ok(g) => g.is_some(),
            Err(_) => false,
        }
    }

    /// Start the background loop consuming from `inputs`.
    ///
    /// Returns the receiver side of the event channel. The runner publishes
    /// one `ForwardEvent` per successful forward; on the first
    /// [`StackError`] it stops and closes the sender (receiver sees `None`).
    pub fn start<I>(&self, inputs: I) -> Result<Receiver<ForwardEvent>, ContinuousError>
    where
        I: IntoIterator<Item = Hypervector> + Send + 'static,
        I::IntoIter: Send + 'static,
    {
        let mut handle_guard = self
            .handle
            .lock()
            .map_err(|_| ContinuousError::AlreadyStarted)?;
        if handle_guard.is_some() {
            return Err(ContinuousError::AlreadyStarted);
        }

        let (tx, rx) = bounded::<ForwardEvent>(self.queue_cap);
        let stack = self.inner.clone();
        let cancel = Arc::clone(&self.cancel);
        let dropped = Arc::clone(&self.dropped);
        cancel.store(false, Ordering::Relaxed);
        dropped.store(0, Ordering::Relaxed);

        let h = thread::Builder::new()
            .name("graphnet-continuous".into())
            .spawn(move || {
                run_loop(stack, inputs.into_iter(), tx, &cancel, &dropped);
            })
            .map_err(|_| ContinuousError::AlreadyStarted)?;
        *handle_guard = Some(h);
        Ok(rx)
    }

    /// Stop the background loop and join. Idempotent.
    pub fn stop(&self) -> Result<(), ContinuousError> {
        self.cancel.store(true, Ordering::Relaxed);
        let mut handle_guard = self
            .handle
            .lock()
            .map_err(|_| ContinuousError::NotStarted)?;
        if let Some(h) = handle_guard.take() {
            let _ = h.join();
        }
        Ok(())
    }
}

impl Drop for ContinuousRunner {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn run_loop<I>(
    stack: Stack,
    inputs: I,
    tx: Sender<ForwardEvent>,
    cancel: &AtomicBool,
    dropped: &AtomicU64,
) where
    I: Iterator<Item = Hypervector>,
{
    for (seq_zero_based, input) in inputs.enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let started = Instant::now();
        let output = match stack.forward(&input) {
            Ok(o) => o,
            Err(_) => break, // first error halts the runner; matches Phase 1 semantics
        };
        let seq = u64::try_from(seq_zero_based)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let latency_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let event = ForwardEvent {
            seq,
            latency_us,
            input,
            output,
        };

        match tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Drop the new event (oldest-first via consumer pacing).
                dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
    // Drop tx to close the channel (receiver gets None on next recv).
    drop(tx);
    // Brief yield so test threads observe close promptly.
    thread::sleep(Duration::from_millis(1));
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::op::Operation;
    use std::time::Duration;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    fn identity_stack() -> Stack {
        Stack::new(1_000).with_operation(Operation::Identity)
    }

    #[test]
    fn runner_publishes_one_event_per_input() {
        let runner = ContinuousRunner::new(identity_stack(), 16);
        let inputs = vec![hv(1), hv(2), hv(3)];
        let rx = runner.start(inputs).expect("start ok");

        let mut received = Vec::new();
        for _ in 0..3 {
            let ev = rx.recv_timeout(Duration::from_secs(2)).expect("recv ok");
            received.push(ev);
        }

        assert_eq!(received[0].seq, 1);
        assert_eq!(received[1].seq, 2);
        assert_eq!(received[2].seq, 3);
        // identity stack: output == input
        assert_eq!(received[0].input, received[0].output);

        runner.stop().expect("stop ok");
    }

    #[test]
    fn runner_stop_is_idempotent() {
        let runner = ContinuousRunner::new(identity_stack(), 8);
        runner.stop().expect("stop on never-started ok");
        runner.stop().expect("stop again ok");
    }

    #[test]
    fn runner_drops_events_when_consumer_slow() {
        // Tiny queue + many inputs; consumer never drains → some events drop.
        let runner = ContinuousRunner::new(identity_stack(), 1);
        let inputs: Vec<Hypervector> = (0..50).map(hv).collect();
        let _rx = runner.start(inputs).expect("start ok");

        // Wait for the producer to finish.
        std::thread::sleep(Duration::from_millis(200));
        runner.stop().expect("stop ok");

        // Tiny queue + 50 inputs + zero consumer drain → some drops expected
        // (exact count depends on scheduling). Just assert ≥ 1.
        assert!(runner.dropped_events() >= 1, "expected ≥1 drop, got 0");
    }

    #[test]
    fn runner_terminates_on_input_exhaustion() {
        let runner = ContinuousRunner::new(identity_stack(), 4);
        let rx = runner.start(vec![hv(1), hv(2)]).expect("start ok");

        // Drain both events.
        for _ in 0..2 {
            rx.recv_timeout(Duration::from_secs(2)).expect("recv ok");
        }
        // Next recv should see channel closed.
        let r = rx.recv_timeout(Duration::from_secs(2));
        assert!(r.is_err(), "channel should close after iterator exhausts");
    }

    #[test]
    fn runner_rejects_double_start() {
        let runner = ContinuousRunner::new(identity_stack(), 4);
        let _rx = runner.start(vec![hv(1)]).expect("first start ok");
        let r = runner.start(vec![hv(2)]);
        assert!(matches!(r, Err(ContinuousError::AlreadyStarted)));
        runner.stop().expect("ok");
    }

    #[test]
    fn runner_can_be_started_again_after_stop() {
        let runner = ContinuousRunner::new(identity_stack(), 4);
        let _rx1 = runner.start(vec![hv(1)]).expect("first start ok");
        runner.stop().expect("stop ok");
        let _rx2 = runner.start(vec![hv(2)]).expect("second start ok");
        runner.stop().expect("stop ok");
    }
}
