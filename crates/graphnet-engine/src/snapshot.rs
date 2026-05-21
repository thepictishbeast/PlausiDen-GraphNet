//! Snapshot + restore — bincode 2.x serde-compat serialisation of a Stack.
//!
//! Used by:
//! - Session save/load (full architecture + intermediate state)
//! - Time-travel debugging (snapshot before each intervention, restore on undo)
//! - Crash recovery (autosave every 30s; restore on startup)
//!
//! Phase 1 ships Stack + Operation snapshots. Phase 9 expands to full
//! Session-level snapshots (model + viz state + recorded inputs).

use bincode::config::Configuration;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::stack::Stack;

/// Errors during snapshot serialisation / restore.
#[derive(Debug, Error)]
pub enum SnapshotError {
    /// Serialisation failed.
    #[error("encode: {0}")]
    Encode(String),

    /// Deserialisation failed.
    #[error("decode: {0}")]
    Decode(String),
}

/// The canonical bincode 2.x config GraphNet uses (little-endian, fixint).
fn config() -> Configuration {
    bincode::config::standard()
}

/// Serialise a Stack to bytes via bincode 2.x serde compat.
pub fn snapshot(stack: &Stack) -> Result<Vec<u8>, SnapshotError> {
    bincode::serde::encode_to_vec(stack, config()).map_err(|e| SnapshotError::Encode(e.to_string()))
}

/// Restore a Stack from bytes previously produced by [`snapshot`].
pub fn restore(bytes: &[u8]) -> Result<Stack, SnapshotError> {
    let (stack, _read) = bincode::serde::decode_from_slice::<Stack, _>(bytes, config())
        .map_err(|e| SnapshotError::Decode(e.to_string()))?;
    Ok(stack)
}

/// Wrapper carrying a snapshot blob + content-hash (blake3) for tamper detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedSnapshot {
    /// Raw bincode bytes of the Stack.
    pub bytes: Vec<u8>,
    /// blake3 hash of `bytes`.
    pub hash: String,
}

/// Snapshot a Stack and attach a blake3 content hash.
pub fn signed_snapshot(stack: &Stack) -> Result<SignedSnapshot, SnapshotError> {
    let bytes = snapshot(stack)?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    Ok(SignedSnapshot { bytes, hash })
}

/// Restore from a signed snapshot; returns `Err(Decode)` if the hash mismatches.
pub fn verify_and_restore(signed: &SignedSnapshot) -> Result<Stack, SnapshotError> {
    let actual = blake3::hash(&signed.bytes).to_hex().to_string();
    if actual != signed.hash {
        return Err(SnapshotError::Decode(format!(
            "hash mismatch: expected {}, got {actual}",
            signed.hash
        )));
    }
    restore(&signed.bytes)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::op::Operation;
    use plausiden_hdc::Hypervector;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn snapshot_then_restore_round_trip() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) });
        let bytes = snapshot(&s).expect("ok");
        assert!(!bytes.is_empty());
        let restored = restore(&bytes).expect("ok");
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.dim(), 1_000);
        assert_eq!(restored.operations()[0].tag(), "identity");
        assert_eq!(restored.operations()[1].tag(), "dense");
    }

    #[test]
    fn snapshot_preserves_forward_output() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) });
        let input = hv(2);
        let original = s.forward(&input).expect("ok");

        let bytes = snapshot(&s).expect("ok");
        let restored = restore(&bytes).expect("ok");
        let restored_out = restored.forward(&input).expect("ok");

        assert_eq!(original, restored_out);
    }

    #[test]
    fn signed_snapshot_verifies_round_trip() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let signed = signed_snapshot(&s).expect("ok");
        let restored = verify_and_restore(&signed).expect("ok");
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn signed_snapshot_detects_tamper() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let mut signed = signed_snapshot(&s).expect("ok");
        // Tamper: flip a byte in the bincode payload.
        if let Some(byte) = signed.bytes.get_mut(0) {
            *byte = byte.wrapping_add(1);
        }
        let err = verify_and_restore(&signed).expect_err("should detect tamper");
        match err {
            SnapshotError::Decode(msg) => assert!(msg.contains("hash mismatch")),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn empty_stack_round_trip() {
        let s = Stack::new(1_000);
        let bytes = snapshot(&s).expect("ok");
        let restored = restore(&bytes).expect("ok");
        assert_eq!(restored.dim(), 1_000);
        assert_eq!(restored.len(), 0);
    }
}
