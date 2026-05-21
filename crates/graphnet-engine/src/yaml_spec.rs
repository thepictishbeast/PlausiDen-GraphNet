//! Versioned YAML architecture specification — Phase 9.
//!
//! Plan §2.7 spec: human-readable, version-controllable, shareable. Owner
//! ask: "make GraphNet so I can export and save my configs so I can show
//! you what works."
//!
//! The YAML spec captures *architecture only* — operation list + types +
//! dimensionality. Weights ride along separately via the snapshot bytes
//! (Phase 1's bincode + Phase 1's signed_snapshot). Splitting structure
//! from weights lets users diff configurations cheaply without touching
//! megabytes of bipolar data.

use plausiden_hdc::Hypervector;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::op::Operation;
use crate::stack::Stack;

/// Current spec schema version. Bump on breaking changes; migrations land
/// in this module alongside the bumped variant.
pub const SPEC_VERSION: &str = "1.0";

/// Errors raised by export/import.
#[derive(Debug, Error)]
pub enum SpecError {
    /// YAML serialisation failed.
    #[error("yaml encode: {0}")]
    Encode(String),

    /// YAML deserialisation failed.
    #[error("yaml decode: {0}")]
    Decode(String),

    /// The spec uses a schema version this binary doesn't know how to load.
    #[error("unknown spec version: {0}; this build supports {SPEC_VERSION}")]
    UnknownVersion(String),

    /// The spec referenced a key index whose ID isn't present in keys map.
    #[error("missing key id `{0}` referenced by operation at index {1}")]
    MissingKey(String, usize),

    /// The reconstructed Stack would have invalid state.
    #[error("invalid spec: {0}")]
    Invalid(String),
}

/// One operation as described in the spec — a tag + optional key reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct OperationSpec {
    /// Operation tag: "identity", "dense", or "hrr_bind".
    pub kind: String,
    /// Key-vector ID referenced from `ArchitectureSpec::keys` (None for Identity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
}

/// Full architecture specification — Stack shape + per-op kinds + key refs.
///
/// Keys are stored separately under a content-addressable id (blake3 of
/// the bipolar bytes) so identical keys reused across ops dedupe naturally,
/// and human readers can spot when two configs share a key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArchitectureSpec {
    /// Schema version (see [`SPEC_VERSION`]).
    pub version: String,
    /// Top-level kind — "stack" today; "stack_of_stacks" / "pipeline" later.
    pub kind: String,
    /// Hypervector dimensionality the Stack operates at.
    pub dim: usize,
    /// Operations in order.
    pub ops: Vec<OperationSpec>,
    /// Content-addressable key vectors referenced by ops. id → bipolar i8 vec.
    #[serde(default)]
    pub keys: std::collections::BTreeMap<String, Vec<i8>>,
    /// Free-form metadata for the spec author to annotate (notes, citations).
    #[serde(default)]
    pub notes: Vec<String>,
    /// Provenance + lineage (plan §21.8) — blake3 content hashes of parent
    /// specs this one was derived from. Empty for root specs.
    #[serde(default)]
    pub lineage: Vec<String>,
}

impl ArchitectureSpec {
    /// Build a spec from an in-memory Stack.
    #[must_use]
    pub fn from_stack(stack: &Stack) -> Self {
        let mut keys: std::collections::BTreeMap<String, Vec<i8>> =
            std::collections::BTreeMap::new();
        let mut ops = Vec::with_capacity(stack.operations().len());

        for op in stack.operations() {
            let (kind, key_id) = match op {
                Operation::Identity => ("identity".to_string(), None),
                Operation::Dense { key } => {
                    let id = blake3::hash(bytemuck::cast_slice(key.as_slice()))
                        .to_hex()
                        .to_string();
                    keys.entry(id.clone())
                        .or_insert_with(|| key.as_slice().to_vec());
                    ("dense".to_string(), Some(id))
                }
                Operation::HrrBind { key } => {
                    let id = blake3::hash(bytemuck::cast_slice(key.as_slice()))
                        .to_hex()
                        .to_string();
                    keys.entry(id.clone())
                        .or_insert_with(|| key.as_slice().to_vec());
                    ("hrr_bind".to_string(), Some(id))
                }
                Operation::Permute { shift } => {
                    // Use the shift value itself as the "key" identifier.
                    (format!("permute:{shift}"), None)
                }
                Operation::Negate => ("negate".to_string(), None),
            };
            ops.push(OperationSpec { kind, key_id });
        }

        Self {
            version: SPEC_VERSION.to_string(),
            kind: "stack".to_string(),
            dim: stack.dim(),
            ops,
            keys,
            notes: Vec::new(),
            lineage: Vec::new(),
        }
    }

    /// Compute a blake3 content hash of this spec.
    ///
    /// Hashes the canonical YAML serialisation. Same spec → same hash, so
    /// two architectures sharing a `content_hash()` are byte-for-byte
    /// identical (modulo lineage, which is excluded — see the
    /// implementation note below).
    ///
    /// BUG ASSUMPTION: lineage is included in the hash today; if you want
    /// lineage-invariant hashing (so two specs from different parents but
    /// identical content collide), strip lineage before serialising. We
    /// keep it included for now because lineage IS part of identity for
    /// provenance audits.
    pub fn content_hash(&self) -> Result<String, SpecError> {
        let yaml = self.to_yaml()?;
        Ok(blake3::hash(yaml.as_bytes()).to_hex().to_string())
    }

    /// Builder: record this spec as having been derived from `parent`.
    ///
    /// Appends `parent.content_hash()` to `self.lineage`. The lineage list
    /// is order-preserving — most recent ancestor at the end.
    pub fn with_parent(mut self, parent: &ArchitectureSpec) -> Result<Self, SpecError> {
        let parent_hash = parent.content_hash()?;
        self.lineage.push(parent_hash);
        Ok(self)
    }

    /// Builder: append a note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Materialise this spec back into a Stack.
    pub fn to_stack(&self) -> Result<Stack, SpecError> {
        if self.version != SPEC_VERSION {
            return Err(SpecError::UnknownVersion(self.version.clone()));
        }
        if self.kind != "stack" {
            return Err(SpecError::Invalid(format!(
                "unsupported top-level kind `{}` (only `stack` in v1)",
                self.kind
            )));
        }
        let mut stack = Stack::new(self.dim);
        for (index, op_spec) in self.ops.iter().enumerate() {
            let op = match op_spec.kind.as_str() {
                "identity" => Operation::Identity,
                "negate" => Operation::Negate,
                s if s.starts_with("permute:") => {
                    let shift: usize = s
                        .strip_prefix("permute:")
                        .and_then(|n| n.parse().ok())
                        .ok_or_else(|| {
                            SpecError::Invalid(format!(
                                "permute op at index {index} has bad shift literal"
                            ))
                        })?;
                    Operation::Permute { shift }
                }
                "dense" | "hrr_bind" => {
                    let id = op_spec
                        .key_id
                        .as_ref()
                        .ok_or_else(|| SpecError::MissingKey("<none>".to_string(), index))?;
                    let key_bytes = self
                        .keys
                        .get(id)
                        .ok_or_else(|| SpecError::MissingKey(id.clone(), index))?;
                    let key = Hypervector::from_bipolar(key_bytes.clone()).ok_or_else(|| {
                        SpecError::Invalid(format!("key `{id}` has non-bipolar values"))
                    })?;
                    if key.dim() != self.dim {
                        return Err(SpecError::Invalid(format!(
                            "key `{id}` dim={} but stack dim={}",
                            key.dim(),
                            self.dim
                        )));
                    }
                    if op_spec.kind == "dense" {
                        Operation::Dense { key }
                    } else {
                        Operation::HrrBind { key }
                    }
                }
                other => {
                    return Err(SpecError::Invalid(format!(
                        "unknown op kind `{other}` at index {index}"
                    )))
                }
            };
            stack.add_operation(op);
        }
        Ok(stack)
    }

    /// Emit this spec as a YAML string.
    pub fn to_yaml(&self) -> Result<String, SpecError> {
        serde_yaml::to_string(self).map_err(|e| SpecError::Encode(e.to_string()))
    }

    /// Parse a spec from a YAML string.
    pub fn from_yaml(s: &str) -> Result<Self, SpecError> {
        serde_yaml::from_str(s).map_err(|e| SpecError::Decode(e.to_string()))
    }
}

/// Convenience: round-trip a Stack through YAML in one call.
pub fn stack_to_yaml(stack: &Stack) -> Result<String, SpecError> {
    ArchitectureSpec::from_stack(stack).to_yaml()
}

/// Convenience: load a Stack from YAML in one call.
pub fn stack_from_yaml(s: &str) -> Result<Stack, SpecError> {
    ArchitectureSpec::from_yaml(s)?.to_stack()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn empty_stack_round_trip_through_yaml() {
        let s = Stack::new(1_000);
        let yaml = stack_to_yaml(&s).expect("yaml ok");
        let restored = stack_from_yaml(&yaml).expect("from yaml ok");
        assert_eq!(restored.dim(), 1_000);
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn three_op_stack_round_trip() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) })
            .with_operation(Operation::HrrBind { key: hv(2) });
        let yaml = stack_to_yaml(&s).expect("yaml ok");
        // YAML is human-readable: should mention identity + dense + hrr_bind.
        assert!(yaml.contains("identity"));
        assert!(yaml.contains("dense"));
        assert!(yaml.contains("hrr_bind"));

        let restored = stack_from_yaml(&yaml).expect("from yaml ok");
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.operations()[0].tag(), "identity");
        assert_eq!(restored.operations()[1].tag(), "dense");
        assert_eq!(restored.operations()[2].tag(), "hrr_bind");
    }

    #[test]
    fn round_trip_preserves_forward_output() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Dense { key: hv(1) })
            .with_operation(Operation::Identity);
        let input = hv(42);
        let original = s.forward(&input).expect("ok");

        let yaml = stack_to_yaml(&s).expect("yaml ok");
        let restored = stack_from_yaml(&yaml).expect("from yaml ok");
        let restored_out = restored.forward(&input).expect("ok");

        assert_eq!(original, restored_out);
    }

    #[test]
    fn keys_dedupe_when_reused() {
        let k = hv(1);
        let s = Stack::new(1_000)
            .with_operation(Operation::Dense { key: k.clone() })
            .with_operation(Operation::HrrBind { key: k.clone() })
            .with_operation(Operation::Dense { key: k });
        let spec = ArchitectureSpec::from_stack(&s);
        // 3 ops referencing the same key → only 1 entry in the keys map.
        assert_eq!(spec.keys.len(), 1);
    }

    #[test]
    fn unknown_version_rejected() {
        let mut spec = ArchitectureSpec::from_stack(&Stack::new(1_000));
        spec.version = "99.99".to_string();
        let err = spec.to_stack().expect_err("should reject");
        assert!(matches!(err, SpecError::UnknownVersion(_)));
    }

    #[test]
    fn unknown_kind_rejected() {
        let mut spec = ArchitectureSpec::from_stack(&Stack::new(1_000));
        spec.kind = "transformer".to_string();
        let err = spec.to_stack().expect_err("should reject");
        assert!(matches!(err, SpecError::Invalid(_)));
    }

    #[test]
    fn missing_key_id_rejected() {
        let s = Stack::new(1_000).with_operation(Operation::Dense { key: hv(1) });
        let mut spec = ArchitectureSpec::from_stack(&s);
        spec.keys.clear(); // strip the key but leave op referencing it
        let err = spec.to_stack().expect_err("should reject");
        assert!(matches!(err, SpecError::MissingKey(_, 0)));
    }

    #[test]
    fn content_hash_is_deterministic() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let spec_a = ArchitectureSpec::from_stack(&s);
        let spec_b = ArchitectureSpec::from_stack(&s);
        let h_a = spec_a.content_hash().expect("ok");
        let h_b = spec_b.content_hash().expect("ok");
        assert_eq!(h_a, h_b);
        // blake3 hex is 64 chars.
        assert_eq!(h_a.len(), 64);
    }

    #[test]
    fn content_hash_differs_on_different_specs() {
        let s1 = Stack::new(1_000).with_operation(Operation::Identity);
        let s2 = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let h1 = ArchitectureSpec::from_stack(&s1)
            .content_hash()
            .expect("ok");
        let h2 = ArchitectureSpec::from_stack(&s2)
            .content_hash()
            .expect("ok");
        assert_ne!(h1, h2);
    }

    #[test]
    fn with_parent_appends_lineage() {
        let parent_stack = Stack::new(1_000).with_operation(Operation::Identity);
        let parent_spec = ArchitectureSpec::from_stack(&parent_stack);
        let parent_hash = parent_spec.content_hash().expect("ok");

        let child_stack = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let child_spec = ArchitectureSpec::from_stack(&child_stack)
            .with_parent(&parent_spec)
            .expect("ok");

        assert_eq!(child_spec.lineage, vec![parent_hash]);
    }

    #[test]
    fn with_parent_chains_multiple_ancestors() {
        let g0 = ArchitectureSpec::from_stack(&Stack::new(1_000));
        let g0_hash = g0.content_hash().expect("ok");

        let g1_stack = Stack::new(1_000).with_operation(Operation::Identity);
        let g1 = ArchitectureSpec::from_stack(&g1_stack)
            .with_parent(&g0)
            .expect("ok");
        let g1_hash = g1.content_hash().expect("ok");

        let g2_stack = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let g2 = ArchitectureSpec::from_stack(&g2_stack)
            .with_parent(&g1)
            .expect("ok");

        // Lineage on g2 only records direct parent (g1), not transitive
        // ancestors. To get transitive history, walk parents via their
        // own lineage fields.
        assert_eq!(g2.lineage, vec![g1_hash]);
        // g1's lineage still has g0.
        assert_eq!(g1.lineage, vec![g0_hash]);
    }

    #[test]
    fn with_note_appends_notes() {
        let spec = ArchitectureSpec::from_stack(&Stack::new(1_000))
            .with_note("initial draft")
            .with_note("revised 2026-05-17");
        assert_eq!(
            spec.notes,
            vec![
                "initial draft".to_string(),
                "revised 2026-05-17".to_string()
            ]
        );
    }

    #[test]
    fn lineage_round_trips_through_yaml() {
        let parent = ArchitectureSpec::from_stack(&Stack::new(1_000));
        let child =
            ArchitectureSpec::from_stack(&Stack::new(1_000).with_operation(Operation::Identity))
                .with_parent(&parent)
                .expect("ok");
        let yaml = child.to_yaml().expect("ok");
        let parsed = ArchitectureSpec::from_yaml(&yaml).expect("ok");
        assert_eq!(parsed.lineage, child.lineage);
    }

    #[test]
    fn yaml_is_human_readable() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let yaml = stack_to_yaml(&s).expect("ok");
        // serde_yaml quotes string fields that look numeric; accept either form.
        assert!(
            yaml.contains(&format!("version: {SPEC_VERSION}"))
                || yaml.contains(&format!("version: '{SPEC_VERSION}'")),
            "yaml missing version: {yaml}"
        );
        assert!(yaml.contains("kind: stack"));
        assert!(yaml.contains("dim: 1000"));
    }
}
