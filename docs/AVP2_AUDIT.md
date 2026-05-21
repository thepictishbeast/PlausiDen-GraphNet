# GraphNet AVP-2 Audit Report

Per [Adversarial Validation Protocol v2](https://github.com/thepictishbeast/PlausiDen-AVP-Doctrine/blob/main/AVP2_PROTOCOL.md)
doctrine, this file records cumulative audit findings against GraphNet's
critical-path code. Default verdict: **STILL BROKEN**. No commit in this
repository has reached `SHIP-DECISION:` status.

## Coverage of Tier 1 — Existence Proof

| Pass | What | Status | Notes |
|---|---|---|---|
| 1 — skeleton audit | Public functions documented, return types match | ✅ DONE | `BUG ASSUMPTION:` comments on engine public API. |
| 2 — null/zero/empty sweep | Empty Stack returns Err(Empty) on forward; D=0 documented | ✅ DONE | `Stack::forward` errors on empty; `Hypervector::random(0, ...)` documented. |
| 3 — boundary sweep | Dim mismatch surfaces as `HdcError::DimMismatch`; index OOB → `InterventionError::IndexOutOfRange` | ✅ DONE | Errors propagated, never silently truncated. |
| 4 — error-path completeness | Every public Result-returning function tested for at least one error path | ✅ DONE | See per-module `#[test]` modules. |
| 5 — type tightening | `#[non_exhaustive]` on `Operation` + `Intervention`; newtype `Hypervector` over `Vec<i8>`; no boolean blindness | ✅ DONE | `OperationError` / `StackError` / `InterventionError` / `HistoryError` / `SpecError` are distinct types. |
| 6 — dependency audit | `cargo audit` runs in CI; advisory database refreshed each run | ✅ DONE | pyo3 0.24, bincode 2.0 (RUSTSEC fixes landed 2026-05-17). |

## Coverage of Tier 2 — Failure Resilience

| Pass | What | Status | Notes |
|---|---|---|---|
| 7 — fault injection on I/O | `snapshot::verify_and_restore` detects tampered bytes via blake3 mismatch | ✅ DONE | `signed_snapshot_detects_tamper` test passes. |
| 8 — concurrency chaos | `ContinuousRunner` tested under double-start, idempotent stop, restart-after-stop | ✅ DONE | 6 continuous tests; no shared mutable state outside `Arc<AtomicBool/U64>`. |
| 9 — resource exhaustion | `ContinuousRunner` bounded queue + drop counter; doesn't OOM on slow consumer | ✅ DONE | `runner_drops_events_when_consumer_slow` test. |
| 10 — graceful degradation | Adapter framework lazy-imports heavy deps; `available_adapters()` reports missing | ✅ DONE | Python `adapters/`. |
| 11 — data integrity under crash | Bincode snapshot + blake3 hash; YAML spec key dedup | ✅ DONE | Tamper detection covered. |
| 12 — combined-chaos dry run | Not yet run on production hosts | ⚠ PENDING | Real bwrap + nft + NVIDIA-SMI integration tests need GPU host. |

## Coverage of Tier 6 — Meta-validation

| Pass | What | Status | Notes |
|---|---|---|---|
| 34 — mutation testing | `cargo mutants` not yet run | ⚠ PENDING | Cron-budget; runs locally outside the 1m loop. |
| 35 — property-based testing | `proptest` ≥100 cases per property on HDC ops + intervention round-trip + yaml round-trip | ✅ DONE | `crates/plausiden-hdc/tests/properties.rs` (6 props × 1k cases) + `crates/graphnet-engine/tests/intervene_properties.rs` (4 props × 200 cases) + `crates/graphnet-engine/tests/yaml_spec_properties.rs` (2 props × 100 cases). |
| 36 — formal verification | Not feasible for this codebase scope | ✗ SKIP | Documented residual risk. |

## SHIP-DECISION: NOT YET

Current state per AVP-2 default verdict: **STILL BROKEN**.

Residual risks accepted to-date: none (no `SHIP-DECISION:` annotations
recorded yet).

To approach `SHIP-DECISION:` for the engine surface, complete:
1. Tier 3 (adversarial security) pass — fuzz the YAML + bincode decoders
2. Tier 4 (UX) pass — first-contact install test on a clean Linux box
3. Mutation survival < 5%
4. Real-hardware GPU/NVML smoke on a GPU host
5. Visual regression suite via PlausiDen-Crawler

Tracked in tasks #694, #696 (native shell), #695 (rolling features).
