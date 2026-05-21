# PlausiDen-GraphNet — doctrine for AI agents

If you (Claude or any other AI agent) are about to write code in this repo,
**read this file first**. Every rule exists because skipping it has produced
bad output in the past or violates owner-stated requirements.

## Source of truth

The full design lives at [`docs/PLAN.md`](./docs/PLAN.md) (symlinked from
`PlausiDen-AI/lfi_vsa_core/GRAPHNET_BUILD_PLAN.md`). When in doubt, read it.
The plan covers:

- 12 build phases (§7)
- 30+ feature requirements (§2, §15-§17, §21)
- Per-feature FOSS dependency picks (§13)
- Test/audit/debug strategy (§9 + §10)
- AVP-2 security model (§21.10)
- Future-proofing abstractions (§18)

## Hard rules

1. **One bounded shippable unit per loop tick.** Don't try to ship a whole
   phase in one tick; smallest viable progress that advances the phase.
2. **Mandatory pre-push gate.** `cargo fmt --all -- --check` + `cargo clippy
   --workspace --all-targets --locked -- -D warnings` + `cargo test
   --workspace --locked` + `pytest python/tests/` (when python touched).
   Any failure → fix, do NOT push.
3. **No `unwrap` / `expect` in library code** without a `SAFETY:` comment
   or `// test-only` justification (per AVP-2 doctrine).
4. **No `unsafe` blocks** unless absolutely necessary; if so, document with
   `// SAFETY:` proof.
5. **Every public function has a `BUG ASSUMPTION:` comment** documenting
   what could go wrong + a test (per AVP-2 doctrine).
6. **AVP-2 §8b branch hygiene.** `main` = active dev; `master` = validated
   tip. Fast-forward `master` from `main` only after CI green for the
   EXACT SHA. Never force-push to either.
7. **Owner-direction is binding.** If the plan and owner direction disagree,
   owner direction wins; update the plan to reflect.

## Layer separation

| Layer | Where | What touches it |
|-------|-------|-----------------|
| Rust core | `crates/graphnet-engine/` | Performance-critical work, Model trait, intervention API, recording |
| Python bindings | `crates/graphnet-bindings/` (PyO3) | Thin FFI layer; no logic |
| Python facade | `python/graphnet/` | User-facing REPL API, visualisation, adapters |
| Tests | `tests/` (Rust int), `python/tests/` (Python) | Mirror the production layer they test |
| Benches | `benches/` (criterion) | Performance regression gate |
| Docs | `docs/` | Plan, architecture, API reference, audit reports |

## Adding dependencies

Any new dep requires:
1. Already listed in `Cargo.toml [workspace.dependencies]` OR `pyproject.toml`
2. Permissive license (MIT / Apache-2.0 / BSD / MPL / FSL — never GPL/AGPL)
3. AVP-2 supply-chain check (`cargo audit` + manual `cargo geiger` if new)
4. Justified against the per-feature FOSS map in `docs/PLAN.md §13`

## Forbidden actions

- Force-push to `main` or `master` without explicit owner sign-off + archive branch
- Skip pre-push gates with `--no-verify`
- Touch repos outside PlausiDen-GraphNet / PlausiDen-Stack scope
- Add features beyond `docs/PLAN.md` scope without owner direction first
- Use Python-version-specific syntax requiring < 3.10
- Add `unsafe` Rust without `// SAFETY:` proof

## When in doubt

Ask the owner. Don't guess and ship.
