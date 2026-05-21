#!/usr/bin/env bash
# Local pre-push gate. Mirrors .github/workflows/ci.yml so a passing run here
# means CI will pass too (modulo environment-specific issues like cargo-audit
# advisory database freshness).
#
# Usage:  scripts/check.sh
#
# Requires a Python virtualenv at .venv/ with ruff + pytest installed:
#   python3 -m venv .venv
#   .venv/bin/pip install ruff pytest hypothesis maturin
#
# AVP-2 doctrine §8b: pre-push gate is mandatory; failures must be FIXED, not
# bypassed.

set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$PWD"
VENV="$REPO_ROOT/.venv"
RUFF="$VENV/bin/ruff"
PYTEST="$VENV/bin/pytest"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

fail() { red "✗ $*"; exit 1; }
ok()   { green "✓ $*"; }
step() { bold "─── $* ───"; }

step "cargo fmt --all -- --check"
cargo fmt --all -- --check || fail "cargo fmt failed"
ok "cargo fmt clean"

step "cargo clippy --workspace --all-targets --locked -- -D warnings"
cargo clippy --workspace --all-targets --locked -- -D warnings 2>&1 | tail -1
ok "cargo clippy clean"

step "cargo test --workspace --locked"
cargo test --workspace --locked 2>&1 | grep -E "^test result|FAILED" | tail -10
ok "cargo test pass"

step "cargo doc --workspace --no-deps --locked"
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked 2>&1 | tail -1
ok "cargo doc clean"

if [ -x "$RUFF" ]; then
    step "ruff check python/"
    "$RUFF" check python/ || fail "ruff check failed"
    ok "ruff clean"
else
    yellow "⚠ ruff not in .venv — run: python3 -m venv .venv && .venv/bin/pip install ruff"
fi

if [ -x "$PYTEST" ]; then
    step "pytest python/tests/ -q"
    "$PYTEST" python/tests/ -q 2>&1 | tail -5
    ok "pytest pass"
else
    yellow "⚠ pytest not in .venv — run: .venv/bin/pip install pytest"
fi

if command -v cargo-audit >/dev/null 2>&1; then
    step "cargo audit"
    cargo audit 2>&1 | tail -3
    ok "cargo audit clean"
else
    yellow "⚠ cargo-audit not installed — run: cargo install --locked cargo-audit"
fi

green "═══════════════════════════════════════"
green "  Pre-push gate: ALL CHECKS PASSED"
green "═══════════════════════════════════════"
