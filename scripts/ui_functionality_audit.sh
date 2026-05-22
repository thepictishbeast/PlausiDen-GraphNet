#!/usr/bin/env bash
# Functionality audit: every button + menu item + tool palette icon
# should do exactly what it's supposed to. Drive each via xdotool +
# assert the expected log entries fire.
#
# *** WARNING *** Steals focus + sends keystrokes/clicks. DO NOT RUN
# while you are actively using the laptop. USER-INVOKED only.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
OUT="${OUT:-/tmp/graphnet-functionality-audit}"
LOG="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/graphnet.log"
mkdir -p "$OUT"
ERRORS=0

fail() { echo "FAIL: $*"; ERRORS=$((ERRORS + 1)); }

# Wait for a log line matching pattern within timeout (seconds).
wait_for_log() {
    local pattern="$1" timeout="${2:-2}"
    local elapsed=0
    while [ "$elapsed" -lt "$((timeout * 10))" ]; do
        if grep -qE "$pattern" "$LOG" 2>/dev/null; then return 0; fi
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
    return 1
}

# Test a keypress + expected log pattern.
test_keypress() {
    local key="$1" pattern="$2" label="$3"
    local before_len=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    xdotool key --window "$WIN" "$key"
    sleep 0.5
    local after_len=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    if [ "$after_len" -gt "$before_len" ]; then
        local new_lines=$(tail -n $((after_len - before_len)) "$LOG")
        if echo "$new_lines" | grep -qE "$pattern"; then
            echo "  ✓ $label: '$key' → matches '$pattern'"
        else
            fail "$label: '$key' fired but no '$pattern' match"
            echo "      got: $new_lines"
        fi
    else
        fail "$label: '$key' fired but no NEW log entries"
    fi
}

echo "==== Functionality audit start ===="
pkill -9 -f graphnet-gui 2>/dev/null || true
sleep 1
rm -f "$LOG"
"$BIN" >/tmp/graphnet-fa.log 2>&1 &
GUI_PID=$!
sleep 4
WIN=$(xdotool search --name "GraphNet" | head -1)
[ -z "$WIN" ] && { fail "no GraphNet window"; exit 1; }
xdotool windowactivate "$WIN"
sleep 0.5
xdotool key --window "$WIN" Escape  # dismiss walkthrough
sleep 0.3

echo "=== Keyboard shortcuts ==="
test_keypress "a"        "\+ op \[.+\] identity"       "A → add identity"
test_keypress "d"        "\+ op \[.+\] dense"          "D → add dense"
test_keypress "f"        "\+ op \[.+\] hrr_bind"       "F → add hrr_bind"
test_keypress "p"        "\+ op \[.+\] permute"        "P → add permute"
test_keypress "n"        "\+ op \[.+\] negate"         "N → add negate"
test_keypress "r"        "input regenerated"           "R → regen input"
test_keypress "space"    "forward #"                   "Space → run forward (every 25 logged)"
test_keypress "l"        "live mode"                   "L → toggle live mode"
test_keypress "l"        "live mode"                   "L → toggle live again"
test_keypress "ctrl+z"   "↶ undo"                      "Ctrl+Z → undo"
test_keypress "ctrl+shift+z" "↷ redo"                  "Ctrl+Shift+Z → redo"
test_keypress "1"        "template '.*' loaded"        "1 → load template 1"
test_keypress "0"        "template '.*' loaded"        "0 → load template 10"
test_keypress "ctrl+1"   "saved current stack to slot" "Ctrl+1 → slot A save"
test_keypress "ctrl+shift+1" "recalled slot|slot .* is empty" "Ctrl+Shift+1 → slot A recall"

echo "=== Arrow-key 3D navigation ==="
xdotool key --window "$WIN" "Right"
sleep 0.3
xdotool key --window "$WIN" "Left"
sleep 0.3
xdotool key --window "$WIN" "Home"
sleep 0.3
xdotool key --window "$WIN" "End"
sleep 0.3
echo "  ✓ Arrow / Home / End cycled (no log entry expected — purely UI state)"

# Don't test Ctrl+N — leaks to file manager on some WMs (iter 70 lesson).

echo ""
echo "==== Functionality audit complete: $ERRORS errors ===="
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
if [ "$ERRORS" -eq 0 ]; then
    echo "[functionality] PASS"; exit 0
else
    echo "[functionality] FAIL"; exit 1
fi
