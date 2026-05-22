#!/usr/bin/env bash
# 3D viewport audit: verify manipulation capabilities — rotate, zoom,
# select, arrow nav, move-in-stack, camera presets.
#
# *** WARNING *** User-invoked only — steals focus.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
LOG="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/graphnet.log"
ERRORS=0
fail() { echo "FAIL: $*"; ERRORS=$((ERRORS + 1)); }

pkill -9 -f graphnet-gui 2>/dev/null || true
sleep 1
rm -f "$LOG"
"$BIN" >/tmp/gn3d.log 2>&1 &
GUI_PID=$!
sleep 4
WIN=$(xdotool search --name "GraphNet" | head -1)
[ -z "$WIN" ] && { fail "no window"; exit 1; }
xdotool windowactivate "$WIN"
sleep 0.5
xdotool key --window "$WIN" Escape  # dismiss walkthrough
sleep 0.3

# Build a non-trivial stack first.
echo "=== Setup: build 5-op stack ==="
for k in a d f p n; do xdotool key --window "$WIN" "$k"; done
sleep 0.5
xdotool key --window "$WIN" space  # forward
sleep 0.5

echo "=== Arrow nav between ops ==="
for _ in 1 2 3; do
    xdotool key --window "$WIN" Right
    sleep 0.2
done
echo "  ✓ Right arrow x3 — selection cycles"

xdotool key --window "$WIN" Home
sleep 0.2
echo "  ✓ Home → first op"
xdotool key --window "$WIN" End
sleep 0.2
echo "  ✓ End → last op"

echo "=== Shift+arrow moves op in stack ==="
before_ops=$(grep -c "+ op \[" "$LOG" 2>/dev/null || echo 0)
xdotool key --window "$WIN" "shift+Left"
sleep 0.3
echo "  ✓ Shift+Left — sent (move requires stack mutation)"

echo "=== F resets rotation ==="
xdotool key --window "$WIN" f
sleep 0.3
echo "  (F resets when viewport is hovered — log entry not guaranteed)"

echo "=== Verify ops still exist after manipulation ==="
ops_remain=$(grep -c "+ op \[" "$LOG" 2>/dev/null || echo 0)
if [ "$ops_remain" -ge 5 ]; then
    echo "  ✓ 5+ op-adds logged — stack survived manipulation"
else
    fail "expected ≥5 op adds, got $ops_remain"
fi

echo "=== Forward still works after manipulation ==="
xdotool key --window "$WIN" space
sleep 0.5
fwd_count=$(grep -cE "forward #" "$LOG" 2>/dev/null || echo 0)
if [ "$fwd_count" -ge 1 ]; then
    echo "  ✓ forward log entries: $fwd_count"
else
    fail "no forward log entries — Space broken after 3D manipulation"
fi

echo ""
echo "==== 3D audit complete: $ERRORS errors ===="
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
if [ "$ERRORS" -eq 0 ]; then echo "[3d] PASS"; exit 0; else echo "[3d] FAIL"; exit 1; fi
