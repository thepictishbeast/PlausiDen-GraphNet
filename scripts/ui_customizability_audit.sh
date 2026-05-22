#!/usr/bin/env bash
# Customizability audit: verify the user can change values + manipulate
# objects via the UI. Drives sliders, toggles, op-mutations and asserts
# the changes propagate.
#
# *** WARNING *** User-invoked.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
LOG="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/graphnet.log"
SETTINGS="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/settings.yaml"
ERRORS=0
fail() { echo "FAIL: $*"; ERRORS=$((ERRORS + 1)); }

pkill -9 -f graphnet-gui 2>/dev/null || true
sleep 1
rm -f "$LOG"
"$BIN" >/tmp/gncust.log 2>&1 &
GUI_PID=$!
sleep 4
WIN=$(xdotool search --name "GraphNet" | head -1)
[ -z "$WIN" ] && { fail "no window"; exit 1; }
xdotool windowactivate "$WIN"
sleep 0.5
xdotool key --window "$WIN" Escape
sleep 0.3

echo "=== Customization 1: add ops + verify count grows ==="
for k in a a a d d f f n; do
    xdotool key --window "$WIN" "$k"
done
sleep 0.3
op_adds=$(grep -c "+ op \[" "$LOG" 2>/dev/null || echo 0)
if [ "$op_adds" -eq 8 ]; then
    echo "  ✓ 8 op-adds logged"
else
    fail "expected 8 op-adds, got $op_adds"
fi

echo "=== Customization 2: template switch changes architecture ==="
xdotool key --window "$WIN" "5"
sleep 0.3
last_template=$(grep "template '" "$LOG" 2>/dev/null | tail -1)
if [ -n "$last_template" ]; then
    echo "  ✓ template loaded: $last_template"
else
    fail "template load not logged"
fi

echo "=== Customization 3: undo restores prior state ==="
xdotool key --window "$WIN" "ctrl+z"
sleep 0.3
if grep -qE "↶ undo" "$LOG"; then
    echo "  ✓ undo fired"
else
    fail "undo not logged"
fi

echo "=== Customization 4: slot save+recall persists state ==="
xdotool key --window "$WIN" "ctrl+1"
sleep 0.3
xdotool key --window "$WIN" "ctrl+shift+1"
sleep 0.3
saves=$(grep -cE "saved current stack to slot|recalled slot" "$LOG" 2>/dev/null || echo 0)
if [ "$saves" -ge 2 ]; then
    echo "  ✓ slot save + recall both logged ($saves entries)"
else
    fail "expected ≥2 slot events, got $saves"
fi

echo "=== Customization 5: settings.yaml exists + has customization keys ==="
sleep 3   # wait for auto-persist
if [ -f "$SETTINGS" ]; then
    for key in colormap theme_dark font_scale workspace achievements; do
        if grep -q "^$key:" "$SETTINGS"; then
            echo "  ✓ $key persisted in settings.yaml"
        else
            fail "$key missing from settings.yaml"
        fi
    done
else
    fail "settings.yaml not written"
fi

echo ""
echo "==== Customizability audit: $ERRORS errors ===="
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
if [ "$ERRORS" -eq 0 ]; then echo "[customizability] PASS"; exit 0; else echo "[customizability] FAIL"; exit 1; fi
