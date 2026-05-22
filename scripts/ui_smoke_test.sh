#!/usr/bin/env bash
# UI smoke test (#759) — drives the GraphNet GUI via xdotool, captures
# a screenshot after each scripted interaction. Run from the repo root.
#
# Requires: xdotool, scrot, an active X display. Reads $DISPLAY +
# $XAUTHORITY from the environment (defaults to :0 + /home/user/.Xauthority).

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
OUT="${OUT:-/tmp/graphnet-ui-test}"
mkdir -p "$OUT"

step() {
    local name="$1"
    sleep 0.6
    scrot "$OUT/$(date +%s)_${name}.png" 2>/dev/null
    echo "  → $name"
}

echo "[ui-smoke] killing any running graphnet-gui"
pkill -9 -f graphnet-gui 2>/dev/null || true
sleep 1

echo "[ui-smoke] launching $BIN"
"$BIN" >/tmp/graphnet-ui-test.log 2>&1 &
GUI_PID=$!
sleep 4

if ! kill -0 "$GUI_PID" 2>/dev/null; then
    echo "ERROR: graphnet-gui exited prematurely"
    tail -20 /tmp/graphnet-ui-test.log
    exit 1
fi

WIN=$(xdotool search --name "GraphNet" | head -1)
if [ -z "$WIN" ]; then
    echo "ERROR: no GraphNet window found"
    kill "$GUI_PID" 2>/dev/null
    exit 1
fi

echo "[ui-smoke] window id=$WIN — running script"
xdotool windowactivate "$WIN"
sleep 0.5
step "00_startup"

# 1. Press Space → run a forward.
xdotool key --window "$WIN" space
step "01_space_forward"

# 2. R → regenerate input.
xdotool key --window "$WIN" r
step "02_r_regen"

# 3. A → add Identity.
xdotool key --window "$WIN" a
step "03_a_add_identity"

# 4. D → add Dense.
xdotool key --window "$WIN" d
step "04_d_add_dense"

# 5. Backtick → open console.
xdotool key --window "$WIN" grave
step "05_console_open"
xdotool key --window "$WIN" Escape
sleep 0.3
xdotool key --window "$WIN" grave
step "06_console_closed"

# 6. H → help overlay.
xdotool key --window "$WIN" h
step "07_help_open"
xdotool key --window "$WIN" h
step "08_help_closed"

# 7. L → live mode.
xdotool key --window "$WIN" l
sleep 1
step "09_live_mode"
xdotool key --window "$WIN" l
step "10_live_off"

# 8. 6 → noise-resilience template.
xdotool key --window "$WIN" 6
step "11_template_6"

# 9. Ctrl+Z undo / Ctrl+Shift+Z redo.
xdotool key --window "$WIN" ctrl+z
step "12_undo"
xdotool key --window "$WIN" ctrl+shift+z
step "13_redo"

# 10. Tab toggles left panel.
xdotool key --window "$WIN" Tab
step "14_left_closed"
xdotool key --window "$WIN" Tab
step "15_left_open"

# 11. Cmd+N opens templates popup.
xdotool key --window "$WIN" ctrl+n
step "16_templates_popup"
xdotool key --window "$WIN" Escape
step "17_templates_closed"

# 12. Cmd+1 saves to slot A.
xdotool key --window "$WIN" ctrl+1
step "18_slot_a_save"

# 13. F key resets 3D viewport rotation.
xdotool key --window "$WIN" f
step "19_f_reset"

echo "[ui-smoke] done — $(ls "$OUT" | wc -l) screenshots in $OUT"
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
echo "[ui-smoke] PASS"
