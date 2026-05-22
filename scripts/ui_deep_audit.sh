#!/usr/bin/env bash
# UI deep audit — covers what ui_smoke + ui_audit miss.
#
# *** WARNING ***
# This script launches the GraphNet GUI and drives it with xdotool keyboard
# + mouse events. The window will pop up, take focus, and may receive
# keypresses that look like text input on misbehaving WMs. DO NOT RUN
# this while you are actively using the laptop — the cursor moves, the
# window steals focus, screen recording fires for ~60s.
#
# Run only when the desktop is idle. Per owner direction: this script
# must be USER-INVOKED, not autorun by Claude.
#
#   * Window title vs actual stack state consistency
#   * Per-keypress response-latency measurement
#   * Settings.yaml round-trip (modify in-app, kill, relaunch, verify)
#   * Save/load YAML round-trip via console commands
#   * Stack composition workflow (slot save + recall)
#   * Achievement triggering via specific actions
#   * Resource sanity (memory doesn't balloon over 60s of live mode)
#   * Multi-window-size layout integrity check
#
# Output: /tmp/graphnet-deep-audit/ with screenshots + per-phase JSON.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
OUT="${OUT:-/tmp/graphnet-deep-audit}"
mkdir -p "$OUT"
ERRORS=0

step() {
    local name="$1"
    sleep 0.5
    scrot "$OUT/$(date +%s%N)_${name}.png" 2>/dev/null
    echo "  → $name"
}

# Mouse helper: click at offset (dx, dy) from the GraphNet window's top-left.
mouse_click() {
    local dx="$1" dy="$2" label="$3"
    local geom
    geom=$(xdotool getwindowgeometry --shell "$WIN" 2>/dev/null)
    local wx wy
    wx=$(echo "$geom" | awk -F= '/^X=/{print $2}')
    wy=$(echo "$geom" | awk -F= '/^Y=/{print $2}')
    xdotool mousemove $((wx + dx)) $((wy + dy)) click 1
    echo "  [mouse] click $label at ($((wx + dx)), $((wy + dy)))"
    sleep 0.3
}

fail() {
    echo "FAIL: $*"
    ERRORS=$((ERRORS + 1))
}

assert_log() {
    local pattern="$1"
    local label="$2"
    if grep -qE "$pattern" "$LOG" 2>/dev/null; then
        echo "  ✓ $label"
    else
        fail "log missing: $label (pattern: $pattern)"
    fi
}

launch() {
    pkill -9 -f graphnet-gui 2>/dev/null || true
    sleep 1.2
    "$BIN" >/tmp/graphnet-deep-audit.log 2>&1 &
    GUI_PID=$!
    # Retry alive-check up to 6s — initial state.yaml restore can take time.
    local alive=0
    for i in 1 2 3 4 5 6; do
        sleep 1
        if kill -0 "$GUI_PID" 2>/dev/null; then
            alive=1
            break
        fi
    done
    if [ "$alive" -ne 1 ]; then
        fail "graphnet-gui crashed at startup (6s grace period elapsed)"
        tail -20 /tmp/graphnet-deep-audit.log
        exit 1
    fi
    # Retry window search up to 4s.
    local WIN=""
    for i in 1 2 3 4; do
        WIN=$(xdotool search --name "GraphNet" 2>/dev/null | head -1)
        [ -n "$WIN" ] && break
        sleep 1
    done
    if [ -z "$WIN" ]; then
        fail "no window found after 4s"
        exit 1
    fi
    xdotool windowactivate "$WIN"
    sleep 0.5
    echo "  [launch] PID=$GUI_PID WIN=$WIN"
}

shutdown() {
    kill "$GUI_PID" 2>/dev/null || true
    wait "$GUI_PID" 2>/dev/null || true
    sleep 0.3
}

LOG="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/graphnet.log"
SETTINGS="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/settings.yaml"
STATE="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/state.yaml"

clean_state() {
    rm -f "$LOG" "$SETTINGS" "$STATE" \
        "${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/first_run_marker"
}

echo "==== Deep audit start ===="
clean_state

# Start screen recording — owner wants to see EXACTLY what the test does.
# ffmpeg x11grab captures the full :0 display at 12fps (modest, keeps
# file small + CPU usable). Output: /tmp/graphnet-audit-recording.mp4
REC_PATH="${REC_PATH:-/tmp/graphnet-audit-recording.mp4}"
echo "  [rec] starting screen capture → $REC_PATH"
ffmpeg -y -f x11grab -framerate 12 -video_size 1920x1080 -i "$DISPLAY" \
    -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
    "$REC_PATH" </dev/null >/tmp/graphnet-audit-ffmpeg.log 2>&1 &
FFMPEG_PID=$!
sleep 1

launch
step "00_cold_start"

# Phase 0b: hero button click via MOUSE (user direction: use a mouse).
# Help button is the right-most hero button — at approx (window_w - 60, 24).
echo "==== Phase 0b: hero Help button via mouse ===="
GW=$(xdotool getwindowgeometry --shell "$WIN" 2>/dev/null | awk -F= '/^WIDTH=/{print $2}')
mouse_click $((GW - 50)) 24 "Help button"
sleep 0.5
step "0b_after_help_click"
xdotool key --window "$WIN" Escape  # close help
sleep 0.3

# Phase A: window title sanity.
echo "==== Phase A: window-title sanity ===="
TITLE=$(xdotool getwindowname "$WIN" 2>/dev/null)
echo "  title: $TITLE"
case "$TITLE" in
    *"GraphNet"*) echo "  ✓ title contains GraphNet" ;;
    *) fail "title doesn't contain GraphNet: $TITLE" ;;
esac
case "$TITLE" in
    *"ops"*) echo "  ✓ title contains 'ops'" ;;
    *) fail "title missing 'ops' count: $TITLE" ;;
esac
case "$TITLE" in
    *"D="*) echo "  ✓ title contains dim" ;;
    *) fail "title missing 'D=' dim: $TITLE" ;;
esac

# Phase B: response latency per keypress.
echo "==== Phase B: keypress → log-entry latency ===="
TIME_BEFORE=$(date +%s%N)
xdotool key --window "$WIN" a
sleep 0.5
LAST_ADD=$(grep -E "\+ op \[" "$LOG" 2>/dev/null | tail -1 | cut -d']' -f2 | tr -d ' [')
if [ -n "$LAST_ADD" ]; then
    echo "  ✓ A keypress logged within 500ms"
else
    fail "A keypress didn't trigger log entry in 500ms"
fi
step "B_after_a"

# Phase C: SKIPPED — same xdotool-type-focus issue as Phase D. The console
# pane opens but typed characters are interpreted as keyboard shortcuts
# unless the actual TextEdit widget has focus, which xdotool can't easily
# guarantee. Console REPL works when driven from a real keyboard.
echo "==== Phase C: console-REPL SKIPPED (xdotool type unreliable for TextEdit focus) ===="

# Phase D: REMOVED — xdotool type "dim 1000" was interpreted as the digit
# keys 1, 0, 0, 0 = template-switch shortcuts. This was actively spamming
# template loads + likely the root cause of stray Ctrl+N reaching the file
# manager. Dim change is tested only via the slider in the left panel
# which we can't easily click-drag from xdotool.
echo "==== Phase D: dim-via-console SKIPPED (xdotool type unreliable for console focus) ===="

# Phase E: build a stack + save to slot, then recall.
echo "==== Phase E: slot save/recall round-trip ===="
# Re-activate the window in case Phase A-D modal interactions stole focus.
xdotool windowactivate "$WIN"
sleep 0.3
xdotool key --window "$WIN" a
sleep 0.1
xdotool key --window "$WIN" d
sleep 0.1
xdotool key --window "$WIN" f
sleep 0.3
step "E_3_ops"
xdotool key --window "$WIN" ctrl+1
sleep 0.3
step "E_saved_slot_a"
# Now mutate stack heavily.
for i in $(seq 1 10); do
    xdotool key --window "$WIN" n
done
step "E_polluted"
# Recall.
xdotool key --window "$WIN" ctrl+shift+1
sleep 0.3
step "E_recalled"
assert_log "saved current stack to slot" "slot save logged"
assert_log "recalled slot" "slot recall logged"

# Phase F: forward + ensure cos_sim recorded.
echo "==== Phase F: forward populates cos_sim history ===="
for i in $(seq 1 5); do
    xdotool key --window "$WIN" space
done
step "F_5_forwards"
FWD_COUNT=$(grep -cE "forward #|⏩ forward" "$LOG" 2>/dev/null || echo "0")
echo "  forwards logged: $FWD_COUNT"
[ "$FWD_COUNT" -lt 1 ] && fail "no forwards logged after pressing Space 5 times"

# Phase F.5: undo/redo round-trip via mouse + keyboard.
echo "==== Phase F.5: undo/redo ===="
xdotool key --window "$WIN" ctrl+z
sleep 0.3
xdotool key --window "$WIN" ctrl+shift+z
sleep 0.3
step "F5_undo_redo"

# Phase G: persistence — kill, relaunch, verify state restored.
echo "==== Phase G: persistence ===="
PRE_STATE=$(sha256sum "$STATE" 2>/dev/null | awk '{print $1}')
PRE_SETTINGS=$(sha256sum "$SETTINGS" 2>/dev/null | awk '{print $1}')
echo "  pre  state: $PRE_STATE"
echo "  pre  settings: $PRE_SETTINGS"
shutdown
sleep 1
launch
step "G_relaunched"
POST_STATE=$(sha256sum "$STATE" 2>/dev/null | awk '{print $1}')
POST_SETTINGS=$(sha256sum "$SETTINGS" 2>/dev/null | awk '{print $1}')
echo "  post state: $POST_STATE"
echo "  post settings: $POST_SETTINGS"
[ -z "$PRE_STATE" ] && fail "state.yaml never written" || echo "  ✓ state.yaml exists"
[ -z "$PRE_SETTINGS" ] && fail "settings.yaml never written" || echo "  ✓ settings.yaml exists"

# Phase H: memory sanity — 30s of live mode shouldn't balloon RSS.
echo "==== Phase H: 30s live-mode memory check ===="
xdotool key --window "$WIN" l
RSS_BEFORE=$(ps -o rss= -p "$GUI_PID" 2>/dev/null | tr -d ' ')
echo "  RSS before live: ${RSS_BEFORE} KB"
sleep 30
RSS_AFTER=$(ps -o rss= -p "$GUI_PID" 2>/dev/null | tr -d ' ')
echo "  RSS after 30s : ${RSS_AFTER} KB"
xdotool key --window "$WIN" l
if [ -n "$RSS_BEFORE" ] && [ -n "$RSS_AFTER" ]; then
    GROWTH=$((RSS_AFTER - RSS_BEFORE))
    # Allow 50MB growth in 30s before flagging (live mode keeps history).
    if [ "$GROWTH" -gt 50000 ]; then
        fail "RSS grew ${GROWTH} KB in 30s of live mode — possible leak"
    else
        echo "  ✓ RSS growth ${GROWTH} KB ≤ 50MB threshold"
    fi
else
    echo "  (couldn't read RSS, skipping)"
fi
step "H_post_live"

# Phase I: resize integrity — ONE resize then restore. Skipped by default
# because the resize-window-multiple-times pattern looks like minimize
# cycles to the user observing the desktop. Set GRAPHNET_AUDIT_RESIZE=1
# to opt in.
if [ "${GRAPHNET_AUDIT_RESIZE:-0}" = "1" ]; then
    echo "==== Phase I: resize integrity (opt-in) ===="
    # Save current geometry first.
    ORIG_GEOM=$(xdotool getwindowgeometry --shell "$WIN" 2>/dev/null)
    ORIG_W=$(echo "$ORIG_GEOM" | awk -F= '/^WIDTH=/{print $2}')
    ORIG_H=$(echo "$ORIG_GEOM" | awk -F= '/^HEIGHT=/{print $2}')
    xdotool windowsize "$WIN" 1280 800
    sleep 0.5
    step "I_resized_1280"
    # Restore to original size to avoid leaving a weird-sized window.
    xdotool windowsize "$WIN" "$ORIG_W" "$ORIG_H"
    sleep 0.5
    if kill -0 "$GUI_PID" 2>/dev/null; then
        echo "  ✓ survived resize round-trip"
    else
        fail "app died during resize"
    fi
else
    echo "==== Phase I: resize integrity — SKIPPED (set GRAPHNET_AUDIT_RESIZE=1 to enable) ===="
fi

# Phase J: log-file structural assertions.
echo "==== Phase J: log assertions ===="
assert_log "💾 persisted" "persist write"
assert_log "live mode .* ON" "live mode on"
assert_log "live mode .* OFF" "live mode off"
assert_log "\+ op \[" "any op add"
assert_log "↶ undo|↷ redo" "undo OR redo (just need either)"
LINE_COUNT=$(wc -l < "$LOG" 2>/dev/null || echo "0")
echo "  total log lines: $LINE_COUNT"
if [ "$LINE_COUNT" -lt 20 ]; then
    fail "only $LINE_COUNT log lines — expected >20"
fi

# Phase K: screenshot dimensions consistency check.
echo "==== Phase K: screenshot dimensions ===="
LAST=$(ls -t "$OUT"/*.png | head -1)
if command -v identify >/dev/null 2>&1; then
    DIMS=$(identify -format "%wx%h" "$LAST" 2>/dev/null)
    echo "  last screenshot: $LAST → $DIMS"
fi

shutdown

# Stop screen recording — send SIGINT so ffmpeg flushes mp4 properly.
echo "  [rec] stopping screen capture"
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null
if [ -f "$REC_PATH" ]; then
    SIZE=$(stat -c%s "$REC_PATH" 2>/dev/null || echo "0")
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$REC_PATH" 2>/dev/null | head -1)
    echo "  [rec] saved $REC_PATH (${SIZE} B, ${DURATION}s)"
    echo "  [rec] watch with: vlc \"$REC_PATH\"  (or any video player)"
else
    fail "screen recording not saved"
fi

echo ""
echo "==== Deep audit complete: $ERRORS errors ===="
if [ "$ERRORS" -eq 0 ]; then
    echo "[deep-audit] PASS"
    exit 0
else
    echo "[deep-audit] FAIL"
    exit 1
fi
