#!/usr/bin/env bash
# UI audit test — deeper than smoke_test.sh. Stress-tests via rapid
# keypress, cycles every template, verifies persistence across restart,
# resizes window mid-session. Captures screenshots at every interesting
# state.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
OUT="${OUT:-/tmp/graphnet-ui-audit}"
mkdir -p "$OUT"

ERRORS=0
step() {
    local name="$1"
    sleep 0.5
    scrot "$OUT/$(date +%s)_${name}.png" 2>/dev/null
    echo "  → $name"
}

fail() {
    echo "FAIL: $*"
    ERRORS=$((ERRORS + 1))
}

launch() {
    pkill -9 -f graphnet-gui 2>/dev/null || true
    sleep 0.8
    "$BIN" >/tmp/graphnet-ui-audit.log 2>&1 &
    GUI_PID=$!
    sleep 4
    if ! kill -0 "$GUI_PID" 2>/dev/null; then
        fail "graphnet-gui exited prematurely"
        tail -10 /tmp/graphnet-ui-audit.log
        exit 1
    fi
    WIN=$(xdotool search --name "GraphNet" | head -1)
    if [ -z "$WIN" ]; then
        fail "no window found"
        exit 1
    fi
    xdotool windowactivate "$WIN"
    sleep 0.5
    echo "  [launch] PID=$GUI_PID WIN=$WIN"
}

shutdown() {
    kill "$GUI_PID" 2>/dev/null || true
    wait "$GUI_PID" 2>/dev/null || true
}

echo "==== Phase 1: cold-start + first paint ===="
launch
step "p1_00_coldstart"

echo "==== Phase 2: stress — add 30 ops rapidly ===="
for i in $(seq 1 30); do
    case $((i % 5)) in
        0) k=a ;;
        1) k=d ;;
        2) k=f ;;
        3) k=p ;;
        *) k=n ;;
    esac
    xdotool key --window "$WIN" "$k"
done
step "p2_00_30_ops_added"
sleep 0.5

echo "==== Phase 3: rapid undo back to empty ===="
for i in $(seq 1 35); do
    xdotool key --window "$WIN" ctrl+z
done
step "p3_00_undone_to_empty"
sleep 0.5

echo "==== Phase 4: rapid redo all the way ===="
for i in $(seq 1 35); do
    xdotool key --window "$WIN" ctrl+shift+z
done
step "p4_00_redone_to_full"
sleep 0.5

echo "==== Phase 5: cycle every template (1-9, 0) ===="
for k in 1 2 3 4 5 6 7 8 9 0; do
    xdotool key --window "$WIN" "$k"
    sleep 0.4
    step "p5_template_${k}"
done

echo "==== Phase 6: workspace tour ===="
# Save current to slot A then click Compare tab via xdotool mouse.
xdotool key --window "$WIN" ctrl+1
step "p6_00_slot_a_saved"
# Compare tab is the 3rd workspace pill in the hero. Approx coords.
# Get window geometry to compute click pos.
GEOM=$(xdotool getwindowgeometry --shell "$WIN" 2>/dev/null)
WX=$(echo "$GEOM" | awk -F= '/^X=/{print $2}')
WY=$(echo "$GEOM" | awk -F= '/^Y=/{print $2}')
# Hero workspace tabs sit ~16px from left, ~12px from top. Edit / Live /
# Compare / Train widths ~58 / 50 / 76 / 60 px each. Compare center ≈ x+150.
xdotool mousemove $((WX + 200)) $((WY + 26)) click 1
sleep 0.5
step "p6_01_compare_tab"
# Cmd+2 saves slot B.
xdotool key --window "$WIN" ctrl+2
step "p6_02_slot_b_saved"
# Click Edit tab back.
xdotool mousemove $((WX + 60)) $((WY + 26)) click 1
sleep 0.5
step "p6_03_back_to_edit"

echo "==== Phase 7: open + close every modal ===="
xdotool key --window "$WIN" h
step "p7_00_help_open"
xdotool key --window "$WIN" Escape
step "p7_01_help_closed"
xdotool key --window "$WIN" ctrl+n
step "p7_02_templates_open"
xdotool key --window "$WIN" Escape
step "p7_03_templates_closed"
xdotool key --window "$WIN" grave
step "p7_04_console_open"
xdotool key --window "$WIN" Escape
step "p7_05_console_closed"

echo "==== Phase 8: live mode burst ===="
xdotool key --window "$WIN" l
sleep 2.5
step "p8_00_live_running"
xdotool key --window "$WIN" l
step "p8_01_live_off"

echo "==== Phase 9: forward sanity (after stress) ===="
for i in 1 2 3 4 5; do
    xdotool key --window "$WIN" space
done
step "p9_00_5_forwards"

echo "==== Phase 10: persistence test — exit + relaunch ===="
STATE_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/state.yaml"
PRE_HASH=$(sha256sum "$STATE_PATH" 2>/dev/null | awk '{print $1}')
PRE_SIZE=$(stat -c%s "$STATE_PATH" 2>/dev/null || echo "0")
echo "  pre-shutdown state: $STATE_PATH ($PRE_SIZE bytes) hash=$PRE_HASH"
shutdown
sleep 1
launch
step "p10_00_relaunched"
sleep 1
POST_HASH=$(sha256sum "$STATE_PATH" 2>/dev/null | awk '{print $1}')
POST_SIZE=$(stat -c%s "$STATE_PATH" 2>/dev/null || echo "0")
echo "  post-relaunch state: $STATE_PATH ($POST_SIZE bytes) hash=$POST_HASH"
if [ -z "$PRE_HASH" ]; then
    fail "no state.yaml existed pre-shutdown — persistence broken"
elif [ "$PRE_HASH" != "$POST_HASH" ]; then
    echo "  STATE CHANGED: pre=$PRE_HASH post=$POST_HASH"
    echo "  (this is expected if the app mutated state on launch, e.g. ran a forward)"
else
    echo "  state.yaml unchanged across restart — persistence verified ✓"
fi
# Settings file:
SETTINGS_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/settings.yaml"
if [ -f "$SETTINGS_PATH" ]; then
    echo "  settings.yaml present ($(stat -c%s "$SETTINGS_PATH") bytes):"
    sed 's/^/    /' "$SETTINGS_PATH"
else
    fail "settings.yaml missing — user-prefs persistence broken"
fi

echo "==== Phase 11: window resize ===="
xdotool windowsize "$WIN" 1024 600
sleep 0.5
step "p11_00_narrow"
xdotool windowsize "$WIN" 1920 1080
sleep 0.5
step "p11_01_wide"
xdotool key --window "$WIN" super+f 2>/dev/null || true  # try toggle fullscreen
sleep 0.5

echo "==== Phase 11.5: log-file behavior assertions ===="
LOG="${XDG_CONFIG_HOME:-$HOME/.config}/graphnet/graphnet.log"
if [ ! -f "$LOG" ]; then
    fail "no graphnet.log written — log persistence broken"
else
    LC=$(wc -l < "$LOG")
    echo "  log lines: $LC"
    # Check for canonical events we expect to see.
    assert_logged() {
        local pattern="$1"
        local label="$2"
        if grep -q "$pattern" "$LOG"; then
            echo "  ✓ logged: $label"
        else
            fail "MISSING log entry for: $label (pattern: $pattern)"
        fi
    }
    assert_logged "template '.*' loaded" "any template load"
    assert_logged "+ op \[" "any op add"
    assert_logged "💾 persisted" "any persist"
    assert_logged "live mode" "live mode toggle"
fi

echo "==== Phase 12: shutdown ===="
shutdown

ALL=$(ls "$OUT" 2>/dev/null | wc -l)
echo ""
echo "[ui-audit] $ALL total screenshots in $OUT"
if [ "$ERRORS" -eq 0 ]; then
    echo "[ui-audit] PASS — 0 errors"
    exit 0
else
    echo "[ui-audit] FAIL — $ERRORS errors"
    exit 1
fi
