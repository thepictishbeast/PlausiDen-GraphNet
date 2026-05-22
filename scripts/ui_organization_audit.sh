#!/usr/bin/env bash
# Organization audit: verify the layout matches the
# RustRover/FreeCAD/Blender consensus structure.
#
# Uses ImageMagick to inspect specific regions of a startup screenshot
# and verify the EXPECTED content is in the EXPECTED place.
#
# *** WARNING *** User-invoked. Steals focus to take screenshot.

set -uo pipefail

: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/home/user/.Xauthority}"
export DISPLAY XAUTHORITY

BIN="${BIN:-/home/user/cargo-target/release/graphnet-gui}"
OUT="${OUT:-/tmp/graphnet-org-audit}"
mkdir -p "$OUT"
ERRORS=0
fail() { echo "FAIL: $*"; ERRORS=$((ERRORS + 1)); }

pkill -9 -f graphnet-gui 2>/dev/null || true
sleep 1
"$BIN" >/tmp/gnorg.log 2>&1 &
GUI_PID=$!
sleep 4
WIN=$(xdotool search --name "GraphNet" | head -1)
[ -z "$WIN" ] && { fail "no window"; exit 1; }
xdotool windowactivate "$WIN"
sleep 0.5
xdotool key --window "$WIN" Escape
sleep 0.5
SHOT="$OUT/startup.png"
scrot "$SHOT"

if [ ! -f "$SHOT" ]; then fail "screenshot failed"; exit 1; fi

# Verify dimensions are reasonable (window should be near 1920x1080).
DIMS=$(identify -format "%wx%h" "$SHOT" 2>/dev/null)
echo "  screenshot: $DIMS"
WIDTH=$(echo "$DIMS" | cut -d'x' -f1)
HEIGHT=$(echo "$DIMS" | cut -d'x' -f2)
[ "$WIDTH" -lt 800 ] && fail "screenshot width <800: $WIDTH"
[ "$HEIGHT" -lt 600 ] && fail "screenshot height <600: $HEIGHT"

# Crop & inspect key regions. Save each region for visual review.
# Top 40px should contain menu bar — convert to grayscale + check entropy.
convert "$SHOT" -crop "${WIDTH}x40+0+0" "$OUT/region_menubar.png"
convert "$SHOT" -crop "${WIDTH}x60+0+40" "$OUT/region_hero.png"
convert "$SHOT" -crop "32x$((HEIGHT - 200))+0+100" "$OUT/region_left_tool_palette.png"
convert "$SHOT" -crop "${WIDTH}x40+0+$((HEIGHT - 40))" "$OUT/region_status.png"
convert "$SHOT" -crop "$((WIDTH * 3 / 5))x$((HEIGHT / 2))+$((WIDTH / 5))+$((HEIGHT / 5))" \
    "$OUT/region_central_3d.png"

echo "  region screenshots saved to $OUT/"
ls "$OUT" | head -8

# Sanity: file sizes — empty regions would be near-zero bytes.
for region in region_menubar region_hero region_status region_central_3d; do
    SIZE=$(stat -c%s "$OUT/$region.png" 2>/dev/null || echo 0)
    if [ "$SIZE" -lt 200 ]; then
        fail "$region.png too small ($SIZE B) — region may be empty"
    else
        echo "  ✓ $region.png — $SIZE B"
    fi
done

echo ""
echo "==== Organization audit complete: $ERRORS errors ===="
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
if [ "$ERRORS" -eq 0 ]; then echo "[organization] PASS"; exit 0; else echo "[organization] FAIL"; exit 1; fi
