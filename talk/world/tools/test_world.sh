#!/usr/bin/env bash
# Refutation-first checks for the World shell (ticket 04). Same style as test_deck_diff.sh: each check states its
# criterion, runs the real tools, and every negative control is a deliberately broken temp copy that must go red.
#   bash talk/world/tools/test_world.sh --unit | --chromium | --firefox | --stock
#   EVIDENCE=<dir> keeps logs and report.json copies.
set -u
export NODE_PATH="${NODE_PATH:-$(npm root -g)}"
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(dirname "$HERE")
TMP=$(mktemp -u /tmp/ticket04.XXXXXX); mkdir -p "$TMP"
fail=0
# ticket 06: S04 and S05 are rebuilt in world.html, so exactly these frames may differ from the Deck (index.html)
REBUILT=S04b0,S04b1,S04b2,S04b3,S04b4,S05b0,S05b1,S05b2,S05b3,S05b4,S05b5
E12='["S04b3","S04b4"]' # notes that differ on purpose: the rebuilt S04 carries story edits E1 + E2
say() { printf '%s\n' "$*"; }
check() { local name=$1 want=$2 got=$3; if [ "$want" = "$got" ]; then say "PASS  $name"; else say "FAIL  $name (want=$want got=$got)"; fail=1; fi; }
# mutate IN OUT FROM TO — copy IN to OUT with FROM→TO; FROM must occur exactly once in IN
mutate() { check "target occurs once in $(basename "$1"): $3" 1 "$(rg -oF -- "$3" "$1" | wc -l)"; sd -F -- "$3" "$4" < "$1" > "$2"; }

if [ "${1:-}" = "--unit" ]; then
  # U1 the unit suite is green on the real module
  out=$(node --test --test-reporter=tap "$HERE/test_world_core.mjs" 2>&1); rc=$?
  check "U1 unit suite green" "0 # fail 0" "$rc $(printf '%s' "$out" | rg '^# fail')"
  [ -n "${EVIDENCE:-}" ] && { mkdir -p "$EVIDENCE"; printf '%s\n' "$out" > "$EVIDENCE/unit.tap.txt"; }
  # U2–U5 each mutation of world-core.js must turn the suite red
  i=0
  while IFS='|' read -r what from to; do
    i=$((i + 1)); m="$TMP/core$i.js"
    mutate "$W/js/world-core.js" "$m" "$from" "$to"
    node -e 'require(process.argv[1])' "$m" 2>/dev/null; check "U$((i + 1)) mutant still loads (red is behaviour, not a crash)" 0 "$?"
    WORLD_CORE="$m" node --test --test-reporter=tap "$HERE/test_world_core.mjs" >/dev/null 2>&1
    check "U$((i + 1)) mutation '$what' is caught" 1 "$?"
  done <<'EOF'
drop PageDown|PageDown: 'next', |
wrong notes index|sc.notes[k - 1]|sc.notes[k]
no stop at the next mark|{ op: 'play', to: marks[cur.k] }|{ op: 'play', to: marks[cur.k + 1] }
0-based backup parameter|.map(x => x - 1)|.map(x => x)
EOF
  # U1b the rebuilt scenes' pure core (ticket 06) is green; U6-U13 each mutation of rebuild-core.js turns it red
  out=$(node --test --test-reporter=tap "$HERE/test_rebuild_core.mjs" 2>&1); rc=$?
  check "U1b rebuild core suite green" "0 # fail 0" "$rc $(printf '%s' "$out" | rg '^# fail')"
  [ -n "${EVIDENCE:-}" ] && printf '%s\n' "$out" > "$EVIDENCE/unit.rebuild.tap.txt"
  i=5
  while IFS='|' read -r what from to; do
    i=$((i + 1)); m="$TMP/rcore$i.js"
    mutate "$W/js/rebuild-core.js" "$m" "$from" "$to"
    node -e 'require(process.argv[1])' "$m" 2>/dev/null; check "U$i mutant still loads" 0 "$?"
    REBUILD_CORE="$m" node --test --test-reporter=tap "$HERE/test_rebuild_core.mjs" >/dev/null 2>&1
    check "U$i mutation '$what' is caught" 1 "$?"
  done <<'EOF'
LO overlaps NLO|LO: [0.18, 0.05, 0.05]|LO: [0.36, 0.05, 0.05]
J hazy|0.45)), J: 0 }|0.45)), J: 0.2 }
camera jumps in S04 b4|seg(prog[3], 0.55, 1)|seg(prog[3], 0.55, 0.56)
E1 typo|at u equals one half.'|at u equal one half.'
3D stops before the glide ends|!want && on && !gliding ?|!want && on ?
shadow replay runs every time|if (!done) { done = true; f(); }|{ f(); }
remainder drawn past l = 2|const REM = [0.23, -0.14, 0.01];|const REM = [0.23, -0.14, 0.01, 0.05];
easing copy drifts from the Deck|return t < 0.5 ? 4 * t * t * t|return t < 0.5 ? 3 * t * t * t
band flies straight through the 𝒥 box|if (k <= h - R) return { x: a[0] + sx * k, y: a[1], scale };|return { x: a[0] + (b[0] - a[0]) * t, y: a[1] + (b[1] - a[1]) * t, scale };
band keeps its old size|const scale = 1 - 0.65 * t;|const scale = 1 - 0.45 * t;
band rises against the "} +"|function bandFlight(a, b, t, xc = b[0]) {|function bandFlight(a, b, t, xc) { xc = b[0];
EOF
fi

# diff ID A B [extra args] — runs deck_diff, sets rc/out, report at $TMP/ID/report.json
diff_run() { local id=$1 a=$2 b=$3; shift 3
  out=$(node "$HERE/deck_diff.mjs" --a "$a" --b "$b" --out "$TMP/$id" "$@" 2>&1); rc=$?
  say "  [$id] rc=$rc $(printf '%s' "$out" | tail -1)"
  if [ -n "${EVIDENCE:-}" ] && [ -f "$TMP/$id/report.json" ]; then mkdir -p "$EVIDENCE"; cp "$TMP/$id/report.json" "$EVIDENCE/$id.report.json"; fi; }
rep() { jq -c "$2" "$TMP/$1/report.json" 2>/dev/null; }
# suite ID [pattern] — the browser tests (env: BROWSER, WORLD_DIR, WORLD_PAGE, REQUIRE_HW_GL), sets rc/out
suite() { local id=$1; shift
  out=$(node --test --test-reporter=tap ${1:+--test-name-pattern="$1"} "$HERE/test_world_browser.mjs" 2>&1); rc=$?
  if [ -n "${EVIDENCE:-}" ]; then mkdir -p "$EVIDENCE"; printf '%s\n' "$out" > "$EVIDENCE/$id.tap.txt"; fi; }
fails() { printf '%s' "$out" | rg '^# fail' ; }

if [ "${1:-}" = "--chromium" ] || [ "${1:-}" = "--firefox" ]; then
  export BROWSER=${1#--}
  # B0 the browser suite (keys, offline, 3D layer, presenter, backup clip) on the real World
  suite "B0-$BROWSER"; check "B0 browser suite green ($BROWSER)" "0 # fail 0" "$rc $(fails)"
  # G1 hosted scenes look exactly as in the Deck: world.html vs index.html, all 129 frames, 0 px, notes equal;
  # the rebuilt S04/S05 (ticket 06) differ on exactly their own frames, no more and no fewer
  diff_run "G1-$BROWSER" "$W/index.html" "$W/world.html" --browser "$BROWSER" --allow "$REBUILT"
  check "G1 exit 0" 0 "$rc"
  # notes: exactly S04b3/S04b4, the rebuilt S04's story edits E1 + E2 (ticket 06); every other beat's notes equal the Deck
  check "G1 129 frames, differing = exactly the rebuilt S04/S05, notes = exactly S04b3,S04b4, no page errors" "129 \"$REBUILT\" $E12 [] []" \
    "$(rep "G1-$BROWSER" '.frames') $(rep "G1-$BROWSER" '.differing_beats | sort | join(",")') $(rep "G1-$BROWSER" '.notes_differing') $(rep "G1-$BROWSER" '.errors.a') $(rep "G1-$BROWSER" '.errors.b')"
  # G1h without the two rebuilt-scene scripts the World is the Deck again, 0 px (the build-level fallback, ticket 06)
  cp -r "$W" "$TMP/w_g1h"
  mutate "$W/world.html" "$TMP/w_g1h/world.html.1" '<script src="js/w04-renormalon.js"></script>' ''
  mutate "$TMP/w_g1h/world.html.1" "$TMP/w_g1h/world.html" '<script src="js/w05-mrs.js"></script>' ''
  diff_run "G1h-$BROWSER" "$W/index.html" "$TMP/w_g1h/world.html" --browser "$BROWSER"
  check "G1h rebuilt scripts removed: 129 frames, 0 differing, 0 notes" "0 129 [] []" "$rc $(rep "G1h-$BROWSER" '.frames') $(rep "G1h-$BROWSER" '.differing_frames') $(rep "G1h-$BROWSER" '.notes_differing')"
  # G7 control: 3D left on after leaving the rebuilt scenes must show up on a hosted frame
  cp -r "$W" "$TMP/w_g7"; mutate "$W/js/rebuild-space.js" "$TMP/w_g7/js/rebuild-space.js" 'const { cam, presence } = RC.blend(parts);' 'const { cam, presence } = RC.blend(parts.length ? parts : [{ w: 1, cam: RC.VIEW.s05 }]);'
  diff_run "G7-$BROWSER" "$W/index.html" "$TMP/w_g7/world.html" --browser "$BROWSER" --allow "$REBUILT"
  check "G7 control: 3D left on is caught on S06b0" "1 true" "$rc $(rep "G7-$BROWSER" '.unexpected | index("S06b0") != null')"
  # G5 control: the backup overlay left on must fail G1
  cp -r "$W" "$TMP/w_g5"; mutate "$W/world.html" "$TMP/w_g5/world.html" '<div id="backup"></div>' '<div id="backup" class="on"></div>'
  diff_run "G5-$BROWSER" "$W/index.html" "$TMP/w_g5/world.html" --browser "$BROWSER" --allow "$REBUILT"
  check "G5 overlay left on is caught (exit 1)" 1 "$rc"
  # K2 control: the Deck itself starts its narrated run on A, so K2 must fail on index.html
  WORLD_PAGE=index.html suite "K2neg-$BROWSER" '^K2'; check "K2 control: fails on the Deck (index.html)" 1 "$rc"
  # O1 controls: one outside URL in the page, one in the CSS
  cp -r "$W" "$TMP/w_o1"; mutate "$W/world.html" "$TMP/w_o1/world.html" '<div id="backup"></div>' '<div id="backup"></div><img src="http://127.0.0.1:9/x.png" style="display:none">'
  WORLD_DIR="$TMP/w_o1" suite "O1neg-$BROWSER" '^O1'; check "O1 control: an http:// image is caught" 1 "$rc"
  cp -r "$W" "$TMP/w_o1c"; printf '%s\n' '@font-face{font-family:X;src:url(https://example.invalid/x.woff2)}' >> "$TMP/w_o1c/world.css"
  WORLD_DIR="$TMP/w_o1c" suite "O1cneg-$BROWSER" '^O1'; check "O1 control: an https:// font in the CSS is caught" 1 "$rc"
  # X3 control: stop() that leaves the canvas on must fail X3's "page as it was"
  cp -r "$W" "$TMP/w_x3"; mutate "$W/js/world-space.js" "$TMP/w_x3/js/world-space.js" "cv.classList.remove('on')" 'void 0'
  WORLD_DIR="$TMP/w_x3" suite "X3neg-$BROWSER" '^X3'; check "X3 control: a canvas left on after stop() is caught" 1 "$rc"
  # P1 control: a World that never sends its state must leave the presenter blank
  cp -r "$W" "$TMP/w_p1"; mutate "$W/js/world.js" "$TMP/w_p1/js/world.js" "if (o && !o.closed) o.postMessage(state(), '*');" 'void o;'
  WORLD_DIR="$TMP/w_p1" suite "P1neg-$BROWSER" '^P1'; check "P1 control: a silent World is caught" 1 "$rc"
  # B1/B4 controls: a clip that is not preloaded, and one that does not stop at its mark, must be caught
  cp -r "$W" "$TMP/w_b1"; mutate "$W/js/world.js" "$TMP/w_b1/js/world.js" "preload: 'auto'" "preload: 'none'"
  WORLD_DIR="$TMP/w_b1" suite "B1neg-$BROWSER" '^B1'; check "B1 control: an unpreloaded clip is caught" 1 "$rc"
  cp -r "$W" "$TMP/w_b4"; mutate "$W/js/world.js" "$TMP/w_b4/js/world.js" 'if (stopAt !== null && v.currentTime >= stopAt)' 'if (false)'
  WORLD_DIR="$TMP/w_b4" suite "B4neg-$BROWSER" '^B4'; check "B4 control: a clip that plays past its mark is caught" 1 "$rc"
  # B0b the rebuilt S04/S05 (ticket 06): dark, 3D lifecycle, seams, crisp branch point, notes, reveal variants, no WebGL,
  # hosted scenes untouched, content boxed, off screen not rendered, and the backup clips (BR)
  out=$(BR=1 node --test --test-reporter=tap "$HERE/test_rebuild_browser.mjs" 2>&1); rc=$?
  [ -n "${EVIDENCE:-}" ] && printf '%s\n' "$out" > "$EVIDENCE/B0b-$BROWSER.tap.txt"
  check "B0b rebuilt-scene suite green ($BROWSER)" "0 # fail 0" "$rc $(fails)"
  # R6 control: without the hosted reveal replay, later hosted scenes get other reveal variants
  cp -r "$W" "$TMP/w_r6"; mutate "$W/js/w04-renormalon.js" "$TMP/w_r6/js/w04-renormalon.js" 'f(p) { shadow(); prog[j] = p;' 'f(p) { prog[j] = p;'
  out=$(WORLD_DIR="$TMP/w_r6" node --test --test-reporter=tap --test-name-pattern=R6 "$HERE/test_rebuild_browser.mjs" 2>&1); rc=$?
  check "R6 control: the reveal replay removed is caught (R6 itself fails, not a crash)" "1 1" "$rc $(printf '%s' "$out" | rg -c '^not ok \d+ - R6 ')"
  # S1 the capture shim still drives the World
  out=$(node /home/gofer/mrs_talk_video/render/deck_shim.mjs --selftest "$W/world.html" 2>&1); rc=$?
  check "S1 shim selftest on world.html" "0 SHIM OK scenes=15 beats=57 sum_d=1014 per_scene=[2,5,4,4,5,4,4,4,4,5,4,4,3,4,1]" "$rc $out"
  if [ "$BROWSER" = chromium ]; then # extract_deck_text.mjs is Chromium-only
    node /home/gofer/mrs_talk_video/render/extract_deck_text.mjs "$W/index.html" "$TMP/text.index.md" >/dev/null 2>&1
    node /home/gofer/mrs_talk_video/render/extract_deck_text.mjs "$W/world.html" "$TMP/text.world.md" >/dev/null 2>&1
    # T1 every scene but the rebuilt S04/S05 reads exactly as in the Deck; the rebuilt ones show exactly the approved words
    sections() { awk '/^## S[0-9][0-9] /{keep = ($2 != "S04" && $2 != "S05")} keep' "$1"; }
    cmp -s <(sections "$TMP/text.index.md") <(sections "$TMP/text.world.md"); rc=$?
    check "T1 extract_deck_text: world == index outside S04/S05 (13 sections each)" "0 13 13" \
      "$rc $(sections "$TMP/text.index.md" | rg -c '^## S') $(sections "$TMP/text.world.md" | rg -c '^## S')"
    out=$(node "$HERE/check_rebuilt_text.mjs" --page world.html --narration /home/gofer/mrs_talk_video/narration_video 2>&1); rc=$?
    check "T1b rebuilt S04/S05: approved words visible, none forbidden, notes == narration" 0 "$rc"
    out=$(node "$HERE/check_rebuilt_text.mjs" --page index.html 2>&1); rc=$?
    check "T1b control: the Deck's S04/S05 fail the rebuilt word list (TEXT FAIL, not a crash)" "1 TEXT FAIL" "$rc $(printf '%s' "$out" | tail -1 | cut -d' ' -f1-2)"
  fi
fi

if [ "${1:-}" = "--stock" ]; then
  say "screen takeover starting: real Chrome and Firefox windows for ~15 min; hands off, on AC power"
  for key in chrome firefox-stock; do
    export BROWSER=$key
    type=$(jq -r --arg k "$key" '.[$k].type' "$HERE/stage_browsers.json")
    bin=$([ "$key" = chrome ] && echo google-chrome || jq -r --arg k "$key" '.[$k].launch.executablePath' "$HERE/stage_browsers.json")
    installed=$("$bin" --version | awk '{print $3}')
    export DECK_DIFF_LAUNCH; DECK_DIFF_LAUNCH=$(jq -c --arg k "$key" '.[$k].launch' "$HERE/stage_browsers.json")
    # N0 noise floor: the stock browser against itself must be 0 px, or its G1 could not be read
    diff_run "N0-$key" "$W/index.html" "$W/index.html" --browser "$type"
    check "N0 $key noise floor: index vs index, 129 frames, 0 px" "0 129 []" "$rc $(rep "N0-$key" '.frames') $(rep "N0-$key" '.differing_frames')"
    check "N0 $key ran the installed browser" "\"$installed\"" "$(rep "N0-$key" '.browser_version')"
    # G1 on the stage browser: hosted scenes == Deck
    diff_run "G1-$key" "$W/index.html" "$W/world.html" --browser "$type" --allow "$REBUILT"
    check "G1 $key: 129 frames, differing = exactly the rebuilt S04/S05, notes = exactly S04b3,S04b4" "0 129 \"$REBUILT\" $E12" \
      "$rc $(rep "G1-$key" '.frames') $(rep "G1-$key" '.differing_beats | sort | join(",")') $(rep "G1-$key" '.notes_differing')"
    unset DECK_DIFF_LAUNCH
    # popup blocker on, installed binary, presenter pairing, real-GPU 3D, backup swap
    REQUIRE_HW_GL=1 suite "stock-$key" '^(N1|N2|P1|P2|P3|X3|B3|B4) '
    check "stock suite $key: N1 N2 P1 P2 P3 X3 B3 B4 all ok" "0 # fail 0 8" \
      "$rc $(fails) $(printf '%s' "$out" | rg -v SKIP | rg -c '^ok [0-9]+ - (N1|N2|P1|P2|P3|X3|B3|B4) ')"
  done
fi

rm -rf "$TMP"
exit $fail
