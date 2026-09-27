#!/usr/bin/env bash
# Refutation-first checks for talk/world (ticket 02). Each check states its criterion and runs the real
# tools on the real Deck, on the rebuilt world/, and on deliberately broken copies of both.
#   bash talk/world/tools/test_deck_diff.sh --tool | --world | --firefox
#   EVIDENCE=<dir> keeps a copy of each run's report.json as <id>.report.json.
set -u
export NODE_PATH="${NODE_PATH:-$(npm root -g)}"
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(dirname "$HERE")
DECK=$(dirname "$W")/mrs-talk-animated.html
DIFF="$HERE/deck_diff.mjs"
TMP=$(mktemp -u /tmp/ticket02.XXXXXX); mkdir -p "$TMP"
fail=0
say() { printf '%s\n' "$*"; }
check() { local name=$1 want=$2 got=$3; if [ "$want" = "$got" ]; then say "PASS  $name"; else say "FAIL  $name (want=$want got=$got)"; fail=1; fi; }
# mutate IN OUT FROM TO — copy IN to OUT with FROM→TO; FROM must occur exactly once in IN
mutate() { check "target occurs once in $(basename "$1"): $3" 1 "$(rg -oF -- "$3" "$1" | wc -l)"; sd -F -- "$3" "$4" < "$1" > "$2"; }
# diff ID A B [extra args] — runs deck_diff, sets rc/out, report at $TMP/ID/report.json
diff_run() { local id=$1 a=$2 b=$3; shift 3
  out=$(node "$DIFF" --a "$a" --b "$b" --out "$TMP/$id" "$@" 2>&1); rc=$?
  say "  [$id] rc=$rc $(printf '%s' "$out" | tail -1)"
  if [ -n "${EVIDENCE:-}" ] && [ -f "$TMP/$id/report.json" ]; then mkdir -p "$EVIDENCE"; cp "$TMP/$id/report.json" "$EVIDENCE/$id.report.json"; fi; }
rep() { jq -c "$2" "$TMP/$1/report.json" 2>/dev/null; }

THANKS_FROM='>Thank you <span'; THANKS_TO='>Thank yoo <span'

if [ "${1:-}" = "--tool" ]; then
  # T0 noise floor: the Deck against itself must be pixel- and notes-identical on every frame
  diff_run T0 "$DECK" "$DECK"
  check "T0 exit 0" 0 "$rc"
  check "T0 129 frames, 0 differing" '129 []' "$(rep T0 '.frames') $(rep T0 '.differing_frames')"
  # T1 one on-screen word changed in S14 → exactly that beat differs, no notes differ
  mutate "$DECK" "$TMP/t1.html" "$THANKS_FROM" "$THANKS_TO"
  diff_run T1 "$DECK" "$TMP/t1.html"
  check "T1 exit 1" 1 "$rc"
  check "T1 differing_beats == [S14b4], notes_differing == []" '["S14b4"] []' "$(rep T1 '.differing_beats') $(rep T1 '.notes_differing')"
  # T2 one word changed in an S07 speaker note (not on screen) → exactly that note differs, no pixels
  mutate "$DECK" "$TMP/t2.html" 'evolve in imaginary time to project' 'evolve in imaginary TIME to project'
  diff_run T2 "$DECK" "$TMP/t2.html"
  check "T2 exit 1" 1 "$rc"
  check "T2 notes_differing == [S07b2], differing_beats == []" '["S07b2"] []' "$(rep T2 '.notes_differing') $(rep T2 '.differing_beats')"
  # T3 Firefox noise floor: the Deck against itself in Firefox, and the shim drives it (15/57/1014)
  diff_run T3 "$DECK" "$DECK" --browser firefox
  check "T3 exit 0" 0 "$rc"
  check "T3 firefox, 129 frames, 0 differing" '"firefox" 129 []' "$(rep T3 '.browser') $(rep T3 '.frames') $(rep T3 '.differing_frames')"
  check "T3 inventory 15/57/1014" '[15,57,1014]' "$(rep T3 '[.inventory.a.scenes,.inventory.a.beats,.inventory.a.sum_d]')"
fi

# broken copies of world/ shared by --world and --firefox
if [ "${1:-}" = "--world" ] || [ "${1:-}" = "--firefox" ]; then
  cp -r "$W" "$TMP/w_s14"; mutate "$W/js/s14-conclusions.js" "$TMP/w_s14/js/s14-conclusions.js" "$THANKS_FROM" "$THANKS_TO"
  cp -r "$W" "$TMP/w_nofont"; rm "$TMP/w_nofont/fonts/P052-Italic.woff2"
fi

if [ "${1:-}" = "--world" ]; then
  # W1 the acceptance gate: rebuilt page == Deck on all 129 frames and all notes, audio loads on both
  diff_run W1 "$DECK" "$W/index.html"
  check "W1 exit 0" 0 "$rc"
  check "W1 129 frames, 0 differing, 0 notes, no errors" '129 [] [] [] []' \
    "$(rep W1 '.frames') $(rep W1 '.differing_frames') $(rep W1 '.notes_differing') $(rep W1 '.errors.a') $(rep W1 '.errors.b')"
  check "W1 audio ok on both" '"ok" "ok"' "$(rep W1 '.audio.a') $(rep W1 '.audio.b')"
  # W2 the negative control on the source folder: one word in js/s14 → exactly S14b4
  diff_run W2 "$DECK" "$TMP/w_s14/index.html"
  check "W2 exit 1" 1 "$rc"
  check "W2 differing_beats == [S14b4]" '["S14b4"]' "$(rep W2 '.differing_beats')"
  # W3 ticket 08's mode: the changed beat is named in --allow → pass, still reported
  diff_run W3 "$DECK" "$TMP/w_s14/index.html" --allow S14b4
  check "W3 exit 0" 0 "$rc"
  check "W3 differing_beats == [S14b4], unexpected == []" '["S14b4"] []' "$(rep W3 '.differing_beats') $(rep W3 '.unexpected')"
  # W4 a missing font file is a setup error, never a quiet fallback-font render
  diff_run W4 "$DECK" "$TMP/w_nofont/index.html"
  check "W4 exit 2" 2 "$rc"
  check "W4 names FONT_NOT_LOADED" 1 "$(printf '%s' "$out" | rg -c FONT_NOT_LOADED)"
  # W5 the capture shim still drives the rebuilt page
  out=$(node /home/gofer/mrs_talk_video/render/deck_shim.mjs --selftest "$W/index.html" 2>&1); rc=$?
  check "W5 shim selftest" "0 SHIM OK scenes=15 beats=57 sum_d=1014 per_scene=[2,5,4,4,5,4,4,4,4,5,4,4,3,4,1]" "$rc $out"
  # W6 the gitignored MP3 missing (fresh checkout) → setup error that says how to restore it
  cp -r "$W" "$TMP/w_nomp3"; rm "$TMP/w_nomp3/assets/narration.mp3"
  diff_run W6 "$DECK" "$TMP/w_nomp3/index.html"
  check "W6 exit 2" 2 "$rc"
  check "W6 names AUDIO_NOT_LOADED + the restore command" 2 "$(printf '%s' "$out" | rg -c -e AUDIO_NOT_LOADED -e '--only assets/narration.mp3' -o)"
  # W7 the restore writes only the MP3 and brings the copy back to world/ byte for byte
  out=$(python3 "$HERE/extract_deck.py" --only assets/narration.mp3 --out "$TMP/w_nomp3" 2>&1); rc=$?
  check "W7 restore" "0 EXTRACT OK written=1 same=0 files=1" "$rc $out"
  diff -r "$TMP/w_nomp3" "$W" >/dev/null; check "W7 copy == world/ after restore" 0 "$?"
fi

if [ "${1:-}" = "--firefox" ]; then
  # F1–F3 the same gate and controls in Firefox (subfolder fonts over file:// included)
  diff_run F1 "$DECK" "$W/index.html" --browser firefox
  check "F1 exit 0" 0 "$rc"
  check "F1 firefox, 129 frames, 0 differing, 0 notes, 57 beats" '"firefox" 129 [] [] 57' \
    "$(rep F1 '.browser') $(rep F1 '.frames') $(rep F1 '.differing_frames') $(rep F1 '.notes_differing') $(rep F1 '.inventory.b.beats')"
  diff_run F2 "$DECK" "$TMP/w_s14/index.html" --browser firefox
  check "F2 exit 1" 1 "$rc"
  check "F2 differing_beats == [S14b4]" '["S14b4"]' "$(rep F2 '.differing_beats')"
  diff_run F3 "$DECK" "$TMP/w_nofont/index.html" --browser firefox
  check "F3 exit 2" 2 "$rc"
  check "F3 names FONT_NOT_LOADED" 1 "$(printf '%s' "$out" | rg -c FONT_NOT_LOADED)"
fi

rm -rf "$TMP"
exit $fail
