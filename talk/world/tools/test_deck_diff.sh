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
# union ID — a run's differing beats ∪ differing notes, sorted, comma-joined
union() { jq -r '[.differing_beats[],.notes_differing[]]|unique|join(",")' "$TMP/$1/report.json" 2>/dev/null; }
# plus LIST... / minus LIST DROP — comma-list set arithmetic, sorted like jq's unique
plus() { printf '%s\n' "$@" | tr ',' '\n' | rg -v '^$' | LC_ALL=C sort -u | paste -sd, -; }
minus() { printf '%s\n' "$1" | tr ',' '\n' | rg -vxF -f <(printf '%s\n' "$2" | tr ',' '\n') | paste -sd, -; }

# Beats that world/ changed on purpose after the extraction, so world/ differs from the monolith Deck there.
# Ticket 08 (Deck-claim wording fixes, 3677113): the union of differing_beats and notes_differing in
# mrs-talk-world/evidence/08/{chromium,firefox}.report.json (both equal to their .allowed), via
#   jq -r '[.differing_beats[],.notes_differing[]]|unique|join(",")'
# Later Deck edits (ticket 10, S09 legend first) append their beats here. W1/F1 demand EQUALITY with this list,
# so a lost edit fails as surely as a stray one.
DECK_EDITS=S02b4,S02b5,S03b4,S04b3,S04b4,S05b2,S05b3,S05b4,S05b5,S06b3,S06b4,S07b4,S08b1,S08b2,S08b3,S08b4,S09b1,S09b2,S10b4,S10b5,S11b2,S11b3,S11b4,S12b2,S12b3,S12b4,S13b0,S13b1,S13b2,S13b3,S14b2,S14b3,S14b4

THANKS_FROM='>Thank you <span'; THANKS_TO='>Thank yoo <span'
# the world/ negative control lives in S15, which DECK_EDITS does not touch (S14b4, T1's target, is in it)
REPT_FROM='Phys. Rept. 317'; REPT_TO='Phys. Rept. 318'

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
  cp -r "$W" "$TMP/w_s15"; mutate "$W/js/s15-references.js" "$TMP/w_s15/js/s15-references.js" "$REPT_FROM" "$REPT_TO"
  cp -r "$W" "$TMP/w_nofont"; rm "$TMP/w_nofont/fonts/P052-Italic.woff2"
fi

if [ "${1:-}" = "--world" ]; then
  # W1 the acceptance gate: rebuilt page == Deck on all 129 frames and all notes except exactly DECK_EDITS
  diff_run W1 "$DECK" "$W/index.html" --allow "$DECK_EDITS"
  check "W1 exit 0" 0 "$rc"
  check "W1 129 frames, nothing unexpected, no errors" '129 [] [] []' \
    "$(rep W1 '.frames') $(rep W1 '.unexpected') $(rep W1 '.errors.a') $(rep W1 '.errors.b')"
  check "W1 differs from the Deck exactly on DECK_EDITS" "$DECK_EDITS" "$(union W1)"
  check "W1 audio ok on both" '"ok" "ok"' "$(rep W1 '.audio.a') $(rep W1 '.audio.b')"
  # W2 the negative control on the source folder: one word in js/s15 → exactly S15b1 on top of DECK_EDITS
  diff_run W2 "$DECK" "$TMP/w_s15/index.html" --allow "$DECK_EDITS"
  check "W2 exit 1" 1 "$rc"
  check "W2 unexpected == [S15b1], union == DECK_EDITS + S15b1" "[\"S15b1\"] $(plus "$DECK_EDITS" S15b1)" "$(rep W2 '.unexpected') $(union W2)"
  # W3 ticket 08's mode: the changed beat is named in --allow → pass, still reported
  diff_run W3 "$DECK" "$TMP/w_s15/index.html" --allow "$DECK_EDITS,S15b1"
  check "W3 exit 0" 0 "$rc"
  check "W3 unexpected == [], union == DECK_EDITS + S15b1" "[] $(plus "$DECK_EDITS" S15b1)" "$(rep W3 '.unexpected') $(union W3)"
  # W8 why W1 demands equality: undo ticket 08's S06 label edit and the run still exits 0 with nothing unexpected,
  # but its union loses S06b3,S06b4, which W1's equality check would flag
  cp -r "$W" "$TMP/w_rev"; mutate "$W/js/s06-mu-r.js" "$TMP/w_rev/js/s06-mu-r.js" \
    ' 2 GeV → fixed order (Fig. 7: frozen <i>α<sub>s</sub></i>)' ' = 2 GeV → fixed order'
  diff_run W8 "$DECK" "$TMP/w_rev/index.html" --allow "$DECK_EDITS"
  check "W8 a lost edit passes --allow alone (exit 0, unexpected [])" '0 []' "$rc $(rep W8 '.unexpected')"
  check "W8 but its union is DECK_EDITS minus S06b3,S06b4 (≠ DECK_EDITS)" "$(minus "$DECK_EDITS" S06b3,S06b4)" "$(union W8)"
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
  diff_run F1 "$DECK" "$W/index.html" --browser firefox --allow "$DECK_EDITS"
  check "F1 exit 0" 0 "$rc"
  check "F1 firefox, 129 frames, nothing unexpected, 57 beats" '"firefox" 129 [] 57' \
    "$(rep F1 '.browser') $(rep F1 '.frames') $(rep F1 '.unexpected') $(rep F1 '.inventory.b.beats')"
  check "F1 differs from the Deck exactly on DECK_EDITS" "$DECK_EDITS" "$(union F1)"
  diff_run F2 "$DECK" "$TMP/w_s15/index.html" --browser firefox --allow "$DECK_EDITS"
  check "F2 exit 1" 1 "$rc"
  check "F2 unexpected == [S15b1], union == DECK_EDITS + S15b1" "[\"S15b1\"] $(plus "$DECK_EDITS" S15b1)" "$(rep F2 '.unexpected') $(union F2)"
  diff_run F3 "$DECK" "$TMP/w_nofont/index.html" --browser firefox
  check "F3 exit 2" 2 "$rc"
  check "F3 names FONT_NOT_LOADED" 1 "$(printf '%s' "$out" | rg -c FONT_NOT_LOADED)"
fi

rm -rf "$TMP"
exit $fail
