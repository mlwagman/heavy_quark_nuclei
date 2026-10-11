# Round 18 tasks (plan: ~/.claude/plans/magical-riding-canyon.md)

Baseline r17 (2026-09-30): check-beats 53/53, check-jumps 0/57, check-colour 2131 neutral, check-text --mid 0 strong,
--frame --mid 0 focus cut, --layout clean, SHIM OK 15/57/1014. Snapshot: `.scratch/world3d-r17`.

- [x] Baseline, snapshot, narration_world3d copy → verify: hashes match, 13 js files in snapshot, 15 scripts copied
- [x] tools/check-transient.mjs (+ --selftest, --strings; a tool of its own: it needs the 30 fps virtual clock) → verify: selftest exit 0; r17 flags S03b1 S03b3 S05b2 S12b1 S10b2 S11b1-MARK, not S02b2
- [x] check-map.mjs → verify: r17 exit 1 listing 6/7, 2/8/9/10/14, 12/13 and the shelf in 2 places (its first glyph shifts between layouts), want 6
- [x] check_narration.py → verify: exit 1 on the copied scripts
- [x] audio3d.py notes [--check] → verify: --check fails before notes.js and after a one-char mutation
- [x] Splits, S05 Λ_R stop, S11 dots, notes wiring, test tuples, shim --want → verify: --transient 0; all step-3 checks green; SHIM OK 15/71/1014 (70 planned; the check also found S15 b1)
- [x] Re-layout (regions, travelling shelf, cameras, OVCAM) → verify: check-map 0 overlaps, shelf 6 places; step-3 checks green; strings diff empty
- [x] Narration rewrite → verify: check_narration exit 0, notes --check exit 0, deck-scripts hash unchanged
- [x] Voice + MP4 → verify: wpm 170-180, 0 gate failures, verify3d PASS (177 wpm, 0, PASS)
- [x] Review at full size, NOTES Round 18, build r18, ADR-001 → verify: independent verifier clean (third round: only stale map PNGs, re-rendered)

# Round 19 (2026-10-03): done
Panels 4, 5, 6, 9, 10, 12 after the author's review; details and numbers in NOTES.md "Round 19". Working tree of branch
`talk-world3d-r18`, uncommitted (commit only when asked). `/tmp/mrs-talk-world3d.r18.mp4` is the r18 video.
- [ ] Author's answers awaited on the open points listed at the end of NOTES.md Round 19 → verify: each answered or changed
