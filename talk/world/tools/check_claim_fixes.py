#!/usr/bin/env python3
"""Gate for ticket 08: the 38 Deck-claim wording fixes of ticket 07 are applied, and nothing else drifted.

    python3 talk/world/tools/check_claim_fixes.py --extract <deck-text.md> [--world DIR] [--narration DIR]

<deck-text.md> comes from `render/extract_deck_text.mjs talk/world/index.html <out.md>`.

G1  per row and file: the old source string occurs 0 times, the new one exactly once
    (notes rows are checked inside their beat, in the JS notes rail and in scene_NN.md).
G2  notes rail == narration script, word for word, on every beat of every scene.
G3  (a) every ticket-07 fragment below is verbatim in ticket 07 (catches copy typos);
    (a') each approved deviation marked punctuation-only changes punctuation only;
    (b) ticket 07's New fragments are present and its Now fragments absent where the audience
        meets them: the rendered screen text (--extract) or the beat's notes + narration.
Exit 0 = CLAIM FIXES PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parent.parent
NARRATION = Path("/home/gofer/mrs_talk_video/narration_video")
TICKET07 = Path("/home/gofer/mrs_talk_video/.scratch/mrs-talk-world/issues/07-deck-claims-to-fix.md")

# Approved deviations from ticket 07's verbatim New cell:
# row -> (ticket 07 text, applied text, punctuation-only?, why). Review = evidence/08/physics-review.md.
DEVIATIONS = {
    4: ("~100 MeV (quarkonium) (§III B)", "~100 MeV (quarkonium) (§III A)", False,
        "user 2026-09-28 after physics review: the ~100 MeV sentence sits in §III A"),
    23: ("342.47 GeV, with a scale range of plus or minus 0.77 GeV — for top, the band is wide — against",
         "342.47 GeV, with a scale range of plus or minus 0.77 GeV, for toponium, the band is wide, against", False,
         "user 2026-09-28: commas for the Piper em-dash ban (render/_check_script.py); "
         "'top' -> 'toponium' after physics review (t c-bar / t b-bar bands are 9 / 24 MeV, Tab. II)"),
    27: ("at 514.46 GeV, with a scale range of plus or minus 0.94 GeV: for top systems the band is wide.",
         "at 514.46 GeV, with a scale range of plus or minus 0.94 GeV: for the triply top Omega the band is wide.", False,
         "user 2026-09-28 after physics review (blocker): only Ω_ttt is wide; other top baryons 6–47 MeV, Tab. IV"),
    34: ("LO apart → all overlap (partial)", "LO apart → overlap (partial)", False,
         "user 2026-09-28 after physics review: MRS LO misses NNLO at m_c, m_b in Figs. 1–2"),
}

# Edit kinds: "js" = raw scene JS (screen), "note" = JS notes beat + scene_NN.md beat,
# "jsnote" / "mdnote" = one side only (the two scripts disagreed before the edit).
# Row: (row, scene, where G3b looks: "screen" | beat | None, edits, t07 Now fragments, t07 New fragments)
ROWS = [
    (1, 2, "screen", [("js", 0, "lattice Ω<sub>QQQ</sub> at ±10 MeV</span>", "lattice Ω<sub>QQQ</sub> at ±10–20 MeV</span>")],
     ["lattice Ω_QQQ at ±10 MeV"], ["lattice Ω_QQQ at ±10–20 MeV"]),
    (2, 2, 4, [("note", 4, "now quote 10 MeV errors.", "now quote 10 to 20 MeV errors.")],
     ["now quote 10 MeV errors."], ["now quote 10 to 20 MeV errors."]),
    (3, 2, "screen", [("js", 0, "<br>~700 MeV (baryons)'", "<br>~700–850 MeV (baryons)'")],
     ["~700 MeV (baryons)"], ["~700–850 MeV (baryons)"]),
    (4, 2, "screen", [("js", 0, "' (quarkonium)<br>", "' (quarkonium) (§III A)<br>")],
     [], [DEVIATIONS[4][1]]),
    (5, 3, "screen", [("js", 0, "~10% for <i>c</i> — dropped here",
                       '~10% for <i>c</i> <span class="small">(§II A)</span> — dropped here')],
     [], ["(§II A)"]),
    (6, 4, "screen", [("js", 0,
                       'Gluons with <i class="m">k ~ Λ</i><sub>QCD</sub> contaminate the potential ⇒ irreducible ambiguity <i class="m">δV ~ Λ</i><sub>QCD</sub>',
                       'Infrared sensitivity (momenta <i class="m">~ Λ</i><sub>QCD</sub>) ⇒ ambiguity <i class="m">δV ~ Λ</i><sub>QCD</sub> — a constant, same at every <i class="m">r</i>')],
     ["Gluons with k ~ Λ_QCD contaminate the potential ⇒ irreducible ambiguity δV ~ Λ_QCD"],
     ["Infrared sensitivity (momenta ~ Λ_QCD) ⇒ ambiguity δV ~ Λ_QCD — a constant, same at every r"]),
    (7, 4, 3, [("note", 3,
                "Gluons with momenta near lambda QCD contaminate the perturbative potential and leave an irreducible ambiguity in it.",
                "The series is sensitive to momenta near lambda QCD, which leaves an ambiguity of order lambda QCD: a constant, the same at every separation.")],
     ["Gluons with momenta near lambda QCD contaminate the perturbative potential and leave an irreducible ambiguity in it."],
     ["The series is sensitive to momenta near lambda QCD, which leaves an ambiguity of order lambda QCD: a constant, the same at every separation."]),
    (8, 4, "screen", [("js", 0, "; LO / NLO / NNLO bands do not overlap</div>",
                       "; the bands jump from order to order — LO does not even overlap NLO/NNLO</div>")],
     ["; LO / NLO / NNLO bands do not overlap"], ["; the bands jump from order to order — LO does not even overlap NLO/NNLO"]),
    (9, 4, "screen", [("js", 0, "gap.textContent='disjoint — no meaningful uncertainty';",
                       "gap.textContent='LO apart — no meaningful uncertainty';"),
                      ("js", 0, "['NLO','var(--amber)',86,30,340,230],['NNLO','var(--c3)',38,26,600,130]",
                       "['NLO','var(--amber)',76,40,340,230],['NNLO','var(--c3)',40,50,500,230]")],
     ["disjoint — no meaningful uncertainty"], ["LO apart — no meaningful uncertainty"]),
    (10, 4, 4, [("jsnote", 4, "and the bands at successive orders do not overlap at all.",
                 "and the bands jump from order to order: leading order does not even overlap the higher orders."),
                ("mdnote", 4, "and for the binding energies the bands at successive orders do not even overlap.",
                 "and the bands jump from order to order: leading order does not even overlap the higher orders.")],
     ["and the bands at successive orders do not overlap at all.", "for the binding energies", "do not even overlap"],
     ["and the bands jump from order to order: leading order does not even overlap the higher orders."]),
    (11, 5, "screen", [("js", 0, "resummed to ∞ in closed form — smooth, unambiguous</div>", "resummed to ∞ in closed form</div>")],
     ["resummed to ∞ in closed form — smooth, unambiguous"], ["resummed to ∞ in closed form"]),
    (12, 5, "screen", [("js", 0, "free of the leading-renormalon ambiguity</b> — the resummed tail is unambiguous.",
                        "free of the leading-renormalon ambiguity</b>.")],
     ["free of the leading-renormalon ambiguity — the resummed tail is unambiguous."], ["free of the leading-renormalon ambiguity."]),
    (13, 5, 2, [("note", 2, "in closed form: smooth and unambiguous.", "in closed form.")],
     ["in closed form: smooth and unambiguous."], ["in closed form."]),
    (14, 5, "screen", [("js", 0, '<span class="tc">✓</span></div>', '<span class="tc">✓</span> <span class="small">(§II B)</span></div>')],
     [], ["(§II B)"]),
    (15, 5, "screen", [("js", 0, "re-expanding reproduces the fixed-order result <b>exactly</b> (Komijani ’17; Kronfeld ’23).",
                        "the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).")],
     ["re-expanding reproduces the fixed-order result exactly (Komijani ’17; Kronfeld ’23)."],
     ["the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23)."]),
    (16, 5, 5, [("note", 5, "re-expanding reproduces the fixed-order result exactly, as shown by Komijani in 2017 and Kronfeld in 2023.",
                 "the series is only reorganized: the renormalon tail is isolated and resummed, following Komijani in 2017 and Kronfeld in 2023.")],
     ["With the same M S bar inputs, re-expanding reproduces the fixed-order result exactly, as shown by Komijani in 2017 and Kronfeld in 2023."],
     ["With the same M S bar inputs, the series is only reorganized: the renormalon tail is isolated and resummed, following Komijani in 2017 and Kronfeld in 2023."]),
    # mlab labels are foreignObject text, which extract_deck_text.mjs does not read: G1 + the pixel diff cover row 17.
    (17, 6, None, [("js", 0, "'<i>μ</i><sub>cut</sub> = 2 GeV → fixed order'",
                    "'<i>μ</i><sub>cut</sub> 2 GeV → fixed order (Fig. 7: frozen <i>α<sub>s</sub></i>)'")], [], []),
    (18, 7, "screen", [("js", 0, 'Bootstrap errors: <span class="num">1–10 MeV</span>.',
                        'Bootstrap errors: <span class="num">≤ 0.4 MeV</span> (most states) · <span class="num">6–11 MeV</span> (tetraquarks) · <span class="num">40–80 MeV</span> (η<sub>t</sub>, Ω<sub>ttt</sub>).')],
     ["Bootstrap errors: 1–10 MeV."],
     ["Bootstrap errors: ≤ 0.4 MeV (most states) · 6–11 MeV (tetraquarks) · 40–80 MeV (η_t, Ω_ttt)."]),
    (19, 7, 4, [("note", 4, "Statistical errors come from bootstrap, typically 1 to 10 MeV, and the imaginary-time extent and step size",
                 "Statistical errors come from bootstrap: below half an MeV for most states, 6 to 11 MeV for the tetraquarks, and 40 to 80 MeV for toponium and the triply top Omega. The imaginary-time extent and step size")],
     ["Statistical errors come from bootstrap, typically 1 to 10 MeV, and"],
     ["Statistical errors come from bootstrap: below half an MeV for most states, 6 to 11 MeV for the tetraquarks, and 40 to 80 MeV for toponium and the triply top Omega. The imaginary-time extent and step size"]),
    (20, 8, "screen", [("js", 0, "'broad bands — and the orders do not overlap'", "'broad bands — LO does not even overlap the higher orders'")],
     ["broad bands — and the orders do not overlap"], ["broad bands — LO does not even overlap the higher orders"]),
    (21, 8, 1, [("note", 1, "and the bands from different orders do not even overlap.", "and leading order does not even overlap the higher orders.")],
     ["and the bands from different orders do not even overlap."], ["and leading order does not even overlap the higher orders."]),
    (22, 8, "screen", [("js", 0, "'2–3× narrower — orders sit on top of each other'", "'2–3× narrower (Fig. 1 caption) — orders sit on top of each other'")],
     [], ["(Fig. 1 caption)"]),
    (23, 9, 1, [("note", 1, "342.47 GeV, with an uncertainty of 77 in the last digits, against", DEVIATIONS[23][1])],
     ["342.47 GeV, with an uncertainty of 77 in the last digits, against"], [DEVIATIONS[23][1]]),
    (24, 9, 2, [("jsnote", 2, "comparable to, though outside, our scale-variation range.", "well outside our scale-variation range: about three times it."),
                ("mdnote", 2, "well outside our scale-variation range.", "well outside our scale-variation range: about three times it.")],
     ["comparable to, though outside, our scale-variation range."], ["well outside our scale-variation range: about three times it."]),
    (25, 10, "screen", [("js", 0, "const gaps=[3.8,1.9,1.1,1.0];",
                         "const gaps=[3.7,1.9,1.1,0.9];\n"
                         "E('text',{x:490,y:24,'text-anchor':'end','font-size':'16',class:'axl'},gs).textContent='gap ÷ lattice';")],
     ["3.8%", "1.0%"], ["3.7%", "0.9%", "gap ÷ lattice"]),
    (26, 10, 4, [("note", 4, "As a fraction of the mass, the gap falls from 3.8 percent for triply charm to 1.0 percent for triply bottom.",
                  "As a fraction of the lattice mass, the gap falls from nearly 4 percent for triply charm to under 1 percent for triply bottom.")],
     ["As a fraction of the mass, the gap falls from 3.8 percent for triply charm to 1.0 percent for triply bottom."],
     ["As a fraction of the lattice mass, the gap falls from nearly 4 percent for triply charm to under 1 percent for triply bottom."]),
    (27, 10, 5, [("note", 5, "at 514.46 GeV, with an uncertainty of 94 in the last digits.",
                  DEVIATIONS[27][1])],
     ["at 514.46 GeV, with an uncertainty of 94 in the last digits."], [DEVIATIONS[27][1]]),
    # Rows 28-29: ticket 07 gives the screen form only; the spoken numbers follow it (user 2026-09-28).
    (28, 11, "screen", [("js", 0, '<span class="num tc">1.502–1.508</span>', '<span class="num tc">1.50–1.51</span>'),
                        ("note", 2, "between 1.502 and 1.508", "between 1.50 and 1.51")],
     ["1.502–1.508"], ["1.50–1.51"]),
    (29, 11, "screen", [("js", 0, "Red points 1.52–1.57,", "Red points 1.52–1.56 (Tab. I + III, lattice [27]),"),
                        ("note", 3, "sit at 1.52 to 1.57", "sit at 1.52 to 1.56")],
     ["Red points 1.52–1.57"], ["1.52–1.56 (Tab. I + III, lattice [27])"]),
    (30, 12, "screen", [("js", 0, "bound below the quarkonium threshold", "bound below the two-quarkonium threshold")],
     ["bound below the quarkonium threshold"], ["bound below the two-quarkonium threshold"]),
    # §V A holds 0.085(5); ≈0.125 is in §VI and ~30% is derived, so the tag sits on 0.085(5) (physics review).
    (31, 12, "screen", [("js", 0, "0.085(5)</span> at NLO", "0.085(5)</span> (§V A) at NLO")],
     [], ["(§V A)"]),
    (32, 12, "screen", [("js", 0, "by 50–150 MeV ⇒", 'by 50–150 MeV <span class="small">(§V B)</span> ⇒')],
     [], ["(§V B)"]),
    (33, 12, 3, [("note", 3, "from 509 to 742 MeV, though with scale ranges of several hundred MeV at this order.",
                  "from 509 to 742 MeV. Here the M R S scale range, several hundred MeV, is wider than in fixed order; what stays small is the shift from one order to the next.")],
     ["from 509 to 742 MeV, though with scale ranges of several hundred MeV at this order."],
     ["from 509 to 742 MeV. Here the M R S scale range, several hundred MeV, is wider than in fixed order; what stays small is the shift from one order to the next."]),
    (34, 13, "screen", [("js", 0, "['LO / NLO / NNLO bands','disjoint','overlapping (partial)']",
                         "['LO / NLO / NNLO bands','LO apart','overlap (partial)']")],
     ["disjoint → overlapping (partial)"], [DEVIATIONS[34][1]]),
    (35, 13, 1, [("note", 1, "And bands that were disjoint now overlap.", "And bands where leading order sat apart now overlap.")],
     ["And bands that were disjoint now overlap."], ["And bands where leading order sat apart now overlap."]),
    (36, 14, 2, [("note", 2, "up to two orders of magnitude, and the results are stable order by order, across every channel we looked at.",
                  "up to two orders of magnitude for charm and bottom states, and the results are stable order by order in every channel we looked at.")],
     ["up to two orders of magnitude, and the results are stable order by order, across every channel we looked at."],
     ["up to two orders of magnitude for charm and bottom states, and the results are stable order by order in every channel we looked at."]),
    (37, 14, "screen", [("js", 0, "<h2>Parameter-free predictions</h2>",
                         '<h2>Predictions — 2 tuned masses + <span style="text-transform:none">m<sub>t</sub></span> input</h2>')],
     ["Parameter-free predictions"], ["Predictions — 2 tuned masses + m_t input"]),
    (38, 14, 3, [("note", 3, "a family of parameter-free predictions.", "a family of predictions from two tuned masses and the top mass.")],
     ["a family of parameter-free predictions."], ["a family of predictions from two tuned masses and the top mass."]),
]

_BEAT_RE = re.compile(r"^##\s+Beat\s+(\d+)\s+·\s+`([^`]+)`\s+·\s+([\d.]+)\s*s\s*$")
_NOTES_RE = re.compile(r"notes:\[\n(.*?)\n\]\}\);", re.S)
_NOTE_LINE = re.compile(r"^\s*'<b>b\d+\.</b> (.*)',?\s*$")


def beats_of(path: Path) -> list[str]:
    """Narration beats of scene_NN.md (same parser as render/update_notes_rail.py)."""
    out, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if _BEAT_RE.match(line):
            if cur is not None:
                out.append(" ".join(cur).strip())
            cur = []
        elif cur is not None and not line.startswith("# "):
            cur.append(line.strip())
    if cur is not None:
        out.append(" ".join(cur).strip())
    return [" ".join(b.split()) for b in out]


def notes_of(js: str) -> list[str]:
    """The notes rail of one scene file, as plain text."""
    m = _NOTES_RE.search(js)
    if not m:
        raise SystemExit("no notes:[ … ]}); block")
    out = []
    for line in m.group(1).splitlines():
        lm = _NOTE_LINE.match(line)
        if not lm:
            raise SystemExit(f"unparsed notes line: {line!r}")
        out.append(html.unescape(lm.group(1).replace("\\'", "'").replace("\\\\", "\\")))
    return out


def norm(s: str) -> str:
    """Extract text → ticket-07 spelling: Ω_{QQQ} → Ω_QQQ, no table pipes or bold marks, one space."""
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s).replace("|", " ").replace("**", "")
    return " ".join(s.split())


def screens_of(extract: str) -> dict[int, str]:
    """Scene number → normalised visible text (the part before '### notes')."""
    parts = re.split(r"^## S(\d\d) ", extract, flags=re.M)
    return {int(n): norm(body.split("### notes")[0]) for n, body in zip(parts[1::2], parts[2::2])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--world", type=Path, default=WORLD)
    ap.add_argument("--narration", type=Path, default=NARRATION)
    ap.add_argument("--ticket07", type=Path, default=TICKET07)
    a = ap.parse_args()

    js = {int(p.name[1:3]): p.read_text(encoding="utf-8") for p in sorted((a.world / "js").glob("s[0-9][0-9]-*.js"))}
    notes = {n: notes_of(t) for n, t in js.items()}
    md = {n: beats_of(a.narration / f"scene_{n:02d}.md") for n in js}
    screens = screens_of(a.extract.read_text(encoding="utf-8"))
    t07 = a.ticket07.read_text(encoding="utf-8")
    fails: list[str] = []

    g1_ok = 0
    for row, sc, _where, edits, _now, _new in ROWS:
        ok = True
        for kind, beat, find, repl in edits:
            texts = {"js": [("js", js[sc])],
                     "note": [("notes", notes[sc][beat - 1]), ("md", md[sc][beat - 1])],
                     "jsnote": [("notes", notes[sc][beat - 1])],
                     "mdnote": [("md", md[sc][beat - 1])]}[kind]
            for side, text in texts:
                nf, nr = text.count(find), text.count(repl)
                if (nf, nr) != (0, 1):
                    ok = False
                    fails.append(f"G1 row {row} S{sc:02d}{f'b{beat}' if beat else ''} {side}: find={nf} repl={nr}")
        g1_ok += ok

    g2 = [f"S{n:02d}b{i + 1}" for n in js for i in range(max(len(notes[n]), len(md[n])))
          if i >= len(notes[n]) or i >= len(md[n]) or notes[n][i] != md[n][i]]
    fails += [f"G2 notes≠narration {b}" for b in g2]

    approved = {v[1] for v in DEVIATIONS.values()}
    for row, _sc, _where, _edits, now, new in ROWS:
        for frag in now + new:
            if frag not in approved and frag not in t07:
                fails.append(f"G3a row {row}: not verbatim in ticket 07: {frag!r}")
    for row, (orig, applied, punct_only, why) in DEVIATIONS.items():
        if orig not in t07:
            fails.append(f"G3a row {row}: deviation source not in ticket 07: {orig!r}")
        if punct_only and re.findall(r"[A-Za-z0-9.]+", orig) != re.findall(r"[A-Za-z0-9.]+", applied):
            fails.append(f"G3a' row {row}: deviation changes more than punctuation")
        print(f"DEVIATION row {row}: {orig!r} -> {applied!r} ({why})")

    for row, sc, where, _edits, now, new in ROWS:
        if where is None:
            continue
        texts = [("screen", screens.get(sc, ""))] if where == "screen" else \
                [("notes", notes[sc][where - 1]), ("md", md[sc][where - 1])]
        for side, text in texts:
            fails += [f"G3b row {row} S{sc:02d} {side}: Now still present: {f!r}" for f in now if norm(f) in text]
            fails += [f"G3b row {row} S{sc:02d} {side}: New missing: {f!r}" for f in new if norm(f) not in text]

    for f in fails:
        print(f)
    print(f"G1 rows ok: {g1_ok}/{len(ROWS)}")
    print(f"G2 notes≠narration: {','.join(g2) or 'none'}")
    print("CLAIM FIXES FAIL" if fails else "CLAIM FIXES PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
