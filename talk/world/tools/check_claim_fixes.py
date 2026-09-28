#!/usr/bin/env python3
"""Gate for tickets 08 and 11: the Deck-claim wording fixes of ticket 07 (38 rows, ROWS) and of
ticket 09 (47 rows, ROWS09) are applied, and nothing else drifted.

    python3 talk/world/tools/check_claim_fixes.py --extract <deck-text.md> [--world DIR] [--narration DIR]
        [--ticket07 FILE] [--ticket09 FILE]

<deck-text.md> comes from `render/extract_deck_text.mjs talk/world/index.html <out.md>`.

G1  per row and file: the old source string occurs 0 times, the new one exactly once
    (notes rows are checked inside their beat, in the JS notes rail and in scene_NN.md).
    A ticket-07 row that a ticket-09 row rewrites again (SUPERSEDED) only needs its old string gone.
G2  notes rail == narration script, word for word, on every beat of every scene.
G3  (a) every ticket-07/09 fragment below is verbatim in its ticket (catches copy typos);
    (a') each approved deviation marked punctuation-only changes punctuation only;
    (b) the New fragments are present and the Now fragments absent where the audience
        meets them: the rendered screen text (--extract) or the beat's notes + narration
        (a SUPERSEDED row: Now fragments only);
    (c) each SUPERSEDED row points at a ticket-09 row in the same scene (same beat for notes)
        whose find string shares ≥ 12 characters with the ticket-07 row's new string.
Exit 0 = CLAIM FIXES PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import difflib
import html
import re
import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parent.parent
NARRATION = Path("/home/gofer/mrs_talk_video/narration_video")
TICKET07 = Path("/home/gofer/mrs_talk_video/.scratch/mrs-talk-world/issues/07-deck-claims-to-fix.md")
TICKET09 = Path("/home/gofer/mrs_talk_video/.scratch/mrs-talk-world/issues/09-deck-claims-physics-review.md")

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
# Ticket 10 (story edit E3) moved S09 b1–b3 to b2–b4; rows 23, 24, M10, M11 use the new beat numbers, their tickets' tables the old ones.
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
    (23, 9, 2, [("note", 2, "342.47 GeV, with an uncertainty of 77 in the last digits, against", DEVIATIONS[23][1])],
     ["342.47 GeV, with an uncertainty of 77 in the last digits, against"], [DEVIATIONS[23][1]]),
    (24, 9, 3, [("jsnote", 3, "comparable to, though outside, our scale-variation range.", "well outside our scale-variation range: about three times it."),
                ("mdnote", 3, "well outside our scale-variation range.", "well outside our scale-variation range: about three times it.")],
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

# Ticket-07 rows that a ticket-09 row rewrites again: row -> ticket-09 id (G1 / G3b / G3c above).
SUPERSEDED = {3: "M1", 5: "T3", 6: "T5", 7: "T6", 8: "M3", 10: "M4", 11: "T9", 13: "T10", 14: "T13",
              17: "T16", 18: "T17", 22: "M8", 23: "M10", 24: "M11", 30: "T22"}

# Ticket 09 (applied by ticket 11), same shape as ROWS with the ticket-09 row id; applied verbatim, no deviations.
# Notes rows use ticket 09's Now/New minus the "…". Markup around the words (colour spans, .small tags) was
# settled in ticket 11's plan. T7, T8, T16 are mlab (foreignObject) labels: G1 + evidence/11/fig_probe.mjs.
ROWS09 = [
    ("B1", 5, "screen", [("js", 0, 'Isolate the growth, <span class="ac">resum it</span>, make the ambiguity <span class="rc">one explicit constant</span>.',
                 'Isolate the leading renormalon into <span class="rc">an explicit power correction</span>, and <span class="ac">resum the growth</span> <span class="small">(§II B)</span>.')],
     ["Isolate the growth, resum it, make the ambiguity one explicit constant."],
     ["Isolate the leading renormalon into an explicit power correction, and resum the growth (§II B)."]),
    ("B2", 5, 1, [("note", 1, "isolate it, resum it in closed form, and push the entire remaining ambiguity into one explicit constant.",
                 "isolate the leading renormalon into an explicit power correction, and resum the growth in closed form.")],
     ["isolate it, resum it in closed form, and push the entire remaining ambiguity into one explicit constant."],
     ["isolate the leading renormalon into an explicit power correction, and resum the growth in closed form."]),
    ("M1", 2, "screen", [("js", 0, "~700–850 MeV (baryons)'",
                 "~700–850 MeV (Ω masses, Tab. III)'")],
     ["~700–850 MeV (baryons)"],
     ["~700–850 MeV (Ω masses, Tab. III)"]),
    ("M2", 2, 5, [("note", 5, "of order 100 MeV in binding energies, and for the triply heavy baryons far more.",
                 "of order 100 MeV in binding energies, and 700 to 850 MeV in the triply heavy baryon masses.")],
     ["of order 100 MeV in binding energies, and for the triply heavy baryons far more."],
     ["of order 100 MeV in binding energies, and 700 to 850 MeV in the triply heavy baryon masses."]),
    ("M3", 4, "screen", [("js", 0, '<i class="m">μ</i> → 2<i class="m">μ</i> moves a triply-heavy baryon mass by <span class="rc num">~700 MeV</span> at NNLO; the bands jump from order to order — LO does not even overlap NLO/NNLO',
                 'Varying <i class="m">μ</i> by 2× moves Ω<sub>bbb</sub> by up to <span class="rc num">~700 MeV</span> at NNLO <span class="small">(Tab. III)</span>; in binding energy at fixed <i class="m">m<sub>Q</sub></i>, LO does not even overlap NLO/NNLO <span class="small">(Figs. 1–2)</span>')],
     ["μ → 2μ moves a triply-heavy baryon mass by ~700 MeV at NNLO; the bands jump from order to order — LO does not even overlap NLO/NNLO"],
     ["Varying μ by 2× moves Ω_bbb by up to ~700 MeV at NNLO (Tab. III); in binding energy at fixed m_Q, LO does not even overlap NLO/NNLO (Figs. 1–2)"]),
    ("M4", 4, 4, [("note", 4, "Doubling the scale mu moves a triply heavy baryon mass by about 700 MeV at next-to-next-to-leading order, and the bands jump from order to order: leading order does not even overlap the higher orders.",
                 "Varying the scale by a factor of two moves the triply bottom Omega mass by up to 700 MeV at next-to-next-to-leading order. And in the binding energy at fixed quark mass, the bands jump from order to order: leading order does not even overlap the higher orders.")],
     ["Doubling the scale mu moves a triply heavy baryon mass by about 700 MeV at next-to-next-to-leading order, and the bands jump from order to order: leading order does not even overlap the higher orders."],
     ["Varying the scale by a factor of two moves the triply bottom Omega mass by up to 700 MeV at next-to-next-to-leading order. And in the binding energy at fixed quark mass, the bands jump from order to order: leading order does not even overlap the higher orders."]),
    ("M5", 6, "screen", [("js", 0, '<span class="num">2 GeV</span> collapses the scheme back to fixed order — nothing gained.',
                 'At <span class="num">2 GeV</span> MRS is rarely used and results fall back to fixed order <span class="small">(Fig. 7, frozen <i>α<sub>s</sub></i>)</span>.')],
     ["2 GeV collapses the scheme back to fixed order — nothing gained."],
     ["At 2 GeV MRS is rarely used and results fall back to fixed order (Fig. 7, frozen α_s)."]),
    ("M6", 6, 3, [("note", 3, "Push it to 2 GeV and the scheme collapses back to fixed order.",
                 "Push it to 2 GeV and it is rarely used: the paper shows the results falling back to fixed order.")],
     ["Push it to 2 GeV and the scheme collapses back to fixed order."],
     ["Push it to 2 GeV and it is rarely used: the paper shows the results falling back to fixed order."]),
    ("M7", 7, 4, [("note", 4, "so that what we quote is a converged ground-state energy.",
                 "so that what we quote is a converged ground-state energy for mesons and baryons; for the tetraquarks, the colour convergence is still under study.")],
     ["so that what we quote is a converged ground-state energy."],
     ["so that what we quote is a converged ground-state energy for mesons and baryons; for the tetraquarks, the colour convergence is still under study."]),
    ("M8", 8, "screen", [("js", 0, "'2–3× narrower (Fig. 1 caption) — orders sit on top of each other'",
                 "'2–3× narrower at <i>m<sub>b</sub></i> (Fig. 1 caption) — orders now partly overlap'")],
     ["2–3× narrower (Fig. 1 caption) — orders sit on top of each other"],
     ["2–3× narrower at m_b (Fig. 1 caption) — orders now partly overlap"]),
    ("M9", 8, 2, [("note", 2, "The bands are two to three times narrower at next-to-leading and next-to-next-to-leading order, and the orders sit on top of one another.",
                 "At the bottom mass the bands are two to three times narrower, and the orders now partly overlap.")],
     ["The bands are two to three times narrower at next-to-leading and next-to-next-to-leading order, and the orders sit on top of one another."],
     ["At the bottom mass the bands are two to three times narrower, and the orders now partly overlap."]),
    ("M10", 9, 2, [("note", 2, "comes out at 342.47 GeV, with a scale range of plus or minus 0.77 GeV, for toponium, the band is wide, against roughly 343 from nonrelativistic threshold treatments.",
                 "comes out at 342.47 GeV, against roughly 343 from nonrelativistic threshold treatments. Its scale range reaches 0.77 GeV, so for toponium the band is wide, even though the shift from one order to the next is only 20 MeV.")],
     ["comes out at 342.47 GeV, with a scale range of plus or minus 0.77 GeV, for toponium, the band is wide, against roughly 343 from nonrelativistic threshold treatments."],
     ["comes out at 342.47 GeV, against roughly 343 from nonrelativistic threshold treatments. Its scale range reaches 0.77 GeV, so for toponium the band is wide, even though the shift from one order to the next is only 20 MeV."]),
    ("M11", 9, 3, [("note", 3, "We find 6.29121 GeV, with an uncertainty of 731 in the last digits, about 25 MeV below the combined experimental and lattice value of 6.316, well outside our scale-variation range: about three times it.",
                 "We find 6.29121 GeV, with a scale range of 7 MeV, about 25 MeV below the combined experimental and lattice value of 6.316, which is itself known to 7 MeV. The gap is about three times our scale range.")],
     ["We find 6.29121 GeV, with an uncertainty of 731 in the last digits, about 25 MeV below the combined experimental and lattice value of 6.316, well outside our scale-variation range: about three times it."],
     ["We find 6.29121 GeV, with a scale range of 7 MeV, about 25 MeV below the combined experimental and lattice value of 6.316, which is itself known to 7 MeV. The gap is about three times our scale range."]),
    ("M12", 10, "screen", [("js", 0, '⇒ missing <span class="rc">𝒪(1/m<sub>Q</sub>) + spin</span> — not noise</div>',
                 '— not noise; consistent with missing <span class="rc">𝒪(1/m<sub>Q</sub>) + spin</span> <span class="small">(§VI)</span></div>')],
     ["fractional gap ∼ 1/m_Q ⇒ missing 𝒪(1/m_Q) + spin — not noise"],
     ["fractional gap ∼ 1/m_Q — not noise; consistent with missing 𝒪(1/m_Q) + spin (§VI)"]),
    ("M13", 10, 4, [("note", 4, "A one over the heavy quark mass trend: exactly what you expect if what is missing are the one over m Q corrections and the spin-dependent terms.",
                 "A one over the heavy quark mass trend, consistent with what the static limit leaves out: the one over m Q corrections and the spin-dependent terms. A direct test is future work.")],
     ["A one over the heavy quark mass trend: exactly what you expect if what is missing are the one over m Q corrections and the spin-dependent terms."],
     ["A one over the heavy quark mass trend, consistent with what the static limit leaves out: the one over m Q corrections and the spin-dependent terms. A direct test is future work."]),
    ("M14", 11, "screen", [("js", 0, '— the 1/<i class="m">m<sub>Q</sub></i> pattern in an independent observable.',
                 '— the same 1/<i class="m">m<sub>Q</sub></i> gap, seen in the ratio.')],
     ["— the 1/m_Q pattern in an independent observable."],
     ["— the same 1/m_Q gap, seen in the ratio."]),
    ("M15", 11, 3, [("note", 3, "That is the one over the heavy quark mass pattern a second time, in an independent observable.",
                 "That is the same one over the heavy quark mass gap, seen again in the ratio.")],
     ["That is the one over the heavy quark mass pattern a second time, in an independent observable."],
     ["That is the same one over the heavy quark mass gap, seen again in the ratio."]),
    ("M16", 13, "screen", [("js", 0, 'A <span class="num">10 MeV</span> band makes the remaining <span class="num">125–175 MeV</span> gap <i>interpretable</i>: missing 1/',
                 'Order-by-order shifts of <span class="num">4–18 MeV</span> make the remaining <span class="num">125–175 MeV</span> gap <i>interpretable</i>: consistent with missing 1/')],
     ["A 10 MeV band makes the remaining 125–175 MeV gap interpretable: missing 1/m_Q physics, not perturbative noise."],
     ["Order-by-order shifts of 4–18 MeV make the remaining 125–175 MeV gap interpretable: consistent with missing 1/m_Q physics, not perturbative noise."]),
    ("M17", 13, "screen", [("js", 0, "textContent='ours ±10 MeV';",
                 "textContent='ours: shifts 4–18 MeV';")],
     ["ours ±10 MeV"],
     ["ours: shifts 4–18 MeV"]),
    ("M18", 13, 3, [("note", 3, "The payoff: because the band is 10 MeV rather than 700, the remaining gap of 125 to 175 MeV to the lattice becomes interpretable. It is missing one over m Q physics, not perturbative noise.",
                 "The payoff: with order-by-order shifts of only 4 to 18 MeV, the remaining gap of 125 to 175 MeV to the lattice becomes interpretable. It is consistent with missing one over m Q physics, not perturbative noise.")],
     ["The payoff: because the band is 10 MeV rather than 700, the remaining gap of 125 to 175 MeV to the lattice becomes interpretable. It is missing one over m Q physics, not perturbative noise."],
     ["The payoff: with order-by-order shifts of only 4 to 18 MeV, the remaining gap of 125 to 175 MeV to the lattice becomes interpretable. It is consistent with missing one over m Q physics, not perturbative noise."]),
    ("M19", 14, "screen", [("js", 0, "Scale sensitivity down by",
                 "Scale sensitivity of charm and bottom hadrons down by")],
     ["Scale sensitivity down by up to two orders of magnitude; results stable order by order."],
     ["Scale sensitivity of charm and bottom hadrons down by up to two orders of magnitude; results stable order by order."]),
    ("T1", 2, "screen", [("js", 0, '<span class="ov">t</span> threshold</span>',
                 '<span class="ov">t</span> threshold (§I)</span>')],
     [],
     ["threshold (§I)"]),
    ("T2", 2, "screen", [("js", 0, "This talk: remove it.",
                 "This talk: remove the leading renormalon.")],
     ["This talk: remove it."],
     ["This talk: remove the leading renormalon."]),
    ("T3", 3, "screen", [("js", 0, 'Spin effects are 𝒪(<i class="m">v</i>²) — ~5% for <i>b</i>, ~10% for <i>c</i> <span class="small">(§II A)</span> — dropped here',
                 '𝒪(1/<i class="m">m<sub>Q</sub></i>) and spin terms dropped here. Spin effects are 𝒪(<i class="m">v</i>²): ~5% (<i>b</i>) / ~10% (<i>c</i>) of the binding energy <span class="small">(§II A)</span>')],
     ["Spin effects are 𝒪(v²) — ~5% for b, ~10% for c (§II A) — dropped here"],
     ["𝒪(1/m_Q) and spin terms dropped here. Spin effects are 𝒪(v²): ~5% (b) / ~10% (c) of the binding energy (§II A)"]),
    ("T4", 3, 4, [("note", 4, "Spin-dependent effects are suppressed by the velocity squared, about 5 percent for bottom and 10 percent for charm, and they are dropped here.",
                 "The one over m Q corrections are dropped, and so are the spin-dependent effects, which are suppressed by the velocity squared: about 5 percent of the binding energy for bottom and 10 percent for charm.")],
     ["Spin-dependent effects are suppressed by the velocity squared, about 5 percent for bottom and 10 percent for charm, and they are dropped here."],
     ["The one over m Q corrections are dropped, and so are the spin-dependent effects, which are suppressed by the velocity squared: about 5 percent of the binding energy for bottom and 10 percent for charm."]),
    ("T5", 4, "screen", [("js", 0, '⇒ ambiguity <i class="m">δV',
                 '⇒ leading ambiguity <i class="m">δV'),
                ("js", 0, ">the same renormalon as the pole mass</div>",
                 ">singlet: the same renormalon as the pole mass</div>")],
     ["⇒ ambiguity δV ~ Λ_QCD — a constant, same at every r"],
     ["⇒ leading ambiguity δV ~ Λ_QCD — a constant, same at every r", "singlet: the same renormalon as the pole mass"]),
    ("T6", 4, 3, [("note", 3, "which leaves an ambiguity of order lambda QCD: a constant, the same at every separation. It is the same renormalon that afflicts the pole mass.",
                 "which leaves a leading ambiguity of order lambda QCD: a constant, the same at every separation. For the colour singlet, it is the same renormalon that afflicts the pole mass.")],
     ["which leaves an ambiguity of order lambda QCD: a constant, the same at every separation. It is the same renormalon that afflicts the pole mass."],
     ["which leaves a leading ambiguity of order lambda QCD: a constant, the same at every separation. For the colour singlet, it is the same renormalon that afflicts the pole mass."]),
    ("T7", 4, None, [("js", 0, "mlab(bsv,74,26,'Δ<i>E</i>',{size:17});",
                 "mlab(bsv,74,26,'|Δ<i>E</i>| (schematic)',{size:17});")],
     [],
     []),
    ("T8", 4, None, [("js", 0, "mlab(psv,74,26,'|<i>v<sub>ℓ</sub></i> <i>α<sub>s</sub></i><sup>ℓ+1</sup>|',{size:18});",
                 "mlab(psv,74,26,'|<i>v<sub>ℓ</sub></i> <i>α<sub>s</sub></i><sup>ℓ+1</sup>| (schematic)',{size:18});")],
     [],
     []),
    ("T9", 5, "screen", [("js", 0, "the full renormalon series, resummed to ∞ in closed form</div>",
                 "the full renormalon series, Borel-summed in closed form (§II B)</div>")],
     ["resummed to ∞ in closed form"],
     ["the full renormalon series, Borel-summed in closed form (§II B)"]),
    ("T10", 5, 2, [("note", 2, "Second, the renormalon series itself, summed from the first term to infinity in closed form.",
                 "Second, the renormalon series itself, Borel-summed over all orders into a closed form.")],
     ["Second, the renormalon series itself, summed from the first term to infinity in closed form."],
     ["Second, the renormalon series itself, Borel-summed over all orders into a closed form."]),
    ("T11", 5, "screen", [("js", 0, 'it is absorbed into the quark mass: <span class="mx">m<sub>Q</sub> + Λ<sub>R</sub></span> is <span class="rc"><b>fixed to the spin-averaged 1S states</b></span>.',
                 'it is absorbed into the quark mass, tuned so that <span class="mx">2m<sub>Q</sub> + E</span> <span class="rc"><b>matches the spin-averaged 1S states</b></span>.')],
     ["m_Q + Λ_R is fixed to the spin-averaged 1S states."],
     ["For the singlet it is absorbed into the quark mass, tuned so that 2m_Q + E matches the spin-averaged 1S states."]),
    ("T12", 5, "screen", [("js", 0, "the renormalon part of <i>v<sub>l</sub></i>, known to",
                 "the renormalon part of <i>v<sub>l</sub></i>; the <i>v<sub>l</sub></i> are known to")],
     ["V_l = the renormalon part of v_l, known to l_max = 2 (NNLO)"],
     ["V_l = the renormalon part of v_l; the v_l are known to l_max = 2 (NNLO)"]),
    ("T13", 5, "screen", [("js", 0, '<span class="tc">✓</span> <span class="small">(§II B)</span>',
                 '<span class="tc">✓</span> <span class="small">(singlet, <i>n<sub>f</sub></i> = 4; §II B)</span>')],
     ["V_0 = 0.771 (NNLO) → 0.772 (N³LO) ✓ (§II B)"],
     ["V_0 = 0.771 (NNLO) → 0.772 (N³LO) ✓ (singlet, n_f = 4; §II B)"]),
    ("T14", 5, "screen", [("js", 0, 'Proven for quark masses — all flavours except top (TUMQCD). <span class="rc">First use inside a bound-state problem.</span>',
                 'Used for quark masses — all flavours except top (Fermilab/MILC/TUMQCD). <span class="rc">First use in a pNRQCD bound-state problem.</span>')],
     ["Proven for quark masses — all flavours except top (TUMQCD). First use inside a bound-state problem."],
     ["Used for quark masses — all flavours except top (Fermilab/MILC/TUMQCD). First use in a pNRQCD bound-state problem."]),
    ("T15", 5, 5, [("note", 5, "It is proven for quark masses, all flavours except top. This is its first use inside a bound-state problem.",
                 "It has been used for quark masses, all flavours except top. This is its first use in a bound-state problem in potential N R Q C D.")],
     ["It is proven for quark masses, all flavours except top. This is its first use inside a bound-state problem."],
     ["It has been used for quark masses, all flavours except top. This is its first use in a bound-state problem in potential N R Q C D."]),
    ("T16", 6, None, [("js", 0, "'<i>μ</i><sub>cut</sub> = 1 GeV — saturates'",
                 "'<span><i>μ</i><sub>cut</sub> = 1 GeV — saturates</span>'"),
                ("js", 0, "'<i>μ</i><sub>cut</sub> 2 GeV → fixed order (Fig. 7: frozen <i>α<sub>s</sub></i>)'",
                 "'<span><i>μ</i><sub>cut</sub> 2 GeV → fixed order (Fig. 7: frozen <i>α<sub>s</sub></i>)</span>'"),
                ("js", 0, "mlab(psv,94,28,'<i>μ</i> [GeV]',{size:18});",
                 "mlab(psv,64,28,'<span><i>μ</i> [GeV]</span>',{anchor:'end',size:18});")],
     [],
     []),
    ("T17", 7, "screen", [("js", 0, '<span class="num">≤ 0.4 MeV</span>',
                 '<span class="num">&lt; 0.5 MeV</span>')],
     ["≤ 0.4 MeV (most states)"],
     ["< 0.5 MeV (most states)"]),
    ("T18", 9, "screen", [("js", 0, "would sharpen exactly",
                 "would stabilize exactly")],
     ["MRS potentials would sharpen exactly"],
     ["MRS potentials would stabilize exactly"]),
    ("T19", 10, 1, [("note", 1, "each with its statistical and its scale uncertainty.",
                 "each with its statistical error and its scale range.")],
     ["each with its statistical and its scale uncertainty."],
     ["each with its statistical error and its scale range."]),
    ("T20", 11, "screen", [("js", 0, "The leading renormalon is <b>universal</b> — it partially cancels in",
                 "The leading renormalon partially cancels in")],
     ["The leading renormalon is universal — it partially cancels in"],
     ["The leading renormalon partially cancels in"]),
    ("T21", 11, 1, [("note", 1, "The leading renormalon is universal, so it partially cancels here.",
                 "The leading renormalon partially cancels here.")],
     ["The leading renormalon is universal, so it partially cancels here."],
     ["The leading renormalon partially cancels here."]),
    ("T22", 12, "screen", [("js", 0, "bound below the two-quarkonium threshold",
                 "bound below the two-meson threshold")],
     ["bound below the two-quarkonium threshold"],
     ["bound below the two-meson threshold"]),
    ("T23", 12, "screen", [("js", 0, "said ≈0.125.",
                 "said ≈0.125 (§VI).")],
     ["fixed-order NLO said ≈0.125."],
     ["fixed-order NLO said ≈0.125 (§VI)."]),
    ("T24", 12, "screen", [("js", 0, "<b>diquark–antidiquark</b> structure</div>",
                 "<b>diquark–antidiquark</b> structure (colour convergence under study)</div>")],
     [],
     ["⇒ consistent with a diquark–antidiquark structure (colour convergence under study)"]),
    ("T25", 12, 2, [("note", 2, "These results stop at next-to-leading order; the four-quark potentials are not known beyond that.",
                 "These results stop at next-to-leading order; beyond that, four-body forces appear that are not yet fully known.")],
     ["These results stop at next-to-leading order; the four-quark potentials are not known beyond that."],
     ["These results stop at next-to-leading order; beyond that, four-body forces appear that are not yet fully known."]),
    ("T26", 13, "screen", [("js", 0, '<div class="big">Every diagnostic moves the same way.',
                 '<div class="big">For charm and bottom hadrons, every diagnostic moves the same way.')],
     ["Every diagnostic moves the same way."],
     ["For charm and bottom hadrons, every diagnostic moves the same way."]),
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
    ap.add_argument("--ticket09", type=Path, default=TICKET09)
    a = ap.parse_args()

    js = {int(p.name[1:3]): p.read_text(encoding="utf-8") for p in sorted((a.world / "js").glob("s[0-9][0-9]-*.js"))}
    notes = {n: notes_of(t) for n, t in js.items()}
    md = {n: beats_of(a.narration / f"scene_{n:02d}.md") for n in js}
    screens = screens_of(a.extract.read_text(encoding="utf-8"))
    t07 = a.ticket07.read_text(encoding="utf-8")
    t09 = a.ticket09.read_text(encoding="utf-8")
    fails: list[str] = []

    g1_ok = {}
    for label, rows in (("t07", ROWS), ("t09", ROWS09)):
        g1_ok[label] = 0
        for row, sc, _where, edits, _now, _new in rows:
            ok = True
            for kind, beat, find, repl in edits:
                texts = {"js": [("js", js[sc])],
                         "note": [("notes", notes[sc][beat - 1]), ("md", md[sc][beat - 1])],
                         "jsnote": [("notes", notes[sc][beat - 1])],
                         "mdnote": [("md", md[sc][beat - 1])]}[kind]
                for side, text in texts:
                    nf, nr = text.count(find), text.count(repl)
                    if nf != 0 or (nr != 1 and row not in SUPERSEDED) or nr > 1:
                        ok = False
                        fails.append(f"G1 row {row} S{sc:02d}{f'b{beat}' if beat else ''} {side}: find={nf} repl={nr}")
            g1_ok[label] += ok

    g2 = [f"S{n:02d}b{i + 1}" for n in js for i in range(max(len(notes[n]), len(md[n])))
          if i >= len(notes[n]) or i >= len(md[n]) or notes[n][i] != md[n][i]]
    fails += [f"G2 notes≠narration {b}" for b in g2]

    approved = {v[1] for v in DEVIATIONS.values()}
    for row, _sc, _where, _edits, now, new in ROWS:
        for frag in now + new:
            if frag not in approved and frag not in t07:
                fails.append(f"G3a row {row}: not verbatim in ticket 07: {frag!r}")
    for row, _sc, _where, _edits, now, new in ROWS09:
        fails += [f"G3a row {row}: not verbatim in ticket 09: {f!r}" for f in now + new if f not in t09]
    for row, (orig, applied, punct_only, why) in DEVIATIONS.items():
        if orig not in t07:
            fails.append(f"G3a row {row}: deviation source not in ticket 07: {orig!r}")
        if punct_only and re.findall(r"[A-Za-z0-9.]+", orig) != re.findall(r"[A-Za-z0-9.]+", applied):
            fails.append(f"G3a' row {row}: deviation changes more than punctuation")
        print(f"DEVIATION row {row}: {orig!r} -> {applied!r} ({why})")

    for row, sc, where, _edits, now, new in ROWS + ROWS09:
        if where is None:
            continue
        texts = [("screen", screens.get(sc, ""))] if where == "screen" else \
                [("notes", notes[sc][where - 1]), ("md", md[sc][where - 1])]
        for side, text in texts:
            fails += [f"G3b row {row} S{sc:02d} {side}: Now still present: {f!r}" for f in now if norm(f) in text]
            if row not in SUPERSEDED:
                fails += [f"G3b row {row} S{sc:02d} {side}: New missing: {f!r}" for f in new if norm(f) not in text]

    by_id = {r[0]: r for r in ROWS09}
    for row, sc, _where, edits, _now, _new in ROWS:
        if row not in SUPERSEDED:
            continue
        rid = SUPERSEDED[row]
        _, sc9, _, edits9, _, _ = by_id[rid]
        beats, beats9 = ({b for k, b, _, _ in e if k != "js"} for e in (edits, edits9))
        share = max(difflib.SequenceMatcher(None, r, f, autojunk=False).find_longest_match(0, len(r), 0, len(f)).size
                    for _, _, _, r in edits for _, _, f, _ in edits9)
        if sc9 != sc or beats != beats9 or share < 12:
            fails.append(f"G3c row {row} -> {rid}: scene {sc}/{sc9}, beats {sorted(beats)}/{sorted(beats9)}, shared {share}")

    for f in fails:
        print(f)
    print(f"G1 rows ok: t07 {g1_ok['t07']}/{len(ROWS)} · t09 {g1_ok['t09']}/{len(ROWS09)}")
    print(f"G2 notes≠narration: {','.join(g2) or 'none'}")
    print("CLAIM FIXES FAIL" if fails else "CLAIM FIXES PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
