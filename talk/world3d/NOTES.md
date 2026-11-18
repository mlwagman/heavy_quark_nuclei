# talk/world3d: the MRS talk as one continuous 3D World

Built 2026-09-30 from the storyboard (`.scratch/mrs-talk-world/storyboard/storyboard.md`, v3), the narration
(`narration_video/scene_01..15.md`; since round 18 the World's own, `~/mrs_talk_video/narration_world3d/`), the approved on-screen strings (`talk/world/js/s01..s15-*.js`, words and
numbers only), the paper's Tables I to V, the picture catalogue, and the capture contract (`render/deck_shim.mjs`).

## How to open it

- **The World:** double-click `talk/world3d/index.html` (Firefox or Chrome, no server). Click, Right, Space or
  PageDown advance one beat. Left or PageUp go back. Down and Up jump a scene. `n` toggles the notes strip, `f` fullscreen.
  `o` opens the overview; in it, click a badge or a legend line, or type a scene's number, to open that scene.
- **Deep link:** `index.html?beat=S07b2` opens settled on that beat (scene 7, beat 2 done). `?fps` shows the fps
  counter and the current beat id at the top left. Both can be combined.
- **The presenter:** double-click `talk/world3d/presenter.html`, press *Open the World*. It opens `index.html` in a
  second window, then shows the current beat's notes, the first sentence of the next beat, scene and beat numbers,
  and a clock. The keys and buttons in the presenter drive the World through postMessage, and a clicker in the
  World's window keeps the presenter in sync the other way.
- **Contact sheet:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs` drives all 75 beats through
  the shim and writes `contact/SNNbM.png`. `--only S07` for one scene, `--play S05` clicks through one scene in real
  time, `--firefox` loads it in Firefox, `--fps S07b2 S10b2` measures fps on the real GPU (headed Chromium).
- **Framing check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs --frame --mid` lists every board the stage
  edge cuts, at the end of each beat and at p = .25, .5, .75 inside it. It fails if a board the beat is showing is cut.
- **Beat check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-beats.mjs` (S12 b3, S10 b3, S07: round 11; the tower after S03: round 12).
- **Jump check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-jumps.mjs`: no beat may jump on its click.
- **Still check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-still.mjs [--only S03] [--selftest]`: between clicks no
  text may move (round 26). It holds every beat end for 12 s and reads the ink of every text.
- **Text check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-text.mjs [--mid] [--only S03]`: works on the ink (the
  line boxes of every text node, the pill of every chip, the shapes of every plot), not on the board boxes. It lists text on
  text, text under or over another panel, text or drawings outside their own panel, text cut by the stage edge, a stroke of a
  plot through one of its labels, and two plain boards set in one place. It fails on any of these at the end of a beat when
  both sides are drawn at 0.3 or more. `--mid` also lists p = .25, .5, .75 inside every beat.
- **Colour check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-colour.mjs`: every hadron as drawn (the colour each
  visible quark is painted with, not the charge it was created with) is colour-neutral at the end of every beat and at three
  moments inside it.
- **Shim self-test:** `NODE_PATH=$(npm root -g) node /home/gofer/mrs_talk_video/render/deck_shim.mjs --selftest --want 15,75,996 talk/world3d/index.html`
  prints `SHIM OK scenes=15 beats=75 sum_d=996` (round 19: S12 lost a beat of 18; without `--want` the shim checks the Deck's 15, 57, 1014).
- **Transient check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-transient.mjs [--only S05] [--strings FILE] [--selftest]`:
  every beat on a 30 fps clock; fails on any text or dot written in a beat and gone (faded or left by the camera) before it ends.
- **Map check:** `NODE_PATH=$(npm root -g) node talk/world3d/tools/check-map.mjs [--boxes]`: the scenes' regions on the overview
  frames, in pixels; fails on two regions closer than 8 px, a region on a legend or off the stage, or a shelf that stays put.
- **Notes:** `js/notes.js` is made from the narration by `python3 ~/mrs_talk_video/world3d/audio3d.py notes` (`--check` fails if stale);
  the engine throws if a scene's notes and beats differ in number.

## What was built

One three.js scene (`vendor/three/three.iife.js`, classic script, offline) laid out in depth along the storyboard's
Zoom loop. Every scene change is a camera move with at least one object carrying across. Nothing advances on its own:
every beat's `f(p)` is a deterministic function of `p`, so a scrubbed beat and a clicked beat look the same.

| file | owns |
|---|---|
| `index.html`, `css/world.css` | stage (1920x1080, scaled to the window), fonts (P052), board styles, math markup |
| `js/util.js` | `scenes`, easing, DOM and SVG helpers, math markup helpers, hadron names |
| `js/space.js` | renderer, bloom (EffectComposer + UnrealBloomPass, no MSAA, pixel ratio 1), a second "crisp" scene drawn after bloom, lights, ambient dust, the camera rig (`cam`, `samplePath`), factories: quark, link, glyph, plane, crisp line |
| `js/boards.js` | DOM boards anchored to 3D points and re-projected every frame (billboards, perspective scale, z-order by depth), one container per scene (`scenes[i].el`), re-measure on font and image load |
| `js/objects.js` | the persistent objects, one owner each: Family shelf (glyphs, labels, chips, named layouts with reflow), static pair, EFT tower, Borel floor + branch point (crisp) + contour, the \|term\| dots board, the band gauge, the number line, the walker cloud (200k points, GPU jitter/diffusion), the \|ψ\|² bump, μ rings and red zone, the lattice wall |
| `js/s01-s02.js` … `js/s13-s15.js` | the 15 scenes, 75 beats; each beat `{ d, cam, win, f(p) }`; `notes: NOTES[SC]` from `js/notes.js` (the narration) |
| `js/engine.js` | the shim API (`scenes`, `si`, `done`, `run`, `auto`, `notesOn`, `show`, `next`, `prev`, `applyUpTo`, `setAuto`, `ui`), deterministic state application, camera keyframes per beat, keys, deep link, presenter sync, fps counter |
| `presenter.html` | the speaker's second window |
| `tools/contact.mjs` | the contact sheet, play-through, Firefox and fps checks |

**State model.** Applying state for scene *s* with *n* settled beats runs every later beat at `f(0)` in reverse order,
then every earlier beat at `f(1)` in order, then the running beat at `f(p)`. The invariant every beat keeps: its
`f(0)` equals the state the objects it touches had at the end of the previous beat. Persistent objects are hidden
explicitly by the first beat of the scene after their last use (the gauge in S03 b1, S05 b1, S11 b1; the number
line in S10 b1, S11 b1, S14 b1; the Borel labels in S05 b4), so their next user finds them in a known state.

**Camera.** One keyframe path per beat (`cam: [K(pos, look), …]`, Catmull-Rom when more than two points), tweened
over the window `win` of the beat's `p` (default the first 55 %). Since round 18 every scene has its own place along one
winding path in talk order (`LAY.T` in `js/objects.js`; each scene's code keeps its own coordinates, `FR(SC)` carries them
into the World): the title top left, S02 to S05 down the left side, S06 to S08 along the bottom, S09 to S11 up the right
side, S12 in the middle, S13 above it, S14 at the top beside the title and S15 above the title. The shelf travels with the
story (S02, S08, S09, S10, S12, S14). A slow ambient camera drift and glow pulses run between clicks.

**Beat durations.** `d` uses the Deck's integer values (sum 1014), because the shim's self-test requires
`sum_d = 1014`. The narration seconds in `scene_NN.md` sum to 995.2 and cannot satisfy that gate; the per-beat
seconds are kept in the narration files and in the presenter's notes, and a beat's click tween lasts `d / 3` s
like the Deck engine.

## Choices the storyboard left open (and what was picked)

- **Palette (W5).** LO `#5b8fc9`, NLO `#e8a33c`, NNLO `#8db83a` from the paper's figures, lattice `#d64545`, warm
  white `#f3ecdf` for quarks and text, grey for remainders and prose. Fixed order is hatched (CSS repeating gradient
  or SVG pattern with the order's colour), MRS solid.
- **Em dashes.** The catalogue's intuition labels use em dashes; the build rule forbids them anywhere, so every
  label uses a colon instead ("colour singlet: schematic", "cartoon: not a computed wavefunction"). Words unchanged.
- **Dead click at scene ends.** None. The click after a scene's last beat moves to the next scene and starts its
  first beat, whose camera move is the scene change. The shim's `show(i); applyUpTo(0)` path still works as before.
- **Glyphs.** Small orbiting quark sets: mesons 2, baryons 3 (triangle links), tetraquarks 4. The generic S02
  tetraquark and the bound S12 states use the compact-centre geometry (two small quarks tight in the middle, two
  outside) so no bound state ever reads as four equal balls on a square. Top-row states are 0.62 scale and dimmer.
- **Shelf reflow.** Named layouts (`fam`, `s09a` … `s12`); when a glyph joins, the row re-spaces with a tween.
- **The S01 hero glyph.** Stays as its own object in S01, throws off the four ghosts that become the shelf, and fades
  during S02 b1 as the families light up.
- **S02 → S03 bars to floors.** The three DOM scale bars fade while three translucent floors grow from thin slabs to
  the tower's floors as the camera dives.
- **S03 chips on the link.** `V_R(r)` above, `C_F = 4/3` below the link; the third quark, the two `C_F/2 = 2/3` links
  and the shaded triangle exist only during b3 and fade at its end, as the storyboard says.
- **S04 board cluster.** Two columns: series and V(r)/F(r) plot on the left, \|term\| dots, band gauge and band stack
  on the right; the pair shrinks into the top-left corner. The Borel floor is at the bottom and the b4 camera tilts
  down to it (the E1 sentence).
- **\|term\| dots.** Schematic heights with the minimum at ℓ = 3, bracket "used (L = 3)" (the pilot's reviewed
  wording), last dots outside the board frame. S05 b1 splits each dot into a white V_l dot and a grey remainder for
  ℓ ≤ 2 only; S05 b2 folds the white column into the 𝒥 box corner and fades the board.
- **Borel plane.** Dark floor with a faint grid, u-axis with ticks 0, ½, 1; the branch point is a crisp white circle
  with a wavy cut drawn in the post-bloom scene (no bloom). The contour from 0 to the branch point is a glowing tube
  in the lit scene; the stretch past u = ½ is a blurred DOM band that flies into the Λ_R box in S05 b3.
- **S05 tags.** "exponent 1 + b" and "at t = 1/(2β₀) (u = ½)" fly from the floor into the arguments of 𝒥 (target
  measured from the equation board's DOM).
- **S05 b3 cancellation.** The Λ_R box carries a CSS glow from S04 b1; the arriving band and the glow fade together
  and the box border turns sharp; the label sits as a line under the equation. The "Λ₁ absorbed into m_Q" chip parks
  right of the equation and flies to the shelf in S08 b3, where it splits into two small dials under cc̄ and bb̄.
- **S06 space.** 1 fm = 12 world units. Rings of radius ∝ 1/μ around one quark (three sets for λ = ½, 1, 2 in b4),
  the |ψ|² bump as a translucent hill r² e^{−2r/a₀} along the axis, the red zone a translucent slab beyond 1 fm.
  The μ′ plot uses r ∈ [0, 1.2] fm, μ = 0.1973/r GeV, all curves labelled "schematic".
- **S07 cloud.** 200 000 points sampled from |ψ|² = e^{−2r/a₀} in the relative coordinate around quark A; the
  partner quark fades. VMC breathing is a damped oscillation of the spread; GFMC steps are hashed Gaussian offsets
  advancing with τ; glow follows exp(−V δτ) ∝ 1/r. Trial-state insets: pair, triangle, regular tetrahedron
  (six equal links, three grey shades for a, b, c).
- **S08 to S12 raster figures.** Paper PNGs on white cards inside dark panels, captions in the World's palette.
  In S11 the fixed-order panel is dimmed until b4 and the MRS panel until b2.
- **Number line.** One persistent SVG board reconfigured per use (η_t, B_c, Ω_bbb, the S13 payoff); bands are
  drawn as solid envelopes with the caption "max deviation", never with error-bar caps.
- **S12 unbound states.** The bbb̄b̄ and ccb̄b̄ glyphs form beside the top row, split into two mesons (links between
  the pairs fade), and disappear.
- **S13 gauge landing.** The band gauge flies from its S10 place onto the table's first row and fades as the row
  reveals. The drift meter and band stack are scene-owned boards and simply become rows.
- **S15 threads.** Crisp lines from B_c, η_t, Ω_ccc, Ω_bbb and bbt̄t̄ to the "Lattice QCD comparisons" and
  "Bounds, tetraquarks, toponium" groups.


## Round 2 (2026-09-30, after the first review): legibility, calmer look, continuity, overview

- **Focus tiers (js/boards.js, js/engine.js).** Every board records the beat that last showed it (`touched`).
  The boards the current beat touched are full size and opacity; the rest of the current scene sits at 42 % and
  0.9 scale; the previous scene's boards stay in the distance at 16 % (continuity while the camera flies); older
  scenes are hidden. A beat that only reveals parts of an existing board calls `board.touch()`. Persistent boards
  (gauge, number line, shelf chips) follow the same rule, so old numbers dim once the talk moves on.
- **Type.** Base board text 28 px (was 24), `.big` 32, `.small` 22, `.tiny` 18, labels 20, chips 22, tables 27.
- **Palette.** Background `#0e1015`, panels rgba(12,14,19,.76) with a 1 px 10 % border, ink `#ece5d8`.
  Bloom strength 0.38, radius 0.3, threshold 0.72 (was 0.5 / 0.3 / 0.72 then 0.85); quark emissive 0.48
  (was 0.85); walker cloud alpha 0.014 (was 0.55); tower floors at 3 % opacity; dust 22 %.
- **Re-layout so consecutive scenes never share space.** S04 boards at x ≈ −32 (left) and 20 (right), S05 boards
  at x ≈ 46 … 84 with the Borel axis moved under them (u = 0 at x = 20, u = ½ at 35). S07 slides the pair 16 units
  right so the cloud and the S07 boards share the frame with S06's boards ghosted between them. Results scenes:
  S08 above-left of the shelf (x −56 … −8, y 114), S09 above-right (x 48, y 90 … 120), S10 above the centre
  (y 92 … 136), S11 (x −122 … −52, y 200 … 232), S12 (x 12 … 134, y 152 … 236), S13 (y 262 … 292),
  S14 (side columns x ±78, y 108 … 172, whole shelf in frame at distance 200), S15 (y 330).
- **Overview (key `o`, presenter button, `?overview`).** A real-time 2.2 s tween to a camera 680 units back
  over the whole World. Every scene up to the current one shows all its boards (90 %), a station label per scene
  (`scene.station`), and an orange Catmull-Rom journey line through the stations. Scenes that share real space
  are exploded only in the overview by `scene.ovOff` (S01 to the upper left, S03 to the left of the tower,
  S10 to the right of the shelf, S12 up-right, S13 left) and S14's boards shrink to 0.5 there (`ovScale`).
  `tools/contact.mjs --overview S15b1` renders it to `contact/overview-S15b1.png`.
- **No dead click** at scene ends (unchanged); the previous scene stays visible as ghosts, so the fly-through
  reads as one space.

### Verified on the final render of round 2
- `SHIM OK scenes=15 beats=57 sum_d=1014`; 57 frames + overview rendered, no page errors.
- `--play S05` real-time click-through: no errors; `--firefox`: no errors; presenter sync OK.
- fps on the AMD GPU: S07 b2 60.0, S10 b2 60.0 (vsync-capped).

### Known small things to polish next (after round 2; all four addressed in round 3 below)
- Overview: station labels near the shelf are dense; the exploded S12/S13 clusters could move further apart.
- S07 cloud reads as a soft glow rather than visible points at 200k; fewer, larger points may read better.
- Top-row shelf labels wrap to two lines at the larger type.

## Round 3 (2026-09-30): the four open items

- **Overview map is readable from the back of the room.** The overview camera is 900 units back (`OVCAM` in
  js/engine.js; the camera far plane went 900 → 2400 in js/space.js, since the old value clipped every anchor
  that sat on it). Station labels are 160 units wide, 34 px type, which lands at 20 px on the 1080p stage
  (was 11 px). Every scene now has its own district; the offsets apply only in the overview (`ovOff`) and the
  boards slide back when you leave it:

  | scene | station (x, y) | ovOff | scene | station (x, y) | ovOff |
  |---|---|---|---|---|---|
  | S01 | −215, 140 | −195, +30 | S09 | 160, 100 | none |
  | S02 | 130, 48 | none | S10 | 300, 36 | +300, 0 |
  | S03 | −210, 30 | −200, 0 | S11 | −270, 262 | −190, 0 |
  | S04 | −46, 12 | none | S12 | 200, 352 | +120, +70 |
  | S05 | 45, −78 | +30, 0 | S13 | −150, 322 | −130, 0 |
  | S06 | −149, −78 | −120, −40 | S14 | 10, 250 | 0, +60 |
  | S07 | 205, −52 | +180, 0 | S15 | 0, 366 | none |
  | S08 | −40, 168 | none | | | |

  A Playwright probe measured every label and every scene's board box on the stage: no label overlaps another
  label, no label sits on another scene's boards (two touch within a 4 px margin), no two scenes' boards overlap.
  The journey line now grows with the talk (`setDrawRange` up to the current station), so a mid-talk overview
  does not loop through empty space.
- **Walker cloud reads as a sample.** 30k points (was 200k), point size 2.4 (was 1.7), per-point alpha 0.13
  (was 0.014), still additive. Single walkers are visible at the edge; the core still saturates behind the pair.
  Knobs: `N`, `uSize` and the `0.13` in the fragment shader of `WALK` in js/objects.js. fps unchanged (60).
- **Top-row shelf label.** The η_t label breaks after "limit" (a `<br>`, same words) instead of inside the
  parenthesis.
- **Ghost level is one constant.** `GHOST = 0.16` at the top of js/boards.js is the opacity of the previous
  scene's boards. It was left at 16 %; tune it in the room.

### Verified on the final render of round 3
- `SHIM OK scenes=15 beats=57 sum_d=1014`; all 57 frames + `overview-S15b1` and `overview-S07b2` rendered,
  no page errors; a real-time `o` toggle from S07 b2 out and back: no errors, camera returns to the beat.
- fps on the AMD GPU: S07 b2 60.0, S10 b2 60.0 (vsync-capped). Presenter sync OK. No em dash in the sources.
- Remaining judgement calls for the room: GHOST 0.16, cloud alpha 0.13, and whether the exploded map should
  come closer (`OVCAM` z) once the station labels are judged on the projector.

## Round 4 (2026-09-30): colour style as themes

The round 2 review asked for a professional colour style; rounds 2 and 3 had only turned the "black space, glowing
orbs, starfield" look down. Round 4 replaces it. Every colour now lives in `css/world.css` as a variable, the 3D
side reads them through `TC()` / `TN()` in `js/util.js`, and three themes exist:

| theme | how to open | look |
|---|---|---|
| `slate` (default) | `index.html` | slate charcoal, matte ivory spheres, no glow, faint dust, neutral off-white ink |
| `paper` | `index.html?theme=paper` | warm paper, dark ink, slate-blue spheres, no glow, normal blending for particles |
| `space` | `index.html?theme=space` | the round 1 to 3 look (black, ivory glow, starfield), kept for comparison |

Per theme: background, ink levels, panel and chip fills and borders, quark colour and emissive share (`--q`,
`--q-glow`), link shades (`--link`, `--link-a/b/c`), crisp wires (`--wire`, `--wire-dim`, `--grid`), fills
(`--floor`, `--soft`), particles (`--dust`, `--dust-a`, `--walk`, `--blend` add|normal), bloom strength
(`--bloom`) and the journey line. The W5 order colours (LO blue, NLO orange, NNLO green, lattice red) are fixed in
every theme. Glow multipliers at call sites (`glow: 1.9` for the S03 pair, `setGlow(k)`) are relative to the
theme's `--q-glow`, so a theme with `--q-glow: 0` has no emissive anywhere.

`tools/contact.mjs --theme paper …` renders any mode with that theme into `contact/theme-paper/`.
Comparison sheets: `contact-sheets/sheet-theme-slate-v2.png`, `-paper-v2.png`, `-space-v2.png` (S03 b2, S05 b2,
S07 b2, S10 b2, S14 b2 and the overview in each). To make another theme the default, move its block into `:root`.

## Round 5 (2026-09-30): white, colour, the map at the end, slower animation

Review 3 asked for a white background, colour, a clearer and prettier overview that plays by itself before the
references, and slower or physically clearer animations.

- **White is the default theme** (`:root` in css/world.css): white stage, dark ink, slate-blue spheres, no glow,
  normal blending for the particles. `?theme=slate` (dark, matte) and `?theme=space` (the round 1 to 3 look) remain.
- **Section colours.** Four accents, fixed per theme: I framework S01 to S03 (indigo), II method S04 to S07 (teal),
  III results S08 to S13 (plum), IV conclusions S14 to S15 (bronze). Each scene container carries `--sec`; panels get a
  3 px top border, a 6 % tint and a coloured `h2`; the map's station labels, cards and rail use the same colours. The
  W5 order colours (LO blue, NLO orange, NNLO green, lattice red) are untouched and still mean what they meant.
- **`setGlow(k)` now also drives the sphere colour** (`--q-lo` at k = 0, `--q` at 1, `--q-hi` above 1), so the dim
  top row and the S08/S14 flashes read on white where emissive light does nothing.
- **The overview is a map.** Fifteen cards in reading order (a 3 x 5 serpentine) 1000 units above the World, camera
  900 back. Each card: the station label (section colour), the scene's boards scaled as one block into the card
  (`ensureCells`: the box of the scene's visible fixed boards fitted into 186 x 104 units), borders and tints kept
  visible at thumbnail scale (`--inv`). A tube rail coloured by section runs above the cards through the stations, rows
  joined outside the cards; it grows with the talk. Persistent objects (shelf, gauge, number line, 3D) stay in the
  World below and are out of frame in the map.
- **The map plays by itself at the end.** S14 b4: "Next" and "Thank you" land, then the World lifts off into the map
  (`OV.beat`, p 0.55 to 0.95) and stays there for questions. S15 b1 descends from the map onto the references
  (p 0 to 0.55). Key `o` still opens the map anywhere; the loop uses k = max(manual, beat).
- **Pace.** `SPEED` 3 → 2: a beat's tween lasts d/2 s (S05 b2: 15 s), every reveal 1.5 x slower, still inside the
  narration. Transients fixed: S03 b1 the "integrate out" words stay ~4 s each; S03 b3 the baryon and its colour-factor
  chips stay through b4 and leave in S04 b1; S08 b1 the walker cloud condenses over p 0.1 to 0.75; S09 b3 the two
  mass chips travel ~3.5 s before the B_c glyph forms; S12 b3 the unbound states fade only at the very end.
- **Physics.** S12 b1 no longer morphs the Detmold bound into the R = 0 threshold: the Detmold line fades where it
  was and the threshold appears on its own, with the same label ("a different line, a different quantity").

### Verified on the final render of round 5
- `SHIM OK scenes=15 beats=57 sum_d=1014`; all 57 frames, the map at S14 b4, the lift-off (p 0.75, 0.9), the descent
  (S15 b1 p 0.25, 0.5) and a manual map at S07 rendered, no page errors.
- `--play S14` (the lift-off tween, 10 s) then the click into S15; `--play S03`, `--play S12`: no errors. Firefox: no
  errors. Presenter sync OK. fps S07 b2 60.0, S10 b2 60.0.
- Sheets: `contact-sheets/sheet-map.png` (transitions), `sheet-final-check.png`, and `sheet-00..14` rebuilt from the
  white frames. `contact/map-*.png` hold the map frames.

## Round 6 (2026-09-30): the colour style of talk/mrs-talk-animated.html

Review 4 asked for the colour style of the reference deck. The themes of rounds 4 and 5 (white, slate, space) and the
four invented section colours are gone; `css/world.css` now carries that deck's two palettes and its type.

- **Palette.** Default = the deck's light theme: stage `#F6F7F4`, white cards with a 1 px `#C9D0D9` border, ink
  `#1A2233`, muted `#5D6779`. `?theme=dark` = the deck's dark theme (`#0A0E17`, panel `#101624`).
- **Accents with the deck's meaning.** Rose = the problem, teal = our result, amber = key parameters (`.rc`, `.tc`,
  `.ac`). Applied to the same headings and phrases as in the deck: The obstacle, Symptom, Factorial growth, Where does
  the renormalon go?, What is new, UNBOUND (rose); MRS, The payoff, BOUND (teal); the three S05 captions
  (teal / amber / rose); ~700 MeV, the lattice gaps, the scale variation (rose); GFMC statistics, the trial forms,
  "two orders of magnitude" (teal); "resum the growth" (amber). The renormalon haze and the haze band are rose.
- **Order colours (W5)** are the deck's: LO `#3B66B0`, NLO `#A06E12`, NNLO `#3E8E4E`, lattice `#C03A3A`
  (variables `--lo`, `--nlo`, `--nnlo`, `--lat`; the lattice wall and the red zone read `--lat`).
- **Type.** Body text is sans (system-ui), math and italics serif (P052), card headings small uppercase letter-spaced
  muted (`h2`; math, numbers and units inside a heading keep their case), chips are pills, banners serif italic rose,
  table heads uppercase. The S01 title is serif, as the deck's `h1`. Number-line titles stay sentences (`h2.nt`).
- **Quark spheres** were coloured by index like the deck's glyph dots in this round; round 9 replaced that by
  colour charge. `setGlow(k)`: k < 1 recedes toward the background, k > 1 is a size pulse.
- **The three scales keep one colour each**: m_Q teal, m_Q v amber, m_Q v² rose, on the S02 bars and on the edges and
  names of the S03 tower floors. S02 also takes the deck's teal triangle nodes and rose excess bump.
- **Map sections** use the deck's four colours: I teal, II amber, III rose, IV blue. Panels no longer carry a section
  tint or top border (the deck's cards are plain); thumbnails in the map still do.
- `tools/contact.mjs --map` renders the lift-off, the settled map and the descent into `contact/map-*.png`.

### Verified on the final render of round 6
- `SHIM OK scenes=15 beats=57 sum_d=1014`; 57 frames + map frames rendered, no page errors; `--play S02`, `--play S14`
  clean; Firefox clean; `--theme dark --only S05` renders; fps S07 b2 60.0, S10 b2 60.0; no em dash in the sources.

## Round 7 (2026-09-30): four 3D animations reworked for an audience that knows basic QCD

The author named four first-half animations as unclear and accepted a replacement for each. No on-screen string was added.

- **S03 b1, the EFT tower.** The camera no longer sinks through the floors in one move. It stops on each floor
  (`camU` in js/s03.js; the beat runs 1.5 x longer, `slow`). On each floor the same pair is shown at that floor's
  resolution, by its own motion only (no gluon cloud): fast and wide on QCD, slower on NRQCD, static on pNRQCD, where
  the link that becomes V_R appears (`PAIR.st.jit`, `fq`, `lkOp`). Each step's words stay while the camera goes down.
  The S02 boards, which hang at the QCD floor's height, fade as the beat starts (`scene.ghost`).
- **S04 b4 to S05 b3, the Borel plane.** It is now a small tilted inset (`BOREL.place`), beside the V(r) plot in
  S04 b4 and under the closed-form box in S05; it leaves once Λ_R is sharp (S05 b3). The |term| vs order dots carry the
  argument: each term splits into its renormalon part (amber) and its remainder (teal); the three remainder dots fly
  into the first box, the renormalon column gathers and flies into the closed form 𝒥, each box takes its colour; what
  the sum cannot fix stays on the plot as a rose haze and slides into the Λ_R box in b3. The two Borel tag chips
  ("exponent 1 + b", "at t = 1/(2β₀)") are gone.
- **S06, the scale.** The μ = 1/r rings are gone. A ruler runs from one quark along the pair axis: teal at short
  distance, red beyond the point where μ = 1/r would fall below Λ_QCD, with a red plane and outline. The second quark
  walks out along the ruler into the red part; the |ψ|² hill sits over both. In b2, as μ′ saturates, the red fill goes
  and the teal extends (`PSI.st.clip`); the outline stays as the place μ′ never goes.
- **S07 b2, S08 b1, the walker cloud.** During GFMC the cloud relaxes in step with ΔE(τ): far walkers die out
  (`uRelax` in the shader) and the cloud tightens (spread 1.06 → 0.8). At S08 b1 it fades in place; nothing flies
  onto the shelf.
- Also: the previous scene's ghost level is 10 % (`GHOST`), `tools/contact.mjs --at S05b2:0.69,…` renders frames
  inside a beat, the engine accepts `camU` (a beat's own camera progress) and `slow` (its own pace).

### Verified on the final render of round 7
- `SHIM OK scenes=15 beats=57 sum_d=1014`; 57 frames, map frames and in-beat frames rendered, no page errors;
  `--play` S03 (b1 now 13.5 s), S05, S06, S07 clean; Firefox clean; fps S07 b2 60.0, S10 b2 60.0.
- Sheets: `contact-sheets/rework-1.png` (S03 stop, S05 b1, b2 mid), `rework-2.png` (S05 b2 end to b4),
  `rework-3.png` (S03 stops, S06 b1, S07 b2).

## Round 8 (2026-09-30): the overview is the real World, not a diagram

Review 6 rejected the card grid with its serpentine rail as a fabricated path. The overview is now a plain pull-back
over the World as it is. For that to be honest, no two scenes may share a place, so the World was re-laid.

- **Re-layout.** `tools/contact.mjs --layout` lists every pair of boards from different scenes that overlap in true
  space. Before: 54 pairs (S01 on the shelf; S03 to S07 piled around the pair; S08, S10, S12, S14 piled above the
  shelf). After: "no two scenes share a place". What moved:
  - S01: its title, hero, QR and thesis sit 150 units left of the shelf. In S01 b2 the four family copies line up
    under the title; S02 b1 carries them to the shelf with the camera.
  - The workshop (tower, pair, S03, S06, S07) sits 25 units lower; the tower floors are 11 apart (`LAY.floorsY`).
    S03's three boards are left of the pair. S04 is 57 lower, S05 is 57 lower and 30 further right.
  - S10's boards form a column right of the shelf (x 114), its row headers sit at the right end of the chip rows.
  - S12's verdict boards and sketch moved clear of S08 and S10. S14 walks along the shelf: two boards left of it
    (b1, b2), the predictions above its centre (b3), Next and Thank you right of it (b4). S15 is 4 higher.
- **The overview** (`o`, or by itself in S14 b4): the camera pulls back to (4, 118, 770). Nothing is moved or scaled.
  The line is the track the camera really flew: the look-at point at the end of every beat, in order, as a tube
  coloured by section and drawn up to the current beat (cut by arc length). A numbered badge sits on the track where
  each scene opens. The titles are a legend fixed to the stage at the left (1 to 7) and right (8 to 15); scenes not
  yet visited are greyed. The cards, the rail and the per-scene offsets are gone.
- The tab title says "build r8".

### Verified on the final render of round 8
- `--layout`: no two scenes share a place. `SHIM OK scenes=15 beats=57 sum_d=1014`. 57 frames, the pull-back, the
  settled overview, the descent and a manual overview at S07 rendered with no page errors. `--play` S01, S03, S10, S14
  clean; Firefox clean; no em dash.
- fps: S07 b2 60, S14 b4 (the overview) 60, S10 b2 60 in the last run; two earlier runs of S10 b2, taken right after
  heavy rendering, read 47 and 49.

## Round 9 (2026-09-30): quark colours are colour charge

Review 7: the sphere colours must mean the quantum colour, and every bound state must mix to white.

- Quarks are red, green, blue (`--qr`, `--qg`, `--qb`); antiquarks are the complements: anti-red cyan, anti-green
  magenta, anti-blue yellow (`--qar`, `--qag`, `--qab`). Both themes define all six.
- `CHARGE` in js/space.js fixes the assignment by hadron type: a meson is red with anti-red; a baryon is red, green,
  blue; a tetraquark is anti-red, anti-green (the compact antiquark pair of the glyph) with red, green. When an unbound
  tetraquark falls apart in S12 b3, each of the two mesons is a colour with its own anticolour.
- The S03 pair is Q red, Q̄ anti-red. When the third quark arrives (S03 b3) the pair is the QQ of a baryon: the second
  sphere turns from cyan to green and the third is blue; it turns back when the third quark leaves in S04 b1.
- `tools/contact.mjs --layout` also prints a colour check: "26 hadron glyphs, all colour-neutral" (the shelf, the S01
  hero, the S07 insets). The S01 label "colour singlet: schematic" now describes what is drawn.
- This departs from rule W5 of the storyboard ("hue never means colour charge") at the author's request. The order
  colours in plots and bands are unchanged; red, green and blue spheres now sit near the blue, green and red of LO,
  NNLO and lattice, so the two codes share hues but never the same kind of object.
- The tab title says "build r9".

### Verified on the final render of round 9
- `SHIM OK scenes=15 beats=57 sum_d=1014`; `--layout`: no two scenes share a place, 26 glyphs colour-neutral; 57 frames
  and the overview frames rendered, no page errors; `--play S03` clean; Firefox clean; dark theme renders; no em dash.
- Sheet: `contact-sheets/sheet-colour.png`.

## Round 10 (2026-09-30): framing. What a beat shows is whole on the stage, and as large as its frame allows

Round 8 gave every scene its own place, at a price: three trade-offs (the overview, S14, S10). This round works on those
three and on the stage-edge cuts the same re-layout left behind. No on-screen string was added, removed or reworded.

- **A check for it: `tools/contact.mjs --frame [--mid]`.** Per beat, every visible board whose box crosses the stage edge.
  FOCUS = a board the beat is showing (opacity >= .8): cut = the run fails. DIM = the rest of the scene and the persistent
  boards: listed. Before this round, `--frame --mid` found 14 focus boards cut; now 0. It also notes focus boards left
  wholly off stage when their beat ends; the seven it lists are intended (S03 b1 leaves the QCD floor, S10 b2 leaves the shelf).
- **Cuts fixed.** S01 b1: the QR block and its URL (4 units left). S02 b1: the "cartoon" shelf label (now right-aligned
  under the line's right end; the line no longer runs through the words, in every scene). S04 b3: the band stack (the
  camera eases back 4 units as it lands). S07 b1, b2: the walker label (under the cloud, between the stage edge and the
  S06 plot). S03 b1: a floor's name comes on as the camera leaves the floor above; at each stop the next name was half
  under the stage edge. S12 b3, b4: the frame takes the whole shelf, the edge used to run through its main row.
- **S10: one camera frame per beat instead of one for the scene.** b1 the shelf (8.5, 75, 102). b2 starts there (wall,
  lattice row), then pulls back over shelf and boards and closes in on the boards (114, 101, 80); the S09 ghosts leave
  with the shelf. b3, b4 stay on the boards. b5 arcs back, stops on the shelf frame while the six top masses land, then
  pulls back over the whole scene (44.5, 88, 162) for the summary board (`camU`, `slow: 1.5`, 9 s).
  On-stage scale of what each beat shows (1.00 = the CSS pixel size): b1 chips 0.70 to 0.76 → 1.01 to 1.04; b2 number
  line and statement 0.77 → 1.38; b3 gauge 0.78 → 1.38; b4 bars 0.77 → 1.39; b5 top masses 0.63 to 0.68 → 0.90 at the
  stop (0.55 in the closing wide frame); the lattice caveat label 0.40 → 0.81 (it is wider and sits on the wall's top edge).
  The summary board is wider (62 units) so it holds 0.78 in the wide frame. The gap percentages sit above their bars.
- **S14: three framings, the whole shelf in each.** b1, b2: the two claims above-left of the shelf (left of S08, above
  S01), camera (-42.5, 101.5, 184). b3: closer, (14.75, 93.75, 119), predictions and thesis in the free slot above S09.
  b4: Next over Thank you below-right of the shelf, camera (29.2, 62.2, 139.5), then the lift-off. Scales: claims 0.97
  and 0.99, predictions 0.96, thesis 0.85, Next 0.84. The b4 frame is tight on purpose: 1 unit lower and the tower's
  top floor shows, 1 unit higher and the b3 boards do.
- **The overview.** The camera is fitted to the World's box as it is now (x -202..173, y -115..359): (-14.5, 122, 735),
  about 5 % closer and centred. While it opens, the track is retraced: it grows from the title to the current beat, and
  each scene's badge and legend line come on as it passes (`OV.trace` in S14 b4, the `o` key does the same; on the way
  back down to the references the whole track stays). `scene.home` places a badge when the scene's first look-at point
  is already another scene's: S10 (its old place) and S14 (on its first claim), both of which now open on the shelf.
- **What the overview cannot do.** Board text needs an on-stage scale of about 0.4 to be read; the overview gives 0.12
  to 0.14. A frame that gives 0.4 shows about 325 x 183 units; the scenes' boards cover about 90 000 square units, the
  frame 59 000. So no layout makes the boards readable in one frame: the overview is read through the legend, the badges
  and the track. Re-laying the World in landscape would buy at most a factor 1.3 (text still 5 to 6 px) and would take
  the side thirds the legend stands in. Not done.

### Left as they are (listed by `--frame`, all DIM)
- S06, S07: the tower's floor outlines and the NRQCD / pNRQCD names stay from S03; in S06 the names are cut by the left
  edge and the rose floor outline sits near the red zone's outline. Hiding the tower after S03 would follow the rule for
  persistent objects, but it also takes the floors out of the overview. A call for the author.
- S07 b3, b4: the camera tilts down to the trial states and the dim VMC board is cut at the top.
- S03 b3, b4: the dim NRQCD name is cut by the left edge. Other dim boards cross the edge only while the camera moves.
- `contact/S08b*-mid.png`, `contact/theme-dark/`, `contact/firefox-S07b2.png` and the older `overview-*` and `map-*-p90 /
  -p50` frames are from earlier rounds and were not re-rendered.

### Verified on the final render of round 10
- `--frame --mid`: 0 focus boards cut (14 before). `--layout`: no two scenes share a place; 26 glyphs colour-neutral.
- `SHIM OK scenes=15 beats=57 sum_d=1014`. 57 frames, the map frames, both overviews and all 53 in-beat frames
  re-rendered, no page errors. `--play` S03, S04, S07, S10, S14 clean (S10 b5 now 9 s). Firefox clean. No em dash.
- fps on the AMD GPU: S07 b2 60.0, S10 b2 60.0, S14 b4 (the overview with the retraced track) 60.0.
- Sheets `contact-sheets/sheet-00..14` and `sheet-map` rebuilt from these frames.

## Round 11 (2026-09-30): three beats reworked

Same rules as round 10: no on-screen string changed, no two scenes share a place, nothing cut by the stage edge.
`tools/check-beats.mjs` states what each of the three must show; it was written first and failed on the r10 build (5 of 8).

- **S12 b3, in two stages** (`camU`, `slow: 1.5`, 19.5 s). First close on the shelf, camera (15, 77, 118): the three bound
  states join the top row, then the two unbound ones form beside it and fall apart; they stay, apart, until the end of
  the beat. Then the camera pulls back to (59, 115, 190) and the two verdict boards come on with the numbers. On-stage
  scale of the bound-state names 0.41 → 0.66, of the "top decays" chips 0.47 → 0.77; the unbound glyphs at 12.2 px per
  unit (8.2 before). The "sketch" caveat moved from under the boards to the right end of the top row (71, 79), beside
  the glyphs it is about.
- **S10 b3, the gauge keeps the 700 MeV width.** A gauge row can carry `ghost: true`: its full width stays as a dashed
  outline and `GAUGE.width()` collapses the hatched fill inside it, so "~700 MeV" stays at the end of the outline and
  the MRS sliver sits under its left end. Before, both rows ended as slivers and the ratio was gone from the board.
  The same gauge flies into the S13 table. The other users of the gauge (S02 b5, S04 b3) are unchanged.
- **S07, the middle of the stage.** The S06 boards stay as ghosts while the camera pans, then leave (`scene.ghost`,
  p .5 to .75 of b1). The pair now slides 20 units right (was 16), as far as the S06 boards' place allows, so the whole
  cloud is on the stage. It cannot go to the centre: that place is S06's, and an overview opened during S07 would show
  the cloud on the S06 plot.

### Verified on the final render of round 11
- `check-beats.mjs`: 8 of 8. `--frame --mid`: 0 focus boards cut. `--layout`: no two scenes share a place.
- `SHIM OK scenes=15 beats=57 sum_d=1014`. 57 frames, map frames, both overviews and the in-beat frames re-rendered,
  no page errors. `--play` S07, S10, S12 (b3 now 19.5 s), S13 clean. Firefox clean. No em dash.
- fps on the AMD GPU: S12 b3 60.0, S10 b3 60.0, S07 b2 60.0.
- Sheets `contact-sheets/sheet-00..14`, `sheet-map` rebuilt; `sheet-r11-ideas.png` shows the three.

## Round 12 (2026-09-30): the tower stays after S03, at rest

The author's call on the open item of round 10: do not hide the tower, improve it. S06 and S07 play on its pNRQCD floor,
so the floors and their names stay. What was wrong: in S06 the NRQCD and pNRQCD names were cut by the left edge ("CD"),
the rose floor outline competed with the red zone's outline, and in S07 the outlines ran through the cloud at full strength.

- **`TOWER.rest(k)`**, set by S04 b1 once the tower is out of the frame (p .45 to .75). At rest the outlines are at half
  strength (0.35, was 0.7; the S06 red zone's is 0.8) and the floor tints at 0.03 (was 0.05). Each name moves from the
  front edge of its floor to the back edge (z 10 → -28), same x and y: the overview and `--layout` see no change. There
  the S06 and S07 frames hold the names whole, and smaller (on-stage scale about 0.7, was 1.3). S03 itself is unchanged.
- **S03 b3, b4:** the camera is 1.5 to 2 units higher and 2 further back, (-54, -11, 56) looking at (-54, -22.5): the
  NRQCD name above the pair is whole (the top edge cut it by 35 px). Boards about 4 % smaller in those two beats.
- **S14:** the predictions board and the thesis line moved 2.5 to 3 units up (the S12 sketch label left that room in
  round 11); Next and Thank you 2 up and 2.5 right; b4 camera (30.3, 65, 141). The b4 frame now clears the tower's top
  floor and its name by more than a unit on both sides (round 10 had 1 unit, and the name at the back edge would have shown).
- `tools/check-beats.mjs` carries the tower checks (12 of its 20), written first: on the r11 build the four S06 ones and
  the two S03 ones failed.

### Verified on the final render of round 12
- `check-beats.mjs`: 20 of 20. `--frame --mid`: 0 focus boards cut; dim boards cut at the end of a beat: only the VMC
  board in S07 b3, b4 (the camera tilts down to the trial states). `--layout`: no two scenes share a place.
- `SHIM OK scenes=15 beats=57 sum_d=1014`. All frames re-rendered, no page errors. `--play` S03, S04, S06, S14 clean.
  Firefox clean. fps S06 b2 60.0, S03 b3 60.0. No em dash.

## Round 13 (2026-09-30): re-analysis; one state bug and two click jumps

- **`tools/check-jumps.mjs`**: the engine's invariant, tested. For every beat it compares the settled state before it with
  the beat at p = 0 (boards, 3D objects, camera, ghosts); a difference is a jump on the click. It found two, now 0 of 57:
  S05 b1 (the Borel branch point dropped from 1 to 0.45 at once; it now dims over the first 30 %) and S09 b3 (the η_t
  number line was swapped for the empty B_c one on the click; it now leaves first, and the B_c line comes once the glyph
  has formed). Checked against an injected jump: it reports it.
- **The renormalon dots leaked into S04.** In S05 b1, `TERMS.fold(0)` ran after `TERMS.split()` and put all ten amber
  dots at their final place at full size from p = 0 on. S04 inherits that state, so S04 b2 to b4 showed the amber dots
  beside the plain ones before the split was introduced, and the split itself had no motion for them. `fold(0)` now runs
  first. S04 shows the plain dots only; in S05 b1 each amber dot leaves its term dot, order by order.
- `tools/check-beats.mjs` has the checks for both (7 more, 27 in all; 6 of the 7 failed before the change).

### Verified on the final render of round 13
- `check-beats.mjs` 27 of 27, `check-jumps.mjs` 0 of 57, `--frame --mid` 0 focus boards cut, `--layout` clean,
  `SHIM OK scenes=15 beats=57 sum_d=1014`, all frames re-rendered with no page errors, `--play` S04, S05, S09 clean.

### Measured in the re-analysis (for the next ideas; nothing changed for these)
- What a beat shows covers little of the stage: median 9 % of the stage area (focus boards, settled beats); 30 of 56
  beats under 10 %. A camera that framed only the beat's own boards could be 1.5 x closer or more in 30 beats.
- Smallest text a beat shows: median 16 px on the 1080p stage, mostly the uppercase card headings (17 px at scale 1).
- Camera: 29 of 57 beats move it. Peak speed of the look-at point, in stage widths per second: S06 b1 2.0, S03 b1 1.5,
  S12 b3 1.1; every other move is under 1.0.

## Round 14 (2026-09-30): smooth zoom-pan camera, main boards that do not dim, the emptied |term| board leaves

Ideas 1, 2 and 8 of the round 13 list. No on-screen string changed.

- **Smooth zoom-pan (`zoomPan` in js/space.js).** A camera leg between two keyframes is no longer a straight line: it
  follows van Wijk and Nuij's path (2003; `RHO` = √2, their value): the view pulls back while it travels and closes in on
  arrival. A short move stays a straight line (S04 b3: 80 → 82 → 84); a long one shows both ends half-way (S02 b1: the
  camera is at 160 half-way between 95 and 108). `samplePath(kfs, u, stops)` uses it for two-keyframe moves and for paths
  the camera halts on at every keyframe (a beat with `camU`: S03 b1, S10 b5, S12 b3). Paths of three or more keyframes
  flown through without a halt stay Catmull-Rom (S08 b1, S11 b1, S13 b1, S14 b1, S15 b1).
  Peak pan speed, in stage widths per second, before → after: S06 b1 1.67 → 0.57, S12 b3 0.98 → 0.66, S03 b1 0.91 → 0.71,
  S01 b1 0.85 → 0.73, S02 b1 0.76 → 0.64. The fastest is now S10 b5 at 0.79. (S03 b1's dive into the tower is a zoom:
  1.3 → 1.2 e-folds per second at its peak.)
  Three hand-placed pull-back keyframes went, because the path now does it by itself: S10 b2, S10 b5 (round 10) and
  S06 b1 (its close pass by the pair before rising; it made the fastest pan of the talk).
- **`keep: true` on a board:** the scene's main board stays at full strength while its scene is the current one, instead
  of dimming to 42 % when the beat turns to another board. Set on the S05 closed form and the S13 table. Once the scene
  is over they are ghosts like any other board.
- **S05 b3:** the |term| board leaves with the Borel inset, once its haze has gone into the Λ_R box (p .45 to .62). It
  used to stay as empty axes for the rest of b3 and leave at the start of b4.

### Verified on the final render of round 14
- `check-beats.mjs` 36 of 36 (9 new, written first; 7 failed before the change). `check-jumps.mjs` 0 of 57.
  `--frame --mid` 0 focus boards cut. `--layout` clean. `SHIM OK scenes=15 beats=57 sum_d=1014`.
- All frames and in-beat frames re-rendered, no page errors. `--play` S01, S02, S03, S05, S06, S10, S12, S13 clean.
  Firefox clean. fps S06 b1 60.0, S05 b4 60.0. No em dash.

## Round 15 (2026-09-30): the rest of the round 13 list (ideas 3 to 7), and two jumps the jump check had not been looking for

No on-screen string changed (the S13 table's text is checked word for word).

- **3. The overview is a menu** (js/engine.js `openScene`). In the overview a click on a badge or on a legend line, or a
  scene's number typed (1 to 15; "1" waits 0.7 s for a second digit), opens that scene complete and leaves the overview;
  the camera comes down from where it is, also from the S14 b4 overview. A click anywhere else still goes on to the next
  beat. Outside the overview nothing changes: badges and legend take no clicks, digits do nothing.
- **4. One ruler** (`NLINE.zoom(A, B, t)`). In S09 b3 the number line no longer leaves and comes back: it stays and travels
  from the η_t window (341.3 to 344.3 GeV) to the B_c window (6.272 to 6.334 GeV) by the camera's own zoom-pan in one
  dimension (`W.vanWijk`, shared with the camera): it opens until it holds both, travels, closes. Unnumbered ticks show
  the motion; η_t's marker rides the axis out; B_c lands on it as before. Only this transition: before S10 b2 the line is
  off the stage for a beat and comes back elsewhere, and the S13 b3 line is schematic (no unit), so there is nothing to
  travel along.
- **5. S13 bars.** Under the numbers of rows 1 to 3, a hatched fixed-order bar and a solid MRS bar, MRS over fixed order
  from the larger number of each range (10/700, 18/55, 1/11). Each MRS bar starts as long as its fixed-order bar and
  shrinks: the same move in every row. Row 4 is a statement and has no bar. The numbers no longer wrap.
- **6. Closer frames, where the text was small.** The round 13 figure (what a beat shows covers 9 % of the stage) was the
  wrong guide: most boards are already at their own size (scale 0.9 to 1.0), so a closer frame would only blow them up.
  Reframed are the beats whose text was under 0.85: S12 b1 0.75 → 1.01 and b2 0.79 → 1.06 (camera (56, 233, 110));
  S12 b4 0.83 → 1.12 (on the three verdict boards, (87, 156, 147)); S08 b4, in two stages like S10 b5 and S12 b3: close
  on the shelf while the two tuned masses land (0.83 → 1.29), then back for the sentence (12 s, was 8 s).
  Not reframed: S03 b3, b4 (0.80 to 0.87), S05 b3 to b5, S07 b2 to b4: their frames are as tight as their content allows.
- **7. Edge fade** (`EDGE` = 90 px in js/boards.js). A board the beat is not showing fades as it crosses the stage edge,
  instead of being shown cut. At the end of a beat it touches one board, the dim VMC board of S07 b3, b4.
- **Two jumps on the click, in the engine.** `check-jumps.mjs` now also compares each board's strength as drawn; 25 of 57
  beats jumped. (a) At every scene change the scene before dropped to its ghost level in one frame: it now goes there over
  the first 45 % of the new scene's first beat, and the ghosts of the scene before that fade out over the same time.
  (b) A board shown by one beat and shown again by the next dipped to 42 % on the click and came back: boards now
  remember the beat that showed them before (`prev`). 0 of 57 now.

### Verified on the final render of round 15
- `check-beats.mjs` 52 of 52 (16 new, written first; 12 failed before the change). `check-jumps.mjs` 0 of 57.
  `--frame --mid` 0 focus boards cut. `--layout` clean. `SHIM OK scenes=15 beats=57 sum_d=1014`.
- All frames and in-beat frames re-rendered, no page errors. `--play` S02, S08, S09, S12, S13, S14 clean. Firefox clean.
  fps S09 b3 60.0, S13 b1 60.0, S14 b4 60.0. No em dash.

## Round 16 (2026-09-30): text on text, text out of place, and hadrons that were not colour-neutral

Two reports from the author: text is misplaced or written on other text; the quark colours do not make colour-neutral
particles. Both were true, and the defects were already in build r9 (the frames kept from it show them): rounds 10 to 15 checked board boxes and
read 2x2 sheets at a quarter of the size, and neither shows a caption set between the two lines of a name, a bar squeezed
by its own value, or a label that runs out of its panel. This round looked at every one of the 57 frames at full size and
wrote two checks that measure what is drawn.

### The checks, written first
- `tools/check-text.mjs` (see "How to open it"). On the r15 build: 42 findings at the ends of beats. After: 0.
  Verified able to fail: it reported the S03 b3 and S07 b1 defects seen by eye before anything was changed.
- `tools/check-colour.mjs`. On the r15 build: 3 drawn states not neutral (below). After: 0 of 2131.
  `contact.mjs --layout` had reported "26 glyphs, all colour-neutral" throughout: it read the charge each quark was
  created with (`userData.c`), which cannot see a quark that is hidden or repainted.

### Colour
- S07 (all four beats) and the start of S08 drew one red quark alone: the antiquark had been faded out for the walker
  cloud. The cloud is the sampled position of that antiquark, so it now carries its anticolour (cyan; `--walk` is gone),
  and the quark leaves together with its cloud in S07 b3, when the camera goes down to the trial states (it had stayed,
  cut in half by the top edge of that frame, until S08 b1).
- S03 b3 turned the anti-red antiquark green while a blue quark faded in from nothing: for about two seconds the three
  spheres were red, blue-green and half a blue. Now the antiquark splits: anti-red is green plus blue, so a green and a
  blue quark leave from inside it (`PAIR.tick`), and the colours change while the two spheres still overlap. S04 b1 runs
  it backwards. The check counts two overlapping spheres as the one they came from.
- S03 b3 left the chip "C_F = 4/3" under the red-green link of the finished baryon. 4/3 is the singlet factor of a
  quark-antiquark pair; in the baryon every pair is QQ. The chip now goes as the antiquark splits, the V_R(r) chip with
  it, and each of the three links carries "C_F/2 = 2/3" (a third copy of an existing chip: the one on-screen string
  added this round). "Q Q-bar singlet: C_F = 4/3" stays on the Matching board beside it.
- The assignment itself is unchanged from round 9 and was measured on the rendered pixels (hue of each sphere): a meson
  is red with cyan (anti-red), a baryon red, green, blue, a tetraquark cyan and magenta (the antiquark pair) with red and
  green; an unbound tetraquark splits into red-cyan and green-magenta.

### Shared code
- `css/world.css`: `.board svg text{font-size:17px}` overrode every `font-size` attribute in the plots (an attribute
  loses to any CSS rule), so labels written at 12 to 24 px all came out at 17 and ran into each other. The rule now
  applies only where no size is given. `.board.panel svg{max-width:100%}`: a plot as wide as its board ran 44 px past
  the panel's padding (|term|, VMC, GFMC, number line).
- `GAUGE`: `set()` wrote the bar's style with `cssText`, which dropped `flex:none`; a long value then squeezed its bar.
  In S02 b5 the ~700 to 850 MeV bar was drawn 35 px long under the 106 px bar of ~100 MeV; in S04 b3 the ~700 MeV bar
  was 55 px. A bar now keeps its length and a value too long for the line goes under it; the label column is as wide
  as its longest label.
- `SHELF`: the two shelf lines stand above the names (they ran through overbars and through the second line of the
  eta_t name). Names and chips hang from their top edge, so every first line of a row is level and a two-line chip no
  longer climbs onto the name above it ("charmonium" under "m_c = 1.555 GeV"). The top row's chips sit on two levels
  2.7 apart (2.2 let neighbours touch). The eta_t name, two lines and wider than its column, stands left of its glyph.
  The note "+-0.94 GeV band" stands beside the Omega_ttt chip instead of under it.
- `BOARDS.topAlign`: a row of boards hangs from one top edge (S02, S07 captions, S11, S12, S13).
- `TERMS`: the last dot lands just above the panel (it drifted 110 px away, onto the S05 headline and the stage edge);
  the first dot is clear of the axis; caption and legend share one place; the legend holds on one line.

### Scene by scene
- S03 b1: "what remains is quantum mechanics" stood between "NRQCD" and "scale m_Q v". It is over the pair.
  S03 b3: the 3-body label is two lines inside the triangle (one line crossed both links and the V_R chip).
- S04: the F(r) label ran 74 px out of its panel (now from the right end); "constant-width haze" was struck through
  by the curve (now under it); the haze itself was white on white (now the haze colour of S05); "NNLO" left the band
  plot; the Symptom gauge sat on the |term| board's lower edge and flew in across it (now in place, 2.5 lower).
- S05 b3: the parked chip "Lambda_1 absorbed into m_Q" is at the panel's edge, beside the sentence it belongs to.
- S06: the plot's three lines of words share one left edge in the free corner; "mu_cut = 1 GeV: saturates" is above the
  curves that come down onto the line; the caption under the hill no longer covers "colour singlet: schematic"; the
  question board and the red-zone label no longer overlap; the three n_f chips are one row.
- S07: "omega slider" was cut by the stage edge (now under the slider's end); the heading of the trial states stands
  over them; the three captions start on one line.
- S08: "4-11%" was 81 px outside the drift panel (wider panel); the panel sat on the lower edge of the two figures;
  the dial's words were under its needle; the m_t chip floated alone at the right edge (now in the chip row);
  the frames hold the two-line chips whole.
- S09: the m_t chip flies to the place where the eta_t mass appears, not across the name; the S08 ghosts leave before
  the top row's first name arrives; m_c and m_b meet side by side along the glyph row.
- S10: the lattice chips with their references were wider than the glyph spacing (0.84 of the size); the lattice wall
  goes at b5, where its grid ran through the top row.
- S11: a figure waiting for its beat was darkened (`brightness(0.6)`), a grey slab on the light theme: it is pale
  now, on either theme. The dashed Detmold line lies on the figure's own 1.50 line (it was 18 px above, across the
  bands), as long as the plot frame (it struck through the "1.50" tick label), and its words are under the panel (they
  were on the plot). The two figures never dim, so the line stays on its figure.
- S12 b3: the second frame holds the two verdict boards alone at 1.17 (one frame over shelf and boards left the boards
  at the top, the shelf at 0.45 in a corner and the middle empty). b4: the third board comes once the camera holds it.
- S13 b1: the gauge is gone before the table comes in its place (they were cross-faded on each other).
- S14: "Thank you : happy to take questions" holds on one line. S15: the threads stop at the panel's lower edge
  (they ran on through the references).

### Left as it is
- Inside beats the text check still lists 44 items: the dots that fly from the |term| plot into the two boxes of the
  equation (S05 b2, on purpose), the legend crossing the boards while the World lifts off to the map (S14 b4, S15 b1),
  and dim boards fading at the stage edge during camera moves.
- S10 b5 ends on the whole scene: nothing collides, and the numbers of the shelf are 11 px high. There is no free place
  near the shelf for the summary board (S02 is under it, S08 and S09 over it).
- The lattice wall stands behind the top row in S10 b2: its grid is a backdrop behind those names.

### Verified on the final render of round 16
- `check-text.mjs` 0 strong, 0 faint at the ends of the 57 beats. `check-colour.mjs` 2131 of 2131 neutral.
  `check-beats.mjs` 52 of 52. `check-jumps.mjs` 0 of 57. `--frame --mid` 0 focus boards cut. `--layout`: no two scenes
  share a place. `SHIM OK scenes=15 beats=57 sum_d=1014`.
- All 57 frames read at full size after the change; sheets `sheet-00` to `sheet-14`, `sheet-map` rebuilt;
  `contact-sheets/sheet-r16-before-after.png`. `--play` S03, S09, S12 clean; Firefox clean; dark theme renders
  (S04, S07, S11 looked at). No em dash. The tab title says "build r16".
- Not run: fps, presenter sync.

## Round 17 (2026-09-30): S10 b5 ends on numbers that can be read

Round 16 left S10 b5 ending 162 back, on the whole scene: nothing collided, and the numbers of the shelf were 9 to 11 px high.
The width came from one board: the summary stood right of the shelf, so the last frame had to hold both.

- The summary ("Top-quark benchmarks up to ...") now stands over the top row it sums up, at [9.5, 90.3], in the one free
  place there: between the S08 drift board and the S09 benchmark board, under the S08 figure. `--layout` still reports
  that no two scenes share a place.
- S10 b5 is one camera move to K([0, 75, 86]) and stays there: the six masses land, then the summary comes. The pull-back
  is gone. The beat lasts 9 s as before.
- The two row headers ("NNLO MRS [GeV]", "lattice QCD [GeV]") stand under the tetraquark, which has no number, instead of
  right of it: beside the rows they name, and inside the frame.
- Top-row chips are 4.6 % larger (w 13 to 13.6), so the dim ones at the ends of the row reach 18 px as well.
- What the frame drops: the two S10 boards far right (gap pattern, band gauge), which b3 and b4 showed, and the shelf's
  "cartoon: not a computed wavefunction" line, which fades at the right edge of this frame only.

### Verified
- `check-beats.mjs`: one new check, written first: at the end of S10 b5 every number chip on the stage is 18 px or more,
  the six top-quark masses and the summary are whole, the summary 20 px or more. Before: 20 chips, smallest 9.2 px,
  all 20 under 18 px. After: smallest 18.2 px, summary 21.9 px. 53 of 53.
- `check-text.mjs` 0 at the ends of beats; `check-colour.mjs` all neutral; `check-jumps.mjs` 0 of 57; `--frame --mid`
  0 focus boards cut; `--layout` clean; fastest camera pan 0.90 stage widths per second (S10 b5; the limit is 1.0).
- The tab title says "build r17".

## Round 18 (2026-10-01): beats that stop on their text, a World in talk order, natural narration at 175 wpm

The author's review of the r17 video: text was written and gone inside one animation; the narration read out every
particle name and was slow; the closing zoom-out showed the scenes on top of each other; in S11 b1 dots flew into the MRS
panel from nowhere; the third term of the S05 equation (Λ_R) was only a red band flying into a box.

- **Every text gets a stop.** New `tools/check-transient.mjs` plays every beat on a 30 fps clock and fails on any text or
  dot that is written in a beat (shown further than at its start) and gone before it ends, by fading or because the camera
  left it. On r17: 41 texts and 5 dots in 12 beats. Each became its own beat: `cut(beat, [durations], cams)` in
  `js/util.js` plays the old `f` and the old camera path over each piece's share of `p` (engine: `pathCam`, a beat's
  `camAt`, `endCam`), so a piece starts where the last one stopped. Splits: S03 b1 into three (QCD, NRQCD, pNRQCD, each
  with its words), S03 b3 (the singlet's 4/3, then the baryon), S05 b2 into four (see below), S06 b2 (μ = 1/r, then the
  morph), S08 b3 (the Λ₁ chip, then the dials), S09 b3 (the two mass chips meet, then B_c), S10 b2 (the lattice row, then
  the boards), S12 b1 (the R = 0 line, then it goes), S12 b3 (the shelf stage, then the verdicts), S13 b1 (the gauge on
  row 1, then the table), S14 b4 (the outlook and the thank-you, then the lift-off), S15 b1 (the map with its last legend line,
  then the references). Now 0. 72 beats, `sum_d` 1014.
- **S05, the three terms.** b2a the equation, the teal dots wait above box 1; b2b they go in, its caption, the amber
  dots wait above box 2; b2c they go in, the contour to the branch point, the captions; b2d Λ_R alone: a rose ring on its
  box, the other boxes and captions at 0.45, "one power correction (Eq. A.15)", the haze on the |term| plot. b3 (whole)
  takes the haze into Λ_R and brings the others back. No new on-screen text.
- **S11 b1:** the four points that flew from the Ω glyphs into the MRS panel are gone.
- **A place per scene** (ADR-001). `LAY.T` moves each scene as one block along a winding path in talk order: title top
  left, down the left side, along the bottom, up the right side, S12 in the middle, S13 above it, S14 at the top beside
  the title, S15 above the title. Scene code keeps its coordinates (`FR(SC)`: `P`, `V`, `KS`). The shelf travels with the
  camera into S02, S08, S09, S10, S12 and S14 (`SHELF.travel`, `SHELF.wp`). The overview camera stands at 1051
  (x −313…308, y −270…400); badges 24 wide (~36 px). New `tools/check-map.mjs` measures the regions on the overview
  frames in pixels: on r17 23 overlaps and the shelf in 2 places; now 0 overlaps, 6 places. Dust spread over the whole
  World. The World opens on the title's frame (CAM0), so no beat starts on an empty stage; the QR code now fades in at
  the end of S01 b1 as intended (`qr.show`: an `rv` on a board's own element is overwritten every frame).
- **Narration** (the World's own: `~/mrs_talk_video/narration_world3d/`, the deck's `narration_video/` untouched):
  rewritten, one section per beat, pointing at the screen instead of listing names; a Hylleraas sentence in S07 (helium,
  all pair distances); S11 b1 says each point is a baryon mass divided by the quarkonium mass of the same flavour; S05
  says what Λ_R is (one constant of the size of Λ_QCD, absorbed into the quark mass for the singlet, neglected for baryons
  and tetraquarks). `js/notes.js` is generated from it (`audio3d.py notes`); the engine throws if notes and beats differ.
  New gate `~/mrs_talk_video/world3d/check_narration.py` (lists of three or more particle names, the three sentences,
  numbers on screen or in the Tables, characters the voice cannot say): it failed on the old scripts, passes now.
- **Voice:** Gerard at length-scale 1.12 (1.10 measured 179 wpm): about 177 wpm overall (see Verified); the per-beat band of the audio gate moved by 175/133 to (113, 264), and a beat cut to half its
  samples still fails it (71 of 71).
- **Video:** `~/mrs_talk_video/world3d/mrs-talk-world3d.mp4`, 13:35 (815.3 s), 24,460 frames, 71 clicks, 0 jumps.
  `verify3d.py` PASS. Its sync gate now counts a neighbour only if the two contact frames differ by more than twice the
  beat's own motion (its last frame against the frame 0.4 s earlier): the first run failed S03 b1, S03 b2, S08 b4, whose
  frames were correct when read at full size (the pair's fast jitter and turning glyphs outweighed a small chip-to-dials
  change). S03 b1 is listed as not separable; with shifted references the gate still fails 66 of 70.

### Found by the independent verification, and done
A fresh agent re-ran every check and read all 71 beat ends and about 40 MP4 frames at full size: NOT CLEAN. Done since:
- S14 b4 wrote the outlook and "Thank you", then the lift-off shrank them to 4 px in the same beat: `check-transient` now
  also needs a text's tallest line at 8 px or more to count it visible (it flagged S14 b4, 4 texts, and nothing else);
  S14 b4 is cut into the stop and the lift-off (b5).
- The map: the line now joins the scenes, one point per badge, in talk order (it followed every beat's look-at point and
  drew a V under S12, loops at S10 and S14, and crossings); in S15 b1 it grows on to badge 15 during the stop. The legends
  come only once the map has formed (`seg(k, .93, 1)`): at .55 a second verification still found a legend line over the
  S02 boards for about 0.3 s each way, between the .25 / .5 / .75 samples of `--mid`; `check-text --grid 50` now samples
  every 1/50 of a beat and finds none in S14 or S15. The S12 caption
  "sketch: t̄t̄ compact ..." goes when the shelf leaves (S14 b1); on the map it was a stray pill.
- S03 stops: the tower's names stand 1.1 higher and 1.5 further right (the NRQCD floor's far edge ran through "scale
  m_Q v" in the pNRQCD stop; the floors' side edges touched the first letters); the NRQCD words 1.8 lower, below the whole
  band of the QCD floor (2 higher put their second line on its far edge).
- S14: the outlook is written once the camera has arrived (it was written while the stage edge still cut it); the second
  claim board stands 8 lower: on the map, where thumbnail borders grow, the two touched (now 4.7 px apart).
- Narration: S08 b2 and S13 b2 "partly overlap" as on the screen; S09 b3 "the next meson" (it forms in b4).
- Narration, against the screen and the paper: S10 b6 "close to a GeV" now for the triply top only; S04 b3 says "in the
  binding energy at fixed quark mass"; S05 b6 "scheme choice" is about extending M R S, not about neglecting Λ_R; S05 b8
  drops "nothing is added"; S12 b1 "mixed-flavour meson"; S12 b2 no longer points at the line that is fading; S12 b3
  keeps the reviewed four-body sentence; S07 b4 "40 to 80 MeV for toponium and the triply top Omega"; S08 b3 "from
  leading to next-to-next-to-leading order"; S14 b2 "charm and bottom hadrons"; S02 b5 "700 to 850 MeV"; S06 b2 "falls
  below lambda QCD"; S06 b5 "active flavours ... fixed by"; S09 b1, b2, b5; S11 b1 "what these two plots show, the bands
  and the red lattice points alike"; S13 b4 "consistent with"; two phrases Whisper misheard reworded (S03 b2, S09 b2).
- Left as is, listed by the verifier, all in place before this round: plane edges and triangle sides touching a few
  labels (S02 b3, S03 b6), overbars on chip borders, the α_V fraction bar, the faint S04 caption at the top left, the S05
  |term| panel empty at full strength while the dots are in the equation, mostly empty stops (S13 b1, S12 b1/b2), the
  lattice grid over the dim S10 shelf, the S10 lattice band drawn ±9 MeV and labelled "stat ⊕ sys" (on-screen, not
  reworded here), and the tetraquark glyph turning so that one small quark hides behind a big one in some frames (S02 b2,
  S09 b4, S14 b3: the charges are neutral, a frame can show three of four colours).

### Tests changed, and why
- `check-beats.mjs`: `at()` maps an r17 beat to the beat it now falls in (`CUT`); S09 b3 "half-way" moves from p .28 to
  .245 (the zoom now ends at .44 so the chips stop on arrival); the S12 b3 glyph test projects the World position
  (`SHELF.wp`); the S06 tower test becomes "no tower name cut" (the tower stays in S03's place).
- `check-text.mjs`: `--grid N` samples p every 1/N inside the beats; text drawn under 5 px (line box) is skipped: the overview's thumbnails (about 1 px at distance 1051)
  gave "line through text" hits that are not letters. It still flags the S05 dots put back on the equation (2 strong).
- `contact.mjs`: `--layout` exits 1 on a shared place; `--map` renders the new S15 b1 and b2. `check-colour.mjs`: the
  beat count in its message. `deck_shim.mjs`: optional `--want` (the Deck keeps 15, 57, 1014).

### Verified (final build, 72 beats)
- `check-transient` 0 texts, 0 dots over 15,794 frames (self-test flags a rising-and-falling probe and not a rising one);
  `check-map` clean (15 regions, shelf in 6 places); `check-colour` 2736 drawn hadrons over 72 beats, all neutral;
  `check-text --mid` 0 strong at the ends of beats (12 items inside beats: flights and fades, listed by the tool), and
  `--grid 50` on S14 and S15 0 strong; `--frame --mid` 0 focus boards cut; `--layout` clean; `check-beats` 53 of 53, fastest pan 0.90 widths/s (S10 b6);
  `check-jumps` 0 of 72; `SHIM OK scenes=15 beats=72 sum_d=1014` (the Deck: 57, 1014); on-screen string set identical
  to r17; `check_narration` PASS (2272 words); `notes --check` PASS.
- Video `~/mrs_talk_video/world3d/mrs-talk-world3d.mp4`: 14:02 (841.7 s), 25,250 frames, 72 clicks, 0 jumps, 0 page
  errors; 767.7 s of speech, **177 wpm** overall (133 to 241 per beat), 0 audio gate failures; `verify3d.py` PASS (sync:
  71 beats separated, S03 b1 not separable and read by eye; with shifted references the gate fails 67 of 71). Two beats'
  animations outlast their narration (S08 b5 by 1.7 s, S11 b3 by 1.0 s).
- Frames read at full size: every new stop, the map frames, the shelf's flights, the opening, S11 b1, the Λ_R stop,
  S14 b4, and frames of the MP4. Left as is: the dim S14 b3 thesis line cut by the top edge in S14 b4 (as in r17); the
  S15 threads leave the frame towards the shelf, now right of the references; the line from badge 14 to 15 passes under
  S14's two claim boards. In the S14 "Next" board the bar over the second t of "tt̄" sits under the L of "NNLO" on the line
  above (as in r17).
- Independent verification: three rounds. The first found the S14 b4 shrink, the map's line and caption, legends over
  boards, the S03 edges and the narration items above; the second, the legends' 0.3 s overlaps and the S03 b2 second line;
  the third passed the legends (every frame of the lift-off and descent scanned), the S03 stops and the tools, and failed
  only on two map PNGs rendered before the last S14 edit: re-rendered and read at full size since.
- The tab title says "build r18".

## Round 19 (2026-10-03), first pass: S05 shows the renormalon series as a sum before its Borel sum, one letter for the order, the haze rises from the cut, the same tuned masses in every channel

The author's review of the r18 video, scene 5: the Borel sum should be seen as the infinite sum it is before its result,
and turn into it; "L = 3" under the dots beside a sum over l = 0 to 2 reads strange; the red bar comes from nowhere in
the graph; and Λ_R is not "neglected" for baryons and tetraquarks: the same tuned quark masses are used there, and they
already contain it ("it is neglected because we use the tuned quark mass where it is included").

- **The second box is the sum first.** `#b2` holds two layers in one grid cell: Σ over l = 0 to ∞ of V_l α_s^(l+1)(μ′(r))
  (Eq. A.10, at the scale of Eq. 6) and the closed form (Eq. A.11). The equation is written with the sum (b2); the amber
  dots of every order go into it (b4, a new stop); then `borelSum(k)` turns it into the closed form while the contour
  lights up to the branch point on the inset, with the two captions (b5, a new beat; `slow: 3.5`, timed on its sentence). Both layers share the cell, so no
  box moves or changes size. S05 has 9 beats (the 7 of the old third stop became 3 + 4; `sum_d` stays 1014): 73 beats at that point.
- **One letter for the order.** The bracket reads "used: l = 0, 1, 2" (was "used (L = 3)"); the axes carry the italic l of
  the equation (were the script letter); the subscripts of the legend and of the Borel label are italic too.
- **The haze rises on the cut.** At the Λ_R stop (b6; `slow: 3.6`, so it comes with its sentence) the rose haze rises on the
  branch cut of the Borel inset, past u = ½, where the contour stopped: what the Borel sum cannot reach (`BOREL.at(u)`).
  In b7 it leaves the cut, narrows and goes into the Λ_R box. It is drawn over the equation panel (`z: 300`, as the flying
  dots): until r18 it slid behind that panel (boards are stacked by distance) and was never seen arriving. The |term|
  graph, with nothing left on it, leaves when its last dots do (b4), not after the haze.
- **Where the renormalon goes.** The board now reads: "Λ_R comes out free of the leading-renormalon ambiguity. It is
  absorbed into the quark mass, tuned so that 2m_Q + E matches the spin-averaged 1S states. Baryons and tetraquarks use
  the same tuned masses: the same fit to experiment absorbs it there too." Chosen by the author, without the "scheme
  choice, future work" line. The paper's Sec. II B, step 3, says "neglected in other cases"; the author's reading: nothing
  is added there because the tuned masses already contain it. S14's "Next" board still lists the non-singlet derivation.
- **Narration** (`narration_world3d/scene_05.md`, 9 beats): b4 names the infinite sum, b5 its Borel sum, b6 ends on the
  part the Borel sum cannot reach, b7 says the same tuned masses carry Λ_R and the fit to experiment takes care of it.
- Not done, by the author's decision of 2026-10-01: the one-frame jumps at seven scene joins of the video (ambient phases
  that restart with the page) stay as they are.

### Tests changed, and why
- `check-beats.mjs`: `raw()` addresses the build's own beats beside `at()` (r17 names); `CUT` for S05 b2 is
  [10, 7, 3, 4, 6]; 14 new tests (section 8): 13 written first, 10 of them failed on r18. One of them ("the bar is on its way to
  the Λ_R box") was tightened to "drawn over the equation panel" when the first frames showed it passing behind, and
  failed again before the fix. "No box moves while the sum transforms" first compared stage positions to 0.1 px and
  failed on the ambient camera's drift of the whole board; it now compares sizes and places relative to the first box, and a
  14th test plants the fault (the closed form taken out of the shared cell: box 3 moves 48 px) to show it can fail. The round 14 test "the |term| board is
  there while the haze leaves it" is gone with the thing it tested; its place is taken by "there while the dots leave it
  (b3 end), gone once they are in the equation (b4 end)".
- `check_narration.py`: the S05 rule required the word "neglect"; it now requires "same tuned masses" and "infinite sum"
  and fails on "neglect". It failed on the r18 narration (3 failures) before the rewrite.

### Second pass: panels 4, 6, 9, 10, 12 (same day, after the author's review of the first r19 video)
- **S04 (7 beats).** The series board follows Eq. A.1 as two lines: V_R is the series, then 𝒱_R = V_R + Λ_R. The old
  fourth beat is four stops, timed on the speech: the smallest-term mark (rose double arrow, `TERMS.mark`) is carried
  from the |term| plot onto the V(r) plot (a TICK follower with `asDrawn`, as the S05 haze); the curve slides between
  two dashed copies, with the band and two equal arrows; three parallel tangent marks for the force; the Borel inset.
  The thin F(r) curve is gone. The bands figure has the order on its x axis (LO, NLO, NNLO), caption "binding energy
  at fixed m_Q (Figs. 1–2 left, schematic)".
- **S06.** The smooth cutoff only: no hard-cutoff curve, no frozen α_s curve, no line about them; the 2 GeV curve is the
  smooth μ′, labelled "μcut 2 GeV → fixed order (Fig. 7)".
- **S09.** "exp. B_c 6.2745(3)" (paper, text on Tab. II, ref. [92]) on the number line, window from 6.262; the narration
  says it is the lighter spin state and that we compare with the spin average.
- **S10.** GeV on every mass chip; headers "NNLO MRS" and "lattice QCD"; no "±0.94 GeV band" note beside Ω_ttt. With the
  unit the chips of the main row and of the lattice row touched (`check-text`: 4 strong): padding 4 px, no side margin,
  letter-spacing -0.4 px. A smaller font failed the 18 px test at b5.
- **S12 (5 beats).** The colour-structure board and its beat are gone (`sum_d` 996). The bound tetraquarks carry their
  MRS masses of Tab. V, marked NLO, where they said "top decays"; the BOUND header lost "· top decays" too (the author:
  "gone everywhere" on panel 12; the η_t label "zero-width limit (top decays)" and the S09 sentence stay, by the author's answer).
- **S13 b1.** S12 now ends on the verdict boards, further left: the gauge is shown once it is whole on the stage
  (q .3 to .4; it crossed the edge as a focus board), and the S12 sketch caption is a focus board only in the first
  piece of its cut (it counted as a click jump at S13 b1).
- **presenter.html** needed no edit; `tools/check-presenter.mjs` (4 tests, shown failing on a planted narration change).
- 75 beats; `--want 15,75,996`.

### Third pass: the open points of the third review, changed at the author's request ("change all of them", then the small ones)
- S05: the sum rolls up out of its box while the closed form rolls in from below (the box clips them; it was a fade through
  an empty box). Board: "For baryons and tetraquarks it is neglected, because they use the same tuned masses, which already
  include it." (the author's sentence; the narration says the same). The chip and the S08 dials say Λ_R (was Λ₁).
- S04: a "V" on the y axis of the V(r) plot; "δV" over the landed mark; the LO box of the bands figure is narrow (the paper's
  LO band collapses, Sec. III A); narration b2: "Even if you stop at the smallest term…".
- S06: no "(Fig. 7)" (that figure is the hard-cutoff and frozen-α_s study); "μcut = 2 GeV → fixed order"; a curve above the
  frame starts where it enters it (clamped, the λ = 2 curve drew a flat top with a corner).
- S09: the gap label is "24.8 MeV gap" and the narration stops at "about 25 MeV below it" (the paper's prose and its
  Table II disagree on how the gap compares with the scale range); GeV on the three values; the exp. B_c label above the line.
- S10: GeV on the three masses of the Ω_bbb line; the lattice band is stat ⊕ sys in quadrature (11.4 MeV; it was 9); the
  summary board no longer repeats "±0.94 GeV".
- S12: "509(10)(340) MeV" and the two others carry both errors of Tab. V; the figure title ends "fixed order (left), MRS
  (right)" (Fig. 5; it said "MRS at LO/NLO"); narration b3: three-body and four-body forces, the subtraction for them left
  for future work.
- S14: "Thank you: happy to take questions".
- S04 band label: "same width at every r: δV ~ Λ_QCD" (was "constant-width haze", the author's choice of 2026-10-03).
- Left as they are (taste, or the author's wording): n_f shown three times on S06; the unlabelled λ
  curves; "~135 MeV" above the fixed-order label on S10 b3; the staggered top-row chips; the blended t̄t̄ in the bound
  tetraquark glyphs; the blank number-line panel at S09 b3; "target precision ~10 MeV"; small pale captions; S14 b4's dim
  thesis line at the top edge.
- `check-beats.mjs` section 13 and three changed tests, written first: 8 failed before the changes; 101 tests.
  `check_narration.py`: S05 must say "neglected, because they use the same tuned masses".

### Tests changed in the second pass
- `check-beats.mjs` sections 9 to 12 (S04, S06, S09/S10, S12), written before the changes: 96 tests. Corrected tests: S06
  "distinct heights" became "under 30 % of the points at one height"; S09 "labels touch" needs 4 px of vertical overlap;
  S12 chip text is read with `innerText`; the S09 window assertion is 0.072; S10 b5 expects the strings with " GeV".
- `check_narration.py`: S04 needs "smallest term", "slide", "force", "plus a constant"; S12 fails on triplet, sextet, diquark.

### Verified (final build r19, 75 beats)
- `check-beats` 101 of 101; `check-transient` 0 texts, 0 marks over 16,523 frames (`--selftest` passes); `check-jumps`
  0 of 75; `check-colour` all neutral; `check-text --mid` 0 strong at the ends of beats (13 items inside beats, listed);
  `--frame --mid` 0 focus boards cut; `--layout` and `check-map` clean; `SHIM OK scenes=15 beats=75 sum_d=996`;
  `check-presenter` 4 of 4; `check_narration` PASS (2407 words); `notes --check` PASS.
- On-screen strings against r18 (`.scratch/r19/strings-r19.txt` at the repository root): only the changes above.
- Video `~/mrs_talk_video/world3d/mrs-talk-world3d.mp4`: 14:45 (885.2 s), 26,555 frames, 75 clicks, 0 jumps, 0 page
  errors; 808.0 s of speech, 0 audio gate failures; 172 subtitle cues; `verify3d.py` PASS (74 beats separated, S03 b1
  read by eye). S04 in the video: the mark inside "its smallest term", the landing on "it is this much", the slide in "slide up or down";
  S13 b1: the gauge with "This band, first" (measured on the second-pass recording; the third pass shifts everything by under 4 s).
- Frames read at full size: S04 b4 to b7 and in-beat, S06 b5, S09 b4, S10 b2 and b5, S12 b4 and b5, S13 b1 in-beat;
  MP4 frames at 200.0 and 206.2 s.
- Third independent review (fresh context, frames at full size, numbers against the paper): every number of the changed
  panels matches the paper; no dash added; no finding needed a change inside what was asked. Left for the author: the
  Ω_bbb number line (S10 b3) has three masses without GeV beside a GeV axis; the S10 summary board still says "the band
  is wide, ±0.94 GeV"; "(Fig. 7)" on S06 is the paper's hard-cutoff and frozen-α_s study; the chip "Λ₁ absorbed into
  m_Q"; the paper's §III C prose (B_c gap "comparable to the scale range", 6.2865(2)(218)) against its Table II, which
  the talk follows; S12 b5 "509(340) MeV" drops the statistical error of Tab. V.
- Not done, by the author's decision: the one-frame jumps at seven scene joins of the video.

### First pass: verified at the time (73 beats, superseded)
- `check-beats` 66 of 66 (fastest pan 0.90 widths/s, S10 b6); `check-transient` 0 texts, 0 marks over 16,200 frames;
  `check-jumps` 0 of 73; `check-colour` 2756 drawn hadrons over 73 beats, all neutral; `check-text --mid` 0 strong at the
  ends of beats (12 items inside beats: the dots' flights, listed); `--frame --mid` 0 focus boards cut (15 dim listed: one
  more than r18, a dim shelf chip 2 px over the edge at S14 b1 p = .5, where r18 already had chips at the edge);
  `--layout` and `check-map` clean; `SHIM OK scenes=15 beats=73 sum_d=1014`; `check_narration` PASS (2328 words);
  `notes --check` PASS.
- On-screen strings against r18 (`check-transient --strings`, `.scratch/r19/strings-r19.txt`): only the bracket, the two
  axis labels, the board on where the renormalon goes, and the new sum (10 changed lines).
- Video `~/mrs_talk_video/world3d/mrs-talk-world3d.mp4`: 14:20 (859.8 s), 25,795 frames, 73 clicks, 0 jumps, 0 page
  errors; 785.0 s of speech, 178 wpm overall, 0 audio gate failures; 168 subtitle cues; `verify3d.py` PASS (sync: 72
  beats separated, S03 b1 read by eye). In the video the contour lights from 236.1 s and the sum turns into the closed
  form from 238.3 s, inside "along this path up to the branch point, it turns into this closed form" (sentence from
  234.5 s); the haze rises at 250.5 s, its sentence starts at 250.7 s. Three beats' animations outlast their narration
  (S05 b5 by 0.4 s, S08 b5 by 1.7 s, S11 b3 by 1.0 s).
- Frames read at full size: every S05 beat end and the in-beat frames of the World, and frames of the MP4 (the sum with
  the contour lit, mid-transformation, the closed form with its captions, the haze on the cut, its way into Λ_R, the board).
- The first recording of this round had the transformation 3 s ahead of its words and the haze 1.5 s ahead: `slow` on
  b5 and b6 was set from the sentence timings and the video recorded again.
- The tab title says "build r19".

## Speed

Measured through the `?fps` counter with `tools/contact.mjs --fps` in headed Chromium on this laptop
(renderer reported: ANGLE, AMD Radeon Graphics, radeonsi renoir, OpenGL 4.6), 1920x1080, pixel ratio 1, bloom on:

| beat | fps (median of 8 samples) |
|---|---|
| S07 b2 (30k walkers; 60.0 also at 200k with bloom) | 60.0 (vsync-capped) |
| S10 b2 (shelf + lattice wall + boards) | 60.0 (vsync-capped) |

Both sit at the display's 60 Hz cap, above the 30 fps floor. Headless Chromium (SwiftShader) renders the contact
sheet correctly but is not a speed reference.

## Checks run

- Shim self-test: `SHIM OK scenes=15 beats=57 sum_d=1014`.
- All 57 settled beats rendered to `contact/`, reviewed as 2x2 sheets, layout fixed until no board was cut or
  overlapping and every carry object was present.
- `--play S05`: real-time click-through, tweens 4 to 10 s, no page errors; the click after b5 moves to S06 b1.
- `--firefox`: loads from `file://`, no page errors, S05 b2 renders (bloom, boards, floor).
- Presenter: opens the World, shows scene/beat, notes and the next line, stays in sync with keys pressed in
  either window.
- No em dash and no "MSR" anywhere in the sources.

## Open wording (kept as is, per issue 12)

All on-screen strings come from the approved scene files, the storyboard or the Tables. The items flagged for the
speaker in `issues/12-physics-majors-for-speaker.md` (the S02 "~700–850 MeV" wording, the S03 spin percentages,
the S06 "2 GeV" sentence, the S05 non-singlet sentence, the S04 band cartoon caption) are unchanged here.

## Round 20 (2026-10-04): particles move only in the beats about them; the presenter shows keynotes

After the author rehearsed the talk live (20 minutes for a 15-minute slot, the first half spent reading the notes).

- **Still particles.** A glyph moves only while the current beat shows its name or its number, or flashes it; the static pair
  jitters only in the beats that set its state. `u.live` (0 to 1) is the same focus the boards use (`boardState`), so it eases
  with the click and is a pure function of the beat; a glyph's place is its pose at t = 0 mixed with its orbit (`space.js`),
  so a still glyph has one pose on every load (the phase was random) and nothing jumps as `live` comes and goes. `PAIR.st` is
  a Proxy that records the beat that set it. The walkers, the dust and the glow pulses are unchanged.
- **Keynotes.** `js/keys.js`, written by hand: one to three keynotes per beat (10 words or fewer), and `show`: what is on the
  screen and needs no words. `presenter.html` shows them instead of the narration, with the time the panel should start
  (`KEY_T`, the pace of the video, 15 minutes) and a clock that turns red once the panel should be over. The full sentences
  stay in `js/notes.js` (key `n` in the World). The engine throws if keynotes and beats disagree.
- **Tests, written first.** `tools/check-live.mjs` (what moves at each settled beat; 8 tests, 6 failed before the change);
  `tools/check-presenter.mjs` rewritten for keynotes (12 tests; it failed on the missing `keys.js`).
- **Verified.** check-live 8 of 8 (most moving at once: 6); check-presenter 12 of 12; check-beats 101; check-jumps 0 of 75;
  check-colour, check-text --mid (0 strong at beat ends), --frame --mid (0 focus cut), check-map, check-transient clean;
  `SHIM OK scenes=15 beats=75 sum_d=996`.
- **The MP4 is not re-recorded** (the author: live talk only): `mrs-talk-world3d.mp4` is the round 19 World, with every
  particle moving.

### Round 20b (2026-10-05): a particle moves for as long as the talk is about it
The author, rehearsing: particles "stop moving" while still being talked about. A real-click replay of the whole talk
showed no engine fault: the rule of round 20 (a particle moves only in the beat that writes its name or number) froze the
four families after S02 b1, the Ω baryons once the lattice row came, cc̄ and bb̄ during the tuning. Now `LIVE` in
`js/keys.js` lists, per beat, the particles the talk is about (`liveOf(id)` in the engine, eased like a board's focus); the
old rule stays as a floor. The title baryon turns in S01 (it never did). At S14 b3 the whole family moves.
`check-live.mjs`: 13 tests, 8 failed before the change. Suite green (jumps 0 of 75, beats 101, presenter 12, colour, text,
frame, transient, shim 15/75/996).

### Round 21 (2026-10-05): the S04 potential plot reads as −C/r, with the script 𝒱 on its axis
The author: the axis letter was a plain V, not the 𝒱 of Eq. A.1, and the curve did not look like 1/r. It was −1/r + 0.2
drawn from r = 0.5 with no zero line, so only its flat part showed. Now V = −1/r from r = 0.4: steep at short distance,
flattening under a dashed zero line marked 0; the band and the lower copy are clipped at the r axis; the two force labels sit
above the zero line. Two tests written first in `check-beats.mjs` (section 14), both failed before; 103 pass. Jumps, text,
frame, transient and live checks clean. The MP4 is not re-recorded.
- Same day, at the author's request: the scale is larger and the zero line higher (V = 0 at y 56, 96 px per unit, from
  r = 0.55), so the upper dashed copy of the curve stays under zero (test added; it reached above it). The "steep" test's
  share of the drop in the first quarter of the arc went from 0.6 to 0.55: the steep part is longer at this scale.

### Round 21b (2026-10-05): the video narration does not read the screen
The author, about the keynotes: the video should drop the same things (names, the list of predictions, numbers on screen).
`check_narration.py` gained three rules, written first: no person's name but the speaker's (Komijani, Kronfeld, the
coauthors, Detmold), no sentence listing mesons, baryons and tetraquarks, and no number read off the screen outside
S09 b4 (the B_c gap), S10 b2 (125 to 175 MeV) and S10 b4 (700 to 10 MeV). The old narration fails 31 times, the new one
passes; 2407 → 2240 words. The facts stay, said qualitatively ("hundreds of MeV", "barely moves from one order to the next",
"nearly two orders of magnitude"). The Hylleraas sentence stays (round 18 request), without the channel list. On-screen
strings for the gate: `.scratch/r21/strings-r21.txt`. Backup of the narration before: `/tmp/r21/narration-before/`.
Four renders of this round died (ffmpeg exit 255 or EPIPE, every running encoder in the same second). Not memory (no
earlyoom or kernel kill in syslog) and not my session's signals (it died under `setsid` too): another Claude session on this
machine, working on `~/program/tools/footage`, ran `pkill -x ffmpeg` at 18:12, 18:38 and 18:42, matching the crashes to the
second. The render now runs ffmpeg under another process name (`/tmp/w3d-bin/ffmpeg` execs a symlink `ffmpeg-w3d`, put first
on PATH), which `pkill -x ffmpeg` does not match. `run3d.sh` also takes `JOBS` (default 4).
- **Video** (round 21, 2026-10-05): `~/mrs_talk_video/world3d/mrs-talk-world3d.mp4` rebuilt from this World (still particles,
  the S04 plot, the trimmed narration): 13:43 (822.7 s), 24,681 frames, 75 clicks, 0 jumps, 0 page errors; 736.9 s of
  speech, 0 audio gate failures; 167 subtitle cues, none naming a person; `verify3d.py` PASS (74 beats separated, S03 b1 read
  by eye). MP4 frames read: S04 b6 end (𝒱 axis, curve under zero); S02 b2 against S02 b5, 0.5 s apart: whole balls move
  in the first, only the camera drift's edges in the second. `run3d.sh` makes the renamed encoder itself. The schedule line
  rounded the minutes (13:43 printed as 14:43; also in earlier rounds' logs): now truncated. Backup of the round 19 video:
  `/tmp/mrs-talk-world3d.r19.mp4`.

## Round 22 (2026-10-07): the talk travels as one zip
- The sans face is bundled: `fonts/NotoSans-{Regular,Bold,Italic,BoldItalic}.woff2` (from /usr/share/fonts/truetype/noto,
  OFL), family `NotoSansW`, first in `--sans`. Until now `system-ui` was the system's Noto Sans in headless Chrome (same width
  to the pixel), Ubuntu in a desktop Chrome on this machine, Segoe UI on Windows, SF on a Mac: every layout check measured
  one font and other devices would have drawn another. The web copy is drawn unhinted (regular text 1.5 % narrower than the
  hinted system copy, italic identical). Checks on this build: check-text --mid 0 strong at beat ends, --frame --mid 0 focus
  cut, check-map, check-beats 104, check-jumps 0 of 75, check-transient 0, check-presenter all pass.
- `talk/mrs-talk-world3d.zip` (1.7 MB): index.html, presenter.html, css, js, vendor, fonts, assets and a README. Unzipped
  alone in /tmp it ran in Chromium and Firefox from file:// with every font loaded, no request outside the folder, no page
  error, and the presenter driving the World. Not tested on Windows, macOS, Safari or Android. The MP4 is not rebuilt
  (the font is only in the World; the video's frames are unchanged in content).
- **Phone practice file** `talk/mrs-talk-practice.html` (3.9 MB, built by `tools/build-practice.py`): every script, the
  stylesheet, fonts and figures inlined, so it opens alone, offline. It starts in practice mode (`?practice` does the same in
  the normal World): a strip under the World with the beat's keynotes, the panel's target time and a clock; a tap on the
  left third of the screen goes back. Upright, the World sits on top and the keynotes fill the rest. `tools/check-practice.mjs`
  (written first, failed on the missing file): emulated Pixel 7, portrait and landscape, 12 of 12 (taps forward and back,
  keynotes of the current beat at 16 px or more, World above the strip, nothing loaded besides the file). Not tried on a
  real phone. The zip carries it too.

## Round 23 (2026-10-07): S06's ruler tells the confinement story right; the GdR QCD 2026 logo on the title
- **S06 red zone at 0.56 fm** (was 1 fm). It starts where 1/r = 0.35 GeV, the edge of the plot's shaded band; at 1 fm the
  ruler had implied Λ ≈ 0.2 GeV (the n_f = 5 value) and disagreed with the plot. One-loop α_s(1/r) is already ~1 at 0.3 to
  0.4 fm, so the zone is if anything generous.
- **S06 b3: the red zone stays red.** Before, as μ′ saturated, the red fill went and the ruler turned teal to its end, which
  read as "the far region is now perturbative". μ′ only keeps α_s finite there (α_s(1 GeV) ≈ 0.5); V still has no linear
  confining term (σ ≈ 0.18 GeV², ~0.9 GeV/fm). The fill and red ticks stay; the label "μ′ never goes there" gives way to
  "μ′ keeps α_s finite here, but V has no linear confining term (string tension · r): a systematic of the method". The
  symbol σr was tried and dropped: in the label's italic serif σ reads as "o" at full size.
- **S06 |ψ|² hill** is r²e^{−2r/a₀} in fm with a₀ = 0.3 fm, between the Bohr radii of cc̄ (0.49 fm) and bb̄ (0.21 fm); it
  peaked at 0.74 fm with 81 % of its weight in the red, now 0.31 fm and 27 %. The base mesh is 12 units (to 1.1 fm), was 22:
  the longer base drew a flat grey sheet past the hill.
- **S01: the GdR QCD 2026 logo** (`assets/gdr-qcd-2026.png`, the emblem cropped from the meeting's Indico poster,
  indico.in2p3.fr/event/40045), top left over the title, 240 px; it dims with the title in b2 (0.42, the engine's focus level).
- check-beats, written first and failing on the r22 build: 10b (S06 ruler, label, hill: 5 checks) and 10c (logo, b1 and b2).
- Not rebuilt: `talk/mrs-talk-world3d.zip`, `talk/mrs-talk-practice.html`, the MP4.
- **PowerPoint** `talk/mrs-talk-world3d.pptx` (60 MB, built by `tools/build_pptx.py` from the round 21 MP4): 75 slides, one
  per beat; each holds a silent H.264 clip of its beat (click to end of the animation + 0.3 s) that starts by itself when the
  slide opens (main sequence, onBegin, playFrom) and holds its last frame; the poster is the clip's first frame; speaker
  notes = the keynotes. `tools/check_pptx.py` (written first, failed on the missing file): 75 slides at 16:9, one full-slide
  video each, durations within 0.25 s, autoplay, notes, seams between clips under 4 grey levels, LibreOffice opens it
  (75 pages). Two planted faults (no autoplay on one slide, emptied notes on another) were both caught. Autoplay in
  PowerPoint itself is not tested here (no PowerPoint on this machine). Rebuild after a new render: build_pptx then check_pptx.

## Round 24 (2026-10-07): S06 shows the cutoff with a gauge for α_s; the λ variation is a band; Fermilab and CPT logos
- Supersedes the S06 parts of round 23 (red zone kept red; a 3D copy of the μ(r) plot, tried and dropped: "the animation is
  the same as the plot, it doesn't help a non-expert").
- **S06 gauge** (`PSI` in js/objects.js): a dial for the coupling α_s at the scale used for the pair's separation, 0 at the
  left, ∞ at the right end stop, angle 180°·α/(1+α). α_s at two loops, n_f = 3, Λ = 0.35 GeV: the pole sits at 1/r = Λ
  (0.56 fm, the ruler's red edge) and α_s(1 GeV) = 0.48. b1: quark B comes in to 0.2 fm, then walks out to 1.25 fm with
  μ = 1/r; past 0.56 fm the needle slams past the end stop, "α_s → ∞" in red. b3: as the plot's curve becomes μ′ the needle
  falls back to 0.47; B walks in and out again and it stays there. No live number under the dial (a counter is text written
  and gone, check-transient): "α_s → ∞" when it blows up, then "α_s ≤ 0.48 at any r"; the needle does not shake (check-jumps). The label keeps the caveat: V still has no linear
  confining term (string tension · r), a systematic of the method. Three versions (bond, gauge, lens) were compared on the
  phone with buttons in the practice strip; the author chose the gauge and the other two and the buttons are gone.
- **S06 b5 plot**: the λ = ½ and λ = 2 curves are one band (`#lamband`, ink2 at 22 %) under the λ = 1 line: the systematic
  uncertainty of the scale choice.
- **S01**: Fermilab (Wikimedia Commons, "Fermilab Logo.svg") and CPT (`assets/cpt.png`) on one white card, top right.
- **Phone**: `talk/mrs-talk-gallery.html`, every beat as a picture (no WebGL needed), built by a one-off script.
- check-beats 10b (gauge: broken at b1 end, calm at b3 end, B starts close, labels on stage), 10c (three logos), 10d
  (λ band); check-practice: no version buttons. All written first and seen failing.
- **S12 BOUND board**, cleaner, at the author's request: one MeV axis from 0, a dot at each binding energy and a teal band for
  the scale range (± half the full range of Tab. V). Every band stays clear of 0, which is the board's one message: all three
  are bound. The fixed-order bars and the "MRS is wider" note are gone. check-beats 12b, written first.
- **Phone PDF** `talk/mrs-talk-slides.pdf`: the 75 beat frames, one per page (the phone reads PDF, not HTML).

## Round 24b (2026-10-07): an independent review (Fable, read-only) and its fixes
- **S06 ruler** now takes the gauge's zones: teal while α_s(1/r) < 0.82 (to 0.28 fm), amber to 1.5 (0.35 fm), red beyond, a taller
  red tick at the pole (0.56 fm); with μ′ the amber and red parts turn grey. Before, the ruler stayed teal to 0.56 fm while the
  gauge was already deep in its red arc. Ruler ends at 1.7 fm (was 2.8 fm, under the plot). Round 23's "red stays red" is
  superseded: after μ′ the far ruler is grey (α_s finite there), and the label carries the confinement caveat.
- **S06 words**: gauge caption "(schematic: two loops, Λ ≈ 0.35 GeV)" (the paper gives neither Λ nor α_s(1 GeV); with Λ = 0.33
  the pole moves to 0.59 fm and α_s(1 GeV) to 0.45); b1 label "about 1 by 0.3 fm, and at 0.56 fm it blows up"; b3 label "at or
  below 0.48" (one number with the gauge); hill label "schematic r²|ψ|², bottomonium-like (a₀ ≈ 0.3 fm)" (the hill is the radial
  density; for charmonium, a₀ ≈ 0.5 fm, some 60 % of it lies past 0.56 fm). The gauge comes up once B is close (p .55, was .5:
  it swung red, then teal). λ line: "the band probes the residual scale dependence" (the paper: a diagnostic of scheme
  stability, not a truncation error).
- **S12**: header "B ≈ 509–742 MeV below threshold" (B of Tab. V; ΔE is another quantity on the R board); caption "bound at
  every λ". Keynote and narration no longer say "small shift between orders" (the board does not show it): "they stay bound at
  every value of the scale we tried". Open: the band is ± half of Tab. V's second parenthesis ("full range"); §V B prose reads
  better with ± the full value. The claim holds both ways; the λ = ½ and 2 numbers settle it.
- **S01**: the University of Cincinnati logo joins Fermilab and CPT (the three affiliations under the authors).
- **Checks**: 10b reads the drawn needle (#ndl), labels may not overlap each other, the ruler zones are tested against α_s;
  10d tests the band's edges against μ′(r; ½) and μ′(r; 2); 12b tests positions in MeV against the 1000 MeV tick; the phone file
  carries a source hash (build-practice.py) and check-practice fails on a stale build. The gauge-timing check was run against
  the old timing and failed (0.50), then passed on the new.
- Not done (nits): the deck title's hyphens differ from the arXiv title; the 2 GeV curve hides in the band at r < 0.19 fm.
- **Zip for Windows** (`tools/build-zip.py`, checked by `tools/check-zip.mjs`, written first: the r22 zip failed 6 of its checks):
  everything in one folder `mrs-talk-world3d/` (Windows "Extract all" gives a folder, not loose files), README.txt with CRLF line
  ends and "Extract all" first, names legal on Windows, no case clashes, short paths; the sources inside byte-identical to
  talk/world3d; unzipped, index.html and presenter.html open from file:// with no page error and nothing loaded from outside.
  Not tried on a real Windows machine.
- **Round 24c, S12 bands are ± the second parenthesis** (the author: 509(10)(340) is 509 ± 340; it was drawn ± half). The
  axis runs to 1300 MeV (the widest band ends at 1221); the lowest edge is 169 MeV, so "bound at every λ" holds. 12b now
  checks the band ends in MeV against the 1000 MeV tick; it caught the 1000 MeV label left at the old end of the axis.
  S09 narration: toponium's band "reaches about three quarters of a GeV on either side" (the number line already drew ±).

## Round 24d–e (2026-10-08): the scale-band board once; one conclusion panel (14 scenes)
- **Scale-band board only in S10** ("this plot appears too often"): S02 b5 says the obstacle in words (`obst`), S04 b3 the
  symptom (`sym`), S13 no longer flies the S10 board in; its table's first row is written in the table's first stop, where
  the narration names it ("The first row: … nearly two orders of magnitude narrower"). The S03 and S05 lines that faded the
  board out are gone (with nothing to fade they made it pop in, check-jumps).
- **One conclusion panel**: the old S13 (what MRS bought) and S14 (conclusions) are scene 12, "Conclusions", 8 beats: the claim
  (new b1, top right of the table's frame, gives way to the payoff's number line), the table (2 stops), hallmarks, payoff, the
  family (the shelf now travels here), next + thank you, the map. The old S14 b2 board ("scale sensitivity down by two
  orders") is gone: row 1 says it. The old S15 is S14. Places in the World do not move: the old S14 boards keep place 13
  (`FR(PL)`), the references place 14; scene and place indices differ from 12 on. KEY_T loses 818; SEC: conclusion from 12.
  Narration: scene_13.md merged (the old b2 sentence dropped), scene_14.md = the old scene_15 ("number fourteen"); the old
  files are in /tmp/narr-bak. Tools moved to 14 scenes: check_narration, check-presenter, check-practice, check-zip,
  check-map, contact.mjs (map frames), check-beats (remap: S13 opens one beat early).

## Round 25 (2026-10-11): the panel counter
- `#panel` on the stage (index.html, css/world.css, engine.js `ui()`): "N / 14" in the section colour of the map's badges
  (`SECC[SEC(si)]`), bottom left, 22 px; it fades with the overview blend (the badges number the panels on the map). Check in
  check-beats (written first, failed with no element): every panel's end, text, colour against a probe of the CSS var, whole on
  stage, on no board; hidden at S13 b8. check-text, map, jumps, transient, presenter clean. Zip rebuilt; the phone file and the
  PDF were left alone at the author's request (check-practice's source hash will say the phone file is stale).
- The method and the standing rules of rounds 19 to 25 are now in the user-wide skill `~/.claude/skills/animated-world-video/`
  (SKILL.md: method items 8 and 9, nine new standing rules; WORLD-ENGINE.md: the counter, 14 scenes, the shelf scenes of
  check-map), so a World talk in any folder gets them; the project memory keeps the same list. check-map's shelf scenes
  were `[1,7,8,9,11,13]` (13 was the old S14): now 12, the conclusion; the count of places was right by luck (the shelf does
  not move in the references).

## Round 26 (2026-11-18): between clicks no text moves
- The author: "even after the animation ends, the text is slightly moving. Can you make it stop moving when no animation is
  on going?"
- **The check, written first** (`tools/check-still.mjs`): the opening frame, every beat end and the map opened by hand are each
  held for 12 s on a virtual clock (the World's own loop in 50 ms steps, nothing running), and the ink of every text on the
  stage (the box of each text node, not the board's box) is read every 0.2 s. It fails on a text whose edge travels more than
  0.05 px, or that is drawn in some of the readings only. On the round 25 build: 75 of 76 stops (all but the opening frame,
  which draws one fixed text), up to 14.2 px (S06). `--selftest`: a probe swaying by 2 px is flagged, a still one is not.
- **Cause 1, every panel: the camera's ambient drift** (`space.js` `applyCamera`: 0.35 units across on a 30 s period, 0.25 up
  and down on a 37 s one). Every board is projected from the camera at every frame, so every text moved, for ever. Removed:
  the camera stands exactly where the beat put it. After this alone: 10 of 76 stops.
- **Cause 2, S03 b4 to b7, S04 b1, S06 b1 to b3, S07 b1 and b3: labels placed from the jittering pair** (up to 10.6 px, for as
  long as the beat is about the pair). `PAIR.rest()` gives where the quarks stand without their own motion; `PAIR.posA()` now
  returns that, and the S03 labels use it. So the pair's labels, the S06 ruler, hill, gauge and labels, and the S07 cloud and
  its labels stand still; the quarks and their links alone jitter, as before (check-live's nine tests of S01 to S10 give the
  same result as on round 25).
- Still moving at rest, on purpose: the dust, the particles the beat is about, the S07 insets.
- **Verified:** check-still 0 of 76. A second measurement that shares neither the check's clock nor its stubbed draw: a page on
  the real clock with real drawing, 5 s at rest, counting the places each drawn board takes. On the round 25 build every board
  changes place (S02 b3 7 of 7, S03 b7 11 of 11, S06 b3 6 of 6, S07 b1 2 of 2, S10 b1 23 of 23); on this build none does.
  check-beats 142 of 142 (run alone: with other browsers running, its "2.6 s later it has landed" test of the overview fails,
  it waits on the wall clock), check-jumps 0 of 74, check-text 0 strong and 0 faint, check-transient 0 and 0 over 16343 frames,
  check-colour all neutral, check-map clean, check-presenter all pass, shim `14,74,984`. S03, S06 and S07 frames re-rendered;
  S03 b7 and S06 b3 read at full size.
- **Zip rebuilt** (check-zip 10 of 10). Not rebuilt: the phone file and the PDF (left alone since round 25), the gallery, the
  PowerPoint and the MP4 (the MP4 and the PowerPoint clips were rendered with the drift).
- Found, not changed: `check-live.mjs` still names the beats of the 15-scene talk. Since round 24e it fails on "S11, S13, S15:
  nothing moves" (S13 b6 is now the family) and then stops on `S14b3`, a beat that no longer exists. The same on the round 25
  build.
