---
date: 2026-10-01
status: accepted
---

# ADR-001: one place per scene along a path in talk order, and a shelf that travels

## Context

The talk ends on an overview of the whole World (S14 b4, S15 b1). Up to build r17 the scenes kept coming back to one
shared shelf of hadron glyphs at the World's centre, so on that overview scenes 2, 8, 9, 10 and 14 lay on top of each
other, and 6 on 7, and 12 on 13. The author asked for separate parts, placed in talk order. `--layout` had passed all
along: it compares world boxes of scene-owned boards, while the overview is a projection that also shows the shared shelf.

## Decision

- Every scene has its own place: `LAY.T[sc]`, a translation of the whole scene. The places follow one winding path in talk
  order: title top left, S02 to S05 down the left side, S06 to S08 along the bottom, S09 to S11 up the right side, S12 in
  the middle, S13 above it, S14 at the top beside the title, S15 above the title.
- Scene code keeps its own coordinates. `FR(SC)` gives each scene `P` (a point), `V` (a vector) and `KS` (a camera keyframe)
  that add the scene's `T`. A persistent object that flies between scenes names both ends in their own scenes' frames
  (`TW(3, …)` to `V(…)`).
- The shelf has an origin (`SHELF.at`) and travels with the camera into each scene that uses it (S02, S08, S09, S10, S12,
  S14): `SHELF.travel(a, b, u)` in that scene's first beat. Glyph positions stay local to the shelf; `SHELF.wp(id)` gives
  the World position. Boards that belong to a scene but ride on the glyphs (the S10 lattice chips) follow them only while
  the shelf is in that scene, then stay where the glyphs stood (`SHELF.layoutPos`).
- `tools/check-map.mjs` measures the regions on the overview frames themselves, in pixels, and fails on any two closer than
  8 px; it also fails if the shelf does not reach six places.

## Consequences

- The overview is clean (15 regions, 0 overlaps) and reads as the talk's route. The World is larger (x −313…308,
  y −270…400), so the overview camera stands farther back (1051) and the thumbnails are smaller; badges were scaled to stay
  about 36 px.
- Camera moves between scenes are longer. The shelf's travel is long in S08 b1 (from S02's place) and S14 b1.
- The S05 chip "Λ₁ absorbed into m_Q" now appears where it lands in S08, since its S05 place is across the World.
- Text drawn under 5 px (the overview thumbnails) is no longer measured by `check-text.mjs`: at that size its boxes are not
  its letters.

## Alternatives considered

- A finale that gathers each scene's main board into a clean timeline at the end, leaving the World as it was: rejected by
  the author in favour of re-laying the World itself.
- Copies of the shelf in each results scene: the author chose one shelf that moves.
- Dropping the automatic zoom-out at the end: rejected.
