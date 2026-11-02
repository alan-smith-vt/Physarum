# Cradle supports for curved faces printed face down: test log

**Status (user, 2026-11-01): a long-running SIDE QUEST, not on the critical path.** The camera cartridges (the rig's
"eyes", cosmetic) will be printed **face up** with conventional supports behind, plus a round or two of compensation
calibration so their inserts still fit. The cradle work continues until it works and gets used later.

Goal (user, 2026-11-01): a support system for printing curved, visible faces **face down**, so the camera cartridges
(R30 face) and later parts can be printed in that orientation without tipping them vertical, which would affect
other tolerances. Requirements: the support crushes or crumbles off **by hand**, and leaves an **even, bed-like
texture**. Point-contact supports (arrays of dots) are ruled out: they always look bad.

All coupons are a slice of the cartridge's **R30 face, 3 mm shell**, printed face down (concave side up).

## Common conditions
- **Printer:** ELEGOO Saturn 4 Ultra 16K. Pixels 14 x 19 µm, layer **50 µm**.
- **Resin:** ELEGOO ABS-like clear blue + Siraya Tenacious clear, 75/25.
- **Exposure:** 2.3 s normal, 20 s bottom layers.
- **Bed:** sheet gyroid. `t` = sheet thickness, measured with a first-order distance (|g| / |∇g|), so it's in real mm at
  every period.
- **Gap:** distance from the bed's top surface to the face, measured **normal to the face**, unless stated otherwise.
  Under a flat face the gap is effectively whole layers (0.05 mm steps).
- **Face angle θ:** 0° at the face's lowest line (the flat middle), up to 41.8° at the edge of a 40 mm chord.
- **Export:** every .goo from 2026-11-01 is through the fixed `to_goo` (the mirror fix). `cradle_mirror_plate` passed the
  F-marker check.

## Coupon #1: `cradle_coupon.py` (strips). Printed 2026-11-01 on `cradle_mirror_plate.goo`
- **Layout:** 4 strips, each 80 mm long and 41 mm wide (the chord), lying face down on a continuous gyroid bed.
  Face lifted 3 mm above the plate, 0.6 mm foot.
- **Matrix:** 32 cells.

  | Variable | Values |
  |---|---|
  | Gyroid period (one per strip) | 5 / 6 / 8 / 10 mm |
  | Uniform normal gap (4 bands along the strip) | 0.05 / 0.10 / 0.15 / 0.20 |
  | Sheet thickness (2 halves across the chord) | 0.3 / 0.4 |

- **Results:**
  - Only the **gap 0.20** cells (4 and 8 dots) could be detached at all.
  - Every strip's **flat middle debonded** during the print, and the curved face deformed there.
  - **Every edge bonded solid.** Even the 0.20 "success" scarred lightly.
  - Period 10 (4 notches) gave the best finish, with some scarring at the edges.
- **Takeaways:**
  - Under a flat face, one gap value can't both hold and release.
  - The middle (prints first, as a floating layer) and the steep edges behave differently.
  - The coarse period finishes best.

## Coupon #2: `cradle_squares.py` (separate cells). Commit `3be3ddb`
- **Layout:** 12 separate cells, each 40 (chord) x 20 mm, each on its own bed, so no cell affects another.
  - Fixed: period 10, sheet 0.3.
  - Gap blended by angle: `G_MID + (G_EDGE - G_MID) * sin(θ) / sin(41.8°)`.
- **Matrix:**

  | Variable | Values |
  |---|---|
  | G_MID | 0.05 / 0.10 / 0.15 |
  | G_EDGE | 0.20 / 0.40 |
  | Interface gray (bed's top 0.3 mm) | 100 % / 60 % |

- **Results:**

  | Cells | Outcome |
  |---|---|
  | **1, 2, 4, 5** (G_MID 0.05 / 0.10 at 100 %) | Face printed, **fully welded** |
  | 3, 6 (G_MID 0.15 at 100 %) | Face never printed (only the gyroid) |
  | 7–12 (every 60 % cell) | Face never printed |

- **Takeaways:**
  - **A cliff in the flat middle:** at 2 layers (0.10) the face welds; at 3 layers (0.15) it never attaches. No
    geometric value lies between them.
  - Once the middle welds, the edge gap doesn't matter (0.20 and 0.40 both stuck).
  - **Gray on the bed tips doesn't help.** The tips are already cured before the face is exposed, so under-exposing them
    only makes them shorter, which widens the gap.

## Coupon #3: `cradle_squares3.py` (tapered contact). Commit `28f22a8`
- **Layout:** 9 cells, 40 x 20 mm.
  - Fixed: period 10, full exposure, G_EDGE 0.40 (blended).
  - Each sheet: body 0.3, thinning to the CONTACT width over the last 1 mm under the face.
- **Matrix:**

  | Variable | Values |
  |---|---|
  | G_MID (columns) | 0.05 / 0.08 / 0.10 |
  | CONTACT (rows) | 0.20 / 0.15 / 0.10 |

- **Results:**
  - **#9** (G_MID 0.10, contact 0.10): failed.
  - **#1–8:** survived, but **all needed a metal scraper**.
  - **Column 1** (G_MID 0.05, cells 1 / 4 / 7): the **edges came off very easily**, possibly by hand. **#7** (contact
    0.10) left a **nearly perfect finish on the edges**.
  - **Columns 1–2:** the middle couldn't be removed, and the finish degrades quickly toward the middle.
  - **Column 3:** the middle broke off with the scraper, but left **~1.2 mm raised scarring**.
  - **#8:** the middle broke off with snippers, with a very bad finish.
- **Takeaways:**
  - **The edges are solved:** thin contact lines (0.10) plus a large edge gap (0.40) at full exposure.
  - **The middle welds at every setting,** whatever the contact width.
  - A 0.3 sheet is far too rigid to crush.

## Coupon #4: `cradle_squares4.py` (crushable bed). Commit `a6eb502`
- **Layout:** 6 cells, 40 x 20 mm.
  - Fixed: **period 5**, G_MID 0.05, G_EDGE 0.40, full exposure, taper over the last 1 mm. No root reinforcement.
- **Matrix:**

  | Variable | Values |
  |---|---|
  | BODY (rows) | 0.10 / 0.15 / 0.20 |
  | CONTACT (columns) | 0.05 / 0.10 |

- **Results:**
  - **Survived:** **#4** (body 0.15, contact 0.10) and **#6** (body 0.20, contact 0.10). One of them partly lost
    adhesion between the gyroid and the foot (which one wasn't noted).
  - **All failures were at the foot-to-gyroid root,** so contact 0.05 and body 0.10 weren't really tested.
  - **Crushable by hand: achieved.**
  - The finish is still poor:
    - breaking the middle off by hand left **~2 mm raised bits**, so it went to the scraper;
    - the sides' finish is poor too.
- **Takeaways:**
  - Thin sheets (0.15) at period 5 crush by hand.
  - The root needs reinforcement.
  - **The welded face/tip joint is now stronger than the sheets,** so the bed tears inside itself and leaves stubs. The
    interface has to become the weakest plane.

## Coupon #5: `conformal_squares5.py` (conformal stack + face gray). Commit `1e53fca`. PRINTING (results pending)
- **Layout:** 8 cells, 40 x 20 mm. Stack, bottom to top:

  | Layer | Spec |
  |---|---|
  | Foot | 0.6 |
  | Egg-crate ribs | 0.8 walls, 8 mm grid, arched drains |
  | Perforated conformal shell | 1.2, Ø2 holes / 6 mm |
  | Crush layer | 5 mm, period 5 |
  | Face | R30, 3 mm |

- **Crush sheet profile:** **root 0.4 at the shell**, thinning to body 0.15 over 1 mm, then to CONTACT over the last
  1 mm.
- **Gaps:** G_MID 0.10 (2 layers), G_EDGE 0.40.
- **Face skin gray:** the face's bottom 2 layers (0.10 normal depth) printed at a gray blended by angle, from GRAY_MID
  in the middle to GRAY_EDGE at 41.8°. This lowers the face's own overcure into the tips.
- **Matrix:**

  | Cells | Contact | GRAY_MID | GRAY_EDGE |
  |---|---|---|---|
  | 1 / 2 / 3 | 0.10 | 100 / 85 / 70 % | 100 % |
  | 4 / 5 / 6 | 0.10 | 100 / 85 / 70 % | 85 % |
  | 7 / 8 | 0.05 | 85 % | 100 / 85 % |

- **Judge:** the root held; the bed crushes off; the break is **at the face** (no stubs); finish in the middle and on
  the sides.
- **Results (2026-11-01): every cell failed.** The ribs, shell and gyroids printed fine (and looked very good), but **no
  curved face printed in any cell**: the gyroids end in empty space. That includes the 100 % / 100 % cells, so face gray
  isn't the cause.
- **Takeaway:**
  - Contact 0.10 at a middle gap of **0.10 (2 layers) doesn't hold the face's floating first layer**. That matches #3
    #9 (G_MID 0.10 + contact 0.10: failed).
  - The cells that attached with thin contacts all had G_MID **0.05 (1 layer)**: #3 column 1, #4.
  - **With thin contacts, the middle needs a 1-layer gap,** and gray has to bring the weld down from there, not up from
    2 layers.
- **Post-processing note:** a 5 s IPA dip followed by an immediate water rinse turned the thin gyroids **opaque white**.
  Curing one coupon under water (10 min) to see whether the clear mesh can be preserved.

## What we understand so far
1. **The flat middle prints first, as a floating layer.** It's joined to the bed only by the face's overcure, which cures
   through the gap onto the tips. Layers are 0.05 mm, so the flat-face gap is quantised: 2 layers welds, 3 layers loses
   the face. **Gray on the face's first layers** (not the bed) is the only continuous control of that reach.
2. **Steep faces** meet the bed mostly sideways, where pixel resolution is fine (14–19 µm). There, a large normal gap
   (0.40) with thin contact (0.10) releases at full exposure.
3. **Crushability** comes from the sheet thickness of the whole bed (0.15 at period 5), not the contact. A double-curved
   sheet is much stiffer than its thickness suggests.
4. **The weakest plane has to be the face/tip interface.** Otherwise the bed tears inside itself and leaves stubs.
5. **Roots** need reinforcement (thicken the sheets toward the foot or shell).
6. **Printability floor,** under reasonable conditions: body 0.15, contact 0.10. Contact 0.05 and body 0.10 are unproven
   (they failed only at the root).

## Plan if #5 doesn't settle it (user, 2026-11-01)
Reduce the problem before reintroducing curves:
1. **10 mm squares, two cases tested separately:**
   - a **flat** face (θ = 0);
   - a **40°** face (a flat plate tilted 40°).

   No rib grid, no shell: a reinforced-root gyroid bed straight on the foot. Small cells allow many more combinations per
   plate (several dozen), for example:
   - **flat:** gap 0.10 / 0.15 × face skin gray 100–60 % in small steps × contact 0.05 / 0.10 × body 0.15 / 0.20;
   - **40°:** gap 0.20–0.50 × skin gray × contact.
   - **The contact spectrum (user):** needles (~0 % contact) → gyroid (~1–5 %) → **solid sheet (100 %, peeled off)**. A
     sheet could leave the best finish, but with full contact the joint strength is the only control. At 100 % gray the
     flat cliff (2 layers weld, 3 layers lose the face) leaves no peel window. **Gray moves the cliff**, so 80 % with a
     1- or 2-layer gap may be exactly "tacked everywhere, peels". Add to the flat matrix: **sheet** and
     **perforated sheet** (holes ~2 mm pitch: less suction, drains), gap **1 / 2 layers × face gray 100 / 90 / 80 / 70 /
     60 %**. The 1-layer, 100 % cell is the known-weld control. Watch for suction on lifts and peel force from the
     sheet's full cross-section.
2. **Calibrate each case on its own** until each releases cleanly at the face with an even texture.
3. **Interpolate** gap, gray and contact by face angle between the two solutions.
4. **Only then reintroduce curves** (the R30 cells), to check curve-specific effects: curling and warp, peel load on a
   large face, the first floating layer of a curved face.
5. **Then the full stack** (conformal shell + rib grid, `conformal_cradle.py`) for real parts.

## Files
- `cradle_coupon.py`, `cradle_mirror_plate.py`, `cradle_squares.py`, `cradle_squares3.py`, `cradle_squares4.py`,
  `conformal_cradle.py` (review model), `conformal_squares5.py`. All in `slicer_preview/designs/`.
- Previews for the fine features: `PREVIEW_SCALE=6 python generate.py <design>`.
- Printer monitor: `python printer_monitor.py` → http://localhost:5556.
