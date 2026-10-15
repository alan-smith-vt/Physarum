# TPMS notes (gyroid & friends) — 2026-10-14

Collected from a design discussion. Numbers are typical literature values, not measurements on our resins.
Interactive toys (claude.ai artifacts, private to the user):
- TPMS section box: https://claude.ai/artifact/7iW6qwurQPekxQgMFqpwtb
- Support comparison (gyroid fill / trees / lattice, with resin volumes): https://claude.ai/artifact/ELDsp62QmApCWk1Cz348Qg

## How sections look
- The gyroid has no preferred axis. g = sin x cos y + sin y cos z + sin z cos x is unchanged by x→y→z→x.
- Why the camera walls (constant-X slabs) and the panels (constant-Y slabs) look different:
  - Under the x→y→z cycle, a constant-X section equals a constant-Y section **rotated 90°**.
  - The slab's **phase** within the period matters. A 3–4 mm slab is under ⅓ of a 12 mm period.
- **4₁ screw axes** (space group I4₁32): a quarter period along an axis gives the same pattern turned 90°, so
  sections alternate between waves one way and waves the other way.
  - At ⅛ period the section passes through the surface's **saddle points** (a critical level, Morse theory). The
    contours touch in X's and swap partners (a topological transition / reconnection), which looks like a diamond
    checkerboard.
- **[111]** (body-diagonal) sections show a hexagonal array of near-circles.
  - [111] is a 3-fold axis, and both labyrinths run along it as loose helices (3-fold screws) with a clear
    straight line of sight.
  - [100]: waves / diamonds. [110]: "double wave". [111]: circles.
  - In the toy: Z 45°, then X or Y ±35.3° (arctan 1/√2).
- In thin slabs (< ⅓ period, like our walls and panels), air crosses almost straight in any orientation. The
  zig-zag path only develops over several periods.

## Mechanics
- **Sheet** (|g| < t) TPMS: doubly curved, so it carries load mostly by in-plane stretching. Among the stiffest and
  strongest cellular structures per weight.
  - At ~25 % solid: roughly 10–20 % of solid stiffness. A foam or strut lattice at the same density is a few %.
  - Stiffness is close to the same in all directions (cubic symmetry).
- **Network** (g < t): strut-like and bending-dominated, so much softer and weaker at the same density.
- Stiffness tunes with density: relative density ≈ 3.1 × wall thickness / period. Stiffness ∝ ρ^~1.5–2.
- **Squishy is a material question.**
  - Rigid resin (~2 GPa): a "hand-squeeze squishy" 50 mm cube needs ~2 MPa effective stiffness, i.e. 1–3 %
    density, i.e. ~0.08 mm walls at a 12 mm period. That's below the 0.6 mm minimum, and it would crack first.
  - For sponginess use elastic resin or TPU (~15 % density, 0.6–0.8 mm walls).
  - At our wall thickness the gyroid acts as a perforated plate, not a bulk lattice.
- In nature, minimal surfaces come from **surface-energy minimisation**: soap films, lipid cubic membranes,
  block-copolymer gyroid phases. Evolution then uses the side benefits:
  - butterfly scales: structural colour
  - sea urchin stereom (≈ Schwarz D/P): a stiff, light skeleton with tissue in the pores
  - cubic membranes: exchange surface

## Other patterns worth trying
- **Schwarz D (diamond):** stiffest common TPMS per density. Self-supporting. It's an option in our wall fill.
- **Schwarz P:** cubic round holes; anisotropic; pores close easily.
- **Neovius:** very stiff; tight curves, harder to print fine.
- **I-WP:** two channel sizes; a distinctive look, big drainage plus fine structure.
- **Fischer–Koch S, lidinoid:** intricate, mostly aesthetic.
- **Spinodoids:** random, bone-like, tunably anisotropic; no moiré.
- **Voronoi foams:** organic; easy density grading.
- **Strut lattices:** octet (stiff, but horizontal struts are bad in resin), BCC (printable, softer), Kelvin.
- Fields can be **blended**, e.g. a stiff D in the bulk turning into a soft network gyroid at an interface.

## Supports: flood-fill gyroid (the user's direction)
Plan and coupon tests: see `designs/PI_BOX_HANDOFF.md` and `designs/tpms_coupon_v2.py`. In summary:
- **Pipeline:** encase → cull to the part's shadow → distance-clear → fine-lattice interface only under
  down-facing (>45°) surfaces → grade sparse→dense → drop islands → layer-area check.
- **Pros:** no judgement needed, smooth layer area, very stiff (accurate), redundant, drains freely.
- **Cons:** the most resin, the largest layer area, slow removal, poor in tight spaces.
- **Interface:** the fine lattice must let go at less force than the part's weakest feature. A network-gyroid
  interface probably breaks away more easily than a sheet one, so test it.
- **Trees** remain best for delicate figures (few point contacts, least resin). Tips/cones on a gyroid platform
  were rejected: if we want tips, use trees.

## Infill for hollow resin parts (with drainage)
1. **Hollow shell** (2–3 mm walls + drain holes), if it's stiff enough. Usually the right answer in resin.
2. **Schwarz D or gyroid sheet** for general strength. ⚠ A sheet TPMS splits the inside into **two separate
   labyrinths, and each needs its own drain hole**. Otherwise one stays full of uncured resin, which can crack
   the part later.
3. **Vertical honeycomb** for a single known load axis: vertical walls print perfectly and drain straight down.
   It's weak sideways, and cell tops need peaked or supported roofs.
4. **Network gyroid** when one connected drain path matters most (a single void).
- Winding TPMS channels wash out harder than straight ones. Use larger periods and agitation, and check that the
  solvent runs clear before gluing or sealing.
