# Automatic support suite: design (2026-11-01, not started)

**Why:** `designs/pi_box_v4_cart_tilt.py` proved generated supports work. But it evaluates every column over the whole
layer image on every layer: 328 columns × ~12 M px per layer, about 4 G operations per layer, so the cartridge plate
export takes ~13 min, almost all of it supports. We also want trees/lattices off the plate and needles for part-to-part
supports, as in the standard slicers.

## Standard practice (Chitubox, Lychee, PrusaSlicer SLA, Formlabs PreForm)
1. **Orient** the part: cosmetic faces away from the plate, often a 15–45° tilt (smaller cross-sections, less peel
   force, no suction pockets).
2. **Find what needs support:** islands (always), and overhangs flatter than a set angle. Density goes up as the
   overhang gets flatter, with a minimum spacing between points.
3. **Heads:** a tapered cone with a small tip at each contact point, sized by the area it carries.
4. **Routing down:**
   - straight pillars where the path is clear;
   - **merging** nearby heads into shared pillars (PrusaSlicer "pillar merging"), taken further into **trees**;
   - **cross-bracing** between pillars every few mm against wobble and peel (the lattice);
   - **part-to-part mini supports:** short, thin needles.
5. **Base:** a raft or pads with a chamfered edge so it pries off easily.
6. Supports are **separate geometry** (cylinders and cones), drawn into each layer on their own.

## Proposed module: `slicer_preview/supports/`
### Stage 1: analysis → contact points
- Voxel overhang and island detection in print orientation (exists in `pi_box_v4_cart_tilt._plan`, and as
  `designs/pi_box_overhangs.py`).
- Density scaled by overhang angle, a minimum spacing, and mandatory island points.
- Regions narrower than MIN_SPAN (hole and slot roofs) are left to bridge.
- **Keep-out zones**, declared by the design: cosmetic faces (e.g. the cartridge R30 face), fitted features (insert
  slots, bores, board pocket). Points there are moved to the nearest allowed spot, or reported.

### Stage 2: routing → support graph
- Output: a list of segments `(p0, p1, r0, r1, kind)`, kind = head / trunk / brace / needle / foot.
- **Avoidance volume:** the part grown by a clearance. Routes never enter it.
- **Trees off the plate:** heads drop and **merge** within a merge radius while going down, and branches thicken as they
  merge. Fewer, stronger trunks reach the plate.
- **Bracing:** diagonal braces between neighbouring trunks every BRACE_DZ, skipped where they'd enter the avoidance volume.
- **Part-to-part:** needles (tip at both ends, thin shaft), used when the path to the plate is blocked.
- **Base:** pads per trunk, or a raft. A 45° chamfer on the raft edge; elephant's-foot compensation on the part's plate
  face (as `EF_CH` in cart_tilt).

### Stage 3: rendering
- **For each layer:** for each segment that crosses that height, draw its (elliptical, if tilted) cross-section **only
  inside its small bounding box**. That's a few hundred tiny stamps per layer instead of full-image passes, so support
  time should go from minutes to seconds.
- **GPU:** one batched kernel for all segments in a layer (CuPy). CPU: vectorised numpy stamps.
- The output merges into the part's slice by max, so gray / anti-aliased parts are unaffected.
- **Viewer:** supports as mesh overlays (instanced cylinders and cones), not voxels, so previews are fast and the
  supports can be toggled (and grouped, as in the legend groups).

### Parameters and presets
Tip radius and dig, head length, shaft radius, density per angle, minimum spacing, merge radius, brace spacing, raft vs.
pads, needle radius. Grouped into **light / medium / heavy presets per resin** (clear blue + Tenacious, smoky + Tenacious).

### Checks
- Re-run island detection with the supports included: nothing floating.
- Report the peel area per layer (largest cross-section) and plate contact area.
- Overlay of spots left to bridge (as cart_tilt does now).

## Build order
1. **Graph + stamp renderer**, so speed is solved at once and cart_tilt's columns can be ported to it (compare export
   time and slices against the current .goo).
2. **Trees with merging + bracing** off the plate; **needles** part-to-part.
3. **Keep-out zones**, presets and checks.
4. Hook into `compose.py` / `to_goo.py` so any design can say `SUPPORTS = dict(preset='medium', keepout=[...])`.
   `to_goo` currently refuses `SUPPORTS` at full resolution.

## Baseline to compare against
`pi_box_v4_cart_tilt` (STAND 'picked'): 308 plate columns + 20 part-to-part, 62 spots left to bridge, ~13 min export.
