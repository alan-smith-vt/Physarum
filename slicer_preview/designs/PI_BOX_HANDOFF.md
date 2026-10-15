# Pi box (lidar rig): handoff, 2026-10-14 (evening), for the core reprint

The decision history lives in the Notes vault (`Reference/lidar-photo-rig-cage-handoff.md`), but **the desktop
doesn't have the vault**. Everything needed for the reprint is repeated here.

## How the user wants you to work (hard rules)
- **Show a design change visually before modelling it.** Use a dimensioned sketch, section or figure, then wait
  for approval. If a fix has more than one plausible shape, approval covers only the direction, so sketch the
  concrete geometry first.
- **After modelling, show it in the viewer first** (`python generate.py pi_box_bat_core`, then open
  http://localhost:5555/battery). Run the clash check, path checks, area scans or exports only after the user
  has looked at it.
- **No extra pieces unless asked.** For example, a separate camera cartridge or plug was rejected outright.
- Commit and push as you go. Push with `git -c sendpack.sideband=false push`.

## Status
- **Core v2 printed clean (2026-10-15).** See "Print results: core v2". It's in use. The camera seats were filed
  to fit; there are fixes for a v3.
- Core reprint started 2026-10-14 night from the commit after `e572396` (flat seat; cord and lidar cable re-routed
  inboard over it, |X| < 22; clash check clean apart from the bank's intended contact at d 22). Layer areas, read
  from the .goo: peak 7,117 mm² (flange, 0–4 mm), a single −25% step to 5,319 at 4.00 mm (plate top, the fillet
  ramps it after that), and 950–1,800 above 25 mm. 3,600 layers at 2.6 s. Check on it with `python sdcp.py status`.
- **First core printed 2026-10-13/14** (Physarum `1b3f47e`). It stays as the mock-up. Problems found:
  1. The flange delaminated ~5 mm up.
  2. The inset camera seats can't be assembled.
  3. The Al spacers just barely don't fit.
- **Reprint planned for tonight (2026-10-14)**, but there is a blocker (see "Uncommitted / open" below).
- The floor and panels aren't printed yet. Panels come last, after everything else works.

## Changes since the first print (committed, on main)
- `32d5072` **Spacer bores Ø8.6:** `SPACER_FIT = 0.25` on top of `RESIN_HOLE_COMP` 0.35, for both plate and boss bores.
- `fa95b66` **Cameras moved to the −Y corners**, over the corner M4s:
  - `CAM_YR ≈ −40.1`, `CAM_D = 30`. The board's −Yr edge sits on the panel-slot solid band.
  - `CAM_RIB_Y = CAM_YR + 14`. Ribs are unclipped; the +X rib stops above the UE300.
  - Ø `HEAD_D` vertical bores run through the camera blocks over the M4s.
  - `CLIP_RIB_DS = (55, 80, 104)`. Clips and the USB-C/ribbon routes were re-routed.
  - Assembly order: box on, corner M4s up through the bores, then cameras, then the Pi. To take the box off,
    the cameras come out first.
- `d5e86cb` / `89bbec6` **Power switch** at y = `CAM_YR`, with a square solid pad (`SWITCH_PAD_R = 12.5`).
- `3a24da0` **Open flange**, the fix for the delamination:
  - The plate keeps only what the body hangs off: walls, rims, bosses, webs, tube and camera blocks, plus rings
    round the M3 recesses, a 3 mm perimeter band, a ring round the well, panel-midpoint spokes to the tube, and a
    diamond between the edge midpoints.
  - Organic windows (`_flange_field`: gaussian blur + EDT at 0.1 mm), plus a 3 mm quarter-round fillet into the body.
  - The plate went from ~9,700 to ~6,400 mm² per layer. The suction vents were dropped.

## Uncommitted / open in this working tree (committed with this handoff, read carefully)
1. **Tube bore Ø52 → Ø43 (`WELL_R` 26 → 21.5, `TUBE_T` 2 unchanged). APPROVED; the user inspected it in the viewer.**
   - The tube is a safety grip over the monopod's hard ring (~Ø42, 10 mm below the top); the foam grip below it
     gets padded.
   - Webs, Pi standoffs, flange ring and spokes, and the floor collar all follow `WELL_R` automatically.
   - The monopod stand-in in the rig overlay has its ring set to Ø42. It's a clash-check reference only.
2. **Flat battery seat. BLOCKER, NOT RESOLVED. Don't print it as it is.**
   - Problem: the bank sits 5–10 mm lower than designed. Its rounded corners slide past the 45° shoulders.
   - The user decided on a flat seat (no 45° slope) reaching 10 mm further in. The current code does that:
     - `BAT_SHOULDER_X = 23.5` (was 33.5)
     - `_rail()` uses `Half(0, 0, 1, BAT['d0'])`, i.e. solid for all d ≤ d0 = 22
   - **The clash check fails:** the PD trigger (d 14–18) and the lidar cable (d 11.7–19.6) run through the new
     solid, and the bank touches the seat at d 22 as intended.
   - The user said "not fixing it right now". Before printing, **sketch the options for the user**. For example:
     - a thin flat ledge at d0 with a 45° taper back up to the wall, kept out of the cable/trigger paths
     - a flat seat only on the rail band that's clear of the cables
     - revert to the 45° shoulders and pad the bank instead
   - Don't pick one yourself.
   - Bank facts (user, 2026-10-14):
     - The bank's **ports are on the bottom** (floor end).
     - A single permanent built-in cable leaves the **top**. It can fold 180° in the worst case.
     - The bank is 74 wide (|X| ≤ 37) and spans Yr 30.8–55.8.
3. **The layer-area scan hasn't been run on the new geometry** (see the printing rules below).

## Checks
- Clash check: `python -m designs.pi_box_check pi_box_bat_core`. It was clean before the seat change.
- Path check: a scratch script on the laptop checked the camera insertion path, the key path and the Ø9.8 head
  path. It isn't in the repo. Re-derive it if needed, or ask the user.

## Resin printing rules (Saturn 4 Ultra 16K, Fast gray / Tenacious Lava black 75/25)
- **Exposure:** 2.4–2.6 s, with 20 s bottom layers. The core printed at 2.6 s.
- **Printable area:** the LCD image is 211.7 × 118.4 mm. `to_goo` refuses anything larger.
- **Keep every layer under ~8,000 mm².** Two prints split near that size: the v2 base plate (~8,000) and the
  first core (~9,700 solid flange, split ~5 mm up). Check the per-layer area before printing.
- **No sharp jumps in layer area.** The core split where it dropped from 9,700 to 3,800 mm². Ramp changes over a
  few mm with fillets, chamfers or tapers.
- **Holes:** vertical holes get +0.35 on the diameter (`RESIN_HOLE_COMP`). Horizontal holes get an extra +0.2 in
  height (`HORIZ_HOLE_EXTRA`).
- **Holes near the build plate need ~+0.25 more** (elephant's foot from the burn-in layers):
  - Spacer bores starting in the flange: Ø8.0 + 0.35 was just too tight, so they're now Ø8.6 (not yet test-fitted).
  - M3 insert holes at the far end: Ø4.0 + 0.35 is slightly loose (~0.1–0.2), which is good for glued inserts.
  - The allowance may scale with hole size; there's not enough data yet.
  - A small chamfer at the hole mouth on the plate face is an alternative.
- **Resin change (planned, no rush):** the black Fast/Tenacious blend runs out after the Monday prints. The
  restock is Fast (any colour) + **clear** Tenacious. Clear cures deeper and bleeds more than black, so redo the
  exposure test and one hole-fit print (spacer bore, insert hole) first. Expect a shorter exposure and different
  hole allowances. Print the coupons in the new blend too.
- Glue inserts with CA (superglue), not threadlocker. Wash and fully post-cure before gluing.

## Print results: core v2 (reprint of 2026-10-14 night, checked 2026-10-15)
- **Clean print.** The open flange held, with no delamination at the flange or at the 4 mm step.
- **Minor delamination under the floor screw seats on the Pi side.** The battery-side seats line up with a rib and
  printed clean. If there's a v3, give the Pi-side seats a rib or web underneath, or ramp them in.
- **Surface:** no terracing on the flange fillet or chamfers, so Z-AA isn't needed (see Modelling notes).
- **Spacers:** fit the Ø8.6 bores. The plate-face allowance is confirmed.
- **Cable clips:** the big clips are exactly right for the M12. The small clips are a little too big to clip onto
  the smaller cords. Tighten them for v3.
- **Battery:** fits on the flat seat with ~1 mm vertical and ~2 mm horizontal clearance (totals, not per side).
  The plan is to pad the bank's edges for a snug fit. No geometry change.
- **Camera seats: the only major issue.** Each camera board has two small nubs, one on either side, where the lens
  screws attach to the PCB. They hold the board ~4 mm off its seat. The user filed the seats down to fit this
  print. **v3 fix:** two horizontal bores in each camera seat, centred vertically between the screw holes, for
  clearance.
- **Not checked yet:** the thread inserts haven't arrived (`PI_INSERT_D` / `CAM_INSERT_D` are still [MEASURE]).

## Test-fit results on the first core (2026-10-14)
- **Pi:** standoff alignment passes with the real board. The insert bores take an M3, so they're ≥ Ø3.0. The deeper
  clearance section is about Ø3 (an M3 would thread it, as expected).
- **Camera bosses:** M2 passes. Hole alignment is **not** checked, because removing the lens would lose its
  focus. The user is gambling on it.
- **Lidar:** the M3 head recesses line up. The printed base plate still uses the lidar's own M3 screws, whose
  heads drop through the recess area. On the metal plate, M3 button heads will sit on top and use the recesses.
- **M3 floor inserts (Ø4):** slightly loose in their holes. Good as it is.
- **Arriving Friday:**
  - power switch: test the hole and pad
  - UE300: test the cradle and jack opening (a floor part)
  - M2.5 / M2 inserts: caliper their OD, then set `PI_INSERT_D` / `CAM_INSERT_D`, which are still [MEASURE]

## Other to-dos (not for tonight's core)
- **Reprint the cross top: a STRUCTURAL requirement for Monday, not a convenience.** It needs a **countersunk ¼"
  head** on one side, and it must be **thickened**.
  - On Monday the ¼" screw through the resin is the only load path holding the GoPro.
  - The existing crosses are 4.76 mm thick. Sinking a ¼" flat head flush (82°, ~12.7 mm head, ~3.6 mm deep)
    leaves only ~1.1 mm of plate under the cone.
  - The user is drilling one of the two existing crosses for **practice and fit only**. Don't fly it.
  - Print the thickened version in the calibrated black blend before Monday.
  - The GoPro mount isn't a fixed orientation; it gets calibrated as-is for that one day. The cross is in
    `CAGE_HANDOFF.md`'s scope.
- **Regenerate the floor and the panels before printing them.** Any outputs from before 2026-10-14 are stale:
  - The floor's collar follows `WELL_R` (now 21.5 for the Ø43 tube).
  - The panel print file is now the gyroid panel.
  - Re-run `generate.py` and `to_goo.py` for `pi_box_bat_lids` and `pi_box_bat_panel` from current main, and check
    their layer areas.
- **Battery fit fallback.** If the bank doesn't quite fit under the new flat seat (the floor won't close, or it
  pinches), cut a **slight recess into the floor** under the bank. That's the agreed fallback; don't re-slope
  the seat. `BAT_H_MARGIN` is +1 mm on an unmeasured bank height, so caliper the bank first.
- **Panels:** don't print them yet. The print file (`pi_box_bat_panel.py`) currently makes a 12 mm gyroid panel
  with a 6 mm solid frame (max layer 6,628 mm²). A plain panel printed flat would be ~16,000 mm² per layer, which
  is too much, but the user prefers the plain look. Run the coupon tests below first. If a gyroid raft with a fine
  lattice interface leaves an acceptable face, print the plain panel on one. Otherwise keep the gyroid panel (or
  print it on edge).

- **Monday site visit with the plastic cage:** the ¼" screw through the resin flange is the only load path. The
  smaller tube is the backup. Also suggested: a tether from the cage to the pole below the coupler.

## Printer network: slow uploads (test when no print is running)
- Measured 2026-10-14 over the same link:
  - Ping: 0 % loss, ~5 ms.
  - Download from the printer (a timelapse): **3.3 MB/s**.
  - Upload of a 241 MB .goo: **0.28 MB/s** (~15 min). That's ~3.7 s per 1 MB chunk, against ~0.3 s for the bytes.
  - So the time goes to the printer's per-chunk handling, not to the network.
- `sdcp.py upload` now prints per-chunk time and rate, and takes `chunk=` / `check=` options for testing.
- Tests (with a ~20 MB .goo, printer idle):
  1. Default upload. Is chunk time constant, or does it grow with offset? Growth means the whole file is
     re-hashed per chunk.
  2. `check=0`.
  3. `chunk=4194304`, then `chunk=8388608`, if the printer accepts them.
  4. Send the same file from ELEGOO's software and time it, to see whether it's just as slow (firmware).
  Keep whatever is fastest and still verifies (the file prints).
- Also found: **telnet (port 23) is open** with a Linux `login:` prompt. The board runs embedded Linux
  (RTSP server "ireader/media-server"). Not logged into; see the firmware discussion with the user before
  trying anything.

## Flood-fill gyroid support: coupon tests (2026-10-14, before the panels)
Background (patterns, mechanics, support and infill trade-offs): [`../TPMS_NOTES.md`](../TPMS_NOTES.md).
Goal (user): supports as a brute-force flood fill that needs no judgement, unlike trees.
- The pipeline: encase the part in gyroid, cull to the part's "shadow" (columns under downward-facing surfaces),
  clear by distance from the part, then grade sparse→dense toward the interface.
- The interface is a **fine-lattice band** only where the surface faces down past ~45°. Every other surface gets an
  air gap.
- Drop lattice islands that don't reach the plate. Leave out the insides of closed cavities. Run the layer-area
  check, then print.
- Tips/cones were rejected: if we wanted tips we'd use trees.

Start from [`tpms_coupon.py`](tpms_coupon.py) (2026-10-11, never printed; not in the printer history):
- 4×4 cells, each a TPMS block with a solid cap floating above it.
- Rows: solid fraction 15–40 %. Columns: cap gap 0.10–0.50 mm.
- ID holes are punched in the caps.

Extend it so each cell is a **graded gyroid block → fine-lattice interface band → flat cap** (the cap stands in
for the part's down-facing face). Variables:
- fine-lattice period and sheet thickness
- interface band height
- the air gap between band and cap (the cap only holds if the gap is under the cure depth, so the gap is part of
  what's being tested)

**Measure two things per cell:**
1. **Finish:** how rough the cap's underside is. Compare it to the plate-side face and to a tree-supported face.
2. **Break-off force per mm² of interface:** a spring scale or weights, until the cap comes off. Also time how long
   cleanup takes.

Pick gap and lattice settings once from these and reuse them.
- Break-off force should become an automatic check: the interface must let go at less force than the part's
  weakest feature (its cross-section × resin strength), so freeing a thin arm doesn't snap it.
- Also try one cell with an enclosed pocket (a lattice-filled gap like between an arm and a torso). It shows
  whether fine lattice can be picked out of confined spaces.
- Show the coupon layout to the user before modelling (hard rule).

## Other context (Pi system, done on the laptop today; no action here)
- Power: Anker 20K bank → PD trigger at 15 V → two WAGO 221-413 → lidar, and → buck → Pi. On 22 AWG solid the Pi
  reads 5.20 V with throttled 0x0.
- Pi host `lidarpi`, user `lidar`, key-only SSH.
- Lidar: Mid-360 at 192.168.1.171, Pi eth0 at 192.168.1.50. The lidar is **right side up** in the rig. The bench
  test read z ≈ −1 g only because it sat upside down on the bench (coupler nut), so the extrinsic needs no flip.
- Livox-SDK2 receives nothing yet (raw UDP works).
- Both OV9281 cameras work (dtoverlay ov9281 cam0 and cam1) and are focused.

## Modelling notes
- `pi_box_bat_core.py` is an SDF tree (`slicer_preview/sdf.py`: Box / Cyl / Half / Fn, union / inter / sub,
  smooth ops). `Half(nx, ny, nd, c)` is the half-space nx·X + ny·Y + nd·d ≤ c.
- Rig constants are in `pi_box_bat_rig.py`. Lines marked BATTERY are overrides. The core imports them with `*`.
- `PI_BOX_AA=1` turns on grayscale edges. It's off by default, because gray pixels cure less and shift fits.
  - **Verdict (2026-10-14): in-plane AA isn't needed at 16K.** The first core's tube bore came out "smooth as
    glass". The pixels are 14 × 19 µm, and light bleed already softens the steps.
  - Leave it off. It isn't part of the restock calibration.
- **Z anti-aliasing (potential, not built):** the visible artefact left at 16K is **layer terracing on shallow
  slopes** (fillets, chamfers, domes): 50 µm steps. In-plane AA can't fix it.
  - The fix would be greyscale blending between layers: each edge pixel's grey set by where the surface crosses
    the layer, from the SDF.
  - `sdf.py` already has the distance it needs.
  - Look at the reprinted core's flange fillet and chamfers first. Only pursue this if the terracing there
    matters.
  - It needs its own exposure calibration (grey → cure-depth curve), like any greyscale.
- Floor and panels are still mask functions (`implicit_piece`).
- Preview resolution is 16× coarser than the print: 224×304 µm pixels and 160 µm layers, against 14×19 µm and 50 µm.

## Commands (run from `slicer_preview/`)
- Viewer: `python generate.py pi_box_bat_core` (also `pi_box_bat_lids`, `pi_box_bat_panel`), then
  `python server.py` → http://localhost:5555/battery (core), /batteryFloor, /batteryPanel. The server picks up
  regenerated files without a restart.
- Clash check: `python -m designs.pi_box_check pi_box_bat_core`. Overhangs: `python -m designs.pi_box_overhangs`.
- Print file: `python to_goo.py pi_box_bat_core` (`--gpu` is faster) → `out/pi_box_bat_core.goo`.
- Printer camera: `python sdcp.py camera` (live; quit with q) / `snapshot` / `timelapse [N]`. The camera allows 2
  viewers, and killed sessions lock it until a power-cycle: see the notes at the top of `sdcp.py`.
- Printer: `python sdcp.py status` / `watch` / `history`. The printer is at 192.168.0.4 (found by discovery).
  A start right after an upload can return `Ack 1`; just retry.
- Meshes: `python -m designs.pi_box_export > log 2>&1`, then check the log for `Traceback`.
  Use `PI_BOX_EXPORT_ONLY=core` for the core alone.
- Render: `"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -t 4 -P designs/pi_box_render.py -- <outdir> [views] --samples 96 --res 1600 1200`.

## Gotchas
- Never render the web viewer headless: the CPU fallback maxes out the machine.
- Don't edit `pi_box_bat_rig.py` with PowerShell Get/Set-Content. It re-encodes the UTF-8 (Ø × °).
- Print frame: print x = Yr, y = d, z = −X (`to_print` / `from_print`). Viewer "up" is rig depth (the floor end).
- The core prints upright, flange down. d = 0 is the plate face, and d = 180 is the floor end.
