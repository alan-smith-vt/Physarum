# Rig v4: state 2026-11-03 (evening)

Rig 1 is **Alnilam** (fleet names: `RIG_NAMES.md` at the repo root). Plate set (core + both panels, smoky) printed
perfectly and everything assembles; blue resin refilled. Exposure: smoky 80 % (2.8 s) won again on the new FEP, so blue
stays at 2.3 s without a recalibration (the floor plate is the canary).

**Printed / printing (blue, 2.3 s):**
1. `pi_box_v4_floor_plate.goo` (design `pi_box_v4_floor_plate`): floor + 3 glue-in standing lips. Printed.
2. `pi_box_v4_door_cart_plate.goo` (design `pi_box_v4_door_cart_plate`, 770 layers, ~2.2 h): both cartridges
   (`cart_tilt`, unchanged) + door A reprint standing on its grip end. Printing.

**Standing lips** (sketch `sketches/pi_box_v4_floor_lips_sketch.png`, option B): the M3 thumb screw stands 6.5 proud of
the door plate vs the old 5.3 grip lip, so the rig rocked. Now everything stands on one plane at d 190.7 (1.1 below the
screw tip): floor side rails (X 43.5..50, Yr -30..28, R3.5 notches round the M3 button heads) + front bar (X +-18) as
separate lips glued into grooves; the door's full-width standing lip (|X| <= 46) is part of the door.
- Grooves (floor outside face): 4.2 x 1.7, 0.3 mouth bevel, square breaks at the screw bosses and the RJ45 opening,
  solid floor round them. Tongues 3.6 x 1.4 (0.3/face, 0.3 under the tip). Lips print foot down, 0.5 EF bevel.
- **Join plan (user):** wash + dry, a bead of the same blue resin in each groove, press in, wipe squeeze-out, stand the
  floor on its lips on glass, UV-tack, then full cure (the wet resin lets all lips settle onto one plane).

**Door A changes** (`pi_box_v4_door_a.py`; core unchanged): CLEAR 0.2 -> 0.1 (web 0.275/face), tongues 2.4 (-X) /
2.7 (+X) tall (up/down slop 1.15-1.45 -> 0.75 total), Ø3.2 slide-through screw hole (no longer captive), standing lip,
"Alnilam" engraved 0.5 deep in Old English in the -X strip opposite the screw (`NAME`, `NAME_FLIP = True` reads right
from outside; font shortlist `sketches/pi_box_v4_door_name_final_sketch.png`, review design `pi_box_v4_door_names`).

**Checks when it comes off:** cartridges as 11-01 (inserts, bores, connector relief); door slides with less wiggle, M3
slides through, name reads correctly, door lip level with the floor lips on glass.

**Tooling (11-03):** `to_goo` trims empty top layers; `print_post.text_inside` (even-odd text fill: counters stay open);
CPU exports over ~5 min are out (user): make designs `--gpu`-safe (fixed: door trees' `ab @ ab` needed cuBLAS). The
door/cart plate still took 8.7 min on the GPU (cartridge support columns looped per layer).

---

# Rig v4: state at end of 2026-11-02 (night)

## Printing overnight
**`pi_box_v4_plate_set.goo`** (smoky, 2.8 s, 3600 layers, ~10.5 h, started ~late evening 11-02): core v4 (unchanged,
same model as `out/pi_box_v4_core_v4.goo`) + **front panel standing** + **back panel standing**, all on one plate.
Design `designs/pi_box_v4_plate_set.py`: panels turned 90° (rotation, not mirror) into the free strips either side of
the core, 3 mm gaps. Peak layer 6,961 mm² (layer 52). **User: keep this core whatever happens;** if a panel doesn't fit,
compensate and reprint the panel.

**Morning checks:**
- All three footprints on the plate. The front panel's floor end was 0.1 mm up (started at layer 2) until a fix tonight
  (`FRONT_PANEL_PIECES` shifted 0.1, now on layer 0, verified in the .goo).
- Back panel standing: the 16 window columns and 4 ledge spikes printed and snap off; rails full height on both sides.
- Core: support trees in the cartridge slots snap out; then the panel fits in the core channels (see older checks below).
- Popping during the flange layers (0–4 mm): quiet = the new film fixed it.

## What happened today
- **New FEP film** (user, 11-02): the old one popped loudly on big layers (lost tension). `exposure_test_smoke` re-exported
  (post-mirror-fix) and printed on the new film: **winner #3 = 80 % = 2.8 s again**, so nothing changes. Bases measured
  ~1.3 mm (design 1.5) evenly across the plate, so the plate is level; ~0.2 mm short at the plate is the squeezed first layers.
- **Back panel printed flat (old film), failed:** body 3.4–4.0 mm thick (should be ~3.2), rail edge 4.2–4.5 (should be
  ~5.2); one rail nearly gone (BL corner worst). Not warp (~1 mm), not exposure. Best theory: the slack film deflected
  under the 5,600 mm² body layers (body overgrew), then the sudden 5,600 → 400 mm² step to the rails left Z behind
  the real top, squeezing the first rail layers. Tilt-release peel may add the side bias. **Fix: panels print standing.**
- **Back panel redesigned** (`pi_box_v4_back_panel.py` flat; `pi_box_v4_back_panel_standing.py` standing):
  - MagSafe disc, ring recess, swoops and logo bar **removed**: the phone mounts on the stick (C-clamp + magnet plate), not the panel.
  - Gyroid fill above the window; window open **5–30 mm from the base edge** (bank screen 5.6–29 mm).
  - Standing version: window corners R4, 16 snap-off columns under the head bar, 2 spikes under each side ledge
    (hook corners stop at d 175.3 for door A's webs, so they start 4.6 mm up), 0.5 mm EF bevel.
  - Flat version keeps 20 needles under the tongue (both corners) + 1 mm outline bevel.
- **Front panel:** centre strip's inner face trimmed flush with the rounds (it was 0.15 proud: the line in the render);
  its hooks end at the step (d 72) where the core channel ends (they ran 10 mm past); 0.5 mm EF bevel; prints standing.
- **GPU export for mask pieces with print post-processing:** `designs/print_post.py` `bevel_piece(piece, ef,
  fill_holes, extra)` gives an EF bevel + extra solids (needles/spikes) as both `make_slice` (numpy) and `slice_xp` (cupy).
  Verified: back panel GPU .goo **byte-identical** to the CPU export (4.4 s vs 254 s). Plate set GPU export ~9 min
  (**close the viewer tab while exporting**: it competes for the GPU and slowed one run ~3×).
- **Uploading during a print is fine:** core upload ran at 0.17 vs 0.19 MB/s idle, and the running print's timing didn't change.
- `pi_box_bat_panel_v3.panel_mask` is now array-safe in y (standing prints); scalar callers unchanged.

## Gotchas (11-02)
- `to_goo --gpu` repoints every `designs.*` module's `np` at cupy: a helper that tests `xp is np` must keep numpy under
  another name (`print_post` uses `_np`).
- An implicit piece whose solid stops 0.1 short of the piece's bottom starts in mid-air: check layer 0 of every piece
  in the .goo (decode with `chromosome/goo_punch_hole_v2.goo_decode`).
- `sdcp.py` discovery can miss once ("No SDCP printer answered"); just retry.

## Side quest: chain textile for resin (11-02)
- Source: **3DSLA "Metal Textile"** (Thingiverse 6029077, CC BY 4.0; 3DSLA = Russian metal-printing firm, printed in
  AlSi10Mg on a RussianSLM FACTORY; 3dsla.ru is gone; Printables 500880 is a re-upload). Local copy
  `C:\Users\Alan\Documents\3d_prints\testing\3DSLA_MetallTextile_1200_Elements.stl` (+ the 2025-07-17 .ctb).
- Geometry: 675 identical links (5,484 tris each), hex lattice pitch 6.6 mm (file units = 0.1 mm), link ~10 × 10.5 × 20 mm,
  standing upright between a bottom and a top ball; links interlock because the pitch is less than the link width.
  ~30 % of down-facing area is steeper than 45° from vertical (round-strand undersides).
- `designs/textile_patch.py` (4 × 4 links from the mesh + a 1.5 mm spike under every ball; viewer /textile_patch) and
  `textile_patch_60.py` (SQUASH 0.6: 12.2 mm links). Interlock scan (100 pull-out directions per neighbour): **locked
  at every squash 1.0–0.5**; min clearance 0.81 → 0.64 mm. Squash's real cost is flatter overhangs and slimmer strands
  in Z. Ideas not done: rebuild as parametric SDF (smooth strands, fillets at the balls, ≥0.7 mm strands), island
  check per layer, coupon at 2–3 heights; a flat-printed panel with the textile tied into its window frame.

---

# Rig v4: state at end of 2026-11-01 (blue resin out until Tue 11-04)

## Where things stand
**Nothing is printing.** Blue resin is used up; more arrives Tuesday. Smoky resin is fine.

| Part | Resin | File | State |
|---|---|---|---|
| **Camera cartridges ×2** | blue, 2.3 s | `out/pi_box_v4_cart_tilt.goo` (design `pi_box_v4_cart_tilt`) | **Ready to upload** (user approved the model 11-01) |
| **Floor** | blue, 2.3 s | design `pi_box_v4_floor` | Not exported yet. Reprint as designed (button +X, nut clearance, no cradle) |
| **Core v4** | smoky, 2.8 s | `out/pi_box_v4_core_v4.goo` (214 MB, 3600 layers) | **Ready to upload** (exported 11-01, mirror fix proven) |
| Panels (front / back) | smoky | `pi_box_v4_front_panel`, `pi_box_v4_back_panel` | Not printed, not exported |

**Order the user set:** blue first (cartridges, then the floor if there's enough), then switch to smoky for the core.
The core *could* go first while blue is out, if the user wants the printer busy: ask.

## Camera cartridges: printed on an end face (decided 11-01)
- **Not face down.** The cradle (face-down supports) is a side quest (`CRADLE_SUPPORT_TESTS.md`). The cartridges are the
  rig's "eyes", so the R30 face must not be touched by supports.
- `designs/pi_box_v4_cart_tilt.py`, `STAND = 'picked'`:
  - Each cartridge stands on the face the user clicked in the **face picker** (`out/pick/pi_box_v4_cam_cartridge.json`).
    It's the large flat face with outward normal (-1, 0, 1)/√2 in the cartridge tree frame: 340 mm², with the R30 face
    tilted away from the plate.
  - **Print-only trim:** the two flange tips that stood 0.88 mm past that plane are cut off (1.9 mm³ total, ~0.5 mm
    slivers at the flange ends, on the pod's back edge). The cartridge model itself is unchanged.
  - **Elephant's foot:** 45° × 0.5 mm bevel round the plate face (`EF_CH`). Plate contact 347 mm² at layer 1.
  - **Supports, auto-generated:** 308 tapered columns to the plate + 20 standing on the part (tip 0.25 dug 0.25, shaft
    0.6, foot 1.2); 62 small spots left to bridge (red overlay in the viewer). Cached in `out/cart_tilt_cache.npz`
    (rebuilt when the cartridge, this file or the pick changes).
  - Includes the 11-01 cartridge changes: FFC connector relief (3.3 deep at the bottom end; 5.0 deep, u 9.4..20 at the
    top end).
  - **No fit strip** (user): print, check the inserts/lens bore, adjust for a reprint if needed. Lens cap vs elephant's
    foot was a face-down problem; on this orientation it should be gone.
- Viewer: `/pi_box_v4_cart_tilt`. Alternatives in the file: `STAND = 'end' / 'slope' / 'bottom'`.

## New tools (11-01)
- **Face picker:** `python face_pick.py <design>` then http://localhost:5555/pick/<design>. Click a face: it shows the
  area, and whether anything sticks out past the plane. "Use this face" saves `out/pick/<design>.json`.
- **Printer monitor:** `python printer_monitor.py` → http://localhost:5556 (read-only: progress, ETA, temps, snapshot).
- `PREVIEW_SCALE=6 python generate.py <design>` for fine features (slow, heavy in the browser).
- `sdf.Box` now accepts an array third coordinate (needed for parts sliced along another axis).

## When we're back
1. Load blue → upload + print `pi_box_v4_cart_tilt.goo` (ask before print) → check the inserts, bores, connector relief.
2. Floor: `python to_goo.py pi_box_v4_floor --gpu` → review → print (blue, if the resin lasts).
3. Smoky → upload `pi_box_v4_core_v4.goo` (~20 min) → print (~10.5 h). Check: the support trees in the cartridge slots
   snap out; the M2 pad inserts (3.2 teardrop pockets) fit; the power board (-X wall, shifted 22 toward the bank,
   straddling the web) clears the Pi with the real harness; the GbE adapter glues in clear of the front panel channel;
   the door slides (+0.15 on the thumb-screw side). Decisions in detail below.
4. Cradle side quest: the 10 mm flat / 40° square matrix (see `CRADLE_SUPPORT_TESTS.md`). Coupon #5's under-water cure
   result (does it keep the thin gyroid clear?) to be noted there.

---

# Rig v4: core reprint, handoff 2026-11-01 (day)

Power board: `power_board/docs/PI_HARNESS.md`. The 2026-10-31 handoff is kept below for history.

## Printing now
- **Blue plate `out/cradle_mirror_plate.goo`** (clear blue + Tenacious 75/25, 2.3 s, 233 layers), started ~13:00.
  `designs/cradle_mirror_plate.py` = the gyroid cradle coupon (unchanged) + a MIRROR TEST marker (raised F + arrow).
  **First print through the fixed to_goo.** Check: marker base-down, arrow pointing right -> a normal F (as the viewer
  seen from above). A backwards F = still mirrored: STOP before the core.
  Then judge the 32 cradle cells (snap-off, no pattern in the face, no sag): the winner sets the cradle defaults.

## The big finding: every .goo before 2026-11-01 printed MIRRORED
The printer reads (image column, row, layer) with the opposite handedness to the viewer. Design, viewer and overlays
were consistent; only the export mirrored. Bought parts (Pi, board, adapter) can't be mirrored, so core v3 put the power
board into the Pi's USB-C. **Fix: `to_goo.py` flips every layer image left-right (MIRROR_FIX).** Parts printed before
the fix are mirror images of their designs (cartridges: symmetric, fine).

## Decisions (user, 2026-11-01), commits 68be921 + b17f22e
- **Door A kept as printed** (its screw is physically at X +41.1): core lock boss moved to +X. `door_a.SCREW_XY` says so.
- **Floor REPRINTED as designed** (the printed one is mirrored). Power button moved to (33, 14) (+X bank side);
  NUT_SWEEP_R 11 clear round it (the M16 nut couldn't turn: collar ring at r 10.3, web rail ~10.7); no GbE cradle
  (adapter glued to the web), so the cradle-vs-Pi-boss clash from the photo is gone.
- **Power board**: original -X wall + orientation (soldered harness lengths), shifted 22 toward the bank (Yr -21..25,
  straddling the web) so the measured lidar plug clears the stick tube. Ledge/hook moved with it.
- **GbE adapter**: Yr -23.95..-0.25, GLUED to the web's Pi face (seat ledge kept); the +X web half moved 1.5 toward the
  bank (WEB_DY) so its front clears the front panel's L channel. Removing it = take the Pi out to unplug (cord maxed).
- **Web = horizontal strips** (`sketches/web_strips_sketch.py` + the user's red boxes): strip A d 48..72 both halves
  (open above, the flange stiffens); -X solid d 158..180 under the board; +X solid d 48..180 (backs the Pi's +X upper
  mounting boss) with one wire hole X 31..45.5, d 76..98 (from 31: the boss + gusset reach X 29.7). Every strip's
  print-underside is a semicircular arch (strip A is ~13 at the crown, 25 at the ends).
- **M12 slack loops dropped** (loop it outside the case if needed). Camera ribbon routes ignored (200 mm, fixed distances).
- **Measured** (back of PCB to plug top, inserted; +2 for wires in the model): lidar 19.5, GPIO 10, button 11.5. Pi USB-C
  90° adapter: 11.5 past the board edge, 25 long, exits UP toward the lidar; its cable plug assumed 22 [MEASURE].
  Pi GPIO harness: female ends 17 off the PCB back (+3 wires).
- **Door**: +0.15 per face on the thumb-screw side (`DOOR_EXTRA`). Back panel stops short of door A's webs (d >= 175.3).
- **Supports**: snap-off trees under both cartridge-slot floors (d 66.9, 166 mm2 bridges that sagged 1-2 mm on v3):
  3 per slot, foot on the slot edge d 17.6, 4 point tips. `TREE` in the core.
- **Cartridge pad M2 inserts** (2.8 OD calipered): pockets 3.2 + comp, teardrop (horizontal in print), screw holes too.
  Core v3's pockets were too small on both sides; no drill access, so v3 stays as is.
- **Cartridge**: the FFC connector is on the board FRONT. Relief 3.3 deep at the bottom end; top end (connector-up)
  5.0 deep, u 9.4..20. Lens cap vs the elephant's foot: no bore change; file it if the cradle reprint still has it.
- Inserts: Pi M2.5 tight, OK; floor M3 slightly loose (CA); power board M2.5 (same pocket as the Pi), M2.5x6 screws.

## Tools added
- `designs/pi_box_assembly_check.py [core]`: every printed part + every overlay voxelised (0.5 mm), all pairs, exit 1 on
  any unexpected overlap. ~10 min. GROUPS / EXPECTED_PAIRS decide what's allowed. Known gaps (TODO): the support trees
  sit in the cartridge slots and core-vs-camera counts as expected, so a real core-vs-cartridge clash would hide; it
  only finds overlaps, not "no room left" (e.g. the missing shroud space): add a min-clearance report.
- `pi_box_check.py`: floor now sampled over its full height (it missed the cradle walls).
- Viewer `/pi_box_v4_assembly`: core + every printed part as overlays. Legend GROUPS (Pi, Power board, Ethernet, Power
  button, Power bank, Lidar + mount, Cameras, Printed parts): click a group to hide it, ▸ to expand.
- Lesson: two flagged clashes were "fixed" by notching the front panel without calling it a workaround. Flag such fixes.

## Next
1. Mirror check on the blue plate (above). Then the cradle verdict -> cartridges reprinted face down on a cradle.
2. **Core v4**: `out/pi_box_v4_core_v4.goo` (from `pi_box_v4_core_flip`, 2.8 s smoky mix). Review the .goo, upload
   (~20 min), print (~10.5 h) only after the F checks out.
3. Floor reprint (blue, 2.3 s). Panels (not printed yet).
4. Open: the board's plug wires vs the Pi GPIO harness are approximate; strip A crown 13 mm (raise A's start to ~37 if
   wanted). Previews take ~100 s now (the trees).

## Gotchas (11-01)
- `sdcp.py print` right after an upload returned Ack 1 again: retry once.
- The cartridge tree's third coordinate is h = FL_D[1] - d, not d: probing it with d gives wrong answers.
- Regenerate both viewers after a core change: `pi_box_v4_core_flip` and `pi_box_v4_assembly`.

---

# (history) Rig v4: printing + power board bring-up, handoff 2026-10-31 (night) → resume 2026-11-01

Design state before this session: `C:\Users\Alan\TetherWork\rig-v3-orch\handoff\` (HANDOFF_*.md, V4_STATE.md).
Power board: `power_board/docs/PI_HARNESS.md` (pin map, colours, Pi config, test results).

## Printing now
- **Core v3** `out/pi_box_v4_core_v3.goo` (3600 layers, 2.8 s, bottom 20 s), started ~20:20, printer ETA 10 h 38 min →
  done ~07:00. Leaving it on the plate a few hours in the UV-blocking tent is fine. Flush the small holes when washing
  (M2 inserts, M4 driver bores, ribbon slots).

## Done today
- **Blue (clear blue + Tenacious 75/25, 2.3 s), all printed:** blue plate (2 cartridges face down, door A standing on
  its grip end with branching trees in the port window, door fit strip) and the floor. Fit strip: door slides
  perfectly; M3x8 into the insert NOT tested yet. Cartridges: slight elephant's foot on the curved face (cosmetic).
  Hand-fixes at work on the printed cartridges: lens bore (Ø15 cap didn't pass the holder zone) - fixed in the model.
- **Dark resin chosen:** smoky black Fast + clear Tenacious 75/25 (wires visible, prototype look). Exposure test
  `exposure_test_smoke` (3.5 s × gray 100..50 %): winner 80 % = **2.8 s**; pins held to 70 %, 50 % failed. Set in
  `pi_box_v4_core.py`, `_front_panel.py`, `_back_panel.py`. Renders: core + panels 'smoke' (lava black = `resin_black`).
- **Model fixes (all committed, Physarum main):** measured lens holder (13 sq x 3.5, Ø3.8 tabs 21.7 o/o along the 25
  side), lens 19 with Ø15 cap, bezel Ø17.5 x 1.8, uni GbE adapter 15.4 x 23.7 x 53, insert slots out to t 17.5, round
  lens bore from the board face, M4 driver bores down to d 26 (cartridge pad frame blocked the front screws), v3
  latch/hinge wall pads removed, board ledge starts at Yr -24 (its front end floated), floor gyroid panes < 150 mm²
  solid. SGH logo dropped (sticker later). Gyroid front panel option dropped (flat).
- **Power board A, bring-up passed** (details in PI_HARNESS.md): Pi boots off the board, INA226 0x40, 4/4 clean soft
  shutdowns, 5 min stress (4.91 V at the Pi, 0x0, 60 °C), lidar eFuse no-load 15.14 V at J3 (left +, right GND).
  Config fix: `gpio_pull=off` for SW_SENSE (the Pi 5 pull-down held it low). Pi: `lidar@lidarpi.local` (key auth;
  this PC's key added), Debian 13, i2c-tools + stress-ng installed, Wi-Fi back on the rig hotspot. Field rule: wait
  ~15 s after latching before unlatching (earlier = hard cut).

## Next
1. **Core** off the plate, wash, cure. Assembly: front M4s first through the driver bores (cartridges out). Cartridge
   boards in with M2x5 now; cartridges to core with **M2x8 socket caps (ordered, arrive Mon 11-02)**.
2. **Panels**: export `pi_box_v4_front_panel` + `pi_box_v4_back_panel` (2.8 s) onto one plate - start the upload
   EARLY (printer link ~0.19 MB/s regardless of wiring: ~20 min per 200 MB). Same smoky mix.
3. **Gyroid cradle calibration** (clear blue mix, 2.3 s, after the dark prints): `cradle_coupon` = 4 strips (period
   5/6/8/10, ID = 1..4 notches) x 4 gaps (0.05..0.20, from the notched end) x 2 sheets (0.3 notch side / 0.4), cell
   dots 1..8. ~55 ml + vat minimum: check clear blue stock (top up 75/25). Judge release, pattern print-through, sag.
   Winner → cradle defaults → reprint the cartridges face down on a cradle (~13 ml each + ~2 ml cradle).
4. **Lidar on J3** (office): `python3 ~/lidar_on.py 20` on lidarpi (start peak, min bus, FLT).
5. **Harness v2** (PEBA 2030M kit + JST-PH assortment ordered): Pi end = Dupont 1x6 + 1x2 + 1x4 glued (odd row pins
   1-15, even row 16-22), colours as locked; later a small Pi adapter board with a JST-PH latch (board v2).
6. Open: the "cut through the side of the inner core near the bottom" (user, 10-31) was never located - check it on
   the printed core. Slot-floor bridges (31 x 18, d 66) left as is: file if they sag. Exposure 2.8 s still to be
   confirmed on the real core.

## Gotchas learned today
- An interrupted foreground tool call does NOT stop `sdcp.py upload`: kill the python process. An unfinished upload
  never appears on the printer. Use a new file name per revision (core_v2 was abandoned → core_v3).
- `sdcp.py print` right after an upload can return Ack 1; retry once a few seconds later.
- The user reviews the viewer and the .goo before any upload/print; ask before starting a print.
- `to_goo --gpu` doesn't work on numpy-only pieces (blue plate, door trees, face-down cartridges): CPU, ~8 min.
- Blender: not on PATH: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"`.
