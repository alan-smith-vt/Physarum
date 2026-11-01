# Rig v4: core reprint, handoff 2026-11-01

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
