# Rig v4: printing + power board bring-up, handoff 2026-10-31 (night) → resume 2026-11-01

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
