# Pi box (lidar rig): handoff, 2026-10-13 (evening)

Design notes / decisions history live in the Notes vault:
`D:\Notes\Reference\lidar-photo-rig-cage-handoff.md` (Pi box section, renders in
`Reference/assets/pi-box-renders/`). The vault is NOT updated with this session's changes yet
(it was on the laptop); renders for it are in `slicer_preview/out/renders_2026-10-13/` (gitignored).

## Print in progress (started 2026-10-13 ~22:50)
- `pi_box_bat_core.goo` (3600 layers, ~11-12 h, 2.6 s / 20 s bottom, 75/25 Fast gray / Tenacious) is running
  on the Saturn 4 Ultra 16K from the printer's own storage (`/local/pi_box_bat_core.goo`); the desktop
  can be off. Check it: `python sdcp.py status` (PrintInfo.Status: 13/3/4 printing, 9 complete,
  6 paused, 8 stopped; ErrorNumber 0 = OK) or `python sdcp.py watch`; `history` lists finished jobs.
  Printer 192.168.0.4, found by discovery. A start right after an upload can return `Ack 1`: retry.
- After it: wash + post-cure fully before gluing inserts (liquid CA on the insert's knurl, waxed screw in it).

## State
- Chosen design: **battery-in-box** (`pi_box_bat_*`), now the **compact symmetric layout**:
  `Y_MAX = -Y_MIN` (design rule: the box is symmetric about the aluminium plate, ±62).
  Core 281 cm3 (~324 g), 124 × 105 × 180 in print.
- Wall fill: gyroid 12 mm / 0.35; sheets thicken over 6 mm into the bottom band (`FILL_RAMP`), no flat start.
- Bank (Yr 30.8..55.8) right past the tube. Cradle: per wall one **stepped flap** (stop rib + front rail
  merged) and a second rail near +Y; 45° shoulders over the bank's top corners; floor traps it below.
  Rails/flap run solid through the gyroid as ribs. They avoid the M4 bores' screw path (Yr 34.7..45.3).
- Under the Pi: buck + 2 WAGOs against the -Y panel, **power switch on the -X wall** (Yr -45, 155 deep).
- Bank built-in cord (~15 cm) -> USB-C female coupler in the +X bay -> PD trigger's own long cord ->
  battery-side web clip -> low web slot -> switch -> WAGO.
- Lidar cable: M12 plug sticks out +Yr; cable loops 180° outside the +Y end, lies up into a **U-notch in
  the +Y rim** (flange plate stays solid), panel closes it; over the bank, slack coil in the -X bay,
  stop-rib clip, -X low web slot (17 × 22, RJ45 + ferrite pass), Pi-side web clip, up into the jack.
  Rims at both ends are 14 deep (`RIM_D`) so panels stay identical (165.8 tall; DXF regenerated).
- UE300 Ethernet dongle flush on the +X web (Pi side) on the floor, capped by a seat ledge on the web.
  The +X Pi-side dowel/foot was removed for it (3 dowels now).
- Cameras (OV9281, 160°, lens 18 from board face) 8 mm deeper on solid blocks grown from the flange;
  lens ~4 proud, 2 mm guard lip; both connectors face down, ribbons drop out of block pockets.
- Push-in C clips (opening 0.85 Ø): 3 per camera-pad rib, lidar clips on the -X stop rib and -X web,
  big clip on the +X stop rib, two small on the +X web's battery face.

## Fasteners / fits
- Plate: M4×18 truss + 6 mm Al spacer (stepped bores, unchanged).
- Floor: M3×6 button heads into glued M3 inserts (Ø4 × 3, in hand).
- Pi: M2.5×6 socket caps into glued M2.5 inserts (kit). Cameras: M2×5 socket caps into glued M2 inserts.
- Glue inserts with **superglue, not threadlocker** (anaerobic Loctite cures poorly on resin and can craze it).
- Horizontal holes (Pi, cameras, switch) print +0.35 wide, +0.55 tall (`HORIZ_HOLE_EXTRA` 0.2):
  between the fit strip's too-tight round and too-loose ellipse. First core is the fit test.

## Amazon cart (2026-10-13, not yet ordered), for review
In the cart:
- USB-C female-female coupler, 2-pack (bank's built-in cord -> PD trigger)
- M2×5 socket cap ×50 (cameras), M2.5×6 socket cap ×50 (Pi), M3×6 button head ×50 (floor)
- FFVRVSS 300-pc brass insert kit M2..M5 (heat-set kit; used here GLUED, resin can't be heat-set)
- Gebildet 16 mm latching push button, 12-24 V / 5 A, LED, 2-pack (power switch)

Gaps found against these notes (check what's already on hand):
- **Pi 5 camera cables ×2: 22-pin (Pi 5) to 15-pin (OV9281), ~150 mm.** OV9281 boards ship with
  15-15 cables, which don't fit the Pi 5. Not in the cart.
- **Superglue (gel CA)** for the inserts, if none on hand.
- PD trigger (corded male USB-C, 15 V), 15->5 V USB-C buck, 2× WAGO 221-413: from the earlier rig
  (cage handoff power chain); confirm they're on hand.
- Right-angle USB-C lead, buck -> Pi (the route assumes a right-angle plug at the Pi).
- TP-Link UE300 (USB 3 gigabit) for the office link: confirm owned.
- Already owned per the cage handoff: M4×18 truss + 6 mm Al spacers, 3×6 dowels, M3×10 lidar screws.
- Optional: thin foam for the bank / dongle (+1 mm margins), heat-shrink for the switch leads,
  1/8" polycarbonate for the final panels (DXF `out/pi_box_panel.dxf`, 95.2 × 165.8, qty 2).

## [MEASURE] before trusting the fit
Bank height (157 + 1 margin), UE300 height (71, +1 margin), lidar cable Ø (7: notch, clips),
lens holder (Ø19 bore passes round or ≤ ~13 square), M2/M2.5 insert OD (3.2 / 3.5 placeholders),
camera-board keep-out round the holes (Ø6 bearing), buck/WAGO/switch body sizes.

## Next steps
1. Print the core (`out/pi_box_bat_core.goo`), fit-test inserts, horizontal holes, clips, cradle.
2. Take the measurements above; adjust; print floor + panels (blanks or cut PC from the DXF).
3. Copy `out/renders_2026-10-13/` into the vault and update the vault handoff.

## Modelling: SDF core (2026-10-13)
- `pi_box_bat_core.py` is a signed-distance tree (`slicer_preview/sdf.py`: Box / Cyl / Half / Fn, union /
  inter / sub, smooth_union / smooth_sub for fillets, offset, shell). Every node has a bounding box;
  slices are evaluated in tiles (128 px CPU, 2048 px GPU) on only the nodes that reach each tile.
- Verified against the previous mask core: identical at preview res (141 layers) and full res
  (10 layers; 2 tie pixels on a vent edge). Full-res slice 0.2-0.3 s/core CPU (was ~2.5), ~35 ms GPU;
  `to_goo --gpu` 2.1 min (was 3.6). Check 4 s (was ~1 min), overhangs 20 s, mesh export 10 s.
- Anti-aliasing: `PI_BOX_AA=1` writes grayscale edges (~2 % of lit pixels). OFF by default: gray
  pixels cure less than white, so edges/fits shift. Calibrate first (exposure_test with gray edges)
  before using it on fit-critical parts.
- Floor / panels are still mask functions (`implicit_piece`); port them the same way if needed.

## Commands (run from `slicer_preview/`)
- Export meshes: `python -m designs.pi_box_export > log 2>&1`, check the log for `Traceback`
  (a failed export leaves stale meshes that render silently). Core only: `PI_BOX_EXPORT_ONLY=core`.
- Checks: `python -m designs.pi_box_check` (overlay clashes), `python -m designs.pi_box_overhangs`.
- Render: `"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -t 4 -P designs/pi_box_render.py -- <outdir> [views] --samples 96 --res 1600 1200`
  (default = full set incl. side_px/mx/py/my and pi_bay).
- Viewer: `python generate.py pi_box_bat_panel pi_box_bat_lids pi_box_bat_core`, `python server.py` -> /battery.
- Panels DXF: `python -m designs.pi_box_panel_dxf`. Print file: `python to_goo.py pi_box_bat_core`.
- Reference models: `models/README.md` (Pi 5 + Mid-360S STEP -> STL via `models/step2stl.py`).

## Gotchas
- Never render the web viewer headless (CPU fallback maxes the machine).
- Don't edit `pi_box_bat_rig.py` with PowerShell Get/Set-Content: it re-encodes the UTF-8 (Ø × °).
- Print frame: print x=Yr, y=d, z=-X (`to_print` / `from_print`). Viewer "up" = rig depth (floor end).
