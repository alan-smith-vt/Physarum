# Pi box (lidar rig): handoff, 2026-10-13

Design notes / decisions history live in the Notes vault:
`D:\Notes\Reference\lidar-photo-rig-cage-handoff.md` (Pi box section, renders in
`Reference/assets/pi-box-renders/`). Read that first.

## State
- Chosen design: **battery-in-box** (`pi_box_bat_*`). Pole version archived in `designs/archive/pole/`.
- Weight reduction done: 4 mm flange plate, Ø20x10 bore bosses, 10 mm panel-stop rims,
  TPMS camera/switch walls with solid zones round cameras, switch, web and wall top/bottom.
- Ribs (`WALL_RIBS` in `pi_box_bat_rig.py`): one touching the camera pad edge (clipped beside the
  far camera board), one under the power switch, one at Yr 50.
- **Wall fill chosen: gyroid 12 mm, half-thickness 0.35** (now the default). Core ~268 cm3 (~308 g)
  vs solid 359 cm3. Runners-up were gyroid 7 (282) and diamond 9 (272).
  `PI_BOX_FILL="kind:a:b"` still overrides for experiments.
- Clash check clean with gyroid 12 (`python -m designs.pi_box_check`).

## Next steps
1. Full Blender render set with gyroid 12 into the vault (`Reference/assets/pi-box-renders/`):
   box, assembled, assembled_gopro, exploded, internals, pod, pod_no_battery, camera_near,
   camera_far (+ wall_out/wall_in if useful). Then update the vault handoff (fill choice, ribs,
   volumes) and commit/push the vault.
2. Overhang report (`python -m designs.pi_box_overhangs`); ignore round-hole arch flags (user:
   resin bridges circular arches; no teardrops).
3. Open questions offered to the user: drop the Yr 50 rib (duplicates the battery stop rib)?
   Lighter wall settings? Carry-modes write-up for recording notes.
4. Regenerate viewer previews (`generate.py`) so /battery routes show the gyroid.

## Commands (run from `slicer_preview/`)
- Export meshes: `python -m designs.pi_box_export > log 2>&1` then check the log for `Traceback`
  (never pipe it into grep; a failed export leaves stale meshes that render silently).
  Core only: `PI_BOX_EXPORT_ONLY=core` (also `core_cut_far`, `core_cut_near`).
- Render: `"C:/Program Files/Blender Foundation/Blender 4.3/blender.exe" -b -t 4 -P designs/pi_box_render.py -- <outdir> <views...> --samples 96 --res 1200 900`
- Panels DXF: `python -m designs.pi_box_panel_dxf` -> `out/pi_box_panel.dxf` (1/8in PC, qty 2).

## Gotchas
- Never render the web viewer headless (CPU fallback maxes the machine).
- Constants in `pi_box_bat_rig.py` are order-dependent (WALL_RIBS uses literal -15 = CAM_YR because
  CAM_YR is defined later).
- Print frame: print x=Yr, z=-X, y=d (`to_print` / `from_print`).
