# Reference models (overlays only, not printed)

STLs here are gitignored (`*.stl`); re-create them with `step2stl.py`, which needs
OpenCascade (`pip install cadquery-ocp`, in a throwaway venv is fine):

    python step2stl.py <in.step> <out.stl> [deflection_mm]

| File | Source | License |
|---|---|---|
| `rpi5.stl` | Raspberry Pi 5 STEP, https://datasheets.raspberrypi.com/rpi5/RaspberryPi5-step.zip (`rpi-5b_no_graphics.step`) | MIT, `rpi5_LICENSE.txt` |
| `mid360s.stl` | Livox Mid-360S STEP, https://www.livoxtech.com/mid-360s/downloads (`mid-360S.stp`) | Livox, not redistributed |

Model frames as used by `slicer_preview/designs/pi_box_rig.py`:
- Pi 5: board x 0..85 (SD/PCIe end at 0, USB/Ethernet past 85), y 0..56 (micro-HDMI edge at 0), z up (component side). Camera connectors at x ≈ 49 and 57, y ≈ 2..16; microSD on the underside at x 0..8.
- Mid-360S: Y up (base at Y = -25.92), connector toward +X.
