"""Export the v4 PROPOSAL rig (pi_box_v4_core + pi_box_v4_cam_cartridge) as meshes in the rig frame for Blender.

    python -m designs.pi_box_v4_export            (from slicer_preview/)  -> out/render_v4/*.ply|stl + parts.json
Then:  blender -b -P designs/pi_box_v4_render.py -- <abs out dir> [views]

Same frame and helpers as designs/pi_box_export.py (X across, Y forward = -Y, Z up, origin under the base plate).
Printed: the core and both cartridges (and home/v4-floor's floor, if designs/pi_box_v4_floor.py exposes one).
Bought parts: the v4 core's viewer overlays (Pi lowered to d 55, power board, cartridge boards and lenses, ...).
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_slicer import load_stl                                  # noqa: E402
from sdf import solid                                            # noqa: E402
from designs import pi_box_export as E                           # noqa: E402
from designs import pi_box_bat_rig as R                          # noqa: E402
from designs import pi_box_v4_core as C                          # noqa: E402
from designs import pi_box_v4_cam_cartridge as K                 # noqa: E402

OUT = R.ROOT / 'slicer_preview' / 'out' / ('render_v4flip' if C.FLIP else 'render_v4')
VOX = float(os.environ.get('PI_BOX_VOX', '0.35'))
SKIP = ('base_plate', 'v4_cartridges', 'm12_trunk', 'power_board_parts')


def core():
    h = VOX
    return E._march(C.core, int(C.SIZE_X / h), int(C.DEPTH / h), int(C.SIZE_Z / h), h,
                    lambda p: R.from_print(np.stack([p[:, 0] + C.X0, p[:, 1], p[:, 2] + C.Z0], 1)))


def cartridge(s):
    """Tree frame (n, t, h); local x -> n, y -> h, z -> t."""
    h = VOX
    fn = lambda x, y, z: solid(K.ROOT, x + K.N0, z + K.T0, y)

    def to_rig(p):
        land = np.stack([p[:, 0] + K.N0, p[:, 2] + K.T0, C.FL_D[1] - p[:, 1]], 1)
        r = K.land_to_rig(land, s)
        return np.stack([r[:, 0], r[:, 1], -r[:, 2]], 1)
    return E._march(fn, int((K.N1 - K.N0) / h), int(K.H_TOP / h) + 1, int((K.T1 - K.T0) / h), h, to_rig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    alu, z_top = E.aluminium()
    printed = {'core': core(), 'cart_far': cartridge(1), 'cart_near': cartridge(-1)}
    for name, (v, f) in {**printed, **alu}.items():
        E.write_ply(OUT / f'{name}.ply', v, f)
        parts[name] = len(f)
        print(f'{name:14s} {len(f):9,d} tris', flush=True)
    for name, tris in E.gopro(z_top).items():
        E.write_stl(OUT / f'{name}.stl', tris)
        parts[name] = len(tris)
    for ov in C.OVERLAYS:                                        # back from the print frame
        name = Path(ov['file']).stem
        if name in SKIP:
            continue
        t = load_stl(R.ROOT / ov['file']).astype(np.float64)
        rig = R.from_print(t.reshape(-1, 3)).reshape(-1, 3, 3)
        E.write_stl(OUT / f'{name}.stl', rig)
        parts[name] = len(t)
        print(f'{name:14s} {len(t):9,d} tris (overlay)', flush=True)
    (OUT / 'parts.json').write_text(json.dumps(parts, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
