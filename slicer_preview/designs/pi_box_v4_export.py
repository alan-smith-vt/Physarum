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


def floor_parts():
    """home/v4-floor's fixed floor + door A, and the panels: v4 wide front + v4 back (top tongues)."""
    h = VOX
    out = {}
    F = __import__('designs.pi_box_v4_floor', fromlist=['x'])
    D = __import__('designs.pi_box_v4_door_a', fromlist=['x'])
    ft = F._g['FLOOR_T']
    out['floor'] = E._march(F.floor_fixed, int(F._LF / h), int(F._FH / h), int(2 * F._XO / h), h,
                            lambda p: np.stack([p[:, 2] - F._XO, p[:, 0] + R.Y_MIN, p[:, 1] - (R.DEPTH + ft)], 1))
    out['door'] = E._march(D.door, int(2 * D.PLATE_X / h), int((D.GRIP['d1'] - D.D_TOP) / h), int((R.Y_MAX - D.Y_LO) / h), h,
                           lambda p: np.stack([p[:, 0] - D.PLATE_X, p[:, 2] + D.Y_LO, -(p[:, 1] + D.D_TOP)], 1))
    PV, B = C.PV, __import__('designs.pi_box_v4_back_panel', fromlist=['x'])
    d0 = C.FRONT_PANEL_D0                                    # wide front panel (rig solid): x -> X, y -> d, z -> Yr
    out['panel_front'] = E._march(lambda x, y, z: C.front_panel_rig(x - C.X_OUT, z + R.Y_MIN, y + d0),
                                  int(2 * C.X_OUT / h) + 1, int((R.DEPTH - d0) / h), int((C.RC[1] + 12.0 - R.Y_MIN) / h), h,
                                  lambda p: np.stack([p[:, 0] - C.X_OUT, p[:, 2] + R.Y_MIN, -(p[:, 1] + d0)], 1))
    hw, face, th, bd0 = PV.HALF_W, R.Y_MAX - R.PANEL_LIP - PV.CL, R.PANEL_T + PV.HOOK, B.D_TIP   # v4 back panel + tongue
    out['panel_back'] = E._march(lambda x, y, z: B.back_mask(x - hw, y, z + bd0), int(2 * hw / h), int(th / h) + 1,
                                 int((PV.D1 - bd0) / h), h,
                                 lambda p: np.stack([p[:, 0] - hw, face - p[:, 1], -(p[:, 2] + bd0)], 1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    alu, z_top = E.aluminium()
    printed = {'core': core(), 'cart_far': cartridge(1), 'cart_near': cartridge(-1), **floor_parts()}
    for name, (v, f) in {**printed, **alu}.items():
        E.write_ply(OUT / f'{name}.ply', v, f)
        parts[name] = len(f)
        print(f'{name:14s} {len(f):9,d} tris', flush=True)
    B = __import__('designs.pi_box_v4_back_panel', fromlist=['x'])    # MagSafe steel ring insert in the back panel's
    a = np.linspace(0, 2 * np.pi, 97)                                  # outer-face recess (black in the renders)
    cx, cd = B.RING_C
    ri, ro, y1 = B.B3.RING_ID / 2, B.B3.RING_OD / 2, B.B3._FACE
    y0 = y1 - B.B3.RING_DEPTH
    tris = []
    for i in range(96):
        c0, s0, c1, s1 = np.cos(a[i]), np.sin(a[i]), np.cos(a[i + 1]), np.sin(a[i + 1])
        for r in (ri, ro):
            p = [(cx + r * c0, y0, -(cd + r * s0)), (cx + r * c1, y0, -(cd + r * s1)),
                 (cx + r * c1, y1, -(cd + r * s1)), (cx + r * c0, y1, -(cd + r * s0))]
            tris += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
        for y in (y0, y1):
            p = [(cx + ri * c0, y, -(cd + ri * s0)), (cx + ro * c0, y, -(cd + ro * s0)),
                 (cx + ro * c1, y, -(cd + ro * s1)), (cx + ri * c1, y, -(cd + ri * s1))]
            tris += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    E.write_stl(OUT / 'magsafe_ring.stl', np.array(tris))
    parts['magsafe_ring'] = len(tris)
    D = __import__('designs.pi_box_v4_door_a', fromlist=['x'])        # door lock: M3 x 8 socket cap under the door
    sx, sy = getattr(D, 'SCREW_XY', (-41.1, 44.0))
    z0 = -(D.PLATE_D[1])
    E.write_stl(OUT / 'door_screw.stl', R._cyl((sx, sy, 0.0), 'z', 2.75, z0 - 3.0, z0, 24))
    parts['door_screw'] = 1
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
