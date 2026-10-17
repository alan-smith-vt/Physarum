"""Checks for stick_collar: per-layer area of each half (print rules: < ~8,000 mm2, no sharp jumps), minimum resin
walls around the screw/insert holes, and how the cone seats in the core's well tube + floor hole.

    python -m designs.stick_collar_check          (from slicer_preview/)
"""
import numpy as np
from sdf import solid
import designs.stick_collar as C
from designs.pi_box_bat_rig import WELL_R, DEPTH, RESIN_HOLE_COMP
from designs.pi_box_bat_lids import FLOOR_T

step = 0.1
xs = np.arange(-C.BAND_R - 1, C.BAND_R + 1, step)
As = np.arange(-1, C.A_END + 1, step)
X, A = xs[:, None], As[None, :]
print('layer area (mm2) by height above the plate (split face):')
for name, root in (('head half', C.HEAD_HALF), ('insert half', C.INSERT_HALF)):
    areas = []
    for d in np.arange(0.025, C.H_TOP, 0.25):
        m = np.asarray(solid(root, np.broadcast_to(X, (len(xs), len(As))), np.broadcast_to(A, (len(xs), len(As))), float(d))) <= 0
        areas.append((d, m.sum() * step * step))
    a = np.array(areas)
    jumps = a[1:, 1] / np.maximum(a[:-1, 1], 1)
    k = np.argmin(jumps)
    print(f'  {name}: max {a[:, 1].max():.0f} mm2 at {a[np.argmax(a[:, 1]), 0]:.2f} mm, first layer {a[0, 1]:.0f}, '
          f'worst drop x{jumps[k]:.2f} at {a[k + 1, 0]:.2f} mm')

# walls: min resin around each hole, by sampling the hole edge rings against the body (bore + outside)
print('minimum walls (mm):')
for a_row in C.ROWS:
    x = C.SCREW_X
    r_ins = (C.INS_D + RESIN_HOLE_COMP) / 2
    r_cb = (C.HEAD_D + RESIN_HOLE_COMP) / 2
    for label, r, d_lo in (('insert pocket', r_ins, C.SHOULDER), ('head counterbore floor', r_cb, C.HEAD_FLOOR)):
        th = np.linspace(0, 2 * np.pi, 360)
        px, yc = x + r * np.cos(th), d_lo + C.SPLIT / 2
        rr = np.hypot(px, yc)
        print(f'  row a={a_row:.1f} {label}: to bore {np.min(rr) - C.BORE_R:.2f}, to outside {C.BAND_R - np.max(rr):.2f}'
              f'{"  (opens to the outside above: the head entry)" if label.startswith("head") else ""}')
    print(f'  row a={a_row:.1f} shoulder (clearance Ø{C.M4_CLEAR} under the Ø{C.INS_D} insert): '
          f'{(C.INS_D - C.M4_CLEAR) / 2:.2f} mm ledge, {C.SHOULDER:.1f} thick')

# seating: cone radius vs the tube bore (d < DEPTH) and the floor hole (DEPTH..DEPTH+FLOOR_T), cone seated at the
# floor underside (a = 13 at d = DEPTH + FLOOR_T)
a_seat = (2 * WELL_R / 2 - C.CONE_R0) / (C.CONE_R1 - C.CONE_R0) * C.CONE_LEN
top_d = DEPTH + FLOOR_T - a_seat
print(f'seating: cone reaches Ø{2 * WELL_R:.0f} at a={a_seat:.1f} mm -> cone top at rig d {top_d:.1f}, '
      f'{DEPTH - top_d:.1f} mm inside the tube')
for d in (top_d, DEPTH - 0.01, DEPTH + FLOOR_T / 2, DEPTH + FLOOR_T):
    a = d - top_d
    r = C.CONE_R0 + (C.CONE_R1 - C.CONE_R0) * a / C.CONE_LEN
    print(f'  rig d {d:6.1f}: cone r {r:.2f} vs hole r {WELL_R:.2f} -> {"clear" if r < WELL_R - 1e-6 else "contact"} '
          f'({WELL_R - r:+.2f})')
