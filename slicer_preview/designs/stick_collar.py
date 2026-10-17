"""Split cone collar for the monopod stick (2026-10-16, concept approved from sketches/stick_collar_v2_sketch.png).

Clamped on the bare Ø29.3 stick (92 mm circumference) below the core and slid up until its cone seats in the
Ø43 floor hole / well-tube mouth. With the stick's Ø42 ring snug at the top of the tube, the stick braces against
the core at two points ~150 mm apart, so side loads go into the core instead of bending the 1/4-20 screw.

  - Cone Ø41 -> Ø45 over 27 mm (Ø43 at 13 mm: the floor underside), then a Ø56 clamp band below the floor.
  - Two halves split on one plane, 1 mm gap closed by 2 rows x 2 M4 x 18 truss-head screws (rows in the band only,
    so the cone stays smooth). Head half: Ø10.2 counterbores to 8.5 mm from the split face, Ø4.5 clearance.
    Insert half: Ø4.5 clearance for a 3 mm shoulder at the split face, then an insert pocket from the outside:
    the M4 insert is pressed in from outside against the shoulder, so the screw pulls it into the resin and the glue
    only locates it. An M4 x 18 tip then sits at the insert's far end (full 6 mm of thread).
  - Bore Ø29.0 (0.3 squeeze on the stick). Each half prints flat on its split face: screw holes vertical.
Frame: collar (x across, y normal to the split plane, a along the stick from the cone top). Each half's tree is
(X', Y', d') = (x, a, |y| - SPLIT/2): d' = height above the plate = distance from its split face.
"""
import numpy as np
from sdf import Box, Cyl, Fn, union, sub, sdf_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP, HEAD_D
from designs import stick_collar_hardware as hw

SUPPORTS = False
EXPOSURE_S = 2.3                       # ELEGOO ABS-like clear blue + Tenacious clear 75/25
BOTTOM_EXPOSURE_S = 20.0

STICK_D = 92.0 / np.pi                 # 29.28
BORE_R = (STICK_D - 0.3) / 2           # clamp squeeze 0.3 on the diameter
CONE_R0, CONE_R1, CONE_LEN = 20.5, 22.5, 27.0   # Ø41 -> Ø45; Ø43 at 13 mm
BAND_R, BAND_LEN = 28.0, 24.0          # Ø56 clamp band
A_END = CONE_LEN + BAND_LEN
SPLIT = 1.0                            # gap between the halves before clamping
SCREW_X = 20.5                         # screw axes at x = ±20.5 (normal to the split plane)
ROWS = (CONE_LEN + 6.0, CONE_LEN + 17.5)   # a of the two screw rows (Ø10.2 counterbores need ~11 mm between)
M4_CLEAR = 4.5
INS_D, INS_L = 5.5, 6.0                # M4 insert (calipers 2026-10-16): Ø5.5 over the ribs, Ø5.0 at their base, 6 long.
#                                        Pocket cut at 5.5 + comp (prints ~5.5): slides in against the shoulder, CA locates.
SHOULDER = 3.0                         # insert half: clearance from the split face before the insert pocket
SCREW_L = 18.0
HEAD_FLOOR = SCREW_L - (SPLIT / 2 + SHOULDER + INS_L) - SPLIT / 2   # 8.0 above the head half's face (y = 8.5):
#                                       tip at y = 8.5 - 18 = -9.5, the insert's far end (0.5 gap + 3 shoulder + 6)
H_TOP = BAND_R - SPLIT / 2             # half height on the plate
PRINTED_NUTS = True                    # 2026-10-17: brass inserts at the office -> hex pocket for printed nuts
#                                        (stick_collar_hardware), printed screws + nuts on the same plate


def _outer(X, Y, d, xp):
    """Revolved outer profile: cone then band, as a distance bound (slope 4.2 deg, ~exact)."""
    yc = d + SPLIT / 2
    r = xp.sqrt(X * X + yc * yc)
    a = Y
    R = xp.where(a < CONE_LEN, CONE_R0 + (CONE_R1 - CONE_R0) * a / CONE_LEN, BAND_R)
    along = xp.maximum(-a, a - A_END)
    return xp.maximum(r - R, along)


def _bore(X, Y, d, xp):
    yc = d + SPLIT / 2
    return BORE_R - xp.sqrt(X * X + yc * yc)


body = Fn(lambda X, Y, d, xp: xp.maximum(_outer(X, Y, d, xp), _bore(X, Y, d, xp)),
          (-BAND_R, BAND_R, 0.0, A_END, 0.0, H_TOP))
body = sub(body, Box(-BAND_R - 1, BAND_R + 1, -1, A_END + 1, -5, 0.0))      # nothing below the split face


def vhole(x, a, dia, d0, d1):          # vertical (print-up = normal to the split) hole
    return Cyl('d', (x, a), (dia + RESIN_HOLE_COMP) / 2, d0, d1)


def hex_pocket(x, a):                 # printed nut drops in from outside onto the shoulder; flats face the bore
    af = hw.NUT_AF + RESIN_HOLE_COMP + 0.2
    e = af / 2 / 0.8660254
    return Fn(lambda X, Y, d, xp: xp.maximum(hw.hex_dist(X - x, Y - a, af, xp), max(SHOULDER - d, d - H_TOP - 1)),
              (x - e, x + e, a - e, a + e, SHOULDER, H_TOP + 1))


head_cuts, insert_cuts = [], []
for a in ROWS:
    for x in (-SCREW_X, SCREW_X):
        head_cuts += [vhole(x, a, M4_CLEAR, -1, H_TOP + 1), vhole(x, a, HEAD_D, HEAD_FLOOR, H_TOP + 1)]
        insert_cuts.append(vhole(x, a, M4_CLEAR, -1, H_TOP + 1))
        if PRINTED_NUTS:
            insert_cuts.append(hex_pocket(x, a))
        else:
            insert_cuts.append(vhole(x, a, INS_D, SHOULDER, H_TOP + 1))
HEAD_HALF = sub(body, *head_cuts)
INSERT_HALF = sub(body, *insert_cuts)
GAP = 6.0
W = 2 * BAND_R


def _to_model(z0):
    def f(x, y, z):                    # local print (x along a, z across, y up) -> tree (X', Y', d')
        return z - BAND_R, x, y
    return f


PIECES = [sdf_piece(A_END, H_TOP, W, root, _to_model(z0), x0=-A_END / 2, z0=z0)
          for root, z0 in ((HEAD_HALF, -W - GAP / 2), (INSERT_HALF, GAP / 2))]
if PRINTED_NUTS:
    PIECES.append(hw.piece(A_END / 2 + 8.0, -hw.SIZE_Z / 2))
