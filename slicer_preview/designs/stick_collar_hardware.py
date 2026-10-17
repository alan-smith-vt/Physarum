"""Printed stand-ins for the collar's M4 x 18 truss screws and M4 inserts (2026-10-17: the bought ones are at the
office; good for a day, zip ties are the backup).

Not M4: a 0.7 pitch tooth (~0.4 tall) is under the 0.7 mm min feature. Both sides are printed, so a custom thread:
Ø4.2 major, 1.2 pitch, 0.5 deep, flat crest/root with sloped flanks, right-hand. The screw still passes the Ø4.5
holes. The brass pocket (Ø5.5) leaves no wall for a threaded sleeve, so the insert half gets a hex pocket instead
(stick_collar.PRINTED_NUTS) and these are hex nuts, 7.7 AF x 6, dropped in from outside onto the 3 mm shoulder.

Both parts are modelled RESIN_HOLE_COMP/2 smaller/larger (solids print oversize as holes print undersize), then the
nuts get a radial clearance range: sets A-D = 0.10 / 0.20 / 0.30 / 0.40. Each nut row has a tag beside it with
1-4 holes (A = 1). Screws print head down (the head's outer face on the plate), slot for a flat screwdriver.
"""
import numpy as np
from sdf import Box, Cyl, Fn, union, sub, sdf_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP

MAJ_R, PITCH, DEPTH = 2.1, 1.2, 0.5
COMP_R = RESIN_HOLE_COMP / 2
SCREW_L = 18.0
HEAD_R, HEAD_H = 4.7, 2.8               # Ø9.4 in the Ø10.2 counterbore
SLOT_W, SLOT_D = 1.5, 1.5               # cross slot in the head's outer face
NUT_AF, NUT_H = 7.7, 6.0
CLEARANCES = (0.10, 0.20, 0.30, 0.40)   # sets A-D, radial
N_SCREWS, NUTS_PER_SET = 8, 4           # 4 needed of each; spares because resin threads snap
LEAD = 0.5                              # chamfer at the screw tip and both nut mouths


def _thread_r(X, Y, h, xp):
    """Radius of the male tooth surface at angle atan2(Y, X) and height h (right-hand: rises counter-clockwise)."""
    f = h / PITCH - xp.arctan2(Y, X) / (2 * np.pi)
    t = 0.5 + 0.5 * xp.cos(2 * np.pi * f)
    return MAJ_R - DEPTH + DEPTH * xp.clip((t - 0.2) / 0.6, 0.0, 1.0)


def screw(cx, cz):
    top = HEAD_H + SCREW_L

    def f(X, Y, d, xp):
        x, y = X - cx, Y - cz
        r = xp.sqrt(x * x + y * y)
        R = _thread_r(x, y, d - HEAD_H, xp) - COMP_R
        R = xp.minimum(R, MAJ_R - COMP_R - max(0.0, d - (top - LEAD)))
        shank = xp.maximum(r - R, max(-d, d - top))
        return shank * 0.7                 # tooth flanks are steep: keep it a distance bound
    head = Cyl('d', (cx, cz), HEAD_R - COMP_R, 0.0, HEAD_H)
    slots = [Box(cx - HEAD_R - 1, cx + HEAD_R + 1, cz - SLOT_W / 2, cz + SLOT_W / 2, -1, SLOT_D),
             Box(cx - SLOT_W / 2, cx + SLOT_W / 2, cz - HEAD_R - 1, cz + HEAD_R + 1, -1, SLOT_D)]
    shank = Fn(f, (cx - MAJ_R, cx + MAJ_R, cz - MAJ_R, cz + MAJ_R, HEAD_H - 0.5, top))
    return sub(union(head, shank), *slots)


def hex_dist(x, y, af, xp):
    """Hexagonal prism, flats normal to x (and at ±60°)."""
    a = xp.abs(x)
    b = xp.abs(0.5 * x + 0.8660254 * y)
    c = xp.abs(0.5 * x - 0.8660254 * y)
    return xp.maximum(xp.maximum(a, b), c) - af / 2


def nut(cx, cz, clear):
    def f(X, Y, d, xp):
        x, y = X - cx, Y - cz
        r = xp.sqrt(x * x + y * y)
        Rf = _thread_r(x, y, d, xp) + COMP_R + clear
        Rf = xp.maximum(Rf, MAJ_R + COMP_R + clear + LEAD - min(d, NUT_H - d))
        return xp.maximum(xp.maximum(hex_dist(x, y, NUT_AF, xp), (Rf - r) * 0.7), max(-d, d - NUT_H))
    e = NUT_AF / 2 / 0.8660254
    return Fn(f, (cx - e, cx + e, cz - e, cz + e, 0.0, NUT_H))


def tag(cx, cz, n):
    holes = [Cyl('d', (cx - 3.0 + 2.0 * i, cz), (1.2 + RESIN_HOLE_COMP) / 2, -1, 3) for i in range(n)]
    return sub(Box(cx - 4.5, cx + 4.5, cz - 2.5, cz + 2.5, 0.0, 1.5), *holes)


SCREW_P, NUT_P = 12.0, 10.5
parts = [screw(6.0 + SCREW_P * (i % 4), 6.0 + SCREW_P * (i // 4)) for i in range(N_SCREWS)]
for s, c in enumerate(CLEARANCES):
    z = 2 * SCREW_P + 6.0 + NUT_P * s
    parts += [nut(6.0 + NUT_P * j, z, c) for j in range(NUTS_PER_SET)]
    parts.append(tag(6.0 + NUT_P * NUTS_PER_SET + 1.0, z, s + 1))
ROOT = union(*parts)
SIZE_X = 6.0 + NUT_P * NUTS_PER_SET + 6.0
SIZE_Z = 2 * SCREW_P + 6.0 + NUT_P * len(CLEARANCES)
SIZE_Y = HEAD_H + SCREW_L


def piece(x0, z0):
    return sdf_piece(SIZE_X, SIZE_Y, SIZE_Z, ROOT, lambda x, y, z: (x, z, y), x0=x0, z0=z0)
