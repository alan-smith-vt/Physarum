"""Lidar cage resin cross v4: the GoPro load path for the Monday pilot (2026-10-15).

On Monday the 1/4" screw through this cross is the only thing holding the GoPro. v3 (4.76 mm) with a flush
82° flat head left ~1 mm of resin under the cone, and the 13.6 mm screw would bottom out in the CAMVATE
mount (8.6 mm tap) before clamping.

v4 changes only the centre:
  - hub raised to HUB_T = 8.0 on the GoPro side (Ø30 top, 45° flare to Ø36.5, which runs full height so
    nothing overhangs); the underside is unchanged, so the ~5 mm gap to the lidar dome stays
  - 82° countersink from the lidar face (prints on the plate): mouth Ø13.5 (12.7 + comp + plate-face extra),
    then Ø6.7 clearance. The screw clamps the mount down onto the hub: 4.1 mm of resin under the cone,
    5.6 mm of thread in the mount, 3.0 mm short of its bottom
  - arms and M4 pads stay 4.76, so the M4×18 stack through cross + spacer is unchanged
  - the 1/4-20 tap and the anti-twist pin hole are gone (the pin was dropped)
Prints flat, lidar face down, like v3 (which printed with no delamination).
"""
import numpy as np
from geometry import implicit_piece
from designs.cage_v2 import (CROSS_SPAN, CROSS_T, ARM_W, HUB_D, PAD_R, STANDOFF, BASE, M4_CLEAR,
                             _chamfered, _centered, _hole, _csk)

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

HUB_T = 8.0
HUB_TOP_R = HUB_D / 2                        # 15
HUB_BASE_R = HUB_TOP_R + (HUB_T - CROSS_T)   # 18.24: 45° flare
QTR_CLEAR = 6.35                             # + RESIN_HOLE_COMP in _csk -> 6.7
CSK_HEAD = 12.7 + 0.45                       # + RESIN_HOLE_COMP -> 13.5 mouth (plate-face allowance)


def cross_v4(x, y, z):
    def outline(x, z, h):
        inset = BASE / 2 - h                                 # chamfer inset on the bottom CHAMFER mm
        hub = x ** 2 + z ** 2 <= (HUB_BASE_R - inset) ** 2
        if y > CROSS_T:                                      # only the flaring hub above the arms
            return x ** 2 + z ** 2 <= (HUB_BASE_R - (y - CROSS_T)) ** 2
        u, v = (x + z) / np.sqrt(2), (x - z) / np.sqrt(2)
        arm_half = ARM_W / 2 - inset
        arms = ((np.abs(u) <= arm_half) | (np.abs(v) <= arm_half)) & (np.abs(x) <= STANDOFF) & (np.abs(z) <= STANDOFF)
        pads = np.zeros_like(arms)
        for sx in (-1, 1):
            for sz in (-1, 1):
                pads |= (x - sx * STANDOFF) ** 2 + (z - sz * STANDOFF) ** 2 <= (PAD_R - inset) ** 2
        return arms | hub | pads

    solid = _chamfered(x, z, y, outline, BASE / 2)
    for sx in (-1, 1):
        for sz in (-1, 1):
            solid &= ~_hole(x, z, sx * STANDOFF, sz * STANDOFF, M4_CLEAR)
    solid &= ~_csk(x, z, 0, 0, QTR_CLEAR, CSK_HEAD, y)
    return solid


PIECES = [implicit_piece(CROSS_SPAN, HUB_T, CROSS_SPAN, _centered(cross_v4, CROSS_SPAN),
                         x0=-CROSS_SPAN / 2, z0=-CROSS_SPAN / 2)]
