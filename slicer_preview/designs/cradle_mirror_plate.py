"""Blue plate (user 2026-11-01): the gyroid cradle coupon (designs/cradle_coupon.py, unchanged) + a MIRROR TEST marker,
the first print after the to_goo MIRROR_FIX. Clear blue + Tenacious, 2.3 s.

Marker: a 1.5 mm base with a raised F (2 mm) and an arrow, on the plate beside the strips (+z). Check: take the part off,
set it base-down and look from above. It must read like the viewer seen from above (python generate.py
cradle_mirror_plate): a normal F (bars to the right of the stem, top bar away from you when the arrow points right).
A backwards F means the export still mirrors.
"""
import numpy as np
from geometry import implicit_piece
from designs import cradle_coupon as CC

SUPPORTS = False
EXPOSURE_S = CC.EXPOSURE_S
BOTTOM_EXPOSURE_S = CC.BOTTOM_EXPOSURE_S

MW, MD, BASE_T, RAISE = 30.0, 22.0, 1.5, 2.0        # marker base (x, z), base thickness, letter height


def marker(x, y, z):
    """Local x (1, W) right, z (H, 1) toward the viewer, y up. F: stem at x 3..5, z 3..19; bars toward +x."""
    s = np.broadcast_to((y <= BASE_T) & (x >= 0) & (x <= MW) & (z >= 0) & (z <= MD), np.broadcast_shapes(x.shape, z.shape))
    if y <= BASE_T:
        return s
    if y > BASE_T + RAISE:
        return np.zeros_like(s)
    f = ((x >= 3) & (x <= 5) & (z >= 3) & (z <= 19)) \
        | ((x >= 3) & (x <= 13) & (z >= 3) & (z <= 5)) \
        | ((x >= 3) & (x <= 10) & (z >= 10) & (z <= 12))
    shaft = (x >= 16) & (x <= 24) & (np.abs(z - 11) <= 1.0)                       # arrow pointing +x
    head = (x >= 24) & (x <= 28) & (np.abs(z - 11) <= (28 - x) * 0.9)
    return np.broadcast_to(f | shaft | head, s.shape)


_z_end = CC.LENGTH / 2 + 5.0
PIECES = list(CC.PIECES) + [implicit_piece(MW, BASE_T + RAISE, MD, marker, x0=-MW / 2, z0=_z_end)]
