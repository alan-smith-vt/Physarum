"""v4 back (+Y) panel: the v3 back panel (gyroid panes above the window, no window bar; user home:2279) (pi_box_bat_back_panel_v3: L-hook edges, MagSafe disc, braced window) with
a TONGUE on its top edge into the groove under the core's back rim (user approved home:2130; sketch
designs/sketches/pi_box_v4_panel_tongue_sketch.py): 1.6 thick, 3.0 tall, centred on the panel, 45° pointed tip (the
lead-in), |X| 6.2..43.5 (gap at the lidar-cable notch). It still slides in from the floor end and stops on the rim face.
Prints flat, outer face down, hooks up (the tongue lies in the plate plane).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import PANEL_T
from designs import pi_box_bat_panel_v3 as PV
from designs import pi_box_bat_back_panel_v3 as B3
from designs.pi_box_v4_core import TONGUE_T, TONGUE_H, BACK_TONGUE_X

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0
D_TIP = PV.D0 - TONGUE_H                        # tongue tip d (panel top edge at PV.D0)


def tongue_mask(X, y, d):
    """X, d rig; y off the outer face toward the interior."""
    yy = np.abs(y - PANEL_T / 2)                 # off the panel's mid-plane
    ax = np.abs(X)
    return ((yy <= TONGUE_T / 2 - PV.PRINT_GROW) & (d >= D_TIP + yy) & (d <= PV.D0 + 1.0)
            & (ax >= BACK_TONGUE_X[0]) & (ax <= BACK_TONGUE_X[1]))


_FACE = B3._FACE                                 # outer face Yr (interior toward -Y)
WIN_HEAD = B3.WIN[2] - B3.BRACE / 2             # above this: panes gyroid-filled; below: the open window (user home:2279)
RING_C = (0.0, WIN_HEAD - B3.BRACE / 2 - B3.DISC_R)   # MagSafe disc as low as it goes: its edge on the window head bar
#                                                (user home:2285: phone clears the cable at the top, overhangs the bottom)


def frame_mask(X, y, d):
    """v3 frame, MagSafe disc and braces, without the window's vertical bar (user home:2279)."""
    s = PV.panel_mask(X, y, d)
    cx, cd = RING_C
    r = np.hypot(X - cx, d - cd)
    ring = (r <= B3.DISC_R) & (r >= B3.RING_ID / 2 - 1.5)              # solid annulus under the MagSafe ring only;
    keep = (np.abs(X) >= B3.IN_X) | (d <= PV.D0 + B3.FRAME) | (d >= PV.D1 - B3.FRAME) | ring   # inside it the braces
    top, bot, IN = PV.D0 + B3.FRAME, WIN_HEAD, B3.IN_X                 # run through to the centre (user home:2289)
    c = (cx, cd)
    for t in [(0.0, top), (-IN, cd), (IN, cd), (-IN, top), (IN, top), (-IN, bot), (IN, bot)]:
        keep |= B3._seg(X, d, c, t)
    keep |= B3._seg(X, d, (-IN, bot), (IN, bot))                       # window head
    return s & keep


def back_mask(X, y, d):
    s = frame_mask(X, y, d) | (PV.panel_mask(X, y, d, gyroid_face=(_FACE, -1)) & (d <= WIN_HEAD))   # gyroid panes
    if y < B3.RING_DEPTH:                                                  # MagSafe recess in the outer face
        r = np.hypot(X - RING_C[0], d - RING_C[1])
        s &= ~((r >= B3.RING_ID / 2) & (r <= B3.RING_OD / 2 + PV.RESIN_HOLE_COMP / 2))
    return s | tongue_mask(X, y, d)


PIECES = [implicit_piece(PV.D1 - D_TIP, PANEL_T + PV.HOOK, 2 * PV.HALF_W,
                         lambda x, y, z: back_mask(z - PV.HALF_W, y, x + D_TIP),
                         x0=-(PV.D1 - D_TIP) / 2, z0=-PV.HALF_W)]
