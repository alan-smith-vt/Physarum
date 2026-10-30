"""v4 back (+Y) panel: the v3 back panel (pi_box_bat_back_panel_v3: L-hook edges, MagSafe disc, braced window) with
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


def back_mask(X, y, d):
    return B3.back_mask(X, y, d) | tongue_mask(X, y, d)


PIECES = [implicit_piece(PV.D1 - D_TIP, PANEL_T + PV.HOOK, 2 * PV.HALF_W,
                         lambda x, y, z: back_mask(z - PV.HALF_W, y, x + D_TIP),
                         x0=-(PV.D1 - D_TIP) / 2, z0=-PV.HALF_W)]
