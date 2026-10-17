"""v3 side panel (2026-10-17, sketches/pi_box_v3_panel_edge_sketch.py): L-hook long edges that slide in matching
channels in the v3 core walls (pi_box_bat_core_v3), so a bowed panel can't pull its edges out along X.

Edge, from the wall's inner face: a NECK-long neck at the panel's full thickness through the slot, then a hook
HOOK_T thick (X) reaching HOOK past the panel's inner face (toward the box interior: only PANEL_LIP of wall is left
outside the panel). Clearance comes from the core's channel (GW, + CL); this file only takes off PRINT_GROW, the
resin's outward growth on external faces. Prints flat, outer face down: the hook stands on the edge band.
Fill as the v2 panel (open sheet gyroid in a solid frame). The +Y back panel (MagSafe frame) reuses panel_mask().
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import X_IN, Y_MIN, DEPTH, RIM_D, PANEL_LIP, PANEL_T, RESIN_HOLE_COMP, GYROID_PERIOD, GYROID_T
from designs.pi_box_bat_lids import PANEL_LEAD

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

NECK, HOOK_T, HOOK = 1.0, 1.6, 2.0
CL = 0.2 + RESIN_HOLE_COMP / 2                 # channel clearance per side (cut in the core)
PRINT_GROW = 0.175                             # external faces print ~this much proud
HALF_W = X_IN + NECK + HOOK_T - PRINT_GROW     # panel half-width over the hooks
NECK_X = X_IN + NECK + PRINT_GROW              # hook's inner face (|X|)
D0, D1 = RIM_D + 0.1, DEPTH - 0.1              # top (stops on the rim) .. bottom
BORDER = 4.0                                   # solid frame inside the wall line before the gyroid starts


def panel_mask(X, y, d, gyroid_face=None):
    """X rig, y off the outer face toward the interior (0 .. PANEL_T + HOOK), d rig depth."""
    s = np.ones(np.broadcast_shapes(np.shape(X), np.shape(d)), bool)
    body = y <= PANEL_T - PRINT_GROW
    hook = (y <= PANEL_T + HOOK - PRINT_GROW) & (np.abs(X) >= NECK_X)
    s &= (np.abs(X) <= HALF_W) & (body | hook) & (y >= 0) & (d >= D0) & (d <= D1)
    yy = min(y, PANEL_T + HOOK - y)
    s &= d >= D0 + PANEL_LEAD - yy                                   # lead-in on the top (leading) edge
    s &= np.abs(X) <= HALF_W - np.maximum(PANEL_LEAD - (d - D0), 0.0)   # and its corners
    if gyroid_face is not None:
        k = 2 * np.pi / GYROID_PERIOD
        Y = gyroid_face[0] + gyroid_face[1] * y
        g = np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y) * np.cos(k * d) + np.sin(k * d) * np.cos(k * X)
        inner = (np.abs(X) <= X_IN - BORDER) & (d >= D0 + BORDER + 2) & (d <= D1 - BORDER)
        s &= ~inner | (np.abs(g) <= GYROID_T)
    return s


# front (-Y) panel: outer face at the slot's outer face + CL, interior toward +Y. Height (D1 - D0) along print x.
_FACE = Y_MIN + PANEL_LIP + CL
PIECES = [implicit_piece(D1 - D0, PANEL_T + HOOK, 2 * HALF_W,
                         lambda x, y, z: panel_mask(z - HALF_W, y, x + D0, gyroid_face=(_FACE, +1)),
                         x0=-(D1 - D0) / 2, z0=-HALF_W)]
