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
RING_C = (0.0, 100.0)                           # MagSafe ring centre (user 2026-10-30: up a little from 108.5),
#                                                held by four swoops (above, below, both sides)


SWOOP_BITE = 1.5                                # side swoops run this far into the disc rim (they hold it)

# SGH logo (user home:2297; placeholder Arial Bold until real artwork): solid letter ribs replacing the gyroid in the
# upper zone, read from outside (+Y side: rig +X is the viewer's left, so the text is mirrored in X; up = -d)
LOGO = dict(text='SGH', font=r'C:\Windows\Fonts\arialbd.ttf', max_w=72.0, grip=1.0)
# Letters are solid ribs in an open box (no gyroid, single print, user 2026-10-30): the box is exactly as tall as the
# letters, so every letter runs grip mm into the top frame and into the box's base bar; that is what holds them.
_LOGO_H = 0.1                                   # raster pitch (mm)


def _logo_raster():
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), LOGO['text'], size=10.0, prop=FontProperties(fname=LOGO['font']))
    v = tp.vertices
    x0, x1, y0, y1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    k = LOGO['max_w'] / (x1 - x0)
    top = PV.D0 + B3.FRAME
    LOGO['zone'] = (top - LOGO['grip'], top - LOGO['grip'] + (y1 - y0) * k)
    xs = np.arange(-LOGO['max_w'] / 2, LOGO['max_w'] / 2, _LOGO_H)
    ds = np.arange(LOGO['zone'][0], LOGO['zone'][1], _LOGO_H)
    U, D = np.meshgrid(xs, ds)
    dc = (LOGO['zone'][0] + LOGO['zone'][1]) / 2
    u = -U / k + (x0 + x1) / 2                   # mirrored: rig X -> -text x
    w = (dc - D) / k + (y0 + y1) / 2
    inside = tp.contains_points(np.c_[u.ravel(), w.ravel()]).reshape(U.shape)
    return xs[0], ds[0], inside


_LX0, _LD0, _LOGO_M = _logo_raster()
LOGO_BOT = LOGO['zone'][1] - LOGO['grip'] + B3.BRACE / 2   # base bar centre: its top edge grips the letters


def logo_mask(X, d):
    i = np.floor((np.asarray(d) - _LD0) / _LOGO_H).astype(int)
    j = np.floor((np.asarray(X) - _LX0) / _LOGO_H).astype(int)
    ok = (i >= 0) & (i < _LOGO_M.shape[0]) & (j >= 0) & (j < _LOGO_M.shape[1])
    return ok & _LOGO_M[np.clip(i, 0, _LOGO_M.shape[0] - 1), np.clip(j, 0, _LOGO_M.shape[1] - 1)]


def _arc(X, d, p0, p1, pm, w=B3.BRACE):
    """Band of width w along the circle through p0, pm, p1, between p0 and p1 (bounding box of the three)."""
    (ax, ay), (bx, by), (cx, cy) = p0, pm, p1
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / D
    R = np.hypot(ax - ux, ay - uy)
    xs, ys = (ax, bx, cx), (ay, by, cy)
    box = (X >= min(xs) - w) & (X <= max(xs) + w) & (d >= min(ys) - w) & (d <= max(ys) + w)
    return box & (np.abs(np.hypot(X - ux, d - uy) - R) <= w / 2)


def frame_mask(X, y, d):
    """v3 frame, MagSafe disc and braces, without the window's vertical bar (user home:2279)."""
    s = PV.panel_mask(X, y, d)
    cx, cd = RING_C
    r = np.hypot(X - cx, d - cd)
    ring = (r <= B3.DISC_R) & (r >= B3.RING_ID / 2 - 1.5)              # MagSafe annulus only, open middle
    keep = (np.abs(X) >= B3.IN_X) | (d <= PV.D0 + B3.FRAME) | (d >= PV.D1 - B3.FRAME) | ring
    top, bot, IN = PV.D0 + B3.FRAME, WIN_HEAD, B3.IN_X                 # (user home:2302) open panes, no gyroid:
    R0, lt = B3.DISC_R, LOGO_BOT                                       # side swoops flank the disc and merge with
    for p0, p1, pm in [((-IN, lt), (-IN, bot), (cx - R0 + SWOOP_BITE, cd)),   # its rim: four concave swoops,
                       ((IN, lt), (IN, bot), (cx + R0 - SWOOP_BITE, cd)),     # sides from the box corners to the
                       ((-IN, lt), (IN, lt), (cx, cd - R0 + SWOOP_BITE)),     # window-head ends, top from the box
                       ((-IN, bot), (IN, bot), (cx, cd + R0 - SWOOP_BITE))]:  # base, bottom from the window head
        keep |= _arc(X, d, p0, p1, pm)
    keep |= B3._seg(X, d, (-IN, bot), (IN, bot))                       # window head
    keep |= B3._seg(X, d, (-IN, lt), (IN, lt))                         # logo box base
    keep |= logo_mask(X, d)                                            # SGH: solid letters bridging frame and bar
    return s & keep


def back_mask(X, y, d):
    s = frame_mask(X, y, d)                                            # no gyroid: open panes (user home:2302)
    if y < B3.RING_DEPTH:                                              # MagSafe ring recess in the outer face
        r = np.hypot(X - RING_C[0], d - RING_C[1])
        s &= ~((r >= B3.RING_ID / 2) & (r <= B3.RING_OD / 2 + PV.RESIN_HOLE_COMP / 2))
    return s | tongue_mask(X, y, d)


PIECES = [implicit_piece(PV.D1 - D_TIP, PANEL_T + PV.HOOK, 2 * PV.HALF_W,
                         lambda x, y, z: back_mask(z - PV.HALF_W, y, x + D_TIP),
                         x0=-(PV.D1 - D_TIP) / 2, z0=-PV.HALF_W)]
