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
EXPOSURE_S = 2.8                                     # smoky black + clear Tenacious (exposure_test_smoke, 2026-10-31)
BOTTOM_EXPOSURE_S = 20.0
D_TIP = PV.D0 - TONGUE_H                        # tongue tip d (panel top edge at PV.D0)


def tongue_mask(X, y, d):
    """X, d rig; y off the outer face toward the interior."""
    yy = np.abs(y - PANEL_T / 2)                 # off the panel's mid-plane
    ax = np.abs(X)
    return ((yy <= TONGUE_T / 2 - PV.PRINT_GROW) & (d >= D_TIP + yy) & (d <= PV.D0 + 1.0)
            & (ax >= BACK_TONGUE_X[0]) & (ax <= BACK_TONGUE_X[1]))


_FACE = B3._FACE                                 # outer face Yr (interior toward -Y)
WIN_HEAD = PV.D1 - 30.0 - B3.BRACE / 2          # head bar centre: window open to 30 from the base (user 2026-11-02:
#                                                bank screen 5.6..29 from the base edge); gyroid above, open below
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
    # box base where the top swoop dips as far as the bottom one (window head -> ring); solid fill above the letters
    sag = WIN_HEAD - (RING_C[1] + B3.DISC_R - SWOOP_BITE)
    base = RING_C[1] - B3.DISC_R + SWOOP_BITE - sag
    z1 = base - B3.BRACE / 2 + LOGO['grip']
    LOGO['zone'] = (z1 - (y1 - y0) * k, z1)
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
    """Frame (sides, top, bottom) + the window head bar. MagSafe disc, swoops and logo bar removed (user 2026-11-02:
    the phone mounts on the stick, not the panel)."""
    s = PV.panel_mask(X, y, d)
    keep = (np.abs(X) >= B3.IN_X) | (d <= PV.D0 + B3.FRAME) | (d >= PV.D1 - B3.FRAME)
    keep |= B3._seg(X, d, (-B3.IN_X, WIN_HEAD), (B3.IN_X, WIN_HEAD))   # window head
    return s & keep


def back_mask(X, y, d):
    s = frame_mask(X, y, d) | (PV.panel_mask(X, y, d, gyroid_face=(_FACE, -1)) & (d <= WIN_HEAD))   # gyroid everywhere
    # above the window head, round the swoops and in the ring; window open; SGH removed (user 2026-10-30)
    s = s | tongue_mask(X, y, d)
    # door A's webs run up inside each wall through the panel channels (core: 1.8 clear below d 175.5): the panel's
    # hook/neck corners stop short of them (assembly check 2026-11-01)
    from designs import pi_box_v4_door_a as _D
    s = s & ~((np.abs(X) >= _D.HX - _D.WEB_T - 0.3) & (np.asarray(d) >= 175.3))
    return s


PIECES = [implicit_piece(PV.D1 - D_TIP, PANEL_T + PV.HOOK, 2 * PV.HALF_W,
                         lambda x, y, z: back_mask(z - PV.HALF_W, y, x + D_TIP),
                         x0=-(PV.D1 - D_TIP) / 2, z0=-PV.HALF_W)]


# ── print-only (user 2026-11-02): 1 mm elephant's-foot bevel round the outline + needles under the tongue ──
EF_CH = 1.0                                      # 45° x 1.0 round the plate face's OUTER outline (holes untouched)
NEEDLE = dict(r0=0.5, r1=0.15, dig=0.1, pitch=4.0, inset=1.0)   # cone radii at plate / tip, tip dug into the tongue
_T_LOW = PANEL_T / 2 - (TONGUE_T / 2 - PV.PRINT_GROW)            # tongue underside height above the plate
_N_D = D_TIP + (TONGUE_T / 2 - PV.PRINT_GROW) + NEEDLE['inset']  # needle d: 1 mm in from the flat underside's start
_a, _b = BACK_TONGUE_X[0] + 1.0, BACK_TONGUE_X[1] - 1.0      # 1 mm in from each tongue end: both corners get one
_n = int(np.ceil((_b - _a) / NEEDLE['pitch'])) + 1             # evenly spaced, pitch <= 4
_N_X = [sx * v for sx in (-1, 1) for v in np.linspace(_a, _b, _n)]


def _needles(x, y, z):
    top = _T_LOW + NEEDLE['dig']
    if y > top:
        return np.zeros(np.broadcast_shapes(x.shape, z.shape), bool)
    r = NEEDLE['r0'] + (NEEDLE['r1'] - NEEDLE['r0']) * min(y / _T_LOW, 1.0)
    d, X = x + D_TIP, z - PV.HALF_W
    s = np.zeros(np.broadcast_shapes(x.shape, z.shape), bool)
    for xn in _N_X:
        s |= (d - _N_D) ** 2 + (X - xn) ** 2 <= r * r
    return s


def _print_piece(piece):
    import geometry as g
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    base = piece['make_slice']
    k = int(EF_CH / g.LY_MM)
    cache = {}
    x = ((np.arange(piece['W']) + 0.5) * g.PX_MM)[None, :]
    z = ((np.arange(piece['H']) + 0.5) * g.PZ_MM)[:, None]

    def make_slice(layer, *a, **kw):
        out = base(layer, *a, **kw)
        y = (layer + 0.5) * g.LY_MM
        if y < EF_CH:
            if 'dist' not in cache:
                cache['dist'] = distance_transform_edt(binary_fill_holes(base(k) > 0), sampling=(g.PZ_MM, g.PX_MM))
            out = np.where(cache['dist'] >= EF_CH - y, out, 0).astype(out.dtype)
        return np.where(_needles(x, y, z), 255, out).astype(out.dtype)
    p = dict(piece, make_slice=make_slice)
    p.pop('solid_fn', None)                      # to_goo --gpu would skip both: CPU export only
    return p


PIECES = [_print_piece(p) for p in PIECES]
