"""v4 back panel printed STANDING (user 2026-11-02: the flat print's 5,600 -> 400 mm2 body-to-rail step on the old
FEP flattened one rail). Same panel as pi_box_v4_back_panel (gyroid, window open 5..30 mm from the base), plus:
  - floor end (d = D1) on the plate, top tongue pointing up (its 45 deg tip needs nothing);
  - the window's four inner corners rounded R WIN_R;
  - snap-off columns in the window holding up the head bar (the only span): shaft r COL['r'], 0.25 tips dug into
    the head bar, a 0.4 neck on the bottom frame bar;
  - 0.5 mm elephant's-foot bevel on the plate face (print only, print_post.bevel_piece; CPU + GPU).
Print frame: x = rig X + HALF_W, layer height = D1 - d, z = y off the outer face (0 .. PANEL_T + HOOK).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import PANEL_T
from designs import pi_box_bat_panel_v3 as PV
from designs import pi_box_bat_back_panel_v3 as B3
from designs import pi_box_v4_back_panel as BP

SUPPORTS = False
EXPOSURE_S = BP.EXPOSURE_S
BOTTOM_EXPOSURE_S = BP.BOTTOM_EXPOSURE_S
EF_CH = 0.5

WIN_TOP = BP.WIN_HEAD + B3.BRACE / 2               # window top edge (head bar underside), d
WIN_BOT = PV.D1 - B3.FRAME                         # window bottom edge (bottom frame bar top), d
WIN_X = B3.IN_X                                    # window half-width
WIN_R = 4.0                                        # inner corner radius
BODY_T = PANEL_T - PV.PRINT_GROW                   # body thickness (y 0..BODY_T)
COL = dict(r=0.6, tip=0.25, dig=0.25, taper=1.0, neck=0.4, pitch=5.0)
_n = int(np.ceil((2 * (WIN_X - WIN_R - 1.0)) / COL['pitch'])) + 1
COL_X = np.linspace(-(WIN_X - WIN_R - 1.0), WIN_X - WIN_R - 1.0, _n)
COL_Y = BODY_T / 2


def _corners(X, y, d):
    """Solid fillets filling the window's four corners outside an R WIN_R arc (body slab only)."""
    ax = np.abs(X)
    s = np.zeros(np.broadcast_shapes(np.shape(X), np.shape(y), np.shape(d)), bool)
    for dc, sgn in ((WIN_TOP + WIN_R, -1), (WIN_BOT - WIN_R, +1)):     # arc centres, sgn: corner side in d
        cx = WIN_X - WIN_R
        in_corner = (ax >= cx) & (ax <= WIN_X) & (sgn * (d - dc) >= 0) & (sgn * (d - dc) <= WIN_R)
        s |= in_corner & ((ax - cx) ** 2 + (d - dc) ** 2 >= WIN_R ** 2)
    return s & (y >= 0) & (y <= BODY_T)


def _columns(X, y, d):
    """Snap-off columns from the bottom frame bar (d WIN_BOT) up to the head bar (d WIN_TOP)."""
    t0, t1 = WIN_TOP - COL['dig'], WIN_BOT + COL['dig']
    if not (t0 <= d <= t1):
        return np.zeros(np.broadcast_shapes(np.shape(X), np.shape(y)), bool)
    u_top, u_bot = d - t0, t1 - d                                        # distance from each contact
    if u_top < COL['taper'] + COL['dig']:
        r = COL['tip'] + (COL['r'] - COL['tip']) * max(u_top - COL['dig'], 0.0) / COL['taper']
    elif u_bot < COL['taper'] + COL['dig']:
        r = COL['neck'] + (COL['r'] - COL['neck']) * max(u_bot - COL['dig'], 0.0) / COL['taper']
    else:
        r = COL['r']
    s = np.zeros(np.broadcast_shapes(np.shape(X), np.shape(y)), bool)
    for xc in COL_X:
        s |= (X - float(xc)) ** 2 + (y - COL_Y) ** 2 <= r * r
    return s


def standing_mask(X, y, d):
    s = BP.back_mask(X, y, d)
    win = (np.abs(X) <= WIN_X) & (d >= WIN_TOP - 0.01) & (d <= WIN_BOT + 0.01)
    s = s | (win & _corners(X, y, d)) | _columns(X, y, d)
    return s


H_TOTAL = PV.D1 - BP.D_TIP
_raw = implicit_piece(2 * PV.HALF_W, H_TOTAL, PANEL_T + PV.HOOK,
                      lambda x, h, z: standing_mask(x - PV.HALF_W, z, PV.D1 - h),
                      x0=-PV.HALF_W, z0=0.0)

# Spikes under the side-edge ledges: the hook/edge corners stop at d 175.3 (door A's webs), so standing they start
# 4.6 mm above the plate. Two cones per side, plate -> 0.25 dug into the ledge (user 2026-11-02).
from designs import pi_box_v4_door_a as _D    # noqa: E402
LEDGE_D = 175.3
SPIKES = [(sx * 46.2, 1.5) for sx in (-1, 1)] + [(sx * 47.8, 4.0) for sx in (-1, 1)]   # (rig X, y off outer face)
SPIKE = dict(r0=0.6, r1=0.25, dig=0.25)


def _spikes(x, h, z, xp=np):
    top = PV.D1 - LEDGE_D + SPIKE['dig']           # height of the dug-in tip
    if h > top:
        return xp.zeros((1, 1), bool)
    r = SPIKE['r0'] + (SPIKE['r1'] - SPIKE['r0']) * min(h / top, 1.0)
    X = x - PV.HALF_W
    s = xp.zeros(xp.broadcast_shapes(x.shape, z.shape), bool)
    for xs, ys in SPIKES:
        s |= (X - xs) ** 2 + (z - ys) ** 2 <= r * r
    return s


from designs.print_post import bevel_piece    # noqa: E402
PIECES = [bevel_piece(_raw, EF_CH, extra=_spikes)]   # CPU + GPU (to_goo --gpu)
