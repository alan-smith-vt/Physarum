"""v4 camera cartridges standing on an END FACE (user 2026-11-01): instead of fully face up on a raft, each cartridge stands
on one of its cap's flat 45° countersink end faces, directly on the plate. The cosmetic R30 face then points up-ish,
away from the plate, and nothing touches it; only real overhangs and islands get supports, which also stop rocking.

Plate face: STAND = 'bottom' (the chin / ribbon end, half-space -n - h <= -(FL_D0 + SEAT) in the tree frame) or 'top'.
Print frame: up = minus that face's outward normal; x along t; the face's plane on the plate.
Supports (as pi_box_v4_cart_faceup): sampled at SAMP; a voxel is unsupported with no resin in the 45° cone under it;
regions narrower than MIN_SPAN are left to bridge; GRID-spaced columns straight down to the plate (tip r TIP_R dug
TIP_DIG in, cone to COL_R, foot pad FOOT_R), or onto the part below when blocked; spots with neither are overlaid red.
Elephant's foot: a 45° x EF_CH bevel round the plate face (from the SDF of the section just above it).
Two cartridges, the second mirrored (t -> -t). Viewer: python generate.py pi_box_v4_cart_tilt. Blue, 2.3 s.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage
from geometry import implicit_piece
from designs import pi_box_v4_cam_cartridge as K

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

STAND = 'bottom'
SAMP, MIN_SPAN, GRID = 0.2, 2.0, 2.0
TIP_R, TIP_DIG, TIP_L, COL_R, FOOT_R, FOOT_L = 0.25, 0.25, 1.5, 0.6, 1.2, 1.0
CLEAR, GAP = 0.4, 6.0
EF_CH = 0.5                                  # elephant's-foot bevel: 45° x 0.5 round the plate face (20 s bottom layers)
_CACHE = Path(__file__).resolve().parents[1] / 'out' / 'cart_tilt_cache.npz'

UP = np.array([1.0, 0.0, 1.0 if STAND == 'bottom' else -1.0]) / np.sqrt(2)   # tree (n, t, h)
E1 = np.array([0.0, 1.0, 0.0])
E2 = np.cross(UP, E1)
_CORNERS = np.array([[a, b, c] for a in (K.N0, K.N1) for b in (K.T0, K.T1) for c in (0.0, K.H_TOP)])
_P = _CORNERS @ np.stack([E1, UP, E2], 1)                         # print (x, y, z) of the tree box corners
LO, HI = _P.min(0), _P.max(0)


def to_tree(x, y, z):
    """print (x, y, z) mm, y up from the plate -> tree (n, t, h)."""
    X, Y, Z = x + LO[0], y + Y0, z + LO[2]
    return tuple(X * E1[i] + Y * UP[i] + Z * E2[i] for i in range(3))


def _root(xp, n, t, h):
    return xp.asarray(K.ROOT(n, t, h, xp, None, 0)) <= 0


def _sample(y0):
    xs = np.arange(0, HI[0] - LO[0], SAMP); zs = np.arange(0, HI[2] - LO[2], SAMP)
    ys = np.arange(0, HI[1] - y0, SAMP)
    sol = np.zeros((len(ys), len(xs), len(zs)), bool)
    for i, y in enumerate(ys):
        n, t, h = (a for a in (xs[:, None] * E1[k] + (y + y0) * UP[k] + zs[None, :] * E2[k] + LO[0] * E1[k] + LO[2] * E2[k]
                               for k in range(3)))
        sol[i] = np.broadcast_to(_root(np, n, t, h), (len(xs), len(zs)))
    return xs, ys, zs, sol


def _plan():
    src = Path(K.__file__).stat().st_mtime + Path(__file__).stat().st_mtime
    if _CACHE.exists():
        c = np.load(_CACHE)
        if float(c['src']) == src:
            return float(c['y0']), c['pts'], c['drop']
    # the plate plane: the lowest resin along UP
    _, ys, _, sol = _sample(LO[1])
    y0 = LO[1] + ys[int(np.argmax(sol.any(axis=(1, 2))))]
    xs, ys, zs, sol = _sample(y0)
    st = ndimage.generate_binary_structure(2, 1)
    uns = np.zeros_like(sol)
    for i in range(1, len(ys)):
        uns[i] = sol[i] & ~ndimage.binary_dilation(sol[i - 1], st)
    lab, nl = ndimage.label(uns, ndimage.generate_binary_structure(3, 3))
    pts, drop = [], []
    rc = int(np.ceil((CLEAR + COL_R) / SAMP))
    for sl, k in zip(ndimage.find_objects(lab), range(1, nl + 1)):
        if max(sl[1].stop - sl[1].start, sl[2].stop - sl[2].start) * SAMP < MIN_SPAN:
            continue
        ii, jj, kk = np.nonzero(lab[sl] == k)
        cells = {}
        for a, b, c in zip(ii + sl[0].start, jj + sl[1].start, kk + sl[2].start):
            key = (int(xs[b] // GRID), int(zs[c] // GRID))
            if key not in cells or a < cells[key][0]:
                cells[key] = (a, b, c)
        for a, b, c in cells.values():
            win = sol[:max(a - int(TIP_L / SAMP), 0), max(b - rc, 0):b + rc + 1, max(c - rc, 0):c + rc + 1]
            if not win.any():
                pts.append((xs[b], ys[a], zs[c], -1.0)); continue
            below = np.nonzero(sol[:a, b, c])[0]
            if len(below):
                base = int(below[-1])
                lo, hi = base + int(np.ceil((TIP_L + CLEAR) / SAMP)), a - int(TIP_L / SAMP)
                if hi - lo >= 1 and not sol[lo:hi, max(b - rc, 0):b + rc + 1, max(c - rc, 0):c + rc + 1].any():
                    pts.append((xs[b], ys[a], zs[c], ys[base])); continue
            drop.append((xs[b], ys[a], zs[c]))
    pts, drop = np.array(pts, float).reshape(-1, 4), np.array(drop, float).reshape(-1, 3)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE, y0=y0, pts=pts, drop=drop, src=src)
    return y0, pts, drop


Y0, PTS, DROP = _plan()
_W, _H, _D = HI[0] - LO[0], HI[1] - Y0, HI[2] - LO[2]


def _xp(a):
    try:
        import cupy as cp
        return cp.get_array_module(a)
    except ImportError:
        return np


def _solid(x, y, z):
    xp = _xp(x)
    s = _root(xp, *to_tree(x, y, z))
    if y < EF_CH:                                   # 45° bevel: keep what is >= (EF_CH - y) inside the section at EF_CH
        n, t, h = to_tree(x, EF_CH, z)
        s = s & (xp.asarray(K.ROOT(n, t, h, xp, None, 0)) <= -(EF_CH - y))
    for px, py, pz, pb in PTS:
        r2 = (x - px) ** 2 + (z - pz) ** 2
        top = py + TIP_DIG
        if y > top:
            continue
        if pb < 0:
            bot = 0.0
            r = COL_R + (FOOT_R - COL_R) * max(FOOT_L - y, 0.0) / FOOT_L if y < FOOT_L else COL_R
        else:
            bot = pb - TIP_DIG
            r = TIP_R + (COL_R - TIP_R) * (y - bot) / TIP_L if y < bot + TIP_L else COL_R
        if y < bot:
            continue
        if y > top - TIP_L:
            r = TIP_R + (COL_R - TIP_R) * (top - y) / TIP_L
        s = s | (r2 <= r * r)
    return s


def _piece(mirror, x0):
    if mirror:
        return implicit_piece(_W, _H, _D, lambda x, y, z: _solid(_W - x, y, z), x0=x0, z0=-_D / 2)
    return implicit_piece(_W, _H, _D, _solid, x0=x0, z0=-_D / 2)


PIECES = [_piece(False, -_W - GAP / 2), _piece(True, GAP / 2)]


def _overlay():
    if not len(DROP):
        return []
    from designs import pi_box_bat_rig as R
    x0 = -_W - GAP / 2
    tris = np.concatenate([R._box(x0 + x - 0.3, x0 + x + 0.3, y - 0.6, y, -_D / 2 + z - 0.3, -_D / 2 + z + 0.3) for x, y, z in DROP])
    path = R.OVERLAY_DIR / 'cart_tilt_unsupported.stl'
    R._write_stl(path, tris)
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=1.0,
                 label=f'unsupported spots left to bridge ({len(DROP)}), left part')]


OVERLAYS = _overlay()
print(f'cart_tilt ({STAND}): {int((PTS[:, 3] < 0).sum())} plate columns + {int((PTS[:, 3] >= 0).sum())} part-to-part, '
      f'{len(DROP)} left to bridge; part {_W:.1f} x {_D:.1f} x {_H:.1f}')
