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
import math
import numpy as np
from scipy import ndimage
from geometry import implicit_piece
from designs import pi_box_v4_cam_cartridge as K

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

STAND = 'picked'                             # 'picked': the face clicked in the face picker (server.py /pick/<design>)
_PICK = Path(__file__).resolve().parents[1] / 'out' / 'pick' / 'pi_box_v4_cam_cartridge.json'
SAMP, MIN_SPAN, GRID = 0.2, 2.0, 2.0
TIP_R, TIP_DIG, TIP_L, COL_R, FOOT_R, FOOT_L = 0.25, 0.25, 1.5, 0.6, 1.2, 1.0
CLEAR, GAP = 0.4, 6.0
EF_CH = 0.5                                  # elephant's-foot bevel: 45° x 0.5 round the plate face (20 s bottom layers)
_CACHE = Path(__file__).resolve().parents[1] / 'out' / 'cart_tilt_cache.npz'

UP = {'bottom': np.array([1.0, 0.0, 1.0]) / np.sqrt(2),         # the chin end's 45° countersink face (61 mm2)
      'top': np.array([1.0, 0.0, -1.0]) / np.sqrt(2),           # the brow end's 45° face (3 mm2: no)
      'end': np.array([0.0, 0.0, -1.0]),                        # the cap's flat top end (~277 mm2)
      'slope': np.array([np.cos(np.radians(312.75)), 0.0, np.sin(np.radians(312.75))]),
      'picked': (-np.array(__import__('json').loads(_PICK.read_text())['normal']) if _PICK.exists() else None)}[STAND]
#   (~265 mm2): the face tilts up and away from the plate (user's pick, 2026-11-01)
#   tree (n, t, h); found by scanning plate directions for contact area (2026-11-01)
UP = UP / np.linalg.norm(UP)
E1 = np.array([0.0, 1.0, 0.0]) - UP[1] * UP                     # x: along t, made perpendicular to UP
E1 = E1 / np.linalg.norm(E1)
E2 = np.cross(UP, E1)
_CORNERS = np.array([[a, b, c] for a in (K.N0, K.N1) for b in (K.T0, K.T1) for c in (0.0, K.H_TOP)])
_P = _CORNERS @ np.stack([E1, UP, E2], 1)                         # print (x, y, z) of the tree box corners
LO, HI = _P.min(0), _P.max(0)


def to_tree(x, y, z):
    """print (x, y, z) mm, y up from the plate -> tree (n, t, h)."""
    X, Y, Z = x + LO[0], y + Y0, z + LO[2]
    return tuple(X * E1[i] + Y * UP[i] + Z * E2[i] for i in range(3))


# Trim (user 2026-11-01): everything past the picked face's plane goes (the two flange tips stood 0.88 proud), so the
# face sits flat on the plate. Print-only: the cartridge model itself is unchanged.
TRIM = STAND == 'picked' and _PICK.exists()
if TRIM:
    _pk = __import__('json').loads(_PICK.read_text())
    _TN, _TOFF = np.array(_pk['normal']), float(np.dot(_pk['normal'], _pk['point']))


def _root(xp, n, t, h):
    s = xp.asarray(K.ROOT(n, t, h, xp, None, 0)) <= 0
    if TRIM:
        s = s & (_TN[0] * n + _TN[1] * t + _TN[2] * h <= _TOFF)
    return s


COARSE, BAND = 4, 0.25     # coarse-to-fine: the tree on every COARSE-th pixel, then full-res only within BAND mm of the
#   threshold (2026-11-03: full-frame tree evaluation was ~0.26 s/layer per cartridge on the GPU). BAND >> the 0.05 mm
#   block half-diagonal, for distance fields that aren't exact; the export was checked byte-identical.


def _tree_le(xp, x, y, z, c):
    """K.ROOT <= c on the (h, W) slice grid x (1, W), z (h, 1)."""
    H, W = z.shape[0], x.shape[1]
    o = COARSE // 2
    xc, zc = x[:, o::COARSE], z[o::COARSE]
    if xc.shape[1] < 2 or zc.shape[0] < 2:
        return xp.broadcast_to(xp.asarray(K.ROOT(*to_tree(x, y, z), xp, None, 0)) <= c, (H, W))
    v = xp.asarray(K.ROOT(*to_tree(xc, y, zc), xp, None, 0)) - c
    def up(a):                                      # coarse block -> its COARSE x COARSE pixels, cropped to (H, W)
        a = xp.repeat(xp.repeat(a, COARSE, axis=0), COARSE, axis=1)
        a = xp.pad(a, ((o, max(H - o - a.shape[0], 0)), (o, max(W - o - a.shape[1], 0))), mode='edge')
        return a[:H, :W]
    out = up(v <= 0)
    near = up(xp.abs(v) < BAND)
    i, j = xp.nonzero(near)
    if i.size:
        out[i, j] = xp.asarray(K.ROOT(*to_tree(x[0, j], y, z[i, 0]), xp, None, 0)) <= c
    return out


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
    src = Path(K.__file__).stat().st_mtime + Path(__file__).stat().st_mtime + (_PICK.stat().st_mtime if _PICK.exists() else 0)
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
    s = _tree_le(xp, x, y, z, 0.0)
    if TRIM:
        n, t, h = to_tree(x, y, z)
        s = s & (_TN[0] * n + _TN[1] * t + _TN[2] * h <= _TOFF)
    if y < EF_CH:                                   # 45° bevel: keep what is >= (EF_CH - y) inside the section at EF_CH
        s = s & _tree_le(xp, x, EF_CH, z, -(EF_CH - y))
    # Support columns: each disc is stamped into a small window round its column (2026-11-03), not tested against the
    # whole slice (328 full-frame passes per layer made the GPU export 8.7 min). x is (1, W), z is (h, 1), evenly spaced
    # (x may run backwards: the mirrored piece passes _W - x).
    s = xp.array(xp.broadcast_to(s, (z.shape[0], x.shape[1])))
    xs, zs = x[0], z[:, 0]
    x0, z0 = float(xs[0]), float(zs[0])
    dx = float(xs[1] - xs[0]) if xs.shape[0] > 1 else 1.0
    dz = float(zs[1] - zs[0]) if zs.shape[0] > 1 else 1.0
    for px, py, pz, pb in PTS:
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
        ja, jb = sorted(((px - r - x0) / dx, (px + r - x0) / dx))
        ia, ib = sorted(((pz - r - z0) / dz, (pz + r - z0) / dz))
        j0, j1 = max(math.floor(ja), 0), min(math.ceil(jb) + 1, s.shape[1])
        i0, i1 = max(math.floor(ia), 0), min(math.ceil(ib) + 1, s.shape[0])
        if j0 >= j1 or i0 >= i1:
            continue
        w = s[i0:i1, j0:j1]
        w |= (x[:, j0:j1] - px) ** 2 + (z[i0:i1] - pz) ** 2 <= r * r
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
