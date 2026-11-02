"""v4 camera cartridges printed FACE UP (user 2026-11-01): the face (the rig's "eyes", cosmetic) is the last thing printed
and nothing touches it; conventional supports behind. No fit strip: adjust from this print if the inserts don't fit.

Supports are generated, not hand placed: the cartridge (designs/pi_box_v4_cam_cartridge.ROOT) is sampled at SAMP mm with
print up = +n (out through the face). A voxel is UNSUPPORTED if no resin lies within the 45° cone under it (one sample
of growth per sample of height). Connected unsupported regions smaller than MIN_SPAN across (hole and slot roofs:
they bridge) are left alone. Each remaining region gets support points on a GRID mm grid (its lowest voxel per cell);
each point a column straight down to a raft: tip r TIP_R dug TIP_DIG into the part, cone to r COL_R over TIP_L, foot
flare to r FOOT_R. A column whose path is blocked by the part is dropped (reported). Part LIFT above a RAFT_T raft.
Two cartridges: the -X one is the mirror image (t -> -t), supports mirrored with it.
Viewer: python generate.py pi_box_v4_cart_faceup.  Clear blue + Tenacious, 2.3 s.
"""
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
from geometry import implicit_piece
from designs import pi_box_v4_cam_cartridge as K

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

SAMP = 0.2
MIN_SPAN = 2.0
GRID = 2.0
TIP_R, TIP_DIG, TIP_L, COL_R, FOOT_R, FOOT_L = 0.25, 0.25, 1.5, 0.6, 1.2, 1.0
LIFT, RAFT_T, RAFT_MARGIN = 5.0, 1.0, 2.0
CLEAR = 0.4                                  # column clearance to the part along its path
GAP = 6.0                                    # between the two parts on the plate (t)

_CACHE = Path(__file__).resolve().parents[1] / 'out' / 'cart_faceup_cache.npz'


def _sample():
    """solid[n, t, h] on the SAMP grid (tree frame)."""
    ns = np.arange(K.N0, K.N1 + SAMP, SAMP)
    ts = np.arange(K.T0, K.T1 + SAMP, SAMP)
    hs = np.arange(0.0, K.H_TOP + SAMP, SAMP)
    sol = np.zeros((len(ns), len(ts), len(hs)), bool)
    for j, h in enumerate(hs):
        v = np.asarray(K.ROOT(ns[:, None], ts[None, :], float(h), np, None, 0))
        sol[:, :, j] = np.broadcast_to(v <= 0, (len(ns), len(ts)))
    return ns, ts, hs, sol


def _plan():
    src = Path(K.__file__).stat().st_mtime + Path(__file__).stat().st_mtime
    if _CACHE.exists() and False:
        c = np.load(_CACHE)
        if float(c['src']) == src and float(c['samp']) == SAMP:
            return c['pts'], int(c['dropped']), float(c['nmin']), c['ext']
    ns, ts, hs, sol = _sample()
    st = ndimage.generate_binary_structure(2, 1)
    uns = np.zeros_like(sol)
    for i in range(1, len(ns)):                                    # 45°: one sample of growth per sample of height
        uns[i] = sol[i] & ~ndimage.binary_dilation(sol[i - 1], st)
    occupied = sol.any(axis=(1, 2))
    i0 = int(np.argmax(occupied))
    uns[i0] = sol[i0]                                               # the lowest skin needs support too
    lab, nlab = ndimage.label(uns, ndimage.generate_binary_structure(3, 3))
    pts, dropped, drop_pts = [], 0, []
    for sl, k in zip(ndimage.find_objects(lab), range(1, nlab + 1)):
        span = max((sl[1].stop - sl[1].start), (sl[2].stop - sl[2].start)) * SAMP
        if span < MIN_SPAN:
            continue
        sub = lab[sl] == k
        ii, jj, kk = np.nonzero(sub)
        ii, jj, kk = ii + sl[0].start, jj + sl[1].start, kk + sl[2].start
        cells = {}
        for a, b, c in zip(ii, jj, kk):
            key = (int(ts[b] // GRID), int(hs[c] // GRID))
            if key not in cells or a < cells[key][0]:
                cells[key] = (a, b, c)
        rc = int(np.ceil(CLEAR / SAMP)) + int(np.ceil(COL_R / SAMP))
        for a, b, c in cells.values():
            win = sol[:max(a - int(TIP_L / SAMP), 0), max(b - rc, 0):b + rc + 1, max(c - rc, 0):c + rc + 1]
            if not win.any():
                pts.append((ns[a], ts[b], hs[c], -1.0))            # to the raft
                continue
            # blocked: stand it on the part surface straight below (interior, non-cosmetic), if that span is clear
            below = np.nonzero(sol[:a, b, c])[0]
            if not len(below):
                dropped += 1
                drop_pts.append((ns[a], ts[b], hs[c]))
                continue
            base = int(below[-1])                                  # top of the resin below
            lo, hi = base + int(np.ceil((TIP_L + CLEAR) / SAMP)), a - int(TIP_L / SAMP)
            if hi - lo < 1 or sol[lo:hi, max(b - rc, 0):b + rc + 1, max(c - rc, 0):c + rc + 1].any():
                dropped += 1                                       # too short or crowded: let it bridge
                drop_pts.append((ns[a], ts[b], hs[c]))
                continue
            pts.append((ns[a], ts[b], hs[c], ns[base]))
    pts = np.array(pts, float).reshape(-1, 4)
    nmin = float(ns[i0])
    tt, hh = np.nonzero(sol.any(0))
    ext = np.array([ts[tt].min(), ts[tt].max(), hs[hh].min(), hs[hh].max()])
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE, pts=pts, dropped=dropped, nmin=nmin, ext=ext, src=src, samp=SAMP, drop_pts=np.array(drop_pts).reshape(-1, 3))
    return pts, dropped, nmin, ext


PTS, DROPPED, N_MIN, EXT = _plan()
N_RAFT = N_MIN - LIFT                                           # raft top (tree n)


def _solid(xp, n, t, h):
    """Cartridge + supports + raft in the tree frame (arrays broadcast)."""
    s = xp.asarray(K.ROOT(n, t, h, xp, None, 0)) <= 0
    s = s | ((n >= N_RAFT - RAFT_T) & (n <= N_RAFT) & (t >= EXT[0] - RAFT_MARGIN) & (t <= EXT[1] + RAFT_MARGIN)
             & (h >= EXT[2] - RAFT_MARGIN) & (h <= EXT[3] + RAFT_MARGIN))
    for pn, pt, ph, pb in PTS:
        r2 = (t - pt) ** 2 + (h - ph) ** 2
        top = pn + TIP_DIG
        rr = xp.where(n > top - TIP_L, TIP_R + (COL_R - TIP_R) * (top - n) / TIP_L, COL_R)
        if pb < 0:                                                    # raft foot
            rr = xp.where(n < N_RAFT + FOOT_L, COL_R + (FOOT_R - COL_R) * (N_RAFT + FOOT_L - n) / FOOT_L, rr)
            bot = N_RAFT - 0.01
        else:                                                         # tip down onto the part as well
            bot = pb - TIP_DIG
            rr = xp.where(n < bot + TIP_L, TIP_R + (COL_R - TIP_R) * (n - bot) / TIP_L, rr)
        s = s | ((n >= bot) & (n <= top) & (r2 <= rr * rr))
    return s


def _xp(a):
    try:
        import cupy as cp
        return cp.get_array_module(a)
    except ImportError:
        return np


_W = EXT[1] - EXT[0] + 2 * RAFT_MARGIN
_D = EXT[3] - EXT[2] + 2 * RAFT_MARGIN
_HGT = (K.N1 - N_RAFT) + RAFT_T


def _piece(sign, z0):
    def solid(x, y, z):
        xp = _xp(x)
        n = y + N_RAFT - RAFT_T
        t = sign * (x + EXT[0] - RAFT_MARGIN) if sign > 0 else -(x + EXT[0] - RAFT_MARGIN) + (EXT[0] + EXT[1])
        h = z + EXT[2] - RAFT_MARGIN
        return _solid(xp, n, t, h)
    return implicit_piece(_W, _HGT, _D, solid, x0=-_W - GAP / 2 if sign > 0 else GAP / 2, z0=z0)


PIECES = [_piece(1, -_D / 2), _piece(-1, -_D / 2)]
print(f'cart_faceup: {int((PTS[:, 3] < 0).sum())} raft columns + {int((PTS[:, 3] >= 0).sum())} part-to-part per part, {DROPPED} dropped')


# viewer overlay: the dropped (unsupported, left to bridge) spots on the +t part, as small red cubes
def _drop_overlay():
    from designs import pi_box_bat_rig as R
    c = np.load(_CACHE)
    d = c['drop_pts'] if 'drop_pts' in c else np.zeros((0, 3))
    if not len(d):
        return []
    x0 = -_W - GAP / 2
    tris = np.concatenate([R._box(x0 + t - EXT[0] + RAFT_MARGIN - 0.3, x0 + t - EXT[0] + RAFT_MARGIN + 0.3,
                                  n - (N_RAFT - RAFT_T) - 0.6, n - (N_RAFT - RAFT_T),
                                  -_D / 2 + h - EXT[2] + RAFT_MARGIN - 0.3, -_D / 2 + h - EXT[2] + RAFT_MARGIN + 0.3)
                           for n, t, h in d])
    path = R.OVERLAY_DIR / 'cart_faceup_unsupported.stl'
    R._write_stl(path, tris)
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=1.0,
                 label=f'unsupported spots left to bridge ({len(d)}), left part')]


OVERLAYS = _drop_overlay()
