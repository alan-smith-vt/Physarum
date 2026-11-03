"""Resin chain-textile study (user 2026-11-02): a small patch of 3DSLA's "Metal Textile" (Thingiverse 6029077,
CC BY 4.0, by 3DSLA) links, cut from the original mesh, with a breakaway spike under every link's bottom ball.

Source: C:/Users/Alan/Documents/3d_prints/testing/3DSLA_MetallTextile_1200_Elements.stl (675 identical links, 5,484
triangles each, hex lattice pitch 66 units; units read as 0.1 mm -> link ~10 x 10.5 x 20 mm, pitch 6.6 mm).
Parameters: SCALE (mm per source unit), SQUASH (extra scale on the links' up axis), N (patch is N x N links).
The patch STL (+ spikes) is written to out/textile_patch.stl and sliced with stl_slicer.stl_piece.
"""
import struct
import os
from pathlib import Path
import numpy as np

SRC = Path(r'C:/Users/Alan/Documents/3d_prints/testing/3DSLA_MetallTextile_1200_Elements.stl')
SCALE, SQUASH, N = 0.1, float(os.environ.get('TEXTILE_SQUASH', '1.0')), 4   # textile_patch_60 sets SQUASH 0.6
OUT = Path(__file__).resolve().parent.parent / 'out' / f'textile_patch_{int(round(SQUASH * 100))}.stl'
PITCH_U = 66.0                                     # source lattice pitch (units)
SPIKE = dict(h=1.5, r0=0.5, r1=0.3, dig=0.3, seg=16)   # mm: height under the ball, base / tip radius, tip dug in

SUPPORTS = False
EXPOSURE_S = 2.8
BOTTOM_EXPOSURE_S = 20.0


def _load(path):
    raw = path.read_bytes()
    n = struct.unpack('<I', raw[80:84])[0]
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    return np.frombuffer(raw, dt, count=n, offset=84)['v'].astype(np.float64)   # (n, 3, 3)


def _links(T):
    """Split into connected links (shared vertices). Returns per-triangle label."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    q = np.round(T.reshape(-1, 3) * 1000).astype(np.int64)
    u, inv = np.unique(q, axis=0, return_inverse=True)
    f = inv.ravel().reshape(-1, 3)
    r = np.concatenate([f[:, 0], f[:, 1]]); c = np.concatenate([f[:, 1], f[:, 2]])
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(u), len(u))), directed=False)
    return lab[f[:, 0]]


def _spike(cx, cy, z_ball, seg):
    """Frustum from z=0 (r0) to the ball bottom + dig (r1), closed both ends. Source frame after scaling (mm)."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    z1 = z_ball + SPIKE['dig']
    b = np.stack([cx + SPIKE['r0'] * np.cos(a), cy + SPIKE['r0'] * np.sin(a), np.zeros(seg)], 1)
    t = np.stack([cx + SPIKE['r1'] * np.cos(a), cy + SPIKE['r1'] * np.sin(a), np.full(seg, z1)], 1)
    cb, ct = np.array([cx, cy, 0.0]), np.array([cx, cy, z1])
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        tris += [(b[i], b[j], t[j]), (b[i], t[j], t[i]), (cb, b[j], b[i]), (ct, t[i], t[j])]
    return np.array(tris)


def build():
    T = _load(SRC)
    lab = _links(T)
    nl = lab.max() + 1
    cmin = np.full((nl, 3), np.inf); cmax = np.full((nl, 3), -np.inf)
    np.minimum.at(cmin, lab, T.min(1)); np.maximum.at(cmax, lab, T.max(1))
    cen = (cmin + cmax) / 2
    c0 = cen[np.argmin(np.linalg.norm(cen[:, :2] - cen[:, :2].mean(0), axis=1))]
    a, b = np.array([PITCH_U, 0.0]), np.array([PITCH_U / 2, PITCH_U * np.sqrt(3) / 2])
    want = [c0[:2] + i * a + j * b for i in range(N) for j in range(N)]
    pick = [int(np.argmin(np.linalg.norm(cen[:, :2] - w, axis=1))) for w in want]
    keep = np.isin(lab, pick)
    P = T[keep].copy()
    z0 = cmin[pick, 2].min()
    P[..., :2] -= c0[:2]
    P[..., 2] -= z0
    P *= SCALE
    P[..., 2] *= SQUASH
    P[..., 2] += SPIKE['h']                         # lift the links onto their spikes
    spikes = []
    for k in pick:                                  # bottom ball = lowest vertices of the link
        L = T[lab == k].reshape(-1, 3)
        bot = L[L[:, 2] <= L[:, 2].min() + 1e-6]
        cx, cy = (bot[:, :2].mean(0) - c0[:2]) * SCALE
        spikes.append(_spike(cx, cy, SPIKE['h'], SPIKE['seg']))
    P = np.concatenate([P] + spikes)
    # links' up (z) -> print up (y): (x, y, z) -> (x, z, -y), a proper rotation (det +1), not a mirror
    R = np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)
    OUT.parent.mkdir(exist_ok=True)
    nrm = np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec = np.zeros(len(R), dt); rec['n'] = nrm; rec['v'] = R
    with open(OUT, 'wb') as f:
        f.write(b'textile_patch'.ljust(80, b' ')); f.write(struct.pack('<I', len(R))); f.write(rec.tobytes())
    return len(pick), len(R)


build()
from stl_slicer import stl_piece    # noqa: E402
PIECES = [stl_piece(str(OUT))]
