"""Face picker (user 2026-11-01): click a flat face of a part in the browser to choose its print base.

    python face_pick.py pi_box_v4_cam_cartridge      -> out/pick/pi_box_v4_cam_cartridge.stl (the part's own frame)
    then open http://localhost:5555/pick/pi_box_v4_cam_cartridge   (server.py), click a face, "use this face"
    -> out/pick/pi_box_v4_cam_cartridge.json  {"normal": outward unit normal, "point": a point on the face, ...}

Supported parts: a module with an SDF ROOT in a frame (a, b, c) with c the scalar 3rd coordinate and sampling ranges
PICK_RANGE = ((a0, a1), (b0, b1), (c0, c1)), or the cartridge (its tree frame (n, t, h)).
"""
import sys
import struct
import importlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / 'out' / 'pick'


def mesh(name, step=0.25):
    sys.path.insert(0, str(HERE))
    M = importlib.import_module('designs.' + name)
    rng = getattr(M, 'PICK_RANGE', None)
    if rng is None:                                    # cartridge tree frame
        rng = ((M.N0, M.N1), (M.T0, M.T1), (0.0, M.H_TOP))
    a = np.arange(rng[0][0], rng[0][1], step)
    b = np.arange(rng[1][0], rng[1][1], step)
    c = np.arange(rng[2][0], rng[2][1] + step, step)
    vol = np.stack([np.broadcast_to(np.asarray(M.ROOT(a[:, None], b[None, :], float(z), np, None, 0)) <= 0,
                                    (len(a), len(b))) for z in c]).astype(np.float32)
    from skimage import measure
    v = np.pad(vol, 1)
    verts, faces, _, _ = measure.marching_cubes(v, 0.5, spacing=(step, step, step))
    verts = verts - step + np.array([c[0], a[0], b[0]])
    tri = verts[faces][..., [1, 2, 0]]                 # (c, a, b) -> (a, b, c)
    tri = tri[:, ::-1]                                 # outward winding
    return tri


def write_stl(path, tri):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, tri
    with open(path, 'wb') as f:
        f.write(b'face_pick'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tri)))
        f.write(rec.tobytes())


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'pi_box_v4_cam_cartridge'
    tri = mesh(name)
    write_stl(OUT / f'{name}.stl', tri)
    print(f'{len(tri):,} triangles -> {OUT / (name + ".stl")}')
