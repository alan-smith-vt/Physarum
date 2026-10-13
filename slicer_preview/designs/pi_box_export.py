"""Export the chosen (battery-in-box) Pi box rig as meshes in the rig frame, for Blender renders.

    python -m designs.pi_box_export            (from slicer_preview/)  -> out/render/*.stl + parts.json

Rig frame: X across the camera walls, Y from the Pi (-) to the battery pod (+), Z up, origin at
the underside centre of the aluminium base plate (mm). Printed parts come from their implicit
solids via marching cubes; the aluminium plate and cross from cage_v2's solids; everything
else from the viewer overlays (converted back from the print frame) plus a GoPro MAX2 envelope.
Printed + aluminium parts are indexed PLY (smooth shading), the rest STL.
Then:  blender -b -P designs/pi_box_render.py
"""
import json
import sys
from pathlib import Path
import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_slicer import load_stl                                  # noqa: E402
from designs import pi_box_bat_core as C                         # noqa: E402
from designs import pi_box_bat_lids as L                         # noqa: E402
from designs import pi_box_bat_rig as R                          # noqa: E402
from designs import cage_v2 as CG                                # noqa: E402

OUT = R.ROOT / 'slicer_preview' / 'out' / 'render'
VOX = 0.35                                                       # mm, marching-cubes grid


def _march(fn, nx, ny, nz, h, to_rig):
    """fn(x_row, y, z_col) -> bool (nz, nx) in local coords; y is the layer axis."""
    xs = ((np.arange(nx) + 0.5) * h)[None, :]
    zs = ((np.arange(nz) + 0.5) * h)[:, None]
    vol = np.zeros((ny + 2, nz + 2, nx + 2), np.float32)
    for j in range(ny):
        vol[j + 1, 1:-1, 1:-1] = np.broadcast_to(fn(xs, (j + 0.5) * h, zs), (nz, nx))
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5, spacing=(h, h, h))
    # vol axes (y, z, x), padded by one cell
    local = np.stack([verts[:, 2] - h / 2, verts[:, 0] - h / 2, verts[:, 1] - h / 2], 1)
    rig = to_rig(local)
    # keep outward normals: if local -> rig is a mirror (det < 0), reverse the winding
    e = np.eye(3) * 10.0
    jac = np.stack([to_rig(np.vstack([np.zeros(3), e[i]]))[1] - to_rig(np.zeros((1, 3)))[0] for i in range(3)], 1)
    if np.linalg.det(jac) < 0:
        faces = faces[:, ::-1]
    # skimage orients faces toward increasing values; our solid is 1 inside, so flip to point outward
    return rig, faces[:, ::-1].copy()


def write_ply(path, verts, faces):
    """Binary little-endian PLY, indexed (keeps shared vertices so Blender can smooth-shade)."""
    verts = np.asarray(verts, '<f4')
    rec = np.zeros(len(faces), dtype=[('n', 'u1'), ('i', '<i4', 3)])
    rec['n'], rec['i'] = 3, faces
    head = '\n'.join(['ply', 'format binary_little_endian 1.0', f'element vertex {len(verts)}',
                      'property float x', 'property float y', 'property float z',
                      f'element face {len(faces)}', 'property list uchar int vertex_indices',
                      'end_header', ''])
    with open(path, 'wb') as f:
        f.write(head.encode('ascii') + verts.tobytes() + rec.tobytes())


def write_stl(path, tris):
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, tris
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + np.uint32(len(tris)).tobytes() + rec.tobytes())


def printed_parts(only=None):
    h = VOX
    if only in ('core_cut_far', 'core_cut_near'):
        keep = (lambda X: X >= 30.0) if only == 'core_cut_far' else (lambda X: X <= -30.0)
        def fn(x, d, z):
            return C.core(x, d, z) & (d >= R.FLANGE) & keep(-(z + C.Z0))
        return {only: _march(fn, int(C.SIZE_X / h), int(C.DEPTH / h), int(C.SIZE_Z / h), h,
                             lambda p: R.from_print(np.stack([p[:, 0] + C.X0, p[:, 1], p[:, 2] + C.Z0], 1)))}
    # core: local print (x, d, z) -> print (x + X0, d, z + Z0) -> rig
    core = _march(C.core, int(C.SIZE_X / h), int(C.DEPTH / h), int(C.SIZE_Z / h), h,
                  lambda p: R.from_print(np.stack([p[:, 0] + C.X0, p[:, 1], p[:, 2] + C.Z0], 1)))
    # floor: local (x = Yr - Y_MIN, y = up from the outer face, z = X + X_OUT)
    ft = L.FLOOR_T + max(L.COLLAR_H, L.CRADLE_H)
    floor = _march(L.floor, int((R.Y_MAX - R.Y_MIN) / h), int(ft / h), int(2 * R.X_OUT / h), h,
                   lambda p: np.stack([p[:, 2] - R.X_OUT, p[:, 0] + R.Y_MIN, p[:, 1] - (R.DEPTH + L.FLOOR_T)], 1))
    # panels: local (x across, y thickness, z down the panel)
    def panel_to_rig(face_y, sign):
        return lambda p: np.stack([p[:, 0] - L.PANEL_W / 2, face_y + sign * p[:, 1],
                                   -(p[:, 2] + R.FLANGE + 0.1)], 1)
    pm = _march(lambda x, y, z: L.panel(x, y, z, grille=False), int(L.PANEL_W / h), int(L.PANEL_T / h) + 1,
                int(L.PANEL_H / h), h, panel_to_rig(R.Y_MIN + R.PANEL_LIP, +1))
    pp = _march(lambda x, y, z: L.panel(x, y, z, grille=False), int(L.PANEL_W / h),
                int(L.PANEL_T / h) + 1, int(L.PANEL_H / h), h, panel_to_rig(R.Y_MAX - R.PANEL_LIP, -1))
    # half-sections without the flange, for looking straight at each camera wall's inside face
    def cut(keep_x):
        def fn(x, d, z):
            X = -(z + C.Z0)                                   # print z -> rig X (battery layout)
            return C.core(x, d, z) & (d >= R.FLANGE) & keep_x(X)
        return _march(fn, int(C.SIZE_X / h), int(C.DEPTH / h), int(C.SIZE_Z / h), h,
                      lambda p: R.from_print(np.stack([p[:, 0] + C.X0, p[:, 1], p[:, 2] + C.Z0], 1)))
    core_cut_near = cut(lambda X: X <= -30.0)                  # just the -X (near) camera wall + web stub
    core_cut_far = cut(lambda X: X >= 30.0)                    # just the +X (far) camera wall + web stub
    return dict(core=core, floor=floor, panel_pi=pm, panel_pod=pp,
                core_cut_near=core_cut_near, core_cut_far=core_cut_far)


def aluminium():
    h = 0.25
    yaw = lambda p: np.stack([-p[:, 2], p[:, 0], p[:, 1]], 1)    # cage frame (connector X) -> box frame
    half = CG.BASE / 2
    plate = _march(lambda x, y, z: CG.base_plate(x - half, y, z - half), int(CG.BASE / h), int(CG.BASE_T / h),
                   int(CG.BASE / h), h, lambda p: yaw(np.stack([p[:, 0] - half, p[:, 1], p[:, 2] - half], 1)))
    span = CG.CROSS_SPAN / 2
    z0 = 6.35 + 60.0 + 6.0                                       # plate top + standoff + spacer
    cross = _march(lambda x, y, z: CG.cross_top(x - span, y, z - span), int(CG.CROSS_SPAN / h), int(CG.CROSS_T / h),
                   int(CG.CROSS_SPAN / h), h,
                   lambda p: yaw(np.stack([p[:, 0] - span, p[:, 1] + z0, p[:, 2] - span], 1)))
    return dict(base_plate=plate, cross=cross), z0 + CG.CROSS_T


def gopro(z_cross_top):
    """GoPro MAX2 envelope from published dims (64 W × 69.7 H × 48.7 D incl. lenses; body ~39 deep)
    in a thin cage, on a CAMVATE 1/4-20 locking adapter. Lenses face ±X here. [ENVELOPE]"""
    adapter = R._cyl((0, 0, 0), 'z', 11.0, z_cross_top, z_cross_top + 20.0, 32)
    b0 = z_cross_top + 20.0
    cage = R._box(-21.5, 21.5, -34.0, 34.0, b0, b0 + 3.0)        # cage base plate
    body = R._box(-19.5, 19.5, -32.0, 32.0, b0 + 3.0, b0 + 3.0 + 69.7)
    zc = b0 + 3.0 + 69.7 * 0.62
    lenses = np.concatenate([R._cyl((0, 0, zc), 'x', 13.0, 19.5, 24.35, 48),
                             R._cyl((0, 0, zc), 'x', 13.0, -24.35, -19.5, 48)])
    return dict(gopro_adapter=adapter, gopro_cage=cage, gopro_body=body, gopro_lenses=lenses)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    only = __import__('os').environ.get('PI_BOX_EXPORT_ONLY')   # e.g. core_cut_far, for quick reviews
    if only:
        v, f = printed_parts(only)[only]
        write_ply(OUT / f'{only}.ply', v, f)
        print(f'{only:12s} {len(f):9,d} tris')
        return
    alu, z_top = aluminium()
    for name, (v, f) in {**printed_parts(), **alu}.items():
        write_ply(OUT / f'{name}.ply', v, f)
        parts[name] = len(f)
        print(f'{name:12s} {len(f):9,d} tris', flush=True)
    for name, tris in gopro(z_top).items():
        write_stl(OUT / f'{name}.stl', tris)
        parts[name] = len(tris)
        print(f'{name:12s} {len(tris):9,d} tris', flush=True)
    for ov in C.OVERLAYS:                                        # back from the print frame
        name = Path(ov['file']).stem
        if name in ('base_plate', 'standoffs_none'):
            continue
        t = load_stl(R.ROOT / ov['file']).astype(np.float64)
        rig = R.from_print(t.reshape(-1, 3)).reshape(-1, 3, 3)
        write_stl(OUT / f'{name}.stl', rig)
        parts[name] = len(t)
        if name == 'cameras':                                    # also one file per camera
            cx = rig[:, :, 0].mean(1)
            ax = np.abs(rig[:, :, 0]).max(1)                     # lens barrels stick out past the board face
            lens = ax > R.X_IN - R.CAM_BOSS_H + 0.5
            write_stl(OUT / 'camera_near.stl', rig[(cx < 0) & ~lens])
            write_stl(OUT / 'camera_far.stl', rig[(cx > 0) & ~lens])
            write_stl(OUT / 'lens_near.stl', rig[(cx < 0) & lens])
            write_stl(OUT / 'lens_far.stl', rig[(cx > 0) & lens])
    (OUT / 'parts.json').write_text(json.dumps(parts, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
