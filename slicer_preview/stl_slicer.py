"""Slice a binary STL into per-layer images as a piece dict.

The piece integrates directly into the generate.py piece system,
so the STL geometry is OR'd with all other pieces via make_global_slice.
"""
import struct
import numpy as np
from printer import PX_MM, PZ_MM, LY_MM


def load_stl(path):
    """Load a binary STL. Returns (N_tris, 3_verts, 3_xyz) float32 array."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tris = struct.unpack('<I', f.read(4))[0]
        dt = np.dtype([
            ('normal', '<f4', 3),
            ('v0', '<f4', 3), ('v1', '<f4', 3), ('v2', '<f4', 3),
            ('attr', '<u2'),
        ])
        data = np.frombuffer(f.read(), dtype=dt, count=n_tris)
    return np.stack([data['v0'], data['v1'], data['v2']], axis=1)


def stl_piece(stl_path, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    """Create a piece dict from a binary STL file.

    rot_x, rot_y, rot_z: Euler rotations in radians, applied in XYZ order
    (matching Three.js Euler XYZ convention).
    """
    tris = load_stl(stl_path).astype(np.float64)

    # Apply rotations in X → Y → Z order (Three.js Euler XYZ)
    if rot_x != 0.0:
        c, s = np.cos(rot_x), np.sin(rot_x)
        y, z = tris[:, :, 1].copy(), tris[:, :, 2].copy()
        tris[:, :, 1] = y * c - z * s
        tris[:, :, 2] = y * s + z * c

    if rot_y != 0.0:
        c, s = np.cos(rot_y), np.sin(rot_y)
        x, z = tris[:, :, 0].copy(), tris[:, :, 2].copy()
        tris[:, :, 0] = x * c + z * s
        tris[:, :, 2] = -x * s + z * c

    if rot_z != 0.0:
        c, s = np.cos(rot_z), np.sin(rot_z)
        x, y = tris[:, :, 0].copy(), tris[:, :, 1].copy()
        tris[:, :, 0] = x * c - y * s
        tris[:, :, 1] = x * s + y * c

    # Bounding box in world coords
    x_min, x_max = tris[:, :, 0].min(), tris[:, :, 0].max()
    y_min, y_max = tris[:, :, 1].min(), tris[:, :, 1].max()
    z_min, z_max = tris[:, :, 2].min(), tris[:, :, 2].max()

    W = int(np.ceil((x_max - x_min) / PX_MM))
    H = int(np.ceil((z_max - z_min) / PZ_MM))
    N_SLICES = int(np.ceil((y_max - y_min) / LY_MM))

    print(f"STL: {len(tris)} triangles, "
          f"bbox {x_max-x_min:.1f}×{y_max-y_min:.1f}×{z_max-z_min:.1f} mm, "
          f"image {W}×{H}, {N_SLICES} slices")

    # Pre-compute edge arrays for fast slicing
    # 3 edges per triangle: (v0→v1), (v1→v2), (v2→v0)
    ev0 = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]], axis=0)  # (3N, 3)
    ev1 = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]], axis=0)  # (3N, 3)
    edge_tri = np.tile(np.arange(len(tris)), 3)                         # (3N,)

    ey0, ey1 = ev0[:, 1], ev1[:, 1]
    ey_lo, ey_hi = np.minimum(ey0, ey1), np.maximum(ey0, ey1)

    # Precompute row Z centers for vectorized scanline
    z_centers = z_min + (np.arange(H) + 0.5) * PZ_MM

    def make_slice(layer):
        y_mm = y_min + (layer + 0.5) * LY_MM
        img = np.zeros((H, W), dtype=np.uint8)

        # ── 1. Find edges crossing this Y plane (half-open interval) ──
        crosses = (ey_lo < y_mm) & (ey_hi >= y_mm)
        if not crosses.any():
            return img

        ci = np.where(crosses)[0]
        t = (y_mm - ey0[ci]) / (ey1[ci] - ey0[ci])
        ix = ev0[ci, 0] + t * (ev1[ci, 0] - ev0[ci, 0])
        iz = ev0[ci, 2] + t * (ev1[ci, 2] - ev0[ci, 2])
        tri_ids = edge_tri[ci]

        # ── 2. Group crossings by triangle → segments ──
        order = np.argsort(tri_ids, kind='mergesort')
        tri_ids = tri_ids[order]
        ix, iz = ix[order], iz[order]

        breaks = np.where(tri_ids[1:] != tri_ids[:-1])[0] + 1
        g_start = np.concatenate([[0], breaks])
        g_end = np.concatenate([breaks, [len(tri_ids)]])
        g_size = g_end - g_start

        valid = g_size == 2
        vs = g_start[valid]
        if len(vs) == 0:
            return img

        sx0, sz0 = ix[vs], iz[vs]
        sx1, sz1 = ix[vs + 1], iz[vs + 1]

        # ── 3. Vectorized scanline rasterize ──
        sz_lo = np.minimum(sz0, sz1)
        sz_hi = np.maximum(sz0, sz1)

        # Row range per segment (half-open: sz_lo < z_row <= sz_hi)
        r_lo = np.searchsorted(z_centers, sz_lo, side='right')
        r_hi = np.searchsorted(z_centers, sz_hi, side='right')
        n_rows = np.maximum(0, r_hi - r_lo)
        total = n_rows.sum()
        if total == 0:
            return img

        # Expand (segment, row) pairs into flat arrays
        seg_idx = np.repeat(np.arange(len(sx0)), n_rows)
        row_idx = np.concatenate([np.arange(a, b) for a, b in
                                  zip(r_lo[n_rows > 0], r_hi[n_rows > 0])])

        # Compute all x crossings at once
        z_at = z_centers[row_idx]
        dz = sz1[seg_idx] - sz0[seg_idx]
        tz = np.where(dz != 0, (z_at - sz0[seg_idx]) / dz, 0.0)
        x_cross = sx0[seg_idx] + tz * (sx1[seg_idx] - sx0[seg_idx])
        col_f = (x_cross - x_min) / PX_MM

        # Sort by (row, x_crossing)
        order2 = np.lexsort((col_f, row_idx))
        row_s = row_idx[order2]
        col_s = col_f[order2]

        # Find row group boundaries and position within each group
        row_change = np.empty(len(row_s), dtype=bool)
        row_change[0] = True
        row_change[1:] = row_s[1:] != row_s[:-1]
        group_start = np.where(row_change)[0]
        group_id = np.cumsum(row_change) - 1
        pos_in_group = np.arange(len(row_s)) - group_start[group_id]

        # Even-odd pairs: even positions are span starts, odd are span ends
        si = np.where(pos_in_group % 2 == 0)[0]
        ei = si + 1
        ok = (ei < len(row_s)) & (row_s[si] == row_s[np.minimum(ei, len(row_s) - 1)])
        si, ei = si[ok], ei[ok]

        fill_r = row_s[si]
        fill_c0 = np.clip(np.floor(col_s[si]).astype(np.int32), 0, W)
        fill_c1 = np.clip(np.ceil(col_s[ei]).astype(np.int32), 0, W)
        good = fill_c0 < fill_c1

        for r, c0, c1 in zip(fill_r[good].tolist(), fill_c0[good].tolist(),
                              fill_c1[good].tolist()):
            img[r, c0:c1] = 255

        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_min,
        OFFSET_Y_MM=y_min,
        OFFSET_Z_MM=z_min,
        make_slice=make_slice,
    )
