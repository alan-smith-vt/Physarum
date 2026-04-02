"""Quaternion Julia set — 3D cross-section (w=0 plane) rendered as print slices.

The quaternion Julia set iterates  q → q² + c  for quaternions
q = a + bi + cj + dk.  Fixing the k-component to 0 gives a 3D solid
whose XY/Z cross-sections we slice layer-by-layer.

Usage:
    python generate_julia.py          # writes _slices.bin + _meta.json
    python -m http.server             # or just run server.py

Tuning:
    C           — the Julia constant quaternion (cx, ci, cj, ck)
    EXTENT_MM   — half-side of the bounding cube in mm
    MAX_ITER    — iteration depth (more → sharper boundary, slower)
    ESCAPE_R    — escape-radius threshold (≥ 2.0)
"""
import struct, json, time, os
import numpy as np

from printer import PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PX_MM, PZ_MM, LY_MM
from helpers import encode_surface_voxels

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE   = os.path.join(SCRIPT_DIR, '_meta.json')

# ── Julia parameters ──────────────────────────────────────────────────────────
# Classic "quaternion Julia" shape; swap in any of the alternatives below.
C = np.array([-0.2,  0.8,  0.0,  0.0], dtype=np.float64)
# Other nice constants:
#   [-0.2,  0.4,  0.4,  0.4]  — bushy, more protrusions
#   [ 0.3,  0.5,  0.4,  0.2]  — asymmetric blob
#   [-0.4,  0.6,  0.6,  0.6]  — large, very intricate

MAX_ITER  = 16      # 12–20 gives good results; higher = sharper, slower
ESCAPE_R  = 2.0     # standard escape radius for quadratic Julia sets
EXTENT_MM = 15.0    # half-side of the cube (→ 30 mm total per axis)


# ── Piece builder ─────────────────────────────────────────────────────────────

def quaternion_julia_piece(extent_mm=EXTENT_MM, c=C,
                           max_iter=MAX_ITER, escape_r=ESCAPE_R):
    """Return a piece dict representing the 3D Quaternion Julia set.

    The Julia space cube  [-1, 1]³  maps to the printer cube
        X ∈ [-extent_mm, +extent_mm]
        Y ∈ [         0, 2*extent_mm]   (build direction)
        Z ∈ [-extent_mm, +extent_mm]

    Each call to make_slice(layer) tests every pixel in the XZ plane at
    that Y-height and returns 255 for points inside the Julia set.
    """
    size_mm   = 2.0 * extent_mm
    W         = int(np.ceil(size_mm / PX_MM))
    H         = int(np.ceil(size_mm / PZ_MM))
    N_SLICES  = int(np.ceil(size_mm / LY_MM))

    # Local pixel → global mm (offset applied externally via OFFSET_*_MM)
    px_local  = np.arange(W, dtype=np.float64) * PX_MM   # 0 … size_mm  (W,)
    pz_local  = np.arange(H, dtype=np.float64) * PZ_MM   # 0 … size_mm  (H,)

    # Map local coords to Julia space [-1, 1]
    # OFFSET_X_MM = -extent_mm, so global x = -extent_mm + px_local
    qx_row = (px_local - extent_mm) / extent_mm           # (W,)
    qz_col = (pz_local - extent_mm) / extent_mm           # (H,)

    c0, c1, c2, c3 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
    er2 = escape_r * escape_r

    def make_slice(layer):
        # Y coord in Julia space: Y goes 0 → 2*extent_mm → qy ∈ [-1, +1]
        y_mm = (layer + 0.5) * LY_MM
        qy   = (y_mm - extent_mm) / extent_mm

        # Quaternion: q = qx + qy·i + qz·j + 0·k  (broadcast → (H, W))
        a = np.broadcast_to(qx_row[None, :], (H, W)).copy()
        b = np.full((H, W), qy, dtype=np.float64)
        c_ = np.broadcast_to(qz_col[:, None], (H, W)).copy()
        d = np.zeros((H, W), dtype=np.float64)

        escaped = np.zeros((H, W), dtype=bool)

        for _ in range(max_iter):
            active = ~escaped
            if not active.any():
                break
            # q² = (a² - b² - c² - d²) + 2ab·i + 2ac·j + 2ad·k
            a2  = a*a - b*b - c_*c_ - d*d + c0
            b2  = 2.0*a*b            + c1
            c2_ = 2.0*a*c_           + c2
            d2  = 2.0*a*d            + c3
            # Only update pixels that haven't escaped yet
            a  = np.where(active, a2,  a)
            b  = np.where(active, b2,  b)
            c_ = np.where(active, c2_, c_)
            d  = np.where(active, d2,  d)

            norm_sq = a*a + b*b + c_*c_ + d*d
            escaped |= active & (norm_sq > er2)

        inside = (~escaped).astype(np.uint8) * 255
        return inside

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=-extent_mm,
        OFFSET_Y_MM=0.0,
        OFFSET_Z_MM=-extent_mm,
        make_slice=make_slice,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

PIECES = [quaternion_julia_piece()]

# Global bounding box
p = PIECES[0]
OFFSET_X_MM = p['OFFSET_X_MM']
OFFSET_Y_MM = p['OFFSET_Y_MM']
OFFSET_Z_MM = p['OFFSET_Z_MM']
W, H, N_SLICES = p['W'], p['H'], p['N_SLICES']


def encode_all():
    all_chunks = []
    total = 0
    for pi, piece in enumerate(PIECES):
        t0 = time.time()
        dx = round((piece['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        dy = round((piece['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
        dz = round((piece['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        chunks, n = encode_surface_voxels(
            piece['make_slice'], piece['W'], piece['H'], piece['N_SLICES'],
            dx, dz, dy)
        all_chunks.extend(chunks)
        total += n
        print(f"  piece {pi+1}/{len(PIECES)}  {n:,} voxels  ({time.time()-t0:.1f}s)",
              flush=True)
    return struct.pack('<I', total) + b''.join(all_chunks)


if __name__ == '__main__':
    print(f"Quaternion Julia set  c={C.tolist()}  max_iter={MAX_ITER}", flush=True)
    print(f"Volume: {2*EXTENT_MM:.0f}³ mm  →  {W}×{H}×{N_SLICES} voxels", flush=True)
    t0 = time.time()
    blob = encode_all()

    with open(SLICES_FILE, 'wb') as f:
        f.write(blob)

    meta = {
        "width":       W,
        "height":      H,
        "num_slices":  N_SLICES,
        "pixel_x_um":  PIXEL_X_UM,
        "pixel_y_um":  PIXEL_Y_UM,
        "layer_um":    LAYER_UM,
        "offset_x_mm": OFFSET_X_MM,
        "offset_y_mm": OFFSET_Y_MM,
        "offset_z_mm": OFFSET_Z_MM,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    print(f"Done in {time.time()-t0:.1f}s  ({len(blob):,} bytes)", flush=True)
