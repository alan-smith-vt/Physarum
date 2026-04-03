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

from printer import PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX, PX_MM, PZ_MM, LY_MM
from helpers import encode_surface_voxels

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE   = os.path.join(SCRIPT_DIR, '_meta.json')

# ── Config ──
GENERATE_SUPPORTS = True      # add lattice supports after slicing

# ── Julia parameters ──────────────────────────────────────────────────────────
# Classic "quaternion Julia" shape; swap in any of the alternatives below.
C = np.array([-0.2,  0.8,  0.0,  0.0], dtype=np.float64)
# Other nice constants:
#   [-0.2,  0.4,  0.4,  0.4]  — bushy, more protrusions
#   [ 0.3,  0.5,  0.4,  0.2]  — asymmetric blob
#   [-0.4,  0.6,  0.6,  0.6]  — large, very intricate

MAX_ITER  = 16      # 12–20 gives good results; higher = sharper, slower
ESCAPE_R  = 2.0     # standard escape radius for quadratic Julia sets
EXTENT_MM = 30.0    # half-side of the cube (→ 60 mm total per axis)
QSPACE    = 1.6     # Julia space half-extent (set can exceed |q|=1 for some c)


# ── Piece builder ─────────────────────────────────────────────────────────────

def quaternion_julia_piece(extent_mm=EXTENT_MM, c=C,
                           max_iter=MAX_ITER, escape_r=ESCAPE_R,
                           qspace=QSPACE):
    """Return a piece dict representing the 3D Quaternion Julia set.

    The Julia space cube  [-qspace, qspace]³  maps to the printer cube
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

    # Map local coords to Julia space [-qspace, qspace]
    qx_row = (px_local - extent_mm) / extent_mm * qspace  # (W,)
    qz_col = (pz_local - extent_mm) / extent_mm * qspace  # (H,)

    c0, c1, c2, c3 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
    er2 = escape_r * escape_r
    c_norm = np.sqrt(c0*c0 + c1*c1 + c2*c2 + c3*c3)
    # Theoretical bound: filled Julia set ⊂ ball of radius R where
    # R = (1 + sqrt(1 + 4|c|)) / 2.  Any q with |q| > R escapes immediately.
    bail_r = (1.0 + np.sqrt(1.0 + 4.0 * c_norm)) / 2.0

    # Minimum |q|² at each XZ pixel (over all qy): qx² + qz² (qy=0 is best case)
    min_q2_xz = qx_row[None, :] ** 2 + qz_col[:, None] ** 2  # (H, W)

    # Adaptive iteration: pixels far inside converge fast, pixels far outside
    # escape fast.  Only pixels near the boundary need full iterations.
    ITER_LOW = 4
    ITER_MID = 8

    def make_slice(layer):
        # Y coord in Julia space
        y_mm = (layer + 0.5) * LY_MM
        qy   = (y_mm - extent_mm) / extent_mm * qspace

        # Early termination: if minimum possible |q|² at this layer
        # exceeds bail_r² for every pixel, the entire slice is empty.
        min_q2_layer = qy * qy  # qx=0, qz=0 is the closest point to origin
        if min_q2_layer > bail_r * bail_r:
            return np.zeros((H, W), dtype=np.uint8)

        # Per-pixel |q|² at this qy
        q2 = min_q2_xz + qy * qy  # (H, W)

        # Pixels clearly outside: |q| > bail_r → will escape iteration 1
        clearly_outside = q2 > bail_r * bail_r

        # If everything is outside, skip
        if clearly_outside.all():
            return np.zeros((H, W), dtype=np.uint8)

        # Quaternion iteration — only on pixels that might be inside
        a = np.broadcast_to(qx_row[None, :], (H, W)).copy()
        b = np.full((H, W), qy, dtype=np.float64)
        c_ = np.broadcast_to(qz_col[:, None], (H, W)).copy()
        d = np.zeros((H, W), dtype=np.float64)

        escaped = clearly_outside.copy()

        # Adaptive: run ITER_LOW first, check if all resolved
        for it in range(max_iter):
            active = ~escaped
            if not active.any():
                break

            # Quaternion square + c
            aa, bb, cc, dd = a*a, b*b, c_*c_, d*d
            a2  = aa - bb - cc - dd + c0
            b2  = 2.0*a*b           + c1
            c2_ = 2.0*a*c_          + c2
            d2  = 2.0*a*d           + c3

            a  = np.where(active, a2,  a)
            b  = np.where(active, b2,  b)
            c_ = np.where(active, c2_, c_)
            d  = np.where(active, d2,  d)

            norm_sq = a*a + b*b + c_*c_ + d*d
            escaped |= active & (norm_sq > er2)

            # Adaptive early-out: after ITER_LOW iterations, check if
            # remaining active pixels are deep inside (norm still small)
            # or near boundary (need more iterations)
            if it == ITER_LOW - 1 and max_iter > ITER_LOW:
                still_active = ~escaped
                if still_active.any():
                    # Pixels with very small norm are deep inside — they
                    # won't escape.  Skip them by marking as "not escaped"
                    # (they stay inside).  Only continue iterating pixels
                    # with intermediate norm (near boundary).
                    deep_inside = still_active & (norm_sq < 0.5 * er2)
                    if deep_inside.any():
                        # These won't escape — stop iterating them
                        # by pretending they escaped (we'll invert at the end)
                        # Actually: just mark them as resolved-inside
                        pass  # let them continue — the active check handles it

                    # If very few pixels remain active, the np.where overhead
                    # dominates.  Check if we can bail entirely.
                    n_active = int(still_active.sum())
                    if n_active == 0:
                        break

        inside = (~escaped).astype(np.uint8) * 255
        return inside

    # Cache slices if memory allows (preview res only, ~150MB for 8x)
    # At full res (1x) a 60mm cube would need ~81GB — skip caching.
    est_bytes = W * H * N_SLICES
    use_cache = est_bytes < 500_000_000  # 500MB threshold

    if use_cache:
        _cache = {}
        def make_slice_cached(layer):
            if layer not in _cache:
                _cache[layer] = make_slice(layer)
            return _cache[layer]
        slice_fn = make_slice_cached
    else:
        slice_fn = make_slice

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=-extent_mm,
        OFFSET_Y_MM=-8.0,
        OFFSET_Z_MM=-extent_mm,
        make_slice=slice_fn,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

PIECES = [quaternion_julia_piece()]

# Global bounding box
p = PIECES[0]
OFFSET_X_MM = p['OFFSET_X_MM']
OFFSET_Y_MM = p['OFFSET_Y_MM']
OFFSET_Z_MM = p['OFFSET_Z_MM']
W, H, N_SLICES = p['W'], p['H'], p['N_SLICES']


def _compute_globals(pieces):
    extents = [(p['OFFSET_X_MM'], p['OFFSET_Y_MM'], p['OFFSET_Z_MM'],
                p['OFFSET_X_MM'] + p['W'] * PX_MM,
                p['OFFSET_Y_MM'] + p['N_SLICES'] * LY_MM,
                p['OFFSET_Z_MM'] + p['H'] * PZ_MM) for p in pieces]
    ox = min(e[0] for e in extents)
    oy = min(e[1] for e in extents)
    oz = min(e[2] for e in extents)
    w = int(np.ceil((max(e[3] for e in extents) - ox) / PX_MM))
    h = int(np.ceil((max(e[5] for e in extents) - oz) / PZ_MM))
    n = int(np.ceil((max(e[4] for e in extents) - oy) / LY_MM))
    return ox, oy, oz, w, h, n


def make_global_slice(layer):
    """Composite all pieces into a single (H, W) image for the given global layer."""
    img = np.zeros((H, W), dtype=np.uint8)
    for p in PIECES:
        dx = round((p['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        dz = round((p['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        dy = round((p['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
        local_layer = layer - dy
        if local_layer < 0 or local_layer >= p['N_SLICES']:
            continue
        piece_img = p['make_slice'](local_layer)
        ph, pw = piece_img.shape
        sx, sz = max(0, -dx), max(0, -dz)
        ex, ez = min(pw, W - dx), min(ph, H - dz)
        if sx >= ex or sz >= ez:
            continue
        region = img[dz + sz:dz + ez, dx + sx:dx + ex]
        np.maximum(region, piece_img[sz:ez, sx:ex], out=region)
    return img


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


def _write_output():
    """Encode all pieces and write _slices.bin + _meta.json."""
    global OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM, W, H, N_SLICES

    if GENERATE_SUPPORTS:
        from supports import generate_supports, SCAN_SCALE, SCAN_LAYER_STEP
        print("Generating supports ...", flush=True)
        # Build a low-res Julia piece for fast model scanning
        from printer import REAL_PIXEL_X_UM, REAL_PIXEL_Y_UM, REAL_LAYER_UM, PREVIEW_SCALE
        scan_scale = SCAN_SCALE * PREVIEW_SCALE  # total downsample from real
        scan_piece = quaternion_julia_piece()
        # The piece was already built at current PREVIEW_SCALE.
        # For scanning we want it even coarser — rebuild at SCAN_SCALE× coarser.
        import printer as _pr
        _orig_px, _orig_pz, _orig_ly = _pr.PX_MM, _pr.PZ_MM, _pr.LY_MM
        _pr.PX_MM = PX_MM * SCAN_SCALE
        _pr.PZ_MM = PZ_MM * SCAN_SCALE
        _pr.LY_MM = LY_MM * SCAN_LAYER_STEP
        scan_piece = quaternion_julia_piece()
        _pr.PX_MM, _pr.PZ_MM, _pr.LY_MM = _orig_px, _orig_pz, _orig_ly

        support_pieces = generate_supports(
            make_global_slice, W, H, N_SLICES,
            OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM,
            make_scan_slice=scan_piece['make_slice'],
            scan_W=scan_piece['W'], scan_H=scan_piece['H'],
            scan_N_SLICES=scan_piece['N_SLICES'])
        if support_pieces:
            PIECES.extend(support_pieces)
            OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM, W, H, N_SLICES = \
                _compute_globals(PIECES)
            print(f"  Added {len(support_pieces)} support piece(s), "
                  f"new bounds: {W}×{H} px, {N_SLICES} layers")

    print("Encoding slices...", flush=True)
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
        "plate_w_mm":  PLATE_W_PX * 14.0 / 1000,
        "plate_h_mm":  PLATE_H_PX * 19.0 / 1000,
        "offset_x_mm": OFFSET_X_MM,
        "offset_y_mm": OFFSET_Y_MM,
        "offset_z_mm": OFFSET_Z_MM,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    return blob


if __name__ == '__main__':
    print(f"Quaternion Julia set  c={C.tolist()}  max_iter={MAX_ITER}", flush=True)
    print(f"Volume: {2*EXTENT_MM:.0f}³ mm  →  {W}×{H}×{N_SLICES} voxels", flush=True)
    t0 = time.time()
    blob = _write_output()
    print(f"Done in {time.time()-t0:.1f}s  ({len(blob):,} bytes)", flush=True)
