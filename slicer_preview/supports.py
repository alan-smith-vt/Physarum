"""Generate lattice support structures for 3D printing.

Fills the model bounding box with a regular lattice, then subtracts
a clearance shell around the model so supports don't fuse to the part.

Algorithm:
  1. Generate a lattice filling the model's bounding box
  2. For each layer, compute the model slice and dilate it by a clearance
     margin to create an exclusion zone
  3. Subtract the exclusion zone from the lattice
  4. Output the remaining lattice as a support piece

Usage (standalone — re-encodes _slices.bin with supports included):
    cd slicer_preview && python supports.py

Usage (importable — returns support piece dicts):
    from supports import generate_supports
    pieces = generate_supports(make_slice_fn, W, H, N_SLICES, ...)
"""
import sys, os, struct, json, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from printer import PX_MM, PZ_MM, LY_MM, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX

# ── Support Parameters ──────────────────────────────────────────────
SUPPORT_GRAY = 128            # grayscale value (viewer renders as teal)
CLEARANCE_LAYERS = 2          # layers of gap between support and model
CLEARANCE_XZ_MM = 0.5         # lateral clearance around model surfaces
LATTICE_SPACING_MM = 3.0      # distance between lattice struts
STRUT_THICKNESS_MM = 0.4      # strut cross-section diameter
TIE_SPACING_LAYERS = None     # computed: horizontal ties every N layers
TIE_SPACING_MM = 3.0          # vertical distance between horizontal ties


# ── Lattice generation ──────────────────────────────────────────────

def _make_lattice_slice_fn(W, H, N_SLICES):
    """Build a diamond/grid lattice filling a W×H×N_SLICES volume.

    Returns a callable  make_lattice(layer) → uint8 (H, W)
    that generates the lattice pattern for each layer.

    Pattern: vertical columns on a grid + diagonal ties between them.
    """
    spacing_px_x = max(2, int(round(LATTICE_SPACING_MM / PX_MM)))
    spacing_px_z = max(2, int(round(LATTICE_SPACING_MM / PZ_MM)))
    strut_r_x = max(1, int(round(STRUT_THICKNESS_MM / 2 / PX_MM)))
    strut_r_z = max(1, int(round(STRUT_THICKNESS_MM / 2 / PZ_MM)))
    tie_layers = max(2, int(round(TIE_SPACING_MM / LY_MM)))

    # Precompute column positions
    col_xs = np.arange(0, W, spacing_px_x)
    col_zs = np.arange(0, H, spacing_px_z)

    # Vertical columns mask (same every layer)
    vert_mask = np.zeros((H, W), dtype=bool)
    for cx in col_xs:
        x0 = max(0, cx - strut_r_x)
        x1 = min(W, cx + strut_r_x + 1)
        for cz in col_zs:
            z0 = max(0, cz - strut_r_z)
            z1 = min(H, cz + strut_r_z + 1)
            vert_mask[z0:z1, x0:x1] = True

    # Horizontal tie layers: struts along X at each col_z, and along Z at each col_x
    def _tie_mask():
        mask = np.zeros((H, W), dtype=bool)
        # X-direction ties (horizontal bars at each Z column position)
        for cz in col_zs:
            z0 = max(0, cz - strut_r_z)
            z1 = min(H, cz + strut_r_z + 1)
            mask[z0:z1, :] = True
        # Z-direction ties (horizontal bars at each X column position)
        for cx in col_xs:
            x0 = max(0, cx - strut_r_x)
            x1 = min(W, cx + strut_r_x + 1)
            mask[:, x0:x1] = True
        return mask

    tie_mask = _tie_mask()

    # Diagonal ties: connect adjacent columns with X-shaped bracing
    # Diagonals shift position linearly with layer within each tie_layers period
    def make_lattice(layer):
        img = np.zeros((H, W), dtype=np.uint8)

        # Vertical columns — always present
        img[vert_mask] = SUPPORT_GRAY

        # Horizontal ties at regular intervals
        phase = layer % tie_layers
        if phase < max(1, strut_r_z):
            img[tie_mask] = SUPPORT_GRAY

        # Diagonal bracing between tie layers
        # Shift columns by a fraction of spacing per layer to create diagonals
        period = tie_layers
        t = (layer % period) / period  # 0..1 within period
        if layer % (period * 2) >= period:
            t = 1.0 - t  # zigzag

        shift_x = int(round(t * spacing_px_x))
        shift_z = int(round(t * spacing_px_z))

        if shift_x != 0 or shift_z != 0:
            shifted = np.roll(np.roll(vert_mask, shift_x, axis=1),
                              shift_z, axis=0)
            img[shifted] = SUPPORT_GRAY

        return img

    return make_lattice


# ── Clearance dilation ──────────────────────────────────────────────

def _dilate_mask(mask, r_x, r_z):
    """Fast approximate dilation using separable box filters."""
    if r_x == 0 and r_z == 0:
        return mask

    out = mask.copy()
    # Dilate along X
    if r_x > 0:
        for dx in range(1, r_x + 1):
            out[:, dx:] |= mask[:, :-dx]
            out[:, :-dx] |= mask[:, dx:]
    # Dilate along Z
    if r_z > 0:
        tmp = out.copy()
        for dz in range(1, r_z + 1):
            out[dz:, :] |= tmp[:-dz, :]
            out[:-dz, :] |= tmp[dz:, :]
    return out


# ── Public API ──────────────────────────────────────────────────────

def generate_supports(make_slice, W, H, N_SLICES,
                      offset_x_mm, offset_y_mm, offset_z_mm):
    """Generate lattice supports with model clearance subtraction.

    Returns list of piece dicts.
    """
    clearance_r_x = max(1, int(round(CLEARANCE_XZ_MM / PX_MM)))
    clearance_r_z = max(1, int(round(CLEARANCE_XZ_MM / PZ_MM)))
    make_lattice = _make_lattice_slice_fn(W, H, N_SLICES)

    print(f"  Lattice: {LATTICE_SPACING_MM}mm spacing, "
          f"{STRUT_THICKNESS_MM}mm struts", flush=True)
    print(f"  Clearance: {CLEARANCE_XZ_MM}mm XZ, "
          f"{CLEARANCE_LAYERS} layers vertical", flush=True)

    # We need model slices offset by CLEARANCE_LAYERS above and below
    # to carve out the vertical clearance.  Cache a sliding window.
    # exclusion[layer] = dilated union of model slices in
    #   [layer - CLEARANCE_LAYERS, layer + CLEARANCE_LAYERS]
    #
    # Process layer by layer, maintaining a ring buffer of model slices.

    window = 2 * CLEARANCE_LAYERS + 1
    model_cache = {}  # layer → bool mask (only keep what's in the window)

    # Precompute the support slices layer by layer
    support_slices = {}

    print(f"  Subtracting model from lattice ({N_SLICES} layers) ...",
          flush=True)
    t0 = time.time()

    for layer in range(N_SLICES):
        # Load model slices into cache for the window
        for l in range(max(0, layer - CLEARANCE_LAYERS),
                       min(N_SLICES, layer + CLEARANCE_LAYERS + 1)):
            if l not in model_cache:
                model_cache[l] = make_slice(l) > 0

        # Evict old entries outside the window
        evict_below = layer - CLEARANCE_LAYERS - 1
        if evict_below in model_cache:
            del model_cache[evict_below]

        # Build exclusion zone: union of model in vertical window, then dilate XZ
        exclusion = np.zeros((H, W), dtype=bool)
        for l in range(max(0, layer - CLEARANCE_LAYERS),
                       min(N_SLICES, layer + CLEARANCE_LAYERS + 1)):
            exclusion |= model_cache[l]

        exclusion = _dilate_mask(exclusion, clearance_r_x, clearance_r_z)

        # Generate lattice and subtract exclusion
        lattice = make_lattice(layer)
        lattice[exclusion] = 0

        # Only store non-empty layers
        if lattice.any():
            support_slices[layer] = lattice

        if (layer + 1) % 200 == 0 or layer + 1 == N_SLICES:
            elapsed = time.time() - t0
            print(f"    {layer+1}/{N_SLICES}  "
                  f"({len(support_slices)} non-empty support layers)  "
                  f"[{elapsed:.1f}s]", flush=True)

    if not support_slices:
        print("  No support material needed (model fills lattice everywhere).")
        return []

    min_layer = min(support_slices.keys())
    max_layer = max(support_slices.keys())
    n_slices = max_layer - min_layer + 1
    piece_offset_y = offset_y_mm + min_layer * LY_MM

    # Build the piece
    local_slices = {}
    for layer, img in support_slices.items():
        local_slices[layer - min_layer] = img

    empty = np.zeros((H, W), dtype=np.uint8)

    def make_support_slice(layer, _s=local_slices, _e=empty):
        return _s.get(layer, _e)

    total_voxels = sum(int(img.astype(bool).sum()) for img in support_slices.values())
    elapsed = time.time() - t0
    print(f"  Support lattice: {len(support_slices)} layers, "
          f"~{total_voxels:,} voxels, {elapsed:.1f}s", flush=True)

    piece = dict(
        W=W, H=H, N_SLICES=n_slices,
        OFFSET_X_MM=offset_x_mm,
        OFFSET_Y_MM=piece_offset_y,
        OFFSET_Z_MM=offset_z_mm,
        make_slice=make_support_slice,
    )
    return [piece]


# ── Standalone entry point ──────────────────────────────────────────

def main():
    from generate import (make_global_slice, W, H, N_SLICES,
                          OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM,
                          PIECES, _compute_globals)

    SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
    META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')

    print("=" * 60)
    print("Lattice Support Generator")
    print("=" * 60)
    print(f"Model: {W}×{H} px, {N_SLICES} layers")
    print(f"Pixel pitch: {PX_MM:.4f} × {PZ_MM:.4f} mm, layer: {LY_MM:.4f} mm")
    print()

    t0 = time.time()
    support_pieces = generate_supports(
        make_global_slice, W, H, N_SLICES,
        OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM)

    if not support_pieces:
        print("Nothing to do.")
        return

    PIECES.extend(support_pieces)
    new_ox, new_oy, new_oz, new_W, new_H, new_N = _compute_globals(PIECES)

    import generate as gen
    gen.OFFSET_X_MM = new_ox
    gen.OFFSET_Y_MM = new_oy
    gen.OFFSET_Z_MM = new_oz
    gen.W = new_W
    gen.H = new_H
    gen.N_SLICES = new_N

    print()
    print(f"Encoding {len(PIECES)} pieces (model + supports) ...")
    blob = gen.encode_all()

    with open(SLICES_FILE, 'wb') as f:
        f.write(blob)

    meta = {
        "width": new_W, "height": new_H, "num_slices": new_N,
        "pixel_x_um": PIXEL_X_UM, "pixel_y_um": PIXEL_Y_UM,
        "layer_um": LAYER_UM,
        "plate_w_mm": PLATE_W_PX * 14.0 / 1000,
        "plate_h_mm": PLATE_H_PX * 19.0 / 1000,
        "offset_x_mm": new_ox,
        "offset_y_mm": new_oy,
        "offset_z_mm": new_oz,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    sz = os.path.getsize(SLICES_FILE)
    print(f"\nDone in {elapsed:.1f}s — {sz:,} bytes ({sz/1024/1024:.1f} MB)")
    print(f"Output: {SLICES_FILE}")


if __name__ == '__main__':
    main()
