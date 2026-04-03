"""Generate lattice support structures for 3D printing.

Fills the model bounding box with a regular lattice, then subtracts
a clearance shell around the model so supports don't fuse to the part.

Algorithm:
  1. Pre-scan model to build a height map (max solid layer per pixel)
  2. Generate lattice only below the model surface at each XZ position
  3. Subtract a dilated clearance shell around the model
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
FLOATER_GRAY = 5              # viewer maps 1-10 → blue (disconnected pieces)
CLEARANCE_LAYERS = 1          # layers of gap between support and model
CLEARANCE_XZ_MM = 0.2         # lateral clearance around model surfaces
LATTICE_SPACING_MM = 3.0      # distance between lattice struts
STRUT_THICKNESS_MM = 0.4      # strut cross-section diameter
TIE_SPACING_MM = 3.0          # vertical distance between horizontal ties


# ── Lattice generation ──────────────────────────────────────────────

def _precompute_lattice_masks(W, H):
    """Precompute the three lattice mask components.

    Returns (vert_mask, tie_mask, spacing_px_x, spacing_px_z, tie_layers).
    """
    spacing_px_x = max(2, int(round(LATTICE_SPACING_MM / PX_MM)))
    spacing_px_z = max(2, int(round(LATTICE_SPACING_MM / PZ_MM)))
    strut_r_x = max(1, int(round(STRUT_THICKNESS_MM / 2 / PX_MM)))
    strut_r_z = max(1, int(round(STRUT_THICKNESS_MM / 2 / PZ_MM)))
    tie_layers = max(2, int(round(TIE_SPACING_MM / LY_MM)))

    col_xs = np.arange(0, W, spacing_px_x)
    col_zs = np.arange(0, H, spacing_px_z)

    # Vertical columns
    vert_mask = np.zeros((H, W), dtype=bool)
    for cx in col_xs:
        x0, x1 = max(0, cx - strut_r_x), min(W, cx + strut_r_x + 1)
        for cz in col_zs:
            z0, z1 = max(0, cz - strut_r_z), min(H, cz + strut_r_z + 1)
            vert_mask[z0:z1, x0:x1] = True

    # Horizontal ties
    tie_mask = np.zeros((H, W), dtype=bool)
    for cz in col_zs:
        z0, z1 = max(0, cz - strut_r_z), min(H, cz + strut_r_z + 1)
        tie_mask[z0:z1, :] = True
    for cx in col_xs:
        x0, x1 = max(0, cx - strut_r_x), min(W, cx + strut_r_x + 1)
        tie_mask[:, x0:x1] = True

    return vert_mask, tie_mask, spacing_px_x, spacing_px_z, strut_r_z, tie_layers


def _lattice_for_layer(layer, vert_mask, tie_mask,
                       spacing_px_x, spacing_px_z, strut_r_z, tie_layers):
    """Generate lattice pattern for a single layer (no allocation of new masks)."""
    H, W = vert_mask.shape
    img = np.zeros((H, W), dtype=np.uint8)

    # Vertical columns
    img[vert_mask] = SUPPORT_GRAY

    # Horizontal ties at regular intervals
    if layer % tie_layers < max(1, strut_r_z):
        img[tie_mask] = SUPPORT_GRAY

    # Diagonal bracing — zigzag shift of columns
    period = tie_layers
    t = (layer % period) / period
    if layer % (period * 2) >= period:
        t = 1.0 - t

    shift_x = int(round(t * spacing_px_x))
    shift_z = int(round(t * spacing_px_z))
    if shift_x != 0 or shift_z != 0:
        shifted = np.roll(vert_mask, shift_x, axis=1)
        if shift_z != 0:
            shifted = np.roll(shifted, shift_z, axis=0)
        img[shifted] = SUPPORT_GRAY

    return img


# ── Clearance dilation ──────────────────────────────────────────────

def _dilate_mask(mask, r_x, r_z):
    """Fast dilation using separable shifts."""
    if r_x == 0 and r_z == 0:
        return mask
    out = mask.copy()
    if r_x > 0:
        for dx in range(1, r_x + 1):
            out[:, dx:] |= mask[:, :-dx]
            out[:, :-dx] |= mask[:, dx:]
    if r_z > 0:
        tmp = out.copy()
        for dz in range(1, r_z + 1):
            out[dz:, :] |= tmp[:-dz, :]
            out[:-dz, :] |= tmp[dz:, :]
    return out


# ── Public API ──────────────────────────────────────────────────────

def generate_supports(make_slice, W, H, N_SLICES,
                      offset_x_mm, offset_y_mm, offset_z_mm):
    """Generate lattice supports with model clearance subtraction."""

    clearance_r_x = max(0, int(round(CLEARANCE_XZ_MM / PX_MM)))
    clearance_r_z = max(0, int(round(CLEARANCE_XZ_MM / PZ_MM)))

    print(f"  Lattice: {LATTICE_SPACING_MM}mm spacing, "
          f"{STRUT_THICKNESS_MM}mm struts", flush=True)
    print(f"  Clearance: {CLEARANCE_XZ_MM}mm XZ ({clearance_r_x}/{clearance_r_z} px), "
          f"{CLEARANCE_LAYERS} layer(s) vertical", flush=True)

    # ── Pass 1: scan model, build height map + cache slices ──
    print(f"  Pass 1: scanning model ({N_SLICES} layers) ...", flush=True)
    t0 = time.time()

    height_map = np.full((H, W), -1, dtype=np.int32)  # max solid layer per pixel
    model_slices = []  # list of bool arrays, indexed by layer

    for layer in range(N_SLICES):
        solid = make_slice(layer) > 0
        model_slices.append(solid)
        height_map[solid] = layer

        if (layer + 1) % 200 == 0 or layer + 1 == N_SLICES:
            elapsed = time.time() - t0
            print(f"    {layer+1}/{N_SLICES}  [{elapsed:.1f}s]", flush=True)

    max_model_layer = int(height_map.max())
    if max_model_layer < 0:
        print("  No model geometry found.")
        return []

    print(f"  Height map: max layer {max_model_layer}, "
          f"{(height_map >= 0).sum():,} solid pixels", flush=True)

    # ── Pass 2: generate lattice, subtract clearance ──
    print(f"  Pass 2: lattice + subtraction ...", flush=True)
    t1 = time.time()

    vert_mask, tie_mask, sp_x, sp_z, sr_z, tie_ly = _precompute_lattice_masks(W, H)

    support_slices = {}
    n_layers_with_support = 0

    for layer in range(max_model_layer + 1):
        # Height-map cull: only keep lattice below model surface
        below_model = layer < height_map
        if not below_model.any():
            continue

        # Generate lattice
        lattice = _lattice_for_layer(layer, vert_mask, tie_mask,
                                     sp_x, sp_z, sr_z, tie_ly)

        # Mask to only below model
        lattice[~below_model] = 0

        # Build exclusion zone from model slices in vertical window
        lo = max(0, layer - CLEARANCE_LAYERS)
        hi = min(N_SLICES, layer + CLEARANCE_LAYERS + 1)
        exclusion = np.zeros((H, W), dtype=bool)
        for l in range(lo, hi):
            exclusion |= model_slices[l]

        # Dilate XZ
        if clearance_r_x > 0 or clearance_r_z > 0:
            exclusion = _dilate_mask(exclusion, clearance_r_x, clearance_r_z)

        # Subtract
        lattice[exclusion] = 0

        if lattice.any():
            support_slices[layer] = lattice
            n_layers_with_support += 1

        if (layer + 1) % 200 == 0 or layer + 1 == max_model_layer + 1:
            elapsed = time.time() - t1
            print(f"    {layer+1}/{max_model_layer+1}  "
                  f"({n_layers_with_support} support layers)  "
                  f"[{elapsed:.1f}s]", flush=True)

    # Free model slice cache
    del model_slices

    if not support_slices:
        print("  No support material needed.")
        return []

    # ── Pass 3: detect floaters via bottom-up flood fill ──
    print(f"  Pass 3: detecting floaters ...", flush=True)
    t2 = time.time()

    min_layer = min(support_slices.keys())
    max_layer = max(support_slices.keys())
    n_floater_voxels = 0

    # connected_below tracks which XZ pixels at the previous layer
    # were reachable from the build plate
    connected_below = np.zeros((H, W), dtype=bool)

    for layer in range(min_layer, max_layer + 1):
        if layer not in support_slices:
            connected_below[:] = False
            continue

        support_mask = support_slices[layer] > 0

        # Seed: pixels connected from below (vertical continuity)
        layer_connected = support_mask & connected_below

        # Layer 0 (build plate): all support is grounded
        if layer == min_layer:
            layer_connected = support_mask.copy()

        # Propagate horizontally through support pixels in this layer.
        # Iterate single-pixel dilation until stable so connectivity
        # floods through horizontal ties between columns.
        while True:
            prev_count = int(layer_connected.sum())
            dilated = _dilate_mask(layer_connected, 1, 1)
            layer_connected |= support_mask & dilated
            if int(layer_connected.sum()) == prev_count:
                break

        # Mark disconnected pixels
        floaters = support_mask & ~layer_connected
        if floaters.any():
            n_float = int(floaters.sum())
            n_floater_voxels += n_float
            img = support_slices[layer].copy()
            img[floaters] = FLOATER_GRAY
            support_slices[layer] = img

        connected_below = layer_connected

    elapsed2 = time.time() - t2
    print(f"  Floaters: {n_floater_voxels:,} voxels marked [{elapsed2:.1f}s]",
          flush=True)

    # ── Build piece ──
    n_slices = max_layer - min_layer + 1
    piece_offset_y = offset_y_mm + min_layer * LY_MM

    local_slices = {}
    for layer, img in support_slices.items():
        local_slices[layer - min_layer] = img
    del support_slices

    empty = np.zeros((H, W), dtype=np.uint8)

    def make_support_slice(layer, _s=local_slices, _e=empty):
        return _s.get(layer, _e)

    total_voxels = sum(int((img > 0).sum()) for img in local_slices.values())
    elapsed = time.time() - t0
    print(f"  Done: {n_layers_with_support} layers, "
          f"~{total_voxels:,} support voxels "
          f"({n_floater_voxels:,} floaters), {elapsed:.1f}s total", flush=True)

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
