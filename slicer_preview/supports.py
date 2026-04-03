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
FLOATER_GRAY = 5              # viewer maps 1-10 → blue (disconnected, nothing below)
MODEL_GROUNDED_GRAY = 200     # viewer renders yellow/orange (sits on model surface)
CLEARANCE_LAYERS = 1          # layers of gap between support and model
CLEARANCE_XZ_MM = 0.2         # lateral clearance around model surfaces
LATTICE_SPACING_MM = 3.0      # distance between lattice struts
STRUT_THICKNESS_MM = 0.4      # strut cross-section diameter
TIE_SPACING_MM = 3.0          # vertical distance between horizontal ties


# ── TPMS Lattice generation ─────────────────────────────────────────

TPMS_THRESHOLD = 0.3          # iso-surface thickness (0 = surface only, higher = thicker)

def _tpms_init(W, H):
    """Precompute XZ coordinate arrays scaled to TPMS period.

    Returns (sx, sz, period_px_x, period_px_z) where sx/sz are
    the scaled coordinate arrays and period_px is the cell size in pixels.
    """
    period_mm = LATTICE_SPACING_MM
    period_px_x = max(2, int(round(period_mm / PX_MM)))
    period_px_z = max(2, int(round(period_mm / PZ_MM)))

    # Scale pixel coords to [0, 2π) per period
    k_x = 2.0 * np.pi / period_mm
    k_z = 2.0 * np.pi / period_mm

    sx = np.arange(W, dtype=np.float64) * PX_MM * k_x   # (W,)
    sz = np.arange(H, dtype=np.float64) * PZ_MM * k_z   # (H,)

    return sx, sz, period_px_x, period_px_z


def _tpms_lidinoid(layer, sx, sz, threshold=TPMS_THRESHOLD):
    """Evaluate the lidinoid TPMS at a given layer, return uint8 image.

    Lidinoid implicit surface:
      sin(2x)cos(y)sin(z) + sin(2y)cos(z)sin(x) + sin(2z)cos(x)sin(y) = 0

    Solid where |f(x,y,z)| < threshold.
    """
    k_y = 2.0 * np.pi / LATTICE_SPACING_MM
    y = layer * LY_MM * k_y

    # Precompute trig (1D arrays, broadcast to 2D)
    sin_x = np.sin(sx)       # (W,)
    cos_x = np.cos(sx)
    sin_2x = np.sin(2 * sx)
    sin_z = np.sin(sz)       # (H,)
    cos_z = np.cos(sz)
    sin_2z = np.sin(2 * sz)

    sin_y = np.sin(y)        # scalar
    cos_y = np.cos(y)
    sin_2y = np.sin(2 * y)

    # f(x,y,z) = sin(2x)cos(y)sin(z) + sin(2y)cos(z)sin(x) + sin(2z)cos(x)sin(y)
    # Broadcast: (H, W)
    f = (sin_2x[None, :] * cos_y * sin_z[:, None] +
         sin_2y * cos_z[:, None] * sin_x[None, :] +
         sin_2z[:, None] * cos_x[None, :] * sin_y)

    img = np.zeros((len(sz), len(sx)), dtype=np.uint8)
    img[np.abs(f) < threshold] = SUPPORT_GRAY
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

    print(f"  TPMS lidinoid: {LATTICE_SPACING_MM}mm period, "
          f"threshold={TPMS_THRESHOLD}", flush=True)
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

    # ── Cell-aware height map ──
    # Coarse height per grid cell, dilated by 1 cell so a TPMS unit cell
    # stays as long as any neighboring cell needs support.
    sx, sz, sp_x, sp_z = _tpms_init(W, H)
    col_xs = np.arange(0, W, sp_x)
    col_zs = np.arange(0, H, sp_z)
    n_cx, n_cz = len(col_xs), len(col_zs)

    # Max model layer per grid cell region
    cell_height = np.full((n_cz, n_cx), -1, dtype=np.int32)
    for ci, cx in enumerate(col_xs):
        x0 = max(0, cx - sp_x // 2)
        x1 = min(W, cx + sp_x // 2 + 1)
        for cj, cz in enumerate(col_zs):
            z0 = max(0, cz - sp_z // 2)
            z1 = min(H, cz + sp_z // 2 + 1)
            region = height_map[z0:z1, x0:x1]
            if region.size > 0:
                cell_height[cj, ci] = int(region.max())

    # Dilate by 1 cell (max of 3×3 neighborhood)
    padded = np.pad(cell_height, 1, mode='constant', constant_values=-1)
    dilated_ch = cell_height.copy()
    for dz in range(-1, 2):
        for dx in range(-1, 2):
            dilated_ch = np.maximum(
                dilated_ch,
                padded[1 + dz:n_cz + 1 + dz, 1 + dx:n_cx + 1 + dx])

    # Expand back to per-pixel effective height map
    effective_height = np.full((H, W), -1, dtype=np.int32)
    for ci, cx in enumerate(col_xs):
        x0 = max(0, cx - sp_x // 2)
        x1 = min(W, cx + (sp_x + 1) // 2 + 1)
        for cj, cz in enumerate(col_zs):
            z0 = max(0, cz - sp_z // 2)
            z1 = min(H, cz + (sp_z + 1) // 2 + 1)
            effective_height[z0:z1, x0:x1] = np.maximum(
                effective_height[z0:z1, x0:x1],
                dilated_ch[cj, ci])

    print(f"  Cell-aware height: {n_cx}×{n_cz} grid cells", flush=True)

    # ── Pass 2: generate lattice, subtract clearance ──
    print(f"  Pass 2: lattice + subtraction ...", flush=True)
    t1 = time.time()

    support_slices = {}
    n_layers_with_support = 0

    for layer in range(max_model_layer + 1):
        # Cell-aware height cull
        below_model = layer < effective_height
        if not below_model.any():
            continue

        # Generate TPMS lattice
        lattice = _tpms_lidinoid(layer, sx, sz)

        # Mask to only below model (cell-aware)
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

    if not support_slices:
        del model_slices
        print("  No support material needed.")
        return []

    # ── Pass 3: detect floaters + classify by what's below ──
    # Two flood fills:
    #   A) Bottom-up from build plate → finds plate-grounded support (teal)
    #   B) Bottom-up from model surfaces → finds model-grounded floaters (yellow)
    #   Anything left = truly disconnected floater (blue)
    print(f"  Pass 3: detecting floaters ...", flush=True)
    t2 = time.time()

    min_layer = min(support_slices.keys())
    max_layer = max(support_slices.keys())
    n_floater_voxels = 0
    n_model_grounded = 0

    # A) Plate-grounded: flood fill from build plate upward
    plate_connected = np.zeros((H, W), dtype=bool)

    for layer in range(min_layer, max_layer + 1):
        if layer not in support_slices:
            plate_connected[:] = False
            continue

        support_mask = support_slices[layer] > 0

        # Seed from below
        layer_connected = support_mask & plate_connected

        # Build plate: all support is grounded
        if layer == min_layer:
            layer_connected = support_mask.copy()

        # Propagate horizontally through support pixels
        while True:
            prev_count = int(layer_connected.sum())
            dilated = _dilate_mask(layer_connected, 1, 1)
            layer_connected |= support_mask & dilated
            if int(layer_connected.sum()) == prev_count:
                break

        plate_connected = layer_connected

        # Store plate-connected mask for this layer (temporarily on the image)
        # We'll do the model-grounded pass next, then combine
        support_slices[layer] = (support_slices[layer], layer_connected)

    # B) Model-grounded: flood fill from model top surfaces upward
    # A support pixel is model-grounded if the layer below it has solid model
    # and the support connects upward from there.
    model_connected = np.zeros((H, W), dtype=bool)

    for layer in range(min_layer, max_layer + 1):
        if layer not in support_slices:
            model_connected[:] = False
            continue

        img, plate_mask = support_slices[layer]
        support_mask = img > 0

        # Seed: support pixels near model below (accounting for clearance gap)
        # Check layers below within the clearance window + 1, and dilate
        # the model mask by the XZ clearance so we catch support pixels
        # that are offset from the model surface.
        nearby_model = np.zeros((H, W), dtype=bool)
        for l in range(max(0, layer - CLEARANCE_LAYERS - 1), layer):
            if l < len(model_slices):
                nearby_model |= model_slices[l]
        if nearby_model.any() and (clearance_r_x > 0 or clearance_r_z > 0):
            nearby_model = _dilate_mask(nearby_model, clearance_r_x + 1,
                                        clearance_r_z + 1)
        on_model = support_mask & nearby_model

        # Also propagate from previously model-connected pixels below
        layer_model_connected = support_mask & (model_connected | on_model)

        # Propagate horizontally
        while True:
            prev_count = int(layer_model_connected.sum())
            dilated = _dilate_mask(layer_model_connected, 1, 1)
            layer_model_connected |= support_mask & dilated
            if int(layer_model_connected.sum()) == prev_count:
                break

        model_connected = layer_model_connected

        # Classify and color
        floaters = support_mask & ~plate_mask
        model_grounded = floaters & layer_model_connected
        truly_floating = floaters & ~layer_model_connected

        n_model_grounded += int(model_grounded.sum())
        n_floater_voxels += int(truly_floating.sum())

        if model_grounded.any() or truly_floating.any():
            out = img.copy()
            out[model_grounded] = MODEL_GROUNDED_GRAY
            out[truly_floating] = FLOATER_GRAY
            support_slices[layer] = out
        else:
            support_slices[layer] = img

    del model_slices

    elapsed2 = time.time() - t2
    print(f"  Model-grounded: {n_model_grounded:,} voxels (yellow)",
          flush=True)
    print(f"  True floaters:  {n_floater_voxels:,} voxels (blue)",
          flush=True)
    print(f"  Pass 3: {elapsed2:.1f}s", flush=True)

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
