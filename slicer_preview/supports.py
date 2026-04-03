"""Generate lattice support structures for 3D printing.

Fills the model bounding box with a TPMS (lidinoid) lattice, then
subtracts a clearance shell around the model.

Architecture for full-resolution support:
  Pass 1 — Scan model at reduced resolution (SCAN_SCALE × coarser)
            to build a height map and cache clearance slices.
  Pass 2 — Generate TPMS lattice analytically at full resolution,
            subtract clearance.  Parallelised across N_WORKERS cores.
  Pass 3 — Sequential floater detection + classification.

Usage (standalone):  cd slicer_preview && python supports.py
Usage (importable):  from supports import generate_supports
"""
import sys, os, struct, json, time
import numpy as np
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from printer import PX_MM, PZ_MM, LY_MM, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX

# ── Support Parameters ──────────────────────────────────────────────
SUPPORT_GRAY = 128            # grayscale value (viewer renders as teal)
MODEL_GROUNDED_GRAY = 200     # viewer renders yellow/orange
MIN_MODEL_GROUNDED_VOL_MM3 = 0.5
CLEARANCE_LAYERS = 1          # vertical gap (layers at full res)
CLEARANCE_XZ_MM = 0.4         # lateral clearance (mm)
LATTICE_SPACING_MM = 3.0      # TPMS period (mm)
TPMS_THRESHOLD = 0.3          # iso-surface thickness
SCAN_SCALE = 4                # spatial downsample factor for model scanning
SCAN_LAYER_STEP = 4           # layer step for model scanning
N_WORKERS = 8                 # parallel workers for pass 2


# ── TPMS ────────────────────────────────────────────────────────────

def _tpms_lidinoid_layer(layer, W, H, threshold=TPMS_THRESHOLD):
    """Evaluate the lidinoid TPMS for a single layer at full resolution."""
    k = 2.0 * np.pi / LATTICE_SPACING_MM
    sx = np.arange(W, dtype=np.float64) * (PX_MM * k)
    sz = np.arange(H, dtype=np.float64) * (PZ_MM * k)
    y = layer * LY_MM * k

    sin_x, cos_x = np.sin(sx), np.cos(sx)
    sin_2x = np.sin(2 * sx)
    sin_z, cos_z = np.sin(sz), np.cos(sz)
    sin_2z = np.sin(2 * sz)
    sin_y, cos_y, sin_2y = np.sin(y), np.cos(y), np.sin(2 * y)

    f = (sin_2x[None, :] * cos_y * sin_z[:, None] +
         sin_2y * cos_z[:, None] * sin_x[None, :] +
         sin_2z[:, None] * cos_x[None, :] * sin_y)

    img = np.zeros((H, W), dtype=np.uint8)
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


# ── Worker for parallel pass 2 ──────────────────────────────────────

def _process_layer(args):
    """Worker: generate TPMS, apply height cull + clearance for one layer.

    Args is a tuple to work with Pool.imap_unordered:
        (layer, W, H, height_map, exclusion_mask)
    where exclusion_mask is the pre-dilated model clearance at this layer.
    """
    layer, W, H, height_map, exclusion = args

    below_model = layer < height_map
    if not below_model.any():
        return layer, None

    lattice = _tpms_lidinoid_layer(layer, W, H)
    lattice[~below_model] = 0
    if exclusion is not None:
        lattice[exclusion] = 0

    if lattice.any():
        return layer, lattice
    return layer, None


# ── Public API ──────────────────────────────────────────────────────

def generate_supports(make_slice, W, H, N_SLICES,
                      offset_x_mm, offset_y_mm, offset_z_mm,
                      make_scan_slice=None, scan_W=None, scan_H=None,
                      scan_N_SLICES=None):
    """Generate lattice supports with model clearance subtraction.

    If make_scan_slice is provided, pass 1 uses it for the height map
    scan instead of downsampling the full-res make_slice.  This avoids
    computing expensive model slices at full resolution during scanning.
    scan_W, scan_H, scan_N_SLICES are the dimensions of the scan slices.
    """

    clearance_r_x = max(0, int(round(CLEARANCE_XZ_MM / PX_MM)))
    clearance_r_z = max(0, int(round(CLEARANCE_XZ_MM / PZ_MM)))

    # Determine scan resolution
    have_scan_fn = make_scan_slice is not None
    if have_scan_fn:
        sW, sH, sN = scan_W, scan_H, (scan_N_SLICES or N_SLICES)
        sc = max(1, round(W / sW))
    else:
        sc = max(1, SCAN_SCALE)
        sW, sH, sN = (W + sc - 1) // sc, (H + sc - 1) // sc, N_SLICES

    scan_step = max(1, SCAN_LAYER_STEP)
    layer_ratio = N_SLICES / sN if sN != N_SLICES else 1.0
    scan_cl_x = max(0, int(round(CLEARANCE_XZ_MM / (PX_MM * sc))))
    scan_cl_z = max(0, int(round(CLEARANCE_XZ_MM / (PZ_MM * sc))))

    print(f"  TPMS lidinoid: {LATTICE_SPACING_MM}mm period, "
          f"threshold={TPMS_THRESHOLD}", flush=True)
    print(f"  Clearance: {CLEARANCE_XZ_MM}mm XZ, "
          f"{CLEARANCE_LAYERS} layer(s) vertical", flush=True)
    mode = "native scan piece" if have_scan_fn else f"{sc}× downsample"
    print(f"  Scan: {sW}×{sH} ({mode}), "
          f"every {scan_step} layers, {N_WORKERS} workers", flush=True)

    # ── Pass 1: scan model at reduced resolution ──
    print(f"  Pass 1: scanning model ...", flush=True)
    t0 = time.time()

    scan_sampled = list(range(0, sN, scan_step))
    if scan_sampled[-1] != sN - 1:
        scan_sampled.append(sN - 1)

    height_map_scan = np.full((sH, sW), -1, dtype=np.int32)
    model_cache_scan = {}

    for i, scan_layer in enumerate(scan_sampled):
        if have_scan_fn:
            solid = make_scan_slice(scan_layer) > 0
            full_layer = int(round(scan_layer * layer_ratio))
        else:
            full_slice = make_slice(scan_layer)
            if sc > 1:
                ph = ((H + sc - 1) // sc) * sc
                pw = ((W + sc - 1) // sc) * sc
                padded = np.zeros((ph, pw), dtype=np.uint8)
                padded[:H, :W] = full_slice
                solid = padded.reshape(sH, sc, sW, sc).max(axis=(1, 3)) > 0
            else:
                solid = full_slice > 0
            full_layer = scan_layer

        model_cache_scan[full_layer] = solid
        height_map_scan[solid] = full_layer

        if (i + 1) % 50 == 0 or i + 1 == len(scan_sampled):
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(scan_sampled)} sampled  [{elapsed:.1f}s]",
                  flush=True)

    max_model_layer = int(height_map_scan.max())
    if max_model_layer < 0:
        print("  No model geometry found.")
        return []

    height_map = height_map_scan.repeat(sc, axis=0).repeat(sc, axis=1)[:H, :W]

    cached_layers = np.array(sorted(model_cache_scan.keys()), dtype=np.int32)

    def _nearest_scan_slice(layer):
        idx = np.searchsorted(cached_layers, layer)
        if idx >= len(cached_layers):
            return model_cache_scan[int(cached_layers[-1])]
        if idx == 0:
            return model_cache_scan[int(cached_layers[0])]
        if layer - cached_layers[idx - 1] <= cached_layers[idx] - layer:
            return model_cache_scan[int(cached_layers[idx - 1])]
        return model_cache_scan[int(cached_layers[idx])]

    print(f"  Height map: max layer {max_model_layer}, "
          f"scan res {sW}×{sH}", flush=True)

    # ── Pre-compute exclusion masks at full resolution ──
    # Build per-layer exclusion from scan-res model, then upsample + dilate
    plate_layer = max(0, int(round(-offset_y_mm / LY_MM)))
    layers_to_process = [l for l in range(plate_layer, max_model_layer + 1)
                         if (l < height_map).any()]

    print(f"  Building exclusion masks ({len(layers_to_process)} layers) ...",
          flush=True)
    t1 = time.time()

    exclusion_masks = {}
    for layer in layers_to_process:
        lo = max(0, layer - CLEARANCE_LAYERS)
        hi = min(N_SLICES, layer + CLEARANCE_LAYERS + 1)
        excl_scan = np.zeros((scan_H, scan_W), dtype=bool)
        for l in range(lo, hi):
            excl_scan |= _nearest_scan_slice(l)

        if scan_cl_x > 0 or scan_cl_z > 0:
            excl_scan = _dilate_mask(excl_scan, scan_cl_x, scan_cl_z)

        # Upsample to full resolution
        excl_full = excl_scan.repeat(sc, axis=0).repeat(sc, axis=1)[:H, :W]
        # Fine-tune dilation at full res for remaining sub-block clearance
        remainder_x = clearance_r_x - scan_cl_x * sc
        remainder_z = clearance_r_z - scan_cl_z * sc
        if remainder_x > 0 or remainder_z > 0:
            excl_full = _dilate_mask(excl_full, max(0, remainder_x),
                                     max(0, remainder_z))

        exclusion_masks[layer] = excl_full

    elapsed1 = time.time() - t1
    print(f"  Exclusion masks: {elapsed1:.1f}s", flush=True)

    del model_cache_scan

    # ── Pass 2: TPMS + subtraction (parallel) ──
    print(f"  Pass 2: TPMS + subtraction ({N_WORKERS} workers) ...",
          flush=True)
    t2 = time.time()

    work_items = [(layer, W, H, height_map, exclusion_masks.get(layer))
                  for layer in layers_to_process]

    support_slices = {}
    n_done = 0

    with Pool(N_WORKERS) as pool:
        for layer, result in pool.imap_unordered(_process_layer, work_items):
            if result is not None:
                support_slices[layer] = result
            n_done += 1
            if n_done % 200 == 0 or n_done == len(work_items):
                elapsed = time.time() - t2
                print(f"    {n_done}/{len(work_items)}  "
                      f"({len(support_slices)} support layers)  "
                      f"[{elapsed:.1f}s]", flush=True)

    del exclusion_masks

    if not support_slices:
        print("  No support material needed.")
        return []

    elapsed2 = time.time() - t2
    print(f"  Pass 2: {elapsed2:.1f}s", flush=True)

    # ── Pass 3A: plate-grounded flood fill ──
    print(f"  Pass 3A: plate-grounded flood fill ...", flush=True)
    t3 = time.time()

    min_layer = min(support_slices.keys())
    max_layer = max(support_slices.keys())
    n_total_layers = max_layer - min_layer + 1

    plate_connected = np.zeros((H, W), dtype=bool)
    layer_data = {}

    for layer in range(min_layer, max_layer + 1):
        if layer not in support_slices:
            plate_connected[:] = False
            continue

        support_mask = support_slices[layer] > 0
        layer_connected = support_mask & plate_connected
        if layer == min_layer:
            layer_connected = support_mask.copy()

        while True:
            prev_count = int(layer_connected.sum())
            dilated = _dilate_mask(layer_connected, 1, 1)
            layer_connected |= support_mask & dilated
            if int(layer_connected.sum()) == prev_count:
                break

        plate_connected = layer_connected
        layer_data[layer] = (support_slices[layer], layer_connected)

        done = layer - min_layer + 1
        if done % 200 == 0 or done == n_total_layers:
            elapsed = time.time() - t3
            print(f"    3A: {done}/{n_total_layers}  [{elapsed:.1f}s]", flush=True)

    # ── Pass 3B: model-grounded flood fill ──
    print(f"  Pass 3B: model-grounded flood fill ...", flush=True)
    t3b = time.time()
    model_connected = np.zeros((H, W), dtype=bool)

    # Rebuild a lightweight model presence check from the height map
    # (we no longer have model_cache at this point)
    # A pixel has model at layer L if L <= height_map[z, x]
    # For clearance seeding, check if model exists within clearance window below

    for layer in range(min_layer, max_layer + 1):
        if layer not in layer_data:
            model_connected[:] = False
            continue

        img, plate_mask = layer_data[layer]
        support_mask = img > 0

        # Seed: support near model below — use height map as proxy
        # Model exists at (z, x, l) if height_map[z, x] >= l
        # Check if model is present within clearance window below this layer
        check_lo = max(0, layer - CLEARANCE_LAYERS - 1)
        nearby_model = height_map >= check_lo
        # Further restrict: model must actually be BELOW this layer
        nearby_model &= (height_map < layer + CLEARANCE_LAYERS + 1)
        on_model = support_mask & nearby_model

        layer_model_connected = support_mask & (model_connected | on_model)
        while True:
            prev_count = int(layer_model_connected.sum())
            dilated = _dilate_mask(layer_model_connected, 1, 1)
            layer_model_connected |= support_mask & dilated
            if int(layer_model_connected.sum()) == prev_count:
                break

        model_connected = layer_model_connected

        floaters = support_mask & ~plate_mask
        model_grounded = floaters & layer_model_connected
        truly_floating = floaters & ~layer_model_connected

        out = img.copy()
        out[truly_floating] = 0
        out[model_grounded] = MODEL_GROUNDED_GRAY
        layer_data[layer] = out

        done = layer - min_layer + 1
        if done % 200 == 0 or done == n_total_layers:
            elapsed = time.time() - t3b
            print(f"    3B: {done}/{n_total_layers}  [{elapsed:.1f}s]", flush=True)

    # ── Pass 3C: volume-filter model-grounded components ──
    print(f"  Pass 3C: volume-filtering model-grounded ...", flush=True)
    t3c = time.time()
    voxel_vol_mm3 = PX_MM * PZ_MM * LY_MM
    min_voxels = max(1, int(round(MIN_MODEL_GROUNDED_VOL_MM3 / voxel_vol_mm3)))

    class _UF:
        def __init__(self):
            self.parent = {}
            self.size = {}
        def make(self, x):
            self.parent[x] = x
            self.size[x] = 0
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, a, b):
            a, b = self.find(a), self.find(b)
            if a == b: return a
            if self.size[a] < self.size[b]: a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
            return a

    uf = _UF()
    next_id = 1
    label_maps = {}
    prev_labels = np.zeros((H, W), dtype=np.int32)

    for layer in range(min_layer, max_layer + 1):
        if layer not in layer_data:
            prev_labels = np.zeros((H, W), dtype=np.int32)
            continue

        img = layer_data[layer]
        mg_mask = img == MODEL_GROUNDED_GRAY
        if not mg_mask.any():
            prev_labels = np.zeros((H, W), dtype=np.int32)
            label_maps[layer] = prev_labels
            continue

        labels = np.zeros((H, W), dtype=np.int32)
        inherit = mg_mask & (prev_labels > 0)
        labels[inherit] = prev_labels[inherit]

        unlabeled = mg_mask & (labels == 0)
        zs, xs = np.nonzero(unlabeled)
        for z, x in zip(zs.tolist(), xs.tolist()):
            labels[z, x] = next_id
            uf.make(next_id)
            next_id += 1

        for dz, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            sz = slice(max(0, dz), H + min(0, dz))
            sx_s = slice(max(0, dx), W + min(0, dx))
            nz = slice(max(0, -dz), H + min(0, -dz))
            nx = slice(max(0, -dx), W + min(0, -dx))
            here = labels[sz, sx_s]
            there = labels[nz, nx]
            both = (here > 0) & (there > 0) & (here != there)
            if both.any():
                for a, b in zip(here[both].tolist(), there[both].tolist()):
                    uf.union(a, b)

        unique_labels = np.unique(labels[labels > 0])
        remap = {int(l): uf.find(int(l)) for l in unique_labels}
        for old, new in remap.items():
            if old != new:
                labels[labels == old] = new

        for l in np.unique(labels[labels > 0]).tolist():
            root = uf.find(l)
            uf.size[root] = uf.size.get(root, 0) + int((labels == l).sum())

        label_maps[layer] = labels
        prev_labels = labels

        done = layer - min_layer + 1
        if done % 200 == 0 or done == n_total_layers:
            elapsed = time.time() - t3c
            print(f"    3C: {done}/{n_total_layers}  "
                  f"({next_id-1} components)  [{elapsed:.1f}s]", flush=True)

    surviving_roots = set()
    for root, vol in uf.size.items():
        if vol >= min_voxels:
            surviving_roots.add(uf.find(root))

    n_deleted = 0
    n_kept = 0
    for layer in range(min_layer, max_layer + 1):
        if layer not in layer_data:
            continue
        img = layer_data[layer]
        if layer in label_maps:
            labels = label_maps[layer]
            mg_mask = img == MODEL_GROUNDED_GRAY
            if mg_mask.any():
                keep = np.zeros((H, W), dtype=bool)
                for l in np.unique(labels[mg_mask]).tolist():
                    if l > 0 and uf.find(l) in surviving_roots:
                        keep |= (labels == l)
                delete = mg_mask & ~keep
                n_deleted += int(delete.sum())
                n_kept += int(keep.sum())
                if delete.any():
                    img = img.copy()
                    img[delete] = 0
                    layer_data[layer] = img

        support_slices[layer] = layer_data[layer]

    del label_maps

    elapsed3 = time.time() - t3
    print(f"  Model-grounded: {n_kept:,} kept, {n_deleted:,} removed "
          f"(min {MIN_MODEL_GROUNDED_VOL_MM3}mm³ = {min_voxels} voxels)",
          flush=True)
    print(f"  Pass 3: {elapsed3:.1f}s", flush=True)

    # ── Build piece ──
    for layer in list(support_slices.keys()):
        if not (support_slices[layer] > 0).any():
            del support_slices[layer]

    if not support_slices:
        print("  All support removed after cleanup.")
        return []

    min_layer = min(support_slices.keys())
    max_layer = max(support_slices.keys())
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
    print(f"  Done: {len(local_slices)} layers, "
          f"~{total_voxels:,} support voxels, {elapsed:.1f}s total", flush=True)

    return [dict(
        W=W, H=H, N_SLICES=n_slices,
        OFFSET_X_MM=offset_x_mm,
        OFFSET_Y_MM=piece_offset_y,
        OFFSET_Z_MM=offset_z_mm,
        make_slice=make_support_slice,
    )]


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
