"""Generate lattice support structures for 3D printing.

Fills the model bounding box with a TPMS (lidinoid) lattice, then
subtracts a clearance shell around the model.

Architecture for full-resolution support:
  Pass 1 — Scan model at reduced resolution (SCAN_SCALE × coarser)
            to build a height map and cache clearance slices.
  Pass 2 — Generate TPMS lattice at support resolution
            (SUPPORT_SCALE × coarser), subtract clearance.
            Parallelised across N_WORKERS cores.
  Pass 3 — Sequential floater detection + classification.

Checkpoints are saved after each major pass so that a crash can
resume without re-doing completed work.

Usage (standalone):  cd slicer_preview && python supports.py <design>
Usage (importable):  from supports import generate_supports
"""
import sys, os, struct, json, time
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from printer import PX_MM, PZ_MM, LY_MM, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX

# ── Support Parameters ──────────────────────────────────────────────
SUPPORT_GRAY = 255            # full exposure for structural supports
MODEL_INTERFACE_GRAY = 128    # reduced exposure at model interface (easy removal)
_MG_MARKER = 200              # internal marker for model-grounded (remapped after pass 3C)
INTERFACE_LAYERS = 3          # layers near model to use reduced gray (~0.15mm)
MIN_MODEL_GROUNDED_VOL_MM3 = 0.5
CLEARANCE_LAYERS = 1          # vertical gap (layers at full res)
CLEARANCE_XZ_MM = 0.4         # lateral clearance (mm)
LATTICE_SPACING_MM = 5.0      # TPMS period (mm)
TPMS_THRESHOLD = 0.6          # iso-surface threshold (~1mm walls at 5mm period)
BASE_SOLID_MM = 1.0           # solid base height at plate
BASE_TAPER_MM = 1.0           # taper zone above solid base (thick → normal)
BASE_WALL_MM = 1.5            # wall thickness at bottom of taper (mm)
SCAN_SCALE = 2                # spatial downsample factor for model scanning
SCAN_LAYER_STEP = 2           # layer step for model scanning (1 = every layer)
N_WORKERS = 20                # parallel workers for pass 2
SUPPORT_SCALE = 2             # spatial downsample for support generation (1=full)


# ── TPMS ────────────────────────────────────────────────────────────

def _tpms_lidinoid_layer(layer, W, H, threshold=TPMS_THRESHOLD,
                          px_mm=PX_MM, pz_mm=PZ_MM):
    """Evaluate the lidinoid TPMS for a single layer."""
    k = 2.0 * np.pi / LATTICE_SPACING_MM
    sx = np.arange(W, dtype=np.float64) * (px_mm * k)
    sz = np.arange(H, dtype=np.float64) * (pz_mm * k)
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


def _dilate_1_into(src, dst, tmp):
    """1-pixel box dilation of src into dst.  Pre-allocated, no new arrays."""
    np.copyto(dst, src)
    dst[:, 1:]  |= src[:, :-1]
    dst[:, :-1] |= src[:, 1:]
    np.copyto(tmp, dst)
    dst[1:, :]  |= tmp[:-1, :]
    dst[:-1, :] |= tmp[1:, :]


# ── Worker for parallel pass 2 ──────────────────────────────────────

def _process_layer(args):
    """Worker: generate TPMS, apply height cull + clearance for one layer.

    Args is a tuple to work with Pool.imap_unordered:
        (layer, W, H, height_map, exclusion_mask, plate_layer, px_mm, pz_mm)
    where exclusion_mask is the pre-dilated model clearance at this layer.

    Layers within BASE_SOLID_MM of the plate are filled solid.
    The next BASE_TAPER_MM ramps the TPMS threshold from ~1mm walls
    down to the normal TPMS_THRESHOLD (~0.3mm walls).
    """
    layer, W, H, height_map, exclusion, plate_layer, px_mm, pz_mm = args

    below_model = layer < height_map
    if not below_model.any():
        return layer, None

    dist_mm = (layer - plate_layer) * LY_MM

    if dist_mm < BASE_SOLID_MM:
        # Solid base zone
        lattice = np.full((H, W), SUPPORT_GRAY, dtype=np.uint8)
    elif dist_mm < BASE_SOLID_MM + BASE_TAPER_MM:
        # Taper zone: linearly interpolate threshold from thick to normal
        t = (dist_mm - BASE_SOLID_MM) / BASE_TAPER_MM   # 0→1
        thick_threshold = BASE_WALL_MM * np.pi / LATTICE_SPACING_MM
        threshold = thick_threshold + t * (TPMS_THRESHOLD - thick_threshold)
        lattice = _tpms_lidinoid_layer(layer, W, H, threshold=threshold,
                                            px_mm=px_mm, pz_mm=pz_mm)
    else:
        lattice = _tpms_lidinoid_layer(layer, W, H, px_mm=px_mm, pz_mm=pz_mm)

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

    Supports are generated at SUPPORT_SCALE× reduced resolution and
    upsampled in the output piece.  Intermediate results are checkpointed
    to disk so a crash can resume without repeating completed passes.

    If make_scan_slice is provided, pass 1 uses it for the height map
    scan instead of downsampling the full-res make_slice.
    """
    t_start = time.time()

    # ── Support resolution ──
    ss = SUPPORT_SCALE
    sup_W = (W + ss - 1) // ss
    sup_H = (H + ss - 1) // ss
    sup_px = PX_MM * ss
    sup_pz = PZ_MM * ss

    # ── Checkpoint paths ──
    ckpt_path = os.path.join(SCRIPT_DIR, '_support_checkpoint.json')
    sup_path = os.path.join(SCRIPT_DIR, '_support_work.dat')
    hmap_path = os.path.join(SCRIPT_DIR, '_support_heightmap.dat')

    # ── Scan resolution ──
    have_scan_fn = make_scan_slice is not None
    if have_scan_fn:
        sW, sH, sN = scan_W, scan_H, (scan_N_SLICES or N_SLICES)
        sc = max(1, round(W / sW))
    else:
        sc = max(1, SCAN_SCALE)
        sW, sH, sN = (W + sc - 1) // sc, (H + sc - 1) // sc, N_SLICES

    # When using a native scan piece, the layer step is already baked into
    # the piece's layer height — don't skip again.
    scan_step = 1 if have_scan_fn else max(1, SCAN_LAYER_STEP)
    layer_ratio = N_SLICES / sN if sN != N_SLICES else 1.0
    sup_cl_x = max(0, int(round(CLEARANCE_XZ_MM / sup_px)))
    sup_cl_z = max(0, int(round(CLEARANCE_XZ_MM / sup_pz)))

    plate_layer = max(0, int(round(-offset_y_mm / LY_MM)))

    print(f"  TPMS lidinoid: {LATTICE_SPACING_MM}mm period, "
          f"threshold={TPMS_THRESHOLD}", flush=True)
    print(f"  Clearance: {CLEARANCE_XZ_MM}mm XZ, "
          f"{CLEARANCE_LAYERS} layer(s) vertical", flush=True)
    mode = "native scan piece" if have_scan_fn else f"{sc}× downsample"
    print(f"  Scan: {sW}×{sH} ({mode}), "
          f"every {scan_step} layers, {N_WORKERS} workers", flush=True)
    if ss > 1:
        print(f"  Support resolution: {ss}× → {sup_W}×{sup_H} px "
              f"({sup_px:.3f}×{sup_pz:.3f} mm/px)", flush=True)

    # ── Try loading checkpoint ──
    stage = 0
    min_layer = max_layer = n_total_layers = 0

    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            expected_mm_bytes = (ckpt.get('n_total_layers', 0)
                                 * ckpt.get('sup_H', 0)
                                 * ckpt.get('sup_W', 0))
            sup_ok = (os.path.exists(sup_path) and
                      os.path.getsize(sup_path) >= expected_mm_bytes > 0)
            hmap_ok = (os.path.exists(hmap_path) and
                       os.path.getsize(hmap_path) > 0)
            if (ckpt.get('support_scale') == ss and
                ckpt.get('sup_W') == sup_W and
                ckpt.get('sup_H') == sup_H and
                ckpt.get('full_W') == W and
                ckpt.get('full_H') == H and
                ckpt.get('N_SLICES') == N_SLICES and
                sup_ok and hmap_ok):
                stage = ckpt['stage']
                min_layer = ckpt['min_layer']
                max_layer = ckpt['max_layer']
                n_total_layers = ckpt['n_total_layers']
                plate_layer = ckpt['plate_layer']
                print(f"  Resuming from checkpoint (stage {stage}, "
                      f"layers {min_layer}-{max_layer})", flush=True)
            else:
                print(f"  Checkpoint stale, starting fresh", flush=True)
        except Exception as e:
            print(f"  Checkpoint corrupted: {e}", flush=True)

    def _save_checkpoint(stg):
        with open(ckpt_path, 'w') as f:
            json.dump({
                'stage': stg,
                'support_scale': ss,
                'sup_W': sup_W, 'sup_H': sup_H,
                'full_W': W, 'full_H': H,
                'N_SLICES': N_SLICES,
                'min_layer': min_layer,
                'max_layer': max_layer,
                'n_total_layers': n_total_layers,
                'plate_layer': plate_layer,
            }, f)

    # ================================================================
    # Stage 1-2: Pass 1 (scan) + exclusion + Pass 2 (TPMS)
    # ================================================================
    if stage < 2:
        # ── Pass 1: scan model at reduced resolution ──
        print(f"  Pass 1: scanning model ...", flush=True)
        t0 = time.time()

        scan_sampled = list(range(0, sN, scan_step))
        if scan_sampled[-1] != sN - 1:
            scan_sampled.append(sN - 1)

        height_map_scan = np.full((sH, sW), -1, dtype=np.int32)
        model_cache_scan = {}

        def _scan_one(scan_layer):
            """Evaluate one scan layer (numpy releases GIL)."""
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
            return scan_layer, full_layer, solid

        n_scan = len(scan_sampled)
        n_scan_workers = min(N_WORKERS, n_scan)
        print(f"    {n_scan} layers with {n_scan_workers} threads ...",
              flush=True)
        n_collected = 0
        with ThreadPoolExecutor(max_workers=n_scan_workers) as executor:
            futures = {executor.submit(_scan_one, sl): sl
                       for sl in scan_sampled}
            for future in as_completed(futures):
                _, full_layer, solid = future.result()
                model_cache_scan[full_layer] = solid
                # as_completed returns out of order — only update if higher
                update = solid & (full_layer > height_map_scan)
                height_map_scan[update] = full_layer
                n_collected += 1
                if n_collected % 50 == 0 or n_collected == n_scan:
                    elapsed = time.time() - t0
                    print(f"    {n_collected}/{n_scan} sampled  "
                          f"[{elapsed:.1f}s]", flush=True)

        max_model_layer = int(height_map_scan.max())
        if max_model_layer < 0:
            print("  No model geometry found.")
            return []

        # Convert height map: scan → support resolution
        # (safe copy handles off-by-one between scan/support dims)
        height_map = np.full((sup_H, sup_W), -1, dtype=np.int32)
        if sc == ss:
            h, w = min(sH, sup_H), min(sW, sup_W)
            height_map[:h, :w] = height_map_scan[:h, :w]
        elif sc > ss:
            r = max(1, round(sc / ss))
            up = height_map_scan.repeat(r, axis=0).repeat(r, axis=1)
            h, w = min(up.shape[0], sup_H), min(up.shape[1], sup_W)
            height_map[:h, :w] = up[:h, :w]
        else:
            r = max(1, round(ss / sc))
            ph, pw = sup_H * r, sup_W * r
            padded = np.full((ph, pw), -1, dtype=np.int32)
            sh, sw = min(sH, ph), min(sW, pw)
            padded[:sh, :sw] = height_map_scan[:sh, :sw]
            height_map[:] = padded.reshape(sup_H, r, sup_W, r).max(axis=(1, 3))

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
              f"support res {sup_W}×{sup_H}", flush=True)

        # ── Pre-compute exclusion masks at support resolution ──
        layers_to_process = [l for l in range(plate_layer, max_model_layer + 1)
                             if (l < height_map).any()]

        print(f"  Building exclusion masks ({len(layers_to_process)} layers) ...",
              flush=True)
        t1 = time.time()

        exclusion_masks = {}
        for layer in layers_to_process:
            lo = max(0, layer - CLEARANCE_LAYERS)
            hi = min(N_SLICES, layer + CLEARANCE_LAYERS + 1)
            excl_scan = np.zeros((sH, sW), dtype=bool)
            for l in range(lo, hi):
                excl_scan |= _nearest_scan_slice(l)

            # Convert to support resolution
            excl_sup = np.zeros((sup_H, sup_W), dtype=bool)
            if sc == ss:
                h, w = min(sH, sup_H), min(sW, sup_W)
                excl_sup[:h, :w] = excl_scan[:h, :w]
            elif sc > ss:
                r = max(1, round(sc / ss))
                up = excl_scan.repeat(r, axis=0).repeat(r, axis=1)
                h, w = min(up.shape[0], sup_H), min(up.shape[1], sup_W)
                excl_sup[:h, :w] = up[:h, :w]
            else:
                r = max(1, round(ss / sc))
                ph, pw = sup_H * r, sup_W * r
                padded = np.zeros((ph, pw), dtype=bool)
                sh, sw = min(sH, ph), min(sW, pw)
                padded[:sh, :sw] = excl_scan[:sh, :sw]
                excl_sup[:] = padded.reshape(sup_H, r, sup_W, r).max(axis=(1, 3))

            if sup_cl_x > 0 or sup_cl_z > 0:
                excl_sup = _dilate_mask(excl_sup, sup_cl_x, sup_cl_z)

            exclusion_masks[layer] = excl_sup

        elapsed1 = time.time() - t1
        print(f"  Exclusion masks: {elapsed1:.1f}s", flush=True)
        del model_cache_scan

        # ── Pass 2: TPMS + subtraction (parallel) ──
        print(f"  Pass 2: TPMS + subtraction ({N_WORKERS} workers) ...",
              flush=True)
        t2 = time.time()

        work_items = [(layer, sup_W, sup_H, height_map,
                       exclusion_masks.get(layer), plate_layer,
                       sup_px, sup_pz)
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

        min_layer = min(support_slices.keys())
        max_layer = max(support_slices.keys())
        n_total_layers = max_layer - min_layer + 1

        # ── Write support slices to disk-backed memmap ──
        gb = n_total_layers * sup_H * sup_W / 1e9
        print(f"  Writing {len(support_slices)} layers to memmap "
              f"({n_total_layers}×{sup_H}×{sup_W} = {gb:.1f} GB) ...",
              flush=True)
        sup_mm = np.memmap(sup_path, dtype=np.uint8, mode='w+',
                           shape=(n_total_layers, sup_H, sup_W))
        for layer, img in support_slices.items():
            sup_mm[layer - min_layer] = img
        sup_mm.flush()
        del support_slices

        # Save height map to disk for checkpoint recovery
        hm_mm = np.memmap(hmap_path, dtype=np.int32, mode='w+',
                          shape=(sup_H, sup_W))
        hm_mm[:] = height_map
        hm_mm.flush()
        del hm_mm

        _save_checkpoint(2)
        stage = 2
        print(f"  Checkpoint saved (stage 2)", flush=True)

    else:
        # ── Resume: reload memmap + height map from disk ──
        sup_mm = np.memmap(sup_path, dtype=np.uint8, mode='r+',
                           shape=(n_total_layers, sup_H, sup_W))
        height_map = np.array(np.memmap(hmap_path, dtype=np.int32, mode='r',
                                        shape=(sup_H, sup_W)))

    # ================================================================
    # Stage 3: Pass 3AB — merged plate + model flood fill
    # ================================================================
    if stage < 3:
        print(f"  Pass 3AB: merged flood fill ({n_total_layers} layers) ...",
              flush=True)
        t3 = time.time()

        plate_connected = np.zeros((sup_H, sup_W), dtype=bool)
        model_connected = np.zeros((sup_H, sup_W), dtype=bool)
        dilated = np.empty((sup_H, sup_W), dtype=bool)
        tmp_buf = np.empty((sup_H, sup_W), dtype=bool)

        for layer in range(min_layer, max_layer + 1):
            idx = layer - min_layer
            raw = np.array(sup_mm[idx])       # copy from memmap
            support_mask = raw > 0
            if not support_mask.any():
                plate_connected[:] = False
                model_connected[:] = False
                continue

            # ── 3A: plate-grounded ──
            if layer == min_layer:
                layer_pc = support_mask.copy()
            else:
                layer_pc = support_mask & plate_connected

            while True:
                prev_count = int(layer_pc.sum())
                _dilate_1_into(layer_pc, dilated, tmp_buf)
                layer_pc |= support_mask & dilated
                if int(layer_pc.sum()) == prev_count:
                    break

            np.copyto(plate_connected, layer_pc)

            # ── 3B: model-grounded ──
            check_lo = max(0, layer - CLEARANCE_LAYERS - 1)
            nearby_model = (height_map >= check_lo) & \
                           (height_map < layer + CLEARANCE_LAYERS + 1)
            on_model = support_mask & nearby_model

            layer_mc = support_mask & (model_connected | on_model)
            while True:
                prev_count = int(layer_mc.sum())
                _dilate_1_into(layer_mc, dilated, tmp_buf)
                layer_mc |= support_mask & dilated
                if int(layer_mc.sum()) == prev_count:
                    break

            np.copyto(model_connected, layer_mc)

            # ── Classify ──
            floaters = support_mask & ~layer_pc
            model_grounded = floaters & layer_mc
            truly_floating = floaters & ~layer_mc

            raw[truly_floating] = 0
            raw[model_grounded] = _MG_MARKER
            sup_mm[idx] = raw                  # write back to memmap

            done = layer - min_layer + 1
            if done % 200 == 0 or done == n_total_layers:
                elapsed = time.time() - t3
                print(f"    3AB: {done}/{n_total_layers}  [{elapsed:.1f}s]",
                      flush=True)

        del plate_connected, model_connected, dilated, tmp_buf
        sup_mm.flush()
        elapsed3ab = time.time() - t3
        print(f"  Pass 3AB: {elapsed3ab:.1f}s", flush=True)
        _save_checkpoint(3)
        stage = 3
        print(f"  Checkpoint saved (stage 3)", flush=True)

    # ================================================================
    # Stage 4: Pass 3C — volume-filter model-grounded components
    # ================================================================
    if stage < 4:
        print(f"  Pass 3C: volume-filtering model-grounded ...", flush=True)
        t3c = time.time()
        voxel_vol_mm3 = sup_px * sup_pz * LY_MM
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

        def _label_layer(mg_mask, prev_labels, labels, uf, next_id, count_vols):
            """Label MG pixels, union neighbours, optionally count volumes.
            Returns updated next_id."""
            labels[:] = 0
            inherit = mg_mask & (prev_labels > 0)
            labels[inherit] = prev_labels[inherit]

            unlabeled = mg_mask & (labels == 0)
            zs, xs = np.nonzero(unlabeled)
            for z, x in zip(zs.tolist(), xs.tolist()):
                labels[z, x] = next_id
                uf.make(next_id)
                next_id += 1

            for dz, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                sz  = slice(max(0,  dz), sup_H + min(0,  dz))
                sx  = slice(max(0,  dx), sup_W + min(0,  dx))
                nz  = slice(max(0, -dz), sup_H + min(0, -dz))
                nx  = slice(max(0, -dx), sup_W + min(0, -dx))
                here  = labels[sz, sx]
                there = labels[nz, nx]
                both = (here > 0) & (there > 0) & (here != there)
                if both.any():
                    for a, b in zip(here[both].tolist(), there[both].tolist()):
                        uf.union(a, b)

            unique = np.unique(labels[labels > 0])
            remap = {int(l): uf.find(int(l)) for l in unique}
            for old, new in remap.items():
                if old != new:
                    labels[labels == old] = new

            if count_vols:
                for l in np.unique(labels[labels > 0]).tolist():
                    root = uf.find(l)
                    uf.size[root] = uf.size.get(root, 0) + int((labels == l).sum())

            return next_id

        # Sub-pass 1: label + count volumes
        uf = _UF()
        next_id = 1
        prev_labels = np.zeros((sup_H, sup_W), dtype=np.int32)
        labels = np.zeros((sup_H, sup_W), dtype=np.int32)

        for layer in range(min_layer, max_layer + 1):
            idx = layer - min_layer
            img = np.array(sup_mm[idx])
            mg_mask = img == _MG_MARKER
            if not mg_mask.any():
                prev_labels[:] = 0
                continue
            next_id = _label_layer(mg_mask, prev_labels, labels, uf, next_id,
                                   count_vols=True)
            np.copyto(prev_labels, labels)

            done = layer - min_layer + 1
            if done % 200 == 0 or done == n_total_layers:
                elapsed = time.time() - t3c
                print(f"    3C-1: {done}/{n_total_layers}  "
                      f"({next_id-1} components)  [{elapsed:.1f}s]", flush=True)

        surviving_roots = set()
        for root, vol in uf.size.items():
            if vol >= min_voxels:
                surviving_roots.add(uf.find(root))

        # Sub-pass 2: re-label identically, delete small components
        # Fresh UF so uf.make() doesn't destroy sub-pass 1's structure.
        uf1 = uf
        uf2 = _UF()
        prev_labels[:] = 0
        labels[:] = 0
        next_id2 = 1
        root_map = {}
        n_deleted = 0
        n_kept = 0

        for layer in range(min_layer, max_layer + 1):
            idx = layer - min_layer
            img = np.array(sup_mm[idx])
            mg_mask = img == _MG_MARKER
            if not mg_mask.any():
                prev_labels[:] = 0
                continue
            next_id2 = _label_layer(mg_mask, prev_labels, labels, uf2, next_id2,
                                    count_vols=False)
            np.copyto(prev_labels, labels)

            # Delete pixels whose root is not in surviving_roots
            keep = np.zeros((sup_H, sup_W), dtype=bool)
            for l in np.unique(labels[mg_mask]).tolist():
                if l <= 0:
                    continue
                r2 = uf2.find(l)
                if r2 not in root_map:
                    root_map[r2] = uf1.find(r2)
                if root_map[r2] in surviving_roots:
                    keep |= (labels == l)
            delete = mg_mask & ~keep
            n_del = int(delete.sum())
            n_deleted += n_del
            n_kept += int(mg_mask.sum()) - n_del
            if n_del > 0:
                img[delete] = 0
                sup_mm[idx] = img

            done = layer - min_layer + 1
            if done % 200 == 0 or done == n_total_layers:
                elapsed = time.time() - t3c
                print(f"    3C-2: {done}/{n_total_layers}  [{elapsed:.1f}s]",
                      flush=True)

        sup_mm.flush()
        elapsed3c = time.time() - t3c
        print(f"  Model-grounded: {n_kept:,} kept, {n_deleted:,} removed "
              f"(min {MIN_MODEL_GROUNDED_VOL_MM3}mm³ = {min_voxels} voxels)",
              flush=True)
        print(f"  Pass 3C: {elapsed3c:.1f}s", flush=True)
        _save_checkpoint(4)
        stage = 4
        print(f"  Checkpoint saved (stage 4)", flush=True)

    # ================================================================
    # Build piece: remap gray values, collect layers, upsample output
    # ================================================================
    print(f"  Building support piece ...", flush=True)
    local_slices = {}
    total_voxels = 0

    for layer in range(min_layer, max_layer + 1):
        idx = layer - min_layer
        img = np.array(sup_mm[idx])
        if not (img > 0).any():
            continue

        # Remap _MG_MARKER → interface or structural gray
        mg = img == _MG_MARKER
        if mg.any():
            near_interface = (height_map >= layer - INTERFACE_LAYERS) & \
                             (height_map < layer + CLEARANCE_LAYERS + INTERFACE_LAYERS)
            img[mg & near_interface] = MODEL_INTERFACE_GRAY
            img[mg & ~near_interface] = SUPPORT_GRAY

        if (img > 0).any():
            local_slices[layer - min_layer] = img
            total_voxels += int((img > 0).sum())

    # Flush and close memmap but keep checkpoint files on disk — the caller
    # should call piece['cleanup']() after the final output is written,
    # so a crash in later pipeline stages can still resume supports.
    sup_mm.flush()
    del sup_mm
    _cleanup_files = [sup_path, hmap_path, ckpt_path]

    if not local_slices:
        print("  All support removed after cleanup.")
        return []

    first_local = min(local_slices.keys())
    last_local = max(local_slices.keys())
    n_slices = last_local - first_local + 1
    piece_offset_y = offset_y_mm + (min_layer + first_local) * LY_MM

    # Re-key so local_layer 0 = first occupied layer
    if first_local > 0:
        local_slices = {k - first_local: v for k, v in local_slices.items()}

    empty = np.zeros((H, W), dtype=np.uint8)

    if ss > 1:
        # Support slices are at reduced resolution — upsample on demand
        empty_sup = np.zeros((sup_H, sup_W), dtype=np.uint8)

        def make_support_slice(layer, _s=local_slices, _e=empty,
                                _es=empty_sup, _ss=ss, _H=H, _W=W):
            sup = _s.get(layer, _es)
            if sup is _es:
                return _e
            return sup.repeat(_ss, axis=0).repeat(_ss, axis=1)[:_H, :_W]
    else:
        def make_support_slice(layer, _s=local_slices, _e=empty):
            return _s.get(layer, _e)

    elapsed = time.time() - t_start
    print(f"  Done: {len(local_slices)} layers, "
          f"~{total_voxels:,} support voxels, {elapsed:.1f}s total", flush=True)

    def _cleanup():
        for p in _cleanup_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    return [dict(
        W=W, H=H, N_SLICES=n_slices,
        OFFSET_X_MM=offset_x_mm,
        OFFSET_Y_MM=piece_offset_y,
        OFFSET_Z_MM=offset_z_mm,
        make_slice=make_support_slice,
        cleanup=_cleanup,
    )]


# ── Standalone entry point ──────────────────────────────────────────

def main(name):
    """Preview a design with supports forced on: python supports.py <design>"""
    from compose import load_design, compute_bounds, add_supports, write_preview

    pieces = list(load_design(name).PIECES)
    ox, oy, oz, W, H, N = compute_bounds(pieces)

    print("=" * 60)
    print("Lattice Support Generator")
    print("=" * 60)
    print(f"Model: {W}×{H} px, {N} layers")
    print(f"Pixel pitch: {PX_MM:.4f} × {PZ_MM:.4f} mm, layer: {LY_MM:.4f} mm")
    print()

    t0 = time.time()
    bounds = add_supports(pieces)
    print(f"\nEncoding {len(pieces)} pieces (model + supports) ...")
    blob = write_preview(pieces, bounds)
    print(f"\nDone in {time.time() - t0:.1f}s — {len(blob):,} bytes ({len(blob)/1024/1024:.1f} MB)")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python supports.py <design>')
    main(sys.argv[1])
