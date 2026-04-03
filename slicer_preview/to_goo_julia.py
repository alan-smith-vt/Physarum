"""Convert Julia set + supports → .goo file for printing.

Patches printer to real resolution, builds the Julia piece at full res,
generates supports at reduced scan resolution, then multiprocess-encodes
all layers into a .goo file.

Support slices are written to a temp .npy file and memory-mapped by
workers (avoids pickle size limits on Windows).  Completed layer RLEs
are checkpointed to disk so a crashed run can resume.

Usage:
    cd slicer_preview && python to_goo_julia.py
"""
import sys, struct, time, os, tempfile, pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# ── Patch printer to real resolution BEFORE importing anything else ──
import printer
GOO_LAYER_UM = 50.0
GOO_EXPOSURE_S = 2.3
printer.PREVIEW_SCALE = 1
printer.PIXEL_X_UM = printer.REAL_PIXEL_X_UM
printer.PIXEL_Y_UM = printer.REAL_PIXEL_Y_UM
printer.LAYER_UM = GOO_LAYER_UM
printer.PX_MM = printer.PIXEL_X_UM / 1000
printer.PZ_MM = printer.PIXEL_Y_UM / 1000
printer.LY_MM = printer.LAYER_UM / 1000

sys.path.insert(0, str(Path(__file__).parent.parent / 'chromosome'))
from goo_punch_hole_v2 import parse_goo
from generate_julia import (quaternion_julia_piece, make_global_slice,
                             _compute_globals, PIECES,
                             W, H, N_SLICES,
                             OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM)

SCRIPT_DIR = Path(__file__).parent
TEMPLATE = SCRIPT_DIR.parent / 'Siraya Tech Test Model 2021 v5.STL_0.05_2_2026_03_28_20_51_00.goo'
OUTPUT   = SCRIPT_DIR / 'julia.goo'
CHECKPOINT = SCRIPT_DIR / '_julia_checkpoint.pkl'
NUM_LAYERS_OFFSET = 195310

# ── Shared state for workers (set by _init_worker) ──
_sup_file = None    # path to memory-mapped support file
_sup_shape = None   # (n_layers, H, W)
_sup_dy = 0
_sup_dx = 0
_sup_dz = 0
_sup_mmap = None    # the mmap array, opened per-worker


def _init_worker(sup_file, sup_shape, dy, dx, dz):
    global _sup_file, _sup_shape, _sup_dy, _sup_dx, _sup_dz, _sup_mmap
    _sup_file = sup_file
    _sup_shape = sup_shape
    _sup_dy = dy
    _sup_dx = dx
    _sup_dz = dz
    if sup_file and os.path.exists(sup_file):
        _sup_mmap = np.memmap(sup_file, dtype=np.uint8, mode='r',
                              shape=sup_shape)


def goo_encode_placed(img, x0, y0, plate_w, plate_h):
    """RLE-encode a small image placed on a large plate."""
    h, w = img.shape
    right_pad = plate_w - x0 - w
    between = right_pad + x0

    runs = []
    prefix = y0 * plate_w + x0
    if prefix > 0:
        runs.append((0, prefix))

    for r in range(h):
        row = img[r]
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], changes))
        lens = np.diff(np.concatenate((starts, [w])))
        cols = row[starts]
        for c, l in zip(cols.tolist(), lens.tolist()):
            runs.append((c, l))
        if r < h - 1:
            if between > 0:
                runs.append((0, between))
        else:
            suffix = right_pad + (plate_h - y0 - h) * plate_w
            if suffix > 0:
                runs.append((0, suffix))

    merged = [runs[0]]
    for c, l in runs[1:]:
        if c == merged[-1][0]:
            merged[-1] = (c, merged[-1][1] + l)
        else:
            merged.append((c, l))

    rle = bytearray([0x55])
    prev_color = 0
    for cur, stride in merged:
        first = len(rle)
        rle.append(0)
        diff = abs(cur - prev_color)
        if diff <= 0xF and stride <= 255 and 0 < cur < 255:
            rle[first] = (0b10 << 6) | (diff & 0xF)
            if stride > 1:
                rle[first] |= 0x1 << 4
                rle.append(stride & 0xFF)
            if cur < prev_color:
                rle[first] |= 0x1 << 5
        else:
            if cur == 255:
                rle[first] |= 0b11 << 6
            elif cur > 0:
                rle[first] |= 0b01 << 6
                rle.append(cur)
            rle[first] |= stride & 0xF
            if stride <= 0xF:
                prev_color = cur; continue
            if stride <= 0xFFF:
                rle[first] |= 0b01 << 4
                rle.append((stride >> 4) & 0xFF)
                prev_color = cur; continue
            if stride <= 0xFFFFF:
                rle[first] |= 0b10 << 4
                rle.append((stride >> 12) & 0xFF)
                rle.append((stride >> 4) & 0xFF)
                prev_color = cur; continue
            if stride <= 0xFFFFFFF:
                rle[first] |= 0b11 << 4
                rle.append((stride >> 20) & 0xFF)
                rle.append((stride >> 12) & 0xFF)
                rle.append((stride >> 4) & 0xFF)
        prev_color = cur

    checksum = 0
    for b in rle[1:]:
        checksum = (checksum + b) & 0xFF
    rle.append((~checksum) & 0xFF)
    return bytes(rle)


def _process_layer(args):
    """Worker: composite model + support, RLE-encode one layer."""
    z, x0, y0, W_goo, H_goo = args
    img = make_global_slice(z)

    if _sup_mmap is not None:
        local_layer = z - _sup_dy
        n_sup_layers = _sup_shape[0]
        if 0 <= local_layer < n_sup_layers:
            sup = np.array(_sup_mmap[local_layer])  # copy from mmap
            if sup.any():
                sdx, sdz = _sup_dx, _sup_dz
                sh, sw = sup.shape
                sx0 = max(0, -sdx)
                sz0 = max(0, -sdz)
                sx1 = min(sw, img.shape[1] - sdx)
                sz1 = min(sh, img.shape[0] - sdz)
                if sx0 < sx1 and sz0 < sz1:
                    region = img[sdz + sz0:sdz + sz1, sdx + sx0:sdx + sx1]
                    sup_region = sup[sz0:sz1, sx0:sx1]
                    mask = (region == 0) & (sup_region > 0)
                    region[mask] = sup_region[mask]

    rle = goo_encode_placed(img, x0, y0, W_goo, H_goo)
    return z, rle


def main():
    global W, H, N_SLICES, OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM

    from printer import PX_MM, PZ_MM, LY_MM

    print(f"Julia piece: {W}×{H} px, {N_SLICES} layers at full resolution")
    print(f"Pixel: {printer.PIXEL_X_UM}×{printer.PIXEL_Y_UM} µm, "
          f"layer: {GOO_LAYER_UM} µm")

    # ── Generate supports ──
    from supports import generate_supports, SCAN_SCALE, SCAN_LAYER_STEP

    print("\nGenerating supports ...", flush=True)
    scan_piece = quaternion_julia_piece(
        px_mm=PX_MM * SCAN_SCALE,
        pz_mm=PZ_MM * SCAN_SCALE,
        ly_mm=LY_MM * SCAN_LAYER_STEP)
    print(f"  Scan piece: {scan_piece['W']}×{scan_piece['H']} px, "
          f"{scan_piece['N_SLICES']} layers", flush=True)

    support_pieces = generate_supports(
        make_global_slice, W, H, N_SLICES,
        OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM,
        make_scan_slice=scan_piece['make_slice'],
        scan_W=scan_piece['W'], scan_H=scan_piece['H'],
        scan_N_SLICES=scan_piece['N_SLICES'])

    sup_file = None
    sup_shape = (0, 0, 0)
    sup_dy = sup_dx = sup_dz = 0

    if support_pieces:
        sp = support_pieces[0]
        PIECES.extend(support_pieces)
        new_ox, new_oy, new_oz, new_W, new_H, new_N = _compute_globals(PIECES)

        import generate_julia as gj
        gj.OFFSET_X_MM = new_ox
        gj.OFFSET_Y_MM = new_oy
        gj.OFFSET_Z_MM = new_oz
        gj.W = new_W
        gj.H = new_H
        gj.N_SLICES = new_N
        OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM = new_ox, new_oy, new_oz
        W, H, N_SLICES = new_W, new_H, new_N

        sup_dx = round((sp['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        sup_dz = round((sp['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        sup_dy = round((sp['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
        print(f"  Support offset: dx={sup_dx} dz={sup_dz} dy={sup_dy}")

        # Write support slices to a memory-mapped temp file
        # so workers can read without pickle serialization
        print("  Writing support slices to temp file ...", flush=True)
        n_sup = sp['N_SLICES']
        sup_H, sup_W = sp['H'], sp['W']
        sup_shape = (n_sup, sup_H, sup_W)
        sup_file = str(SCRIPT_DIR / '_support_cache.dat')

        fp = np.memmap(sup_file, dtype=np.uint8, mode='w+', shape=sup_shape)
        for local_layer in range(n_sup):
            fp[local_layer] = sp['make_slice'](local_layer)
            if (local_layer + 1) % 200 == 0 or local_layer + 1 == n_sup:
                print(f"    {local_layer+1}/{n_sup}", flush=True)
        fp.flush()
        del fp
        print(f"  Support cache: {os.path.getsize(sup_file)/1024/1024:.0f} MB",
              flush=True)

    # ── Read template .goo ──
    print(f"\nReading template: {TEMPLATE}")
    data = TEMPLATE.read_bytes()
    header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
    print(f"Plate resolution: {W_goo} x {H_goo}")

    template_def = bytearray(orig_layers[0][0])

    x0 = (W_goo - W) // 2
    y0 = (H_goo - H) // 2
    print(f"Placing {W}x{H} pattern at ({x0}, {y0}) on {W_goo}x{H_goo} plate")
    print(f"{N_SLICES} layers at {GOO_LAYER_UM} µm")

    # ── Load checkpoint if available ──
    results = {}
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, 'rb') as f:
                results = pickle.load(f)
            print(f"Resumed from checkpoint: {len(results)}/{N_SLICES} layers done",
                  flush=True)
        except Exception as e:
            print(f"Checkpoint corrupted, starting fresh: {e}", flush=True)
            results = {}

    remaining = [z for z in range(N_SLICES) if z not in results]
    if not remaining:
        print("All layers already computed!", flush=True)
    else:
        # ── Multiprocess encode ──
        n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
        print(f"Encoding {len(remaining)} layers with {n_workers} workers",
              flush=True)

        t0 = time.time()
        args = [(z, x0, y0, W_goo, H_goo) for z in remaining]
        checkpoint_interval = 50  # save every N layers

        with Pool(n_workers, initializer=_init_worker,
                  initargs=(sup_file, sup_shape, sup_dy, sup_dx, sup_dz)) as pool:
            for z, rle in pool.imap_unordered(_process_layer, args):
                results[z] = rle
                done = len(results)
                if done % 10 == 0 or done == N_SLICES:
                    elapsed = time.time() - t0
                    done_this_run = done - (N_SLICES - len(remaining))
                    rate = done_this_run / elapsed if elapsed > 0 else 0
                    left = N_SLICES - done
                    eta = left / rate if rate > 0 else 0
                    print(f"  {done}/{N_SLICES} layers  ({len(rle):,} bytes RLE)  "
                          f"[{elapsed:.1f}s, ~{eta:.0f}s remaining]", flush=True)

                # Periodic checkpoint
                if done % checkpoint_interval == 0:
                    with open(CHECKPOINT, 'wb') as f:
                        pickle.dump(results, f)

        # Final checkpoint
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump(results, f)

    # ── Assemble .goo file ──
    print("Assembling .goo ...", flush=True)
    new_layers = []
    for z in range(N_SLICES):
        rle = results[z]
        ldef = bytearray(template_def)
        z_mm = (z + 1) * GOO_LAYER_UM / 1000.0
        struct.pack_into('>f', ldef, 2, z_mm)
        struct.pack_into('>f', ldef, 6, z_mm)
        struct.pack_into('>I', ldef, 66, len(rle))
        new_layers.append((bytes(ldef), rle))

    header = bytearray(header_bytes)
    struct.pack_into('>I', header, NUM_LAYERS_OFFSET, N_SLICES)
    LAYER_HEIGHT_OFFSET = NUM_LAYERS_OFFSET + 4+2+2+1+1 + 4+4+4
    struct.pack_into('>f', header, LAYER_HEIGHT_OFFSET, GOO_LAYER_UM / 1000.0)
    EXPOSURE_OFFSET = LAYER_HEIGHT_OFFSET + 4
    struct.pack_into('>f', header, EXPOSURE_OFFSET, GOO_EXPOSURE_S)

    print(f"Writing {OUTPUT}")
    with open(OUTPUT, 'wb') as f:
        f.write(header)
        for ldef, rle in new_layers:
            f.write(ldef)
            f.write(rle)
            f.write(b'\x0d\x0a')
        f.write(footer)

    size = OUTPUT.stat().st_size
    print(f"Done! {size:,} bytes ({size/1024/1024:.1f} MB)")

    # Clean up temp files
    if sup_file and os.path.exists(sup_file):
        os.unlink(sup_file)
    if CHECKPOINT.exists():
        os.unlink(CHECKPOINT)


if __name__ == '__main__':
    main()
