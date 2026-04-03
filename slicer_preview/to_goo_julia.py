"""Convert Julia set + supports → .goo file for printing.

Patches printer to real resolution, builds the Julia piece at full res,
generates supports at reduced scan resolution, then multiprocess-encodes
all layers into a .goo file.

Usage:
    cd slicer_preview && python to_goo_julia.py
"""
import sys, struct, time, os
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


def goo_encode_placed(img, x0, y0, plate_w, plate_h):
    """RLE-encode a small image placed on a large plate without allocating the full plate."""
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

TEMPLATE = Path(__file__).parent.parent / 'Siraya Tech Test Model 2021 v5.STL_0.05_2_2026_03_28_20_51_00.goo'
OUTPUT   = Path(__file__).parent / 'julia.goo'
NUM_LAYERS_OFFSET = 195310

# ── Support data (shared with workers via initializer) ──
_sup = {}  # populated by _init_worker


def _init_worker(sup_slices, sup_dy, sup_dx, sup_dz):
    """Initializer for pool workers — receives support data."""
    _sup['slices'] = sup_slices
    _sup['dy'] = sup_dy
    _sup['dx'] = sup_dx
    _sup['dz'] = sup_dz


def _process_layer(args):
    """Worker: composite model + support for one layer, RLE-encode."""
    z, x0, y0, W_goo, H_goo = args
    img = make_global_slice(z)

    # Overlay pre-computed support slice if available
    if _sup:
        local_layer = z - _sup['dy']
        slices = _sup['slices']
        if local_layer in slices:
            sup = slices[local_layer]
            sh, sw = sup.shape
            sdx, sdz = _sup['dx'], _sup['dz']
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
    global _support_slices, _support_min_layer
    global _support_W, _support_H, _support_dx, _support_dz, _support_dy
    global W, H, N_SLICES, OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM

    from printer import PX_MM, PZ_MM, LY_MM

    print(f"Julia piece: {W}×{H} px, {N_SLICES} layers at full resolution")
    print(f"Pixel: {printer.PIXEL_X_UM}×{printer.PIXEL_Y_UM} µm, "
          f"layer: {GOO_LAYER_UM} µm")

    # ── Generate supports ──
    from supports import (generate_supports, SCAN_SCALE, SCAN_LAYER_STEP,
                          SUPPORT_GRAY)

    print("\nGenerating supports ...", flush=True)
    # Build low-res scan piece
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

    if support_pieces:
        sp = support_pieces[0]
        PIECES.extend(support_pieces)
        new_ox, new_oy, new_oz, new_W, new_H, new_N = _compute_globals(PIECES)

        # Update globals in generate_julia module
        import generate_julia as gj
        gj.OFFSET_X_MM = new_ox
        gj.OFFSET_Y_MM = new_oy
        gj.OFFSET_Z_MM = new_oz
        gj.W = new_W
        gj.H = new_H
        gj.N_SLICES = new_N
        OFFSET_X_MM = new_ox
        OFFSET_Y_MM = new_oy
        OFFSET_Z_MM = new_oz
        W, H, N_SLICES = new_W, new_H, new_N

        # Pre-extract support slices for workers
        sup_dx = round((sp['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        sup_dz = round((sp['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        sup_dy = round((sp['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
        print(f"  Support offset: dx={sup_dx} dz={sup_dz} dy={sup_dy}")

        # Cache all support slices for workers
        print("  Caching support slices ...", flush=True)
        sup_slices = {}
        for local_layer in range(sp['N_SLICES']):
            img = sp['make_slice'](local_layer)
            if img.any():
                sup_slices[local_layer] = img
        print(f"  Cached {len(sup_slices)} non-empty support layers",
              flush=True)
    else:
        sup_slices, sup_dy, sup_dx, sup_dz = {}, 0, 0, 0

    # ── Read template .goo ──
    print(f"\nReading template: {TEMPLATE}")
    data = TEMPLATE.read_bytes()
    header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
    print(f"Plate resolution: {W_goo} x {H_goo}")

    template_def = bytearray(orig_layers[0][0])

    # Center pattern on build plate
    x0 = (W_goo - W) // 2
    y0 = (H_goo - H) // 2
    print(f"Placing {W}x{H} pattern at ({x0}, {y0}) on {W_goo}x{H_goo} plate")
    print(f"{N_SLICES} layers at {GOO_LAYER_UM} µm")

    # ── Multiprocess encode ──
    n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    print(f"Encoding with {n_workers} worker processes", flush=True)

    t0 = time.time()
    results = {}
    args = [(z, x0, y0, W_goo, H_goo) for z in range(N_SLICES)]

    with Pool(n_workers, initializer=_init_worker,
              initargs=(sup_slices, sup_dy, sup_dx, sup_dz)) as pool:
        for z, rle in pool.imap_unordered(_process_layer, args):
            results[z] = rle
            done = len(results)
            if done <= 20 or done % 100 == 0 or done == N_SLICES:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (N_SLICES - done) / rate if rate > 0 else 0
                print(f"  {done}/{N_SLICES} layers  ({len(rle):,} bytes RLE)  "
                      f"[{elapsed:.1f}s, ~{eta:.0f}s remaining]", flush=True)

    # ── Assemble .goo file ──
    new_layers = []
    for z in range(N_SLICES):
        rle = results[z]
        ldef = bytearray(template_def)
        z_mm = (z + 1) * GOO_LAYER_UM / 1000.0
        struct.pack_into('>f', ldef, 2, z_mm)
        struct.pack_into('>f', ldef, 6, z_mm)
        struct.pack_into('>I', ldef, 66, len(rle))
        new_layers.append((bytes(ldef), rle))

    elapsed = time.time() - t0
    print(f"Encoded {N_SLICES} layers in {elapsed:.1f}s")

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


if __name__ == '__main__':
    main()
