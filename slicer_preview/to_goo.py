"""Convert generated slices → .goo file using original.goo as template.

Reads the original's header (preview images, printer config, print settings)
and replaces all layer image data with our generated lattice slices.

Patches printer module to real resolution before importing geometry/generate
so all pieces are built at full printer resolution.

Uses multiprocessing to slice layers in parallel.
"""
import sys, struct, time, os
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# ── Patch printer to real resolution BEFORE importing geometry/generate ──
import printer
GOO_LAYER_UM = 50.0
GOO_EXPOSURE_S = 3.75
printer.PREVIEW_SCALE = 1
printer.PIXEL_X_UM = printer.REAL_PIXEL_X_UM
printer.PIXEL_Y_UM = printer.REAL_PIXEL_Y_UM
printer.LAYER_UM = GOO_LAYER_UM
printer.PX_MM = printer.PIXEL_X_UM / 1000
printer.PZ_MM = printer.PIXEL_Y_UM / 1000
printer.LY_MM = printer.LAYER_UM / 1000

sys.path.insert(0, str(Path(__file__).parent.parent / 'chromosome'))
from goo_punch_hole_v2 import parse_goo
from generate import make_global_slice, W, H, N_SLICES

TEMPLATE = Path(__file__).parent.parent / 'Siraya Tech Test Model 2021 v5.STL_0.05_2_2026_03_28_20_51_00.goo'
OUTPUT   = Path(__file__).parent / 'full_box.goo'

NUM_LAYERS_OFFSET = 195310


def goo_encode_placed(img, x0, y0, plate_w, plate_h):
    """RLE-encode a small image placed on a large plate without allocating the full plate.

    Analytically handles the surrounding zero regions, only runs numpy on the
    small pattern image rows.
    """
    h, w = img.shape
    right_pad = plate_w - x0 - w
    between = right_pad + x0           # zeros between consecutive pattern rows

    runs = []                          # (color, length) pairs

    # Leading zeros: full rows above pattern + left pad of first row
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
        # Gap to next row (or suffix after last row)
        if r < h - 1:
            if between > 0:
                runs.append((0, between))
        else:
            suffix = right_pad + (plate_h - y0 - h) * plate_w
            if suffix > 0:
                runs.append((0, suffix))

    # Merge adjacent same-color runs
    merged = [runs[0]]
    for c, l in runs[1:]:
        if c == merged[-1][0]:
            merged[-1] = (c, merged[-1][1] + l)
        else:
            merged.append((c, l))

    # ── Encode using GOO RLE format ──
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
    """Worker function: slice one layer and RLE-encode it."""
    z, x0, y0, W_goo, H_goo = args
    img = make_global_slice(z)
    rle = goo_encode_placed(img, x0, y0, W_goo, H_goo)
    return z, rle


def main():
    print(f"Reading template: {TEMPLATE}")
    data = TEMPLATE.read_bytes()
    header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
    print(f"Plate resolution: {W_goo} x {H_goo}")

    template_def = bytearray(orig_layers[0][0])

    # Center our pattern on the build plate
    x0 = (W_goo - W) // 2
    y0 = (H_goo - H) // 2
    print(f"Placing {W}x{H} pattern at ({x0}, {y0}) on {W_goo}x{H_goo} plate")
    print(f"{N_SLICES} layers at {GOO_LAYER_UM} µm")

    n_workers = max(1, (os.cpu_count() or 4) - 4)
    print(f"Using {n_workers} worker processes", flush=True)

    t0 = time.time()
    results = {}
    args = [(z, x0, y0, W_goo, H_goo) for z in range(N_SLICES)]

    with Pool(n_workers) as pool:
        for z, rle in pool.imap_unordered(_process_layer, args):
            results[z] = rle
            done = len(results)
            if done <= 20 or done % 100 == 0 or done == N_SLICES:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (N_SLICES - done) / rate if rate > 0 else 0
                print(f"  {done}/{N_SLICES} layers  ({len(rle):,} bytes RLE)  "
                      f"[{elapsed:.1f}s, ~{eta:.0f}s remaining]", flush=True)

    # Assemble layers in order
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
    LAYER_HEIGHT_OFFSET = NUM_LAYERS_OFFSET + 4+2+2+1+1 + 4+4+4  # 22 bytes past layerCount
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
