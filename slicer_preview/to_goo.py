"""Build a design at full printer resolution → out/<design>.goo.

Usage:  python to_goo.py <design> [--gpu]        (designs/<design>.py)

--gpu: slice on the GPU with CuPy (implicit pieces only; ~50 ms/layer on a 4090 vs ~2-4 s/layer/core,
and the CPU pool is memory-bandwidth bound). Needs cupy-cuda12x + nvidia-cuda-nvrtc-cu12.

Uses the Siraya test-model .goo as a template: its header (preview images,
printer config, print settings incl. bottom-layer exposure) is kept and all
layer image data is replaced. Only layer height and normal exposure are
overridden below. Base/bottom exposure is deliberately left at the template's
value — the Julia print's custom 30 s base fused it to the plate.

Patches printer module to real resolution before importing any design so
all pieces are built at full printer resolution. Layers are sliced in
parallel; each worker builds the design once in its initializer.
"""
import sys, struct, time, os
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# ── Patch printer to real resolution BEFORE importing geometry/generate ──
import printer
GOO_LAYER_UM = 50.0
GOO_EXPOSURE_S = 2.3          # default; a design can override with EXPOSURE_S. Set Mar 2026 on an old, scratched film
printer.PREVIEW_SCALE = 1
printer.PIXEL_X_UM = printer.REAL_PIXEL_X_UM
printer.PIXEL_Y_UM = printer.REAL_PIXEL_Y_UM
printer.LAYER_UM = GOO_LAYER_UM
printer.PX_MM = printer.PIXEL_X_UM / 1000
printer.PZ_MM = printer.PIXEL_Y_UM / 1000
printer.LY_MM = printer.LAYER_UM / 1000

sys.path.insert(0, str(Path(__file__).parent.parent / 'chromosome'))
from goo_punch_hole_v2 import parse_goo
from compose import load_design, compute_bounds, make_global_slice, _pixel_offsets

TEMPLATE = Path(__file__).parent.parent / 'Siraya Tech Test Model 2021 v5.STL_0.05_2_2026_03_28_20_51_00.goo'
OUT_DIR  = Path(__file__).parent / 'out'

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


_pieces, _bounds = None, None


def _build(name):
    design = load_design(name)
    if getattr(design, 'SUPPORTS', False):
        raise NotImplementedError(
            "Full-res supports aren't wired into to_goo yet — see "
            "archive/julia/to_goo_julia.py for the memmap/checkpoint approach.")
    pieces = list(design.PIECES)
    return pieces, compute_bounds(pieces)


def _init_worker(name):
    global _pieces, _bounds
    _pieces, _bounds = _build(name)


def _process_layer(args):
    """Worker function: slice one layer and RLE-encode it."""
    z, x0, y0, W_goo, H_goo = args
    img = make_global_slice(_pieces, _bounds, z)
    rle = goo_encode_placed(img, x0, y0, W_goo, H_goo)
    return z, rle


def _encode_packed(args):
    """Worker (GPU mode): unpack a 1-bit slice from the GPU and RLE-encode it."""
    z, packed, H, W, x0, y0, W_goo, H_goo = args
    img = np.unpackbits(packed, count=H * W).reshape(H, W) * np.uint8(255)
    return z, goo_encode_placed(img, x0, y0, W_goo, H_goo)


def _gpu_slices(name, pieces, bounds, args):
    """Slice on the GPU (CuPy): each implicit piece's solid_fn runs on whole-slice GPU arrays,
    with the design modules' `np` pointed at cupy. Yields encoder jobs with 1-bit packed slices."""
    os.environ.setdefault('CUPY_ACCELERATORS', '')    # CUB needs newer CUDA headers than the 11.8 toolkit on PATH
    import cupy as cp
    if not hasattr(cp, 'broadcast_shapes'):
        cp.broadcast_shapes = np.broadcast_shapes
    if any('solid_fn' not in p for p in pieces):
        sys.exit("--gpu needs implicit pieces (solid_fn); this design has others")
    for mod in [m for k, m in sys.modules.items() if k.startswith('designs.') and getattr(m, 'np', None) is np]:
        mod.np = cp
    W, H = bounds[3], bounds[4]
    grids = []
    for p in pieces:
        x = cp.asarray(((np.arange(p['W']) + 0.5) * printer.PX_MM)[None, :])
        z = cp.asarray(((np.arange(p['H']) + 0.5) * printer.PZ_MM)[:, None])
        grids.append((x, z) + _pixel_offsets(p, bounds))
    for z_layer, x0, y0, W_goo, H_goo in args:
        img = cp.zeros((H, W), cp.uint8)
        for p, (xg, zg, dx, dy, dz) in zip(pieces, grids):
            local = z_layer - dy
            if not 0 <= local < p['N_SLICES']:
                continue
            m = cp.broadcast_to(p['solid_fn'](xg, (local + 0.5) * printer.LY_MM, zg), (p['H'], p['W']))
            region = img[dz:dz + p['H'], dx:dx + p['W']]
            cp.maximum(region, m[:region.shape[0], :region.shape[1]].astype(cp.uint8), out=region)
        yield z_layer, cp.asnumpy(cp.packbits(img.ravel())), H, W, x0, y0, W_goo, H_goo


def main(name, gpu=False):
    pieces, bounds = _build(name)
    design = load_design(name)
    exposure_s = getattr(design, 'EXPOSURE_S', GOO_EXPOSURE_S)
    bottom_s = getattr(design, 'BOTTOM_EXPOSURE_S', None)   # None → template's profile
    W, H, N_SLICES = bounds[3:]
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT = OUT_DIR / f'{name}.goo'

    print(f"Reading template: {TEMPLATE}")
    data = TEMPLATE.read_bytes()
    header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
    print(f"Plate resolution: {W_goo} x {H_goo}")

    # Per-layer defs carry their own exposure + lift settings. Reuse the
    # template's bottom-layer defs (incl. its exposure ramp) as-is, then its
    # last def for every normal layer with exposure overridden. (Pre-Oct 2026
    # files copied layer 0 everywhere → 30 s per-layer exposure, which the
    # printer evidently ignored in favour of the header.)
    template_defs = [d for d, _ in orig_layers]
    normal_exp = struct.unpack('>f', template_defs[-1][10:14])[0]
    n_bottom = next(i for i, d in enumerate(template_defs)
                    if struct.unpack('>f', d[10:14])[0] == normal_exp)
    print(f"Template: {n_bottom} bottom layers, normal exposure {normal_exp:.2f} s → {exposure_s} s")
    template_bottom = struct.unpack('>f', template_defs[0][10:14])[0]
    bottom_scale = bottom_s / template_bottom if bottom_s else 1.0
    if bottom_s:
        print(f"Bottom exposure {template_bottom:.1f} s → {bottom_s} s (ramp scaled ×{bottom_scale:.2f})")

    # Center our pattern on the build plate
    if W > W_goo or H > H_goo:
        sys.exit(f"Pattern {W}x{H} px doesn't fit the {W_goo}x{H_goo} plate "
                 f"({W * printer.REAL_PIXEL_X_UM / 1000:.1f} x {H * printer.REAL_PIXEL_Y_UM / 1000:.1f} mm "
                 f"vs {W_goo * printer.REAL_PIXEL_X_UM / 1000:.1f} x {H_goo * printer.REAL_PIXEL_Y_UM / 1000:.1f})")
    x0 = (W_goo - W) // 2
    y0 = (H_goo - H) // 2
    print(f"Placing {W}x{H} pattern at ({x0}, {y0}) on {W_goo}x{H_goo} plate")
    print(f"{N_SLICES} layers at {GOO_LAYER_UM} µm")

    t0 = time.time()
    results = {}
    args = [(z, x0, y0, W_goo, H_goo) for z in range(N_SLICES)]

    if gpu:   # slices on the GPU in this process; a few CPU workers only RLE-encode
        n_workers = min(12, max(1, (os.cpu_count() or 4) - 4))
        print(f"GPU slicing, {n_workers} encoder processes", flush=True)
        pool, jobs, fn = Pool(n_workers), _gpu_slices(name, pieces, bounds, args), _encode_packed
    else:
        n_workers = max(1, (os.cpu_count() or 4) - 4)
        print(f"Using {n_workers} worker processes", flush=True)
        pool, jobs, fn = Pool(n_workers, initializer=_init_worker, initargs=(name,)), args, _process_layer

    with pool:
        for z, rle in pool.imap_unordered(fn, jobs):
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
        if z < n_bottom:
            ldef = bytearray(template_defs[z])
            exp = struct.unpack('>f', ldef[10:14])[0] * bottom_scale
            struct.pack_into('>f', ldef, 10, max(exp, exposure_s))
        else:
            ldef = bytearray(template_defs[-1])
            struct.pack_into('>f', ldef, 10, exposure_s)
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
    struct.pack_into('>f', header, EXPOSURE_OFFSET, exposure_s)
    BOTTOM_EXPOSURE_OFFSET = EXPOSURE_OFFSET + 33   # found by inspection; bottom layer count (u32) follows
    if bottom_s:
        struct.pack_into('>f', header, BOTTOM_EXPOSURE_OFFSET, bottom_s)
    print(f"Normal exposure: {exposure_s} s")

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
    names = [a for a in sys.argv[1:] if a != '--gpu']
    if len(names) != 1:
        sys.exit(__doc__)
    main(names[0], gpu='--gpu' in sys.argv)
