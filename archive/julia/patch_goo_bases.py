"""Patch a .goo file: add 1mm solid base + 1mm taper to support feet.

Much faster than regenerating — only decodes and re-encodes the ~40
bottom layers that need modification.

Usage:  cd slicer_preview && python patch_goo_bases.py
"""
import struct, time, sys, os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'chromosome'))
from goo_punch_hole_v2 import parse_goo, goo_decode, goo_encode

INPUT  = SCRIPT_DIR / 'julia_v2.goo'
OUTPUT = SCRIPT_DIR / 'julia_v3.goo'

# ── Parameters (must match generation) ─────────────────────────────
SUPPORT_GRAY        = 255
PX_MM  = 0.014   # real pixel pitch X (mm)
PZ_MM  = 0.019   # real pixel pitch Z (mm)
LY_MM  = 0.05    # layer height (mm)

BASE_SOLID_MM       = 1.0    # solid fill height at plate
BASE_TAPER_MM       = 1.0    # taper zone above solid base
NORMAL_THICKNESS_MM = 1.0    # TPMS wall thickness (threshold=0.6, period=5mm)
BASE_THICKNESS_MM   = 1.5    # wall thickness at bottom of taper

SOLID_LAYERS = int(round(BASE_SOLID_MM / LY_MM))   # 20
TAPER_LAYERS = int(round(BASE_TAPER_MM / LY_MM))   # 20
TOTAL_MODIFY = SOLID_LAYERS + TAPER_LAYERS          # 40

# ── Per-layer exposure fix ─────────────────────────────────────────
FIX_EXPOSURE       = True
BASE_EXPOSURE_S    = 30.0
NORMAL_EXPOSURE_S  = 2.3
BASE_LAYERS        = 20      # 1mm raft at base exposure
TRANSITION_LAYERS  = 10      # gradual ramp


# ── Dilation (same separable shift approach as supports.py) ────────

def dilate_mask(mask, r_x, r_z):
    """Fast separable dilation using shifts."""
    if r_x == 0 and r_z == 0:
        return mask
    out = mask.copy()
    if r_x > 0:
        for dx in range(1, r_x + 1):
            out[:, dx:]  |= mask[:, :-dx]
            out[:, :-dx] |= mask[:, dx:]
    if r_z > 0:
        tmp = out.copy()
        for dz in range(1, r_z + 1):
            out[dz:, :]  |= tmp[:-dz, :]
            out[:-dz, :] |= tmp[dz:, :]
    return out


# ── Main ───────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print(f"Reading {INPUT}")
    data = INPUT.read_bytes()
    header, layers, footer, W_goo, H_goo = parse_goo(data)
    del data
    n_layers = len(layers)
    print(f"{n_layers} layers, plate {W_goo}x{H_goo}")

    # ── Step 1: build support footprint from bottom layers ──
    # Scan enough layers to cover at least one full TPMS period (5mm = 100 layers)
    # above the modified zone, so the union captures the full support region.
    scan_end = min(n_layers, TOTAL_MODIFY + 120)
    print(f"Scanning {scan_end} layers for support footprint ...", flush=True)

    # Work in a cropped region — first pass to find bbox, second to build footprint
    # Initial pass: quick bbox from first layer
    img0 = goo_decode(layers[0][1], W_goo, H_goo)
    # Detect support pixels: anything non-zero that isn't model (255)
    sup0 = (img0 > 0) & (img0 < 255)
    if not sup0.any():
        # Supports might already be 255; fall back to all non-zero
        sup0 = img0 > 0
    if not sup0.any():
        print("No support pixels on layer 0!")
        return

    zs, xs = np.nonzero(sup0)
    bx0, bx1 = int(xs.min()), int(xs.max())
    bz0, bz1 = int(zs.min()), int(zs.max())
    del img0, sup0

    # Expand bbox during scan as needed, plus dilation padding
    max_dil_x = int(round((BASE_THICKNESS_MM - NORMAL_THICKNESS_MM) / 2 / PX_MM))
    max_dil_z = int(round((BASE_THICKNESS_MM - NORMAL_THICKNESS_MM) / 2 / PZ_MM))

    # Full scan to build footprint (use full plate decode — RLE is fast for sparse layers)
    footprint = np.zeros((H_goo, W_goo), dtype=bool)
    for i in range(scan_end):
        img = goo_decode(layers[i][1], W_goo, H_goo)
        sup = (img > 0) & (img < 255)
        footprint |= sup

        if (i + 1) % 20 == 0 or i + 1 == scan_end:
            elapsed = time.time() - t_start
            print(f"  scanned {i+1}/{scan_end}  [{elapsed:.1f}s]", flush=True)

    if not footprint.any():
        print("No support pixels found!")
        return

    # Final tight bbox with dilation padding
    zs, xs = np.nonzero(footprint)
    bx0 = max(0, int(xs.min()) - max_dil_x)
    bx1 = min(W_goo, int(xs.max()) + max_dil_x + 1)
    bz0 = max(0, int(zs.min()) - max_dil_z)
    bz1 = min(H_goo, int(zs.max()) + max_dil_z + 1)

    crop_fp = footprint[bz0:bz1, bx0:bx1]
    n_fp = int(footprint.sum())
    del footprint  # free full-plate array
    print(f"Support footprint: {n_fp:,} pixels")
    print(f"Working crop: ({bx0},{bz0})-({bx1},{bz1}) "
          f"= {bx1-bx0}x{bz1-bz0} px")
    print(f"Max dilation: {max_dil_x}px X, {max_dil_z}px Z")

    # ── Step 2: modify bottom layers ──
    actual_modify = min(TOTAL_MODIFY, n_layers)
    print(f"Modifying {actual_modify} layers "
          f"({SOLID_LAYERS} solid + {TAPER_LAYERS} taper) ...", flush=True)
    t0 = time.time()

    for i in range(actual_modify):
        layer_def, rle_data = layers[i]
        img = goo_decode(rle_data, W_goo, H_goo)

        crop = img[bz0:bz1, bx0:bx1]
        model_mask = crop == 255

        # Identify existing support pixels (may be old 128 or new 255 gray)
        sup_mask = (crop > 0) & ~model_mask

        if i < SOLID_LAYERS:
            # Solid base: fill entire support footprint at full exposure
            fill = crop_fp & ~model_mask
            crop[fill] = 255

        else:
            # Taper zone: dilate existing support walls, clip to footprint
            t = (i - SOLID_LAYERS) / TAPER_LAYERS          # 0 at base, 1 at top
            dil_mm = (BASE_THICKNESS_MM - NORMAL_THICKNESS_MM) / 2 * (1 - t)
            r_x = int(round(dil_mm / PX_MM))
            r_z = int(round(dil_mm / PZ_MM))

            if r_x > 0 or r_z > 0:
                dilated = dilate_mask(sup_mask, r_x, r_z)
                fill = dilated & crop_fp & ~model_mask & ~sup_mask
                crop[fill] = 255

        img[bz0:bz1, bx0:bx1] = crop

        # Re-encode
        new_rle = goo_encode(img)
        new_def = bytearray(layer_def)
        struct.pack_into('>I', new_def, 66, len(new_rle))
        layers[i] = (bytes(new_def), new_rle)

        if (i + 1) % 5 == 0 or i + 1 == actual_modify:
            elapsed = time.time() - t0
            print(f"  {i+1}/{actual_modify}  [{elapsed:.1f}s]", flush=True)

    # ── Step 2b: fix per-layer exposure on ALL layers ──
    if FIX_EXPOSURE:
        print(f"Fixing per-layer exposure ({BASE_LAYERS} base @ {BASE_EXPOSURE_S}s, "
              f"{TRANSITION_LAYERS} transition, normal {NORMAL_EXPOSURE_S}s) ...",
              flush=True)
        for i in range(n_layers):
            layer_def, rle_data = layers[i]
            if i < BASE_LAYERS:
                exp = BASE_EXPOSURE_S
            elif i < BASE_LAYERS + TRANSITION_LAYERS:
                t = (i - BASE_LAYERS) / TRANSITION_LAYERS
                exp = BASE_EXPOSURE_S + t * (NORMAL_EXPOSURE_S - BASE_EXPOSURE_S)
            else:
                exp = NORMAL_EXPOSURE_S
            new_def = bytearray(layer_def)
            struct.pack_into('>f', new_def, 10, exp)
            layers[i] = (bytes(new_def), rle_data)

    # ── Step 3: write output (copy unmodified layers verbatim) ──
    print(f"Writing {OUTPUT} ...", flush=True)
    with open(OUTPUT, 'wb') as f:
        f.write(header)
        for ld, rle in layers:
            f.write(ld)
            f.write(rle)
            f.write(b'\x0d\x0a')
        f.write(footer)

    size = OUTPUT.stat().st_size
    elapsed_total = time.time() - t_start
    print(f"Done in {elapsed_total:.1f}s — {size:,} bytes ({size/1024/1024:.1f} MB)")


if __name__ == '__main__':
    main()
