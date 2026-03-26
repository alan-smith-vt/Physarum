"""Convert generated slices → .goo file using original.goo as template.

Reads the original's header (preview images, printer config, print settings)
and replaces all layer image data with our generated lattice slices.
"""
import sys, struct, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'chromosome'))
from goo_punch_hole_v2 import goo_encode, parse_goo
from generate import make_slice, W, H, N_SLICES, LAYER_UM

TEMPLATE = Path(__file__).parent.parent / 'chromosome' / 'original.goo'
OUTPUT   = Path(__file__).parent / 'test_lattice.goo'

LAYER_DEF_SIZE = 70
NUM_LAYERS_OFFSET = 195310
LDA_OFFSET = (195310 + 4+2+2+1+1 + 4+4+4 + 4+4+1 + 4+4+4+4 +
              4+4+4 + 4+4 + 4+4+4+4 + 4+4+4+4 + 4+4+4+4 + 4+4+4+4 +
              2+2+1+4 + 4+4+4+8)


def main():
    print(f"Reading template: {TEMPLATE}")
    data = TEMPLATE.read_bytes()
    header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
    print(f"Plate resolution: {W_goo} x {H_goo}")

    # Copy print settings from first original layer as template
    template_def = bytearray(orig_layers[0][0])

    # Center our pattern on the build plate
    x0 = (W_goo - W) // 2
    y0 = (H_goo - H) // 2
    print(f"Placing {W}x{H} pattern at ({x0}, {y0}) on {W_goo}x{H_goo} plate")

    t0 = time.time()
    new_layers = []
    for z in range(N_SLICES):
        # Generate slice and embed in full-plate image
        full = np.zeros((H_goo, W_goo), dtype=np.uint8)
        full[y0:y0+H, x0:x0+W] = make_slice(z)
        rle = goo_encode(full)

        # Build layer def: copy template, update z-position + data length
        ldef = bytearray(template_def)
        z_mm = (z + 1) * LAYER_UM / 1000.0           # cumulative z (1-indexed)
        struct.pack_into('>f', ldef, 2, z_mm)         # z pos field 1
        struct.pack_into('>f', ldef, 6, z_mm)         # z pos field 2
        struct.pack_into('>I', ldef, 66, len(rle))    # rle byte count
        new_layers.append((bytes(ldef), rle))

        if (z + 1) % 50 == 0 or z == 0:
            print(f"  {z+1}/{N_SLICES} layers  ({len(rle):,} bytes RLE)")

    elapsed = time.time() - t0
    print(f"Encoded in {elapsed:.1f}s")

    # Patch header: update layer count and layer-def address
    header = bytearray(header_bytes)
    struct.pack_into('>I', header, NUM_LAYERS_OFFSET, N_SLICES)
    # LayerDefAddress stays the same (layers start right after header)

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
