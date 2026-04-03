"""Re-assemble julia.goo from existing checkpoint, skipping sub-plate layers.

Usage: cd slicer_preview && python reassemble_goo.py
"""
import struct, pickle, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'chromosome'))
from goo_punch_hole_v2 import parse_goo

SCRIPT_DIR = Path(__file__).parent
TEMPLATE = SCRIPT_DIR.parent / 'Siraya Tech Test Model 2021 v5.STL_0.05_2_2026_03_28_20_51_00.goo'
OUTPUT   = SCRIPT_DIR / 'julia.goo'
CHECKPOINT = SCRIPT_DIR / '_julia_checkpoint.pkl'
NUM_LAYERS_OFFSET = 195310

GOO_LAYER_UM = 50.0
GOO_EXPOSURE_S = 2.3
OFFSET_Y_MM = -8.0  # must match generate_julia

plate_layer = max(0, int(round(-OFFSET_Y_MM / (GOO_LAYER_UM / 1000.0))))

print(f"Loading checkpoint: {CHECKPOINT}")
with open(CHECKPOINT, 'rb') as f:
    results = pickle.load(f)

total_layers = max(results.keys()) + 1
goo_layers = list(range(plate_layer, total_layers))
print(f"Checkpoint has {len(results)} layers, total={total_layers}")
print(f"Skipping {plate_layer} sub-plate layers, writing {len(goo_layers)}")

# Check all needed layers exist
missing = [z for z in goo_layers if z not in results]
if missing:
    print(f"ERROR: missing {len(missing)} layers: {missing[:10]}...")
    sys.exit(1)

print(f"Reading template: {TEMPLATE}")
data = TEMPLATE.read_bytes()
header_bytes, orig_layers, footer, W_goo, H_goo = parse_goo(data)
template_def = bytearray(orig_layers[0][0])

print("Assembling .goo ...", flush=True)
new_layers = []
for i, z in enumerate(goo_layers):
    rle = results[z]
    ldef = bytearray(template_def)
    z_mm = (i + 1) * GOO_LAYER_UM / 1000.0
    struct.pack_into('>f', ldef, 2, z_mm)
    struct.pack_into('>f', ldef, 6, z_mm)
    struct.pack_into('>I', ldef, 66, len(rle))
    new_layers.append((bytes(ldef), rle))

header = bytearray(header_bytes)
struct.pack_into('>I', header, NUM_LAYERS_OFFSET, len(goo_layers))
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
print(f"{len(goo_layers)} layers, {len(goo_layers) * GOO_LAYER_UM / 1000:.1f}mm total height")
