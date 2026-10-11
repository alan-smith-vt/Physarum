"""Compose a list of pieces into one global volume.

A piece is a dict (see geometry.py):
    W, H, N_SLICES                          — image dimensions and layer count
    OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM   — world position
    make_slice(layer) → uint8 image (H, W)

Pieces are OR'd (max) together. Y is the build axis (layers).
"""
import struct, json, time, os, importlib
import numpy as np
import printer
from helpers import encode_surface_voxels

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')


def load_design(name):
    """Import designs/<name>.py. Must define PIECES; may define SUPPORTS = True."""
    return importlib.import_module(f'designs.{name}')


def compute_bounds(pieces):
    """Global bounding box: (ox, oy, oz, W, H, N_SLICES) in mm / pixels / layers."""
    px, pz, ly = printer.PX_MM, printer.PZ_MM, printer.LY_MM
    extents = [(p['OFFSET_X_MM'], p['OFFSET_Y_MM'], p['OFFSET_Z_MM'],
                p['OFFSET_X_MM'] + p['W'] * px,
                p['OFFSET_Y_MM'] + p['N_SLICES'] * ly,
                p['OFFSET_Z_MM'] + p['H'] * pz) for p in pieces]

    ox = min(e[0] for e in extents)
    oy = min(e[1] for e in extents)
    oz = min(e[2] for e in extents)

    W = int(np.ceil((max(e[3] for e in extents) - ox) / px))
    H = int(np.ceil((max(e[5] for e in extents) - oz) / pz))
    N = int(np.ceil((max(e[4] for e in extents) - oy) / ly))

    return ox, oy, oz, W, H, N


def _pixel_offsets(p, bounds):
    ox, oy, oz = bounds[:3]
    dx = round((p['OFFSET_X_MM'] - ox) / printer.PX_MM)
    dy = round((p['OFFSET_Y_MM'] - oy) / printer.LY_MM)
    dz = round((p['OFFSET_Z_MM'] - oz) / printer.PZ_MM)
    return dx, dy, dz


def make_global_slice(pieces, bounds, layer):
    """Composite all pieces into a single (H, W) image for the given global layer."""
    W, H = bounds[3], bounds[4]
    img = np.zeros((H, W), dtype=np.uint8)
    for p in pieces:
        dx, dy, dz = _pixel_offsets(p, bounds)
        local_layer = layer - dy
        if local_layer < 0 or local_layer >= p['N_SLICES']:
            continue
        piece_img = p['make_slice'](local_layer)
        ph, pw = piece_img.shape
        sx, sz = max(0, -dx), max(0, -dz)
        ex, ez = min(pw, W - dx), min(ph, H - dz)
        if sx >= ex or sz >= ez:
            continue
        region = img[dz + sz:dz + ez, dx + sx:dx + ex]
        np.maximum(region, piece_img[sz:ez, sx:ex], out=region)
    return img


def add_supports(pieces):
    """Append TPMS lattice supports (see supports.py). Returns new bounds."""
    from supports import generate_supports
    bounds = compute_bounds(pieces)
    print("Generating supports ...", flush=True)
    support_pieces = generate_supports(
        lambda layer: make_global_slice(pieces, bounds, layer),
        bounds[3], bounds[4], bounds[5], *bounds[:3])
    if support_pieces:
        pieces.extend(support_pieces)
        bounds = compute_bounds(pieces)
        print(f"  Added {len(support_pieces)} support piece(s), "
              f"new bounds: {bounds[3]}×{bounds[4]} px, {bounds[5]} layers")
    return bounds


def encode_all(pieces, bounds):
    """Encode surface voxels of all pieces into a single binary blob for the viewer."""
    all_chunks = []
    total = 0

    for pi, p in enumerate(pieces):
        t0p = time.time()
        dx, dy, dz = _pixel_offsets(p, bounds)
        chunks, n = encode_surface_voxels(
            p['make_slice'], p['W'], p['H'], p['N_SLICES'], dx, dz, dy)
        all_chunks.extend(chunks)
        total += n
        print(f"  piece {pi+1}/{len(pieces)}  {n:,} voxels  ({time.time()-t0p:.1f}s)", flush=True)

    return struct.pack('<I', total) + b''.join(all_chunks)


def write_preview(pieces, bounds):
    """Encode all pieces and write _slices.bin + _meta.json for server.py/viewer.html."""
    print("Encoding slices...", flush=True)
    blob = encode_all(pieces, bounds)

    with open(SLICES_FILE, 'wb') as f:
        f.write(blob)

    ox, oy, oz, W, H, N = bounds
    meta = {
        "width": W, "height": H, "num_slices": N,
        "pixel_x_um": printer.PIXEL_X_UM, "pixel_y_um": printer.PIXEL_Y_UM,
        "layer_um": printer.LAYER_UM,
        "plate_w_mm": printer.PLATE_W_PX * printer.REAL_PIXEL_X_UM / 1000,
        "plate_h_mm": printer.PLATE_H_PX * printer.REAL_PIXEL_Y_UM / 1000,
        "offset_x_mm": ox, "offset_y_mm": oy, "offset_z_mm": oz,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    return blob
