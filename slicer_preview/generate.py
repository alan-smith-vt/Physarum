"""Compose geometry pieces, encode, and write to disk for the server."""
import struct, json, time, os
import numpy as np
from printer import PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX, PX_MM, PZ_MM, LY_MM
from helpers import encode_surface_voxels
from geometry import (rope_column, base_lattice, hex_column,
                      grid_column, diamond_column, quad_spiral_column,
                      bridge_struts, bridge_struts_x, braid, maypole_braid,
                      horizontal_maypole_braid,
                      solid_column, horizontal_solid_column,
                      diagonal_rope, solid_frame, vertical_frame)

from stl_slicer import stl_piece

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')
STL_FILE = os.path.join(PROJECT_ROOT, 'whiteboard_box_frame_75mm.stl')


def _arch_top_beams(piece, side_walls, front_back_walls,
                    beam_w=10.0, beam_bottom_y=95.0, arch_mm=3.0):
    """Cut parabolic arches from the underside of top beams between explicit column pairs.

    side_walls:       [(wall_x, [(z_lo, z_hi), ...]), ...]
    front_back_walls: [(wall_z, [(x_lo, x_hi), ...]), ...]
    """
    W_p, H_p = piece['W'], piece['H']
    ox, oy, oz = piece['OFFSET_X_MM'], piece['OFFSET_Y_MM'], piece['OFFSET_Z_MM']

    gx = ox + np.arange(W_p) * PX_MM
    gz = oz + np.arange(H_p) * PZ_MM
    beam_half = beam_w / 2

    def _pair_arch(coords, c_lo, c_hi):
        """Parabolic arch heights between c_lo and c_hi, zero outside."""
        t = np.clip((coords - c_lo) / (c_hi - c_lo), 0, 1)
        h = arch_mm * 4.0 * t * (1.0 - t)
        h[(coords < c_lo) | (coords > c_hi)] = 0
        return h

    arch_field = np.zeros((H_p, W_p), dtype=np.float32)

    for wall_x, z_pairs in side_walls:
        x_mask = np.abs(gx - wall_x) <= beam_half
        for z_lo, z_hi in z_pairs:
            h = _pair_arch(gz, z_lo, z_hi)
            arch_field[:, x_mask] = np.maximum(arch_field[:, x_mask], h[:, None])

    for wall_z, x_pairs in front_back_walls:
        z_mask = np.abs(gz - wall_z) <= beam_half
        for x_lo, x_hi in x_pairs:
            h = _pair_arch(gx, x_lo, x_hi)
            arch_field[z_mask, :] = np.maximum(arch_field[z_mask, :], h[None, :])

    orig = piece['make_slice']
    def _arched(layer, _orig=orig, _af=arch_field, _oy=oy,
                _bby=beam_bottom_y, _am=arch_mm):
        y_mm = _oy + (layer + 0.5) * LY_MM
        if y_mm >= _bby + _am or y_mm < _bby:
            return _orig(layer)
        img = _orig(layer)
        mask = y_mm < (_bby + _af)
        if mask.any():
            img = img.copy()
            img[mask] = 0
        return img
    piece['make_slice'] = _arched


def _apply_grayscale(piece, gray):
    """Wrap make_slice to output gray instead of 255."""
    orig = piece['make_slice']
    def _scaled(layer, _orig=orig, _g=gray):
        return np.where(_orig(layer) > 0, _g, np.uint8(0)).astype(np.uint8)
    piece['make_slice'] = _scaled


def _punch_dot_holes(piece, n_dots, dot_depth_mm=0.5, dot_r=1.0, dot_spacing=3.0):
    """Punch identification holes into the bottom layers of an existing piece."""
    W, H = piece['W'], piece['H']
    depth_layers = max(1, int(np.ceil(dot_depth_mm / LY_MM)))

    piece_x = piece['W'] * PX_MM
    piece_z = piece['H'] * PZ_MM

    bar_w = 3.0
    px = np.arange(W) * PX_MM
    pz = np.arange(H) * PZ_MM
    hole_mask = np.zeros((H, W), dtype=bool)
    total_w = (n_dots - 1) * dot_spacing
    for i in range(n_dots):
        cx = piece_x / 2 - total_w / 2 + i * dot_spacing
        cz = bar_w / 2
        dist_sq = (px[None, :] - cx) ** 2 + (pz[:, None] - cz) ** 2
        hole_mask |= dist_sq <= dot_r ** 2

    orig = piece['make_slice']
    def _punched(layer, _orig=orig, _mask=hole_mask, _d=depth_layers):
        img = _orig(layer)
        if layer < _d:
            img = img.copy()
            img[_mask] = 0
        return img
    piece['make_slice'] = _punched


def _vault_base(piece, col_x, col_z, vault_mm=2.0):
    """Checkerboard vault on the base lattice.

    Every other grid intersection (checkerboard by i+j parity) touches the
    build plate; the rest arch up by vault_mm.  Shape is a smooth cosine
    egg-carton so arches span naturally between touching nodes.
    """
    W_p, H_p = piece['W'], piece['H']
    ox, oz = piece['OFFSET_X_MM'], piece['OFFSET_Z_MM']
    oy = piece['OFFSET_Y_MM']

    gx = ox + np.arange(W_p) * PX_MM
    gz = oz + np.arange(H_p) * PZ_MM

    cx = np.sort(col_x)
    cz = np.sort(col_z)

    # Continuous column index along X (0 at first col, N-1 at last)
    ix = np.clip(np.searchsorted(cx, gx), 1, len(cx) - 1)
    sx = (ix - 1) + np.clip((gx - cx[ix - 1]) / (cx[ix] - cx[ix - 1]), 0, 1)

    # Continuous row index along Z
    iz = np.clip(np.searchsorted(cz, gz), 1, len(cz) - 1)
    sz = (iz - 1) + np.clip((gz - cz[iz - 1]) / (cz[iz] - cz[iz - 1]), 0, 1)

    # Egg-carton: 0 at (i+j)-even nodes, vault_mm at (i+j)-odd nodes
    cos_x = np.cos(np.pi * sx)   # (W,)
    cos_z = np.cos(np.pi * sz)   # (H,)
    vault_field = (vault_mm / 2.0) * (1.0 - cos_x[None, :] * cos_z[:, None])

    orig = piece['make_slice']
    def _vaulted(layer, _orig=orig, _vf=vault_field, _oy=oy, _vm=vault_mm):
        y_mm = _oy + (layer + 0.5) * LY_MM
        if y_mm >= _vm:
            return _orig(layer)
        img = _orig(layer)
        mask = y_mm < _vf
        if mask.any():
            img = img.copy()
            img[mask] = 0
        return img
    piece['make_slice'] = _vaulted


# ── Full box (final print settings) ──
HELIX_R = 2.0
COL_SPACING = 2 * np.pi + 2 * HELIX_R   # ≈ 10.283 mm
WALL_START = np.pi / 2   # front/back wall filaments start at ±Z not ±X
SIDE_N_COLS = int(65.0 / COL_SPACING) + 2          # columns along Z
col_z = [(i - (SIDE_N_COLS - 1) / 2) * COL_SPACING for i in range(SIDE_N_COLS)]
FRONT_N_COLS = int(115.0 / COL_SPACING) + 1        # 12 columns along X
col_x = [(i - (FRONT_N_COLS - 1) / 2) * COL_SPACING for i in range(FRONT_N_COLS)]
PIECES = []
_stl = stl_piece(STL_FILE, rot_x=-np.pi/2, rot_y=np.pi/2)
_side_z_pairs = [(col_z[i], col_z[i+1]) for i in range(1, len(col_z) - 2)]
_front_x_pairs = [(col_x[i], col_x[i+1]) for i in range(1, len(col_x) - 2)]
_back_x_pairs = ([(col_x[i], col_x[i+1]) for i in range(1, 3)] +
                 [(col_x[i], col_x[i+1]) for i in range(len(col_x) - 4, len(col_x) - 2)] +
                 [(col_x[10], 52.68), (13.59, col_x[8]),
                  (-52.68, col_x[1]), (col_x[3], -13.59)])
_arch_top_beams(_stl,
    side_walls=[(-60.0, _side_z_pairs), (60.0, _side_z_pairs)],
    front_back_walls=[(-35.0, _front_x_pairs), (35.0, _back_x_pairs)],
    beam_bottom_y=95.0, arch_mm=1.0)
PIECES.append(_stl)
_lattice = base_lattice(col_x=col_x, col_z=col_z, grid_z_mm=65.0, spacing_mm=COL_SPACING)
_vault_base(_lattice, col_x, col_z, vault_mm=2.0)
PIECES.append(_lattice)
for side_x in [-60.0, 60.0]:
    PIECES += [quad_spiral_column(offset_x_mm=side_x, offset_z_mm=z,
                                  base_taper_mm=3.5, base_helix_r=1.2, base_filament_r=0.75)
               for z in col_z[1:-1]]
    PIECES += [bridge_struts(col_z[i], col_z[i+1], offset_x_mm=side_x)
               for i in range(len(col_z) - 1)]
FRONT_Z = -35.0
PIECES += [quad_spiral_column(offset_x_mm=x, offset_z_mm=FRONT_Z,
                              start_angle=WALL_START,
                              base_taper_mm=3.5, base_helix_r=1.2, base_filament_r=0.75)
           for x in col_x]
PIECES += [bridge_struts_x(col_x[i], col_x[i+1], offset_z_mm=FRONT_Z,
                            start_angle=WALL_START)
           for i in range(len(col_x) - 1)]
BACK_Z = 35.0
HOLE_R = 16.0
HOLE_CY = 90.0 - 20.0
back_col_h = []
for x in col_x:
    if abs(x) >= HOLE_R:
        back_col_h.append(90.0)
    else:
        back_col_h.append(HOLE_CY - np.sqrt(HOLE_R ** 2 - x ** 2))
for x, h in zip(col_x, back_col_h):
    PIECES.append(quad_spiral_column(offset_x_mm=x, offset_z_mm=BACK_Z,
                                     wall_y_mm=h, start_angle=WALL_START,
                                     ref_wall_y_mm=90.0,
                                     base_taper_mm=3.5, base_helix_r=1.2,
                                     base_filament_r=0.75))
for i in range(len(col_x) - 1):
    h = min(back_col_h[i], back_col_h[i+1])
    PIECES.append(bridge_struts_x(col_x[i], col_x[i+1],
                                   offset_z_mm=BACK_Z, wall_y_mm=h,
                                   start_angle=WALL_START,
                                   ref_wall_y_mm=90.0))

# # ── Test lattice: frame + base lattice only ──
# TEST_SIZE = 35.0
# COL_SPACING = 2 * np.pi + 2 * 2.0
# BAR_W = 5.0
# PIECES = []
# PIECES.append(solid_frame(outer_x=TEST_SIZE, outer_z=TEST_SIZE,
#                            bar_w=BAR_W, bar_h=BAR_W))
# PIECES.append(base_lattice(grid_x_mm=TEST_SIZE, grid_z_mm=TEST_SIZE,
#                             spacing_mm=COL_SPACING))

# # ── Calibration grid: 3×3, strut thickness (rows) × grayscale exposure (cols) ──
# CELL_SIZE = 25.0
# BAR_W = 3.0
# CELL_GAP = 5.0
# COL_SPACING = 2 * np.pi + 2 * 2.0   # same as full box
#
# STRUT_DIAMETERS = [0.25, 0.35, 0.50]   # rows (Z axis)
# GRAY_LEVELS = [180, 215, 255]           # columns (X axis)

# PIECES = []
# for row, strut_d in enumerate(STRUT_DIAMETERS):
#     bar_h = strut_d / 2 + strut_d * 4   # match lattice total_h
#     for col, gray in enumerate(GRAY_LEVELS):
#         cx = (col - 1) * (CELL_SIZE + CELL_GAP)
#         cz = (row - 1) * (CELL_SIZE + CELL_GAP)
#
#         frame = solid_frame(outer_x=CELL_SIZE, outer_z=CELL_SIZE,
#                             bar_w=BAR_W, bar_h=bar_h)
#         frame['OFFSET_X_MM'] += cx
#         frame['OFFSET_Z_MM'] += cz
#
#         lattice = base_lattice(grid_x_mm=CELL_SIZE, grid_z_mm=CELL_SIZE,
#                                spacing_mm=COL_SPACING, strut_d_mm=strut_d)
#         lattice['OFFSET_X_MM'] += cx
#         lattice['OFFSET_Z_MM'] += cz
#
#         _arch_bridges(lattice, spacing_mm=COL_SPACING, arch_mm=1.0)
#         _punch_dot_holes(frame, col + 1)
#
#         if gray < 255:
#             _apply_grayscale(frame, gray)
#             _apply_grayscale(lattice, gray)
#
#         PIECES.append(frame)
#         PIECES.append(lattice)

# # ── Test piece: base frame + lattice + one wall ──
# HELIX_R = 2.0
# COL_SPACING = 2 * np.pi + 2 * HELIX_R
# TEST_SIZE = 35.0
# BAR_W = 5.0
# N_BASE_COLS = int(TEST_SIZE / COL_SPACING) + 2
# base_col = [(i - (N_BASE_COLS - 1) / 2) * COL_SPACING for i in range(N_BASE_COLS)]
# N_WALL_COLS = int(TEST_SIZE / COL_SPACING)
# wall_col = [(i - (N_WALL_COLS - 1) / 2) * COL_SPACING for i in range(N_WALL_COLS)]
# PIECES = []
# PIECES.append(solid_frame(outer_x=TEST_SIZE, outer_z=TEST_SIZE,
#                            bar_w=BAR_W, bar_h=BAR_W))
# PIECES.append(base_lattice(grid_x_mm=TEST_SIZE, grid_z_mm=TEST_SIZE,
#                             col_x=base_col, col_z=base_col,
#                             spacing_mm=COL_SPACING))
# WALL_Z = -TEST_SIZE / 2 + BAR_W / 2
# WALL_H = 25.12
# FRAME_Y = WALL_H + 2 * BAR_W
# PIECES.append(vertical_frame(frame_x=TEST_SIZE, frame_y=FRAME_Y,
#                               bar_w=BAR_W, y_offset=0.0,
#                               offset_z_mm=WALL_Z))
# WALL_START = np.pi / 2
# PIECES += [quad_spiral_column(offset_x_mm=x, offset_z_mm=WALL_Z, wall_y_mm=WALL_H,
#                               start_angle=WALL_START)
#            for x in wall_col]
# PIECES += [bridge_struts_x(wall_col[i], wall_col[i + 1],
#                             offset_z_mm=WALL_Z, wall_y_mm=WALL_H,
#                             start_angle=WALL_START)
#            for i in range(len(wall_col) - 1)]


# ── Post-process: outer chamfer ──
def apply_chamfer(pieces, outer_x_mm, outer_z_mm, chamfer_mm=1.0):
    """Wrap every piece's make_slice so the outer boundary tapers on low layers.

    At y=0 the outer rect is inset by chamfer_mm; at y=chamfer_mm it's full size.
    Affects all pieces whose bottom layers fall within the chamfer zone.
    """
    # Global outer boundary (centered at origin)
    x_lo, x_hi = -outer_x_mm / 2, outer_x_mm / 2
    z_lo, z_hi = -outer_z_mm / 2, outer_z_mm / 2

    for p in pieces:
        orig = p['make_slice']
        ox_p = p['OFFSET_X_MM']
        oz_p = p['OFFSET_Z_MM']
        oy_p = p['OFFSET_Y_MM']

        # Global coords for each pixel in this piece
        gx = ox_p + np.arange(p['W']) * PX_MM          # (W,)
        gz = oz_p + np.arange(p['H']) * PZ_MM           # (H,)

        def _wrapped(layer, _orig=orig, _gx=gx, _gz=gz, _oy=oy_p):
            y_mm = _oy + (layer + 0.5) * LY_MM
            img = _orig(layer)
            if y_mm >= chamfer_mm:
                return img
            inset = chamfer_mm - y_mm
            mask = ((_gx[None, :] >= x_lo + inset) & (_gx[None, :] <= x_hi - inset) &
                    (_gz[:, None] >= z_lo + inset) & (_gz[:, None] <= z_hi - inset))
            out = img.copy()
            out[~mask] = 0
            return out

        p['make_slice'] = _wrapped


# apply_chamfer(PIECES, outer_x_mm=TEST_SIZE, outer_z_mm=TEST_SIZE, chamfer_mm=1.0)

# ── Global bounding box from all pieces ──
def _compute_globals(pieces):
    extents = [(p['OFFSET_X_MM'], p['OFFSET_Y_MM'], p['OFFSET_Z_MM'],
                p['OFFSET_X_MM'] + p['W'] * PX_MM,
                p['OFFSET_Y_MM'] + p['N_SLICES'] * LY_MM,
                p['OFFSET_Z_MM'] + p['H'] * PZ_MM) for p in pieces]

    ox = min(e[0] for e in extents)
    oy = min(e[1] for e in extents)
    oz = min(e[2] for e in extents)

    W = int(np.ceil((max(e[3] for e in extents) - ox) / PX_MM))
    H = int(np.ceil((max(e[5] for e in extents) - oz) / PZ_MM))
    N = int(np.ceil((max(e[4] for e in extents) - oy) / LY_MM))

    return ox, oy, oz, W, H, N


OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM, W, H, N_SLICES = _compute_globals(PIECES)


def make_global_slice(layer):
    """Composite all pieces into a single (H, W) image for the given global layer."""
    img = np.zeros((H, W), dtype=np.uint8)
    for p in PIECES:
        dx = round((p['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        dz = round((p['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        dy = round((p['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
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


def encode_all():
    """Encode all active pieces into a single binary blob."""
    all_chunks = []
    total = 0

    for pi, p in enumerate(PIECES):
        t0p = time.time()
        dx = round((p['OFFSET_X_MM'] - OFFSET_X_MM) / PX_MM)
        dy = round((p['OFFSET_Y_MM'] - OFFSET_Y_MM) / LY_MM)
        dz = round((p['OFFSET_Z_MM'] - OFFSET_Z_MM) / PZ_MM)
        chunks, n = encode_surface_voxels(
            p['make_slice'], p['W'], p['H'], p['N_SLICES'], dx, dz, dy)
        all_chunks.extend(chunks)
        total += n
        print(f"  piece {pi+1}/{len(PIECES)}  {n:,} voxels  ({time.time()-t0p:.1f}s)", flush=True)

    return struct.pack('<I', total) + b''.join(all_chunks)


if __name__ == '__main__':
    print("Generating slices...", flush=True)
    t0 = time.time()
    blob = encode_all()

    with open(SLICES_FILE, 'wb') as f:
        f.write(blob)

    meta = {
        "width": W, "height": H, "num_slices": N_SLICES,
        "pixel_x_um": PIXEL_X_UM, "pixel_y_um": PIXEL_Y_UM, "layer_um": LAYER_UM,
        "plate_w_mm": PLATE_W_PX * 14.0 / 1000,
        "plate_h_mm": PLATE_H_PX * 19.0 / 1000,
        "offset_x_mm": OFFSET_X_MM,
        "offset_y_mm": OFFSET_Y_MM,
        "offset_z_mm": OFFSET_Z_MM,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    print(f"Done in {time.time() - t0:.1f}s  ({len(blob):,} bytes)")
