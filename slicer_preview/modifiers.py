"""Post-process modifiers: wrap a piece's make_slice to carve or reshape it.

Each modifier mutates piece['make_slice'] in place.
"""
import numpy as np
from printer import PX_MM, PZ_MM, LY_MM


def arch_top_beams(piece, side_walls, front_back_walls,
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


def apply_grayscale(piece, gray):
    """Wrap make_slice to output gray instead of 255."""
    orig = piece['make_slice']
    def _scaled(layer, _orig=orig, _g=gray):
        return np.where(_orig(layer) > 0, _g, np.uint8(0)).astype(np.uint8)
    piece['make_slice'] = _scaled


def punch_dot_holes(piece, n_dots, dot_depth_mm=0.5, dot_r=1.0, dot_spacing=3.0):
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


def vault_base(piece, col_x, col_z, vault_mm=2.0):
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
