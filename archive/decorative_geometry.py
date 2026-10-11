"""Archived decorative geometry: ropes, braids, textured columns.

Moved out of slicer_preview/geometry.py. To use again, copy back or add
slicer_preview to sys.path.
"""
import numpy as np
from printer import PX_MM, PZ_MM, LY_MM


def rope_column(helix_r=1.0, filament_r=1.0, pitch=10.0,
                wall_y_mm=90.0, y_offset=5.0,
                offset_x_mm=-60.0, offset_z_mm=0.0, margin=1.0):
    """Single rope column: two helical filaments twisted around a shared axis."""
    rope_od = 2 * (helix_r + filament_r)
    img_x = rope_od + 2 * margin
    img_z = rope_od + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    rope_cx = img_x / 2
    rope_cz = img_z / 2

    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        theta = 2 * np.pi * y_mm / pitch

        img = np.zeros((H, W), np.uint8)
        for phase in (0, np.pi):
            cx = rope_cx + helix_r * np.cos(theta + phase)
            cz = rope_cz + helix_r * np.sin(theta + phase)
            dist_sq = (X2D - cx)**2 + (Z2D - cz)**2
            img[dist_sq <= filament_r**2] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - rope_cx,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - rope_cz,
        make_slice=make_slice,
    )


def diagonal_rope(x_start=-57.5, x_end=57.5, helix_r=1.0, filament_r=1.0,
                  n_twists=15, wall_y_mm=90.0, y_offset=5.0,
                  offset_z_mm=-45.0, margin=0.5):
    """Two-strand rope along an arbitrary diagonal axis in the XY plane."""
    dx = x_end - x_start
    L = np.sqrt(dx ** 2 + wall_y_mm ** 2)
    ny = wall_y_mm / L

    # Compensate lateral amplitude for axis projection onto X
    lat_r = helix_r / ny if ny > 0.01 else helix_r

    swing = helix_r + filament_r
    x_lo = min(x_start, x_end) - swing - margin
    x_hi = max(x_start, x_end) + swing + margin
    img_x = x_hi - x_lo
    img_z = 2 * swing + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cz = img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]
    filament_r_sq = filament_r ** 2

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        t = y_mm / wall_y_mm
        x_axis = (x_start + t * dx) - x_lo
        omega_t = 2 * np.pi * n_twists * t

        img = np.zeros((H, W), np.uint8)
        for phase in (0.0, np.pi):
            x_fil = x_axis - lat_r * np.cos(omega_t + phase) * ny
            z_fil = cz + helix_r * np.sin(omega_t + phase)
            dist_sq = (X2D - x_fil) ** 2 + (Z2D - z_fil) ** 2
            img[dist_sq <= filament_r_sq] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_lo,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


# ── Textured column helper ──

def _textured_column(texture_fn, base_r=2.0, cut_depth=0.4, groove_width=0.3,
                     wall_y_mm=90.0, y_offset=5.0,
                     offset_x_mm=-60.0, offset_z_mm=0.0, margin=0.5,
                     **tex_kwargs):
    """Solid column with a pluggable surface texture.

    texture_fn(U, y_mm, base_r, cut_depth, groove_width, **tex_kwargs)
        returns eff_r array (H, W) — effective radius at each pixel.
    """
    img_side = 2 * base_r + 2 * margin
    W = int(np.ceil(img_side / PX_MM))
    H = int(np.ceil(img_side / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))
    col_cx = img_side / 2
    col_cz = img_side / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]
    DIST = np.sqrt((X2D - col_cx)**2 + (Z2D - col_cz)**2)
    THETA = np.arctan2(Z2D - col_cz, X2D - col_cx)
    U = THETA * base_r

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        eff_r = texture_fn(U, y_mm, base_r, cut_depth, groove_width, **tex_kwargs)
        img = np.zeros((H, W), np.uint8)
        img[DIST <= eff_r] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - col_cx,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - col_cz,
        make_slice=make_slice,
    )


def _snap_spacing(spacing, base_r):
    """Adjust spacing so an integer number of cells fits the circumference."""
    circ = 2 * np.pi * base_r
    n = max(1, round(circ / spacing))
    return circ / n


def grid_column(spacing=2.0, **kwargs):
    """Column of orthogonal grid lines — no solid base."""
    base_r = kwargs.get('base_r', 2.0)
    spacing = _snap_spacing(spacing, base_r)

    def _tex(U, y_mm, base_r, cut_depth, groove_width, spacing):
        u_mod = np.mod(U, spacing)
        dist_v = np.minimum(u_mod, spacing - u_mod)
        y_mod = y_mm % spacing
        dist_h = min(y_mod, spacing - y_mod)
        on_line = (dist_v < groove_width / 2) | (dist_h < groove_width / 2)
        return np.where(on_line, base_r, 0.0)
    return _textured_column(_tex, spacing=spacing, **kwargs)


def diamond_column(spacing=2.0, core_r=0.5, **kwargs):
    """Column of diamond (45deg) grid lines with a solid core."""
    base_r = kwargs.get('base_r', 2.0)
    spacing = _snap_spacing(spacing, base_r)

    def _tex(U, y_mm, base_r, cut_depth, groove_width, spacing, core_r):
        d1 = np.mod(U + y_mm, spacing)
        dist1 = np.minimum(d1, spacing - d1)
        d2 = np.mod(U - y_mm, spacing)
        dist2 = np.minimum(d2, spacing - d2)
        on_line = (dist1 < groove_width / 2) | (dist2 < groove_width / 2)
        return np.where(on_line, base_r, core_r)
    return _textured_column(_tex, spacing=spacing, core_r=core_r, **kwargs)


def hex_column(hex_size=1.5, **kwargs):
    """Solid column with hexagonal grid on the surface (raised edges)."""
    base_r = kwargs.get('base_r', 2.0)
    # Snap so hex columns tile around circumference
    # Hex column-to-column distance in U direction is sqrt(3) * hex_size
    s3 = np.sqrt(3)
    col_pitch = s3 * hex_size
    circ = 2 * np.pi * base_r
    n = max(1, round(circ / col_pitch))
    hex_size = (circ / n) / s3

    s3_2 = s3 / 2
    s3_3 = s3 / 3
    apothem = hex_size * s3_2

    def _tex(U, y_mm, base_r, cut_depth, groove_width, hex_size):
        qf = (s3_3 * U - y_mm / 3) / hex_size
        rf_val = (2 * y_mm / 3) / hex_size
        sf = -qf - rf_val

        qi = np.round(qf)
        ri = np.full_like(qi, round(rf_val))
        si = np.round(sf)

        q_diff = np.abs(qi - qf)
        r_diff = np.abs(ri - rf_val)
        s_diff = np.abs(si - sf)

        fix_q = (q_diff > r_diff) & (q_diff > s_diff)
        fix_s = ~fix_q & (s_diff > r_diff)
        qi = np.where(fix_q, -ri - si, qi)
        ri = np.where(fix_s, -qi - si, ri)

        u_center = hex_size * (s3 * qi + s3_2 * ri)
        v_center = hex_size * 1.5 * ri

        du = U - u_center
        dv = y_mm - v_center

        p1 = np.abs(du)
        p2 = np.abs(0.5 * du + s3_2 * dv)
        p3 = np.abs(0.5 * du - s3_2 * dv)
        edge_dist = apothem - np.maximum(np.maximum(p1, p2), p3)

        return np.where(edge_dist < groove_width / 2, base_r, base_r - cut_depth)

    return _textured_column(_tex, hex_size=hex_size, **kwargs)


def braid(x_start=-57.5, x_end=57.5, lateral_r=2.0, depth_r=1.0,
          filament_r=0.75, n_strands=3, n_twists=15,
          wall_y_mm=90.0, y_offset=5.0,
          offset_z_mm=-45.0, margin=0.5):
    """N-strand pigtail braid spanning diagonally across a wall in the XY plane.

    Strands oscillate side-to-side (lateral_r) in the wall plane and
    weave over/under each other in depth (depth_r).  lateral_r >> depth_r
    gives the classic braided look; lateral_r == depth_r gives a rope.
    """
    dx = x_end - x_start
    L = np.sqrt(dx ** 2 + wall_y_mm ** 2)
    ny = wall_y_mm / L  # normalized axis Y component

    # Compensate lateral amplitude for axis projection onto X
    lat_r = lateral_r / ny if ny > 0.01 else lateral_r

    # Image X spans the full diagonal sweep + apparent lateral swing
    swing = lateral_r + filament_r
    x_lo = min(x_start, x_end) - swing - margin
    x_hi = max(x_start, x_end) + swing + margin
    img_x = x_hi - x_lo
    img_z = 2 * (depth_r + filament_r) + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cz = img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    filament_r_sq = filament_r ** 2
    phases = [i * 2 * np.pi / n_strands for i in range(n_strands)]

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        t = y_mm / wall_y_mm
        x_axis = (x_start + t * dx) - x_lo  # in image coords

        img = np.zeros((H, W), np.uint8)
        for phase in phases:
            theta = 2 * np.pi * n_twists * t + phase
            x_fil = x_axis - lat_r * np.cos(theta) * ny
            z_fil = cz + depth_r * np.sin(theta)
            dist_sq = (X2D - x_fil) ** 2 + (Z2D - z_fil) ** 2
            img[dist_sq <= filament_r_sq] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_lo,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def maypole_braid(x_start=-57.5, x_end=57.5, helix_r=1.5, filament_r=0.75,
                  n_twists=15, wall_y_mm=90.0, y_offset=5.0,
                  offset_z_mm=-45.0, margin=0.5):
    """4-strand maypole braid: 2 CW + 2 CCW strands weaving around the axis.

    Counter-rotating strands naturally cross over/under each other.
    The trick: CCW strands reverse lateral direction but keep depth direction,
    so at each crossing the two strands are on opposite sides of the wall.
    """
    dx = x_end - x_start
    L = np.sqrt(dx ** 2 + wall_y_mm ** 2)
    ny = wall_y_mm / L

    # Compensate lateral amplitude for axis projection onto X
    lat_r = helix_r / ny if ny > 0.01 else helix_r

    swing = helix_r + filament_r  # apparent swing after projection
    x_lo = min(x_start, x_end) - swing - margin
    x_hi = max(x_start, x_end) + swing + margin
    img_x = x_hi - x_lo
    img_z = 2 * (helix_r + filament_r) + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cz = img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    filament_r_sq = filament_r ** 2

    # 2 CW strands (phases 0, π) + 2 CCW strands (phases π/2, 3π/2)
    strands = [
        (0.0, +1),
        (np.pi, +1),
        (np.pi / 2, -1),
        (3 * np.pi / 2, -1),
    ]

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        t = y_mm / wall_y_mm
        x_axis = (x_start + t * dx) - x_lo
        omega_t = 2 * np.pi * n_twists * t

        img = np.zeros((H, W), np.uint8)
        for phase, direction in strands:
            if direction > 0:
                lat_angle = omega_t + phase
                dep_angle = omega_t + phase
            else:
                lat_angle = -omega_t + phase
                dep_angle = omega_t - phase
            x_fil = x_axis - lat_r * np.cos(lat_angle) * ny
            z_fil = cz + helix_r * np.sin(dep_angle)
            dist_sq = (X2D - x_fil) ** 2 + (Z2D - z_fil) ** 2
            img[dist_sq <= filament_r_sq] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_lo,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def horizontal_maypole_braid(y_center_mm, x_start=-57.5, x_end=57.5, helix_r=1.5,
                              filament_r=0.75, n_twists=12, y_offset=5.0,
                              offset_z_mm=-45.0, margin=0.5):
    """4-strand maypole braid with horizontal axis (along X).

    Same strand topology as maypole_braid but axis runs along X at a fixed Y
    height.  Each slice intersects the helical strands at discrete points;
    the solver finds all crossing t-values analytically.
    """
    dx = x_end - x_start
    swing = helix_r + filament_r
    omega = 2 * np.pi * n_twists          # total winding angle over t ∈ [0,1]

    x_lo = min(x_start, x_end) - filament_r - margin
    x_hi = max(x_start, x_end) + filament_r + margin
    img_x = x_hi - x_lo
    img_z = 2 * swing + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))

    # Piece spans only the narrow Y band where strands exist
    y_band = swing + margin
    N_SLICES = int(np.ceil(2 * y_band / LY_MM))

    cz = img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]
    filament_r_sq = filament_r ** 2

    strands = [
        (0.0, +1),                # CW, phase 0
        (np.pi, +1),              # CW, phase π
        (np.pi / 2, -1),          # CCW, phase π/2
        (3 * np.pi / 2, -1),      # CCW, phase 3π/2
    ]

    def make_slice(layer):
        y_local = (layer + 0.5) * LY_MM
        y_wall = (y_center_mm - y_band) + y_local
        delta = (y_wall - y_center_mm) / helix_r

        img = np.zeros((H, W), np.uint8)
        if abs(delta) > 1.0:
            return img

        base_angle = np.arccos(np.clip(delta, -1.0, 1.0))

        for phase, direction in strands:
            # Solve for all t ∈ [0,1] where lateral cos(angle(t)) = delta
            k_vals = np.arange(-n_twists - 1, n_twists + 2)
            for sign in (+1, -1):
                targets = sign * base_angle + 2 * np.pi * k_vals
                if direction > 0:        # CW: ω·t + φ = target
                    t_vals = (targets - phase) / omega
                else:                    # CCW: −ω·t + φ = target
                    t_vals = (phase - targets) / omega
                t_vals = t_vals[(t_vals >= 0) & (t_vals <= 1)]

                for t in t_vals:
                    x_strand = (x_start + t * dx) - x_lo
                    dep_angle = (omega * t + phase) if direction > 0 else (omega * t - phase)
                    z_strand = cz + helix_r * np.sin(dep_angle)
                    dist_sq = (X2D - x_strand) ** 2 + (Z2D - z_strand) ** 2
                    img[dist_sq <= filament_r_sq] = 255

        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_lo,
        OFFSET_Y_MM=y_offset + y_center_mm - y_band,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


