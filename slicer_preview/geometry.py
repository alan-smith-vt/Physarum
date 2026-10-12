"""Geometry primitives: lattice grids, rope columns, etc.

Each function returns a 'piece' dict with keys:
    W, H, N_SLICES  — image dimensions and layer count
    OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM — world position
    make_slice(layer) — callable returning uint8 image (H, W)
"""
import numpy as np
from printer import PX_MM, PZ_MM, LY_MM


def solid_frame(outer_x=30.0, outer_z=30.0, bar_w=5.0, bar_h=5.0):
    """Solid rectangular frame (square ring) in the XZ plane."""
    W = int(np.ceil(outer_x / PX_MM))
    H = int(np.ceil(outer_z / PZ_MM))
    N_SLICES = int(np.ceil(bar_h / LY_MM))

    px = np.arange(W) * PX_MM
    pz = np.arange(H) * PZ_MM

    inner = ((px[None, :] >= bar_w) & (px[None, :] <= outer_x - bar_w) &
             (pz[:, None] >= bar_w) & (pz[:, None] <= outer_z - bar_w))
    ring = (~inner).astype(np.uint8) * 255

    def make_slice(_layer):
        return ring

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=-outer_x / 2,
        OFFSET_Y_MM=0.0,
        OFFSET_Z_MM=-outer_z / 2,
        make_slice=make_slice,
    )


def vertical_frame(frame_x=30.0, frame_y=30.0, bar_w=5.0,
                   y_offset=5.0, offset_x_mm=0.0, offset_z_mm=0.0):
    """Solid rectangular frame standing vertically in the XY plane.

    frame_x × frame_y outer dimensions, bar_w square bars, bar_w deep in Z.
    """
    W = int(np.ceil(frame_x / PX_MM))
    H = int(np.ceil(bar_w / PZ_MM))
    N_SLICES = int(np.ceil(frame_y / LY_MM))

    px = np.arange(W) * PX_MM

    full_bar = np.full((H, W), 255, np.uint8)
    posts = np.zeros((H, W), np.uint8)
    posts[:, px <= bar_w] = 255
    posts[:, px >= frame_x - bar_w] = 255

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        if y_mm < bar_w or y_mm > frame_y - bar_w:
            return full_bar
        return posts

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - frame_x / 2,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - bar_w / 2,
        make_slice=make_slice,
    )


def base_lattice(grid_x_mm=115.0, grid_z_mm=90.0, spacing_mm=10.0,
                 col_x=None, col_z=None, strut_d_mm=1.0):
    """Flat lattice grid + diamond diagonals in the XZ plane.

    If col_x / col_z are given (world coords, centered on 0) the struts
    align to those positions.  Otherwise positions are centred automatically.
    """
    strut_r = strut_d_mm / 2
    # Cross-section: semicircle bottom (0..strut_r) + square top (strut_r..strut_r+strut_d_mm)
    total_h = strut_r + strut_d_mm * 4        # 0.5 + 1.0 = 1.5 mm
    W = int(np.ceil(grid_x_mm / PX_MM))
    H = int(np.ceil(grid_z_mm / PZ_MM))
    N_SLICES = int(np.ceil(total_h / LY_MM))

    half_x, half_z = grid_x_mm / 2, grid_z_mm / 2
    eps = 0.01

    # World → local, clip to strict interior (no edge struts)
    def _to_local(world_pos, half, span):
        local = [w + half for w in world_pos]
        return np.array([v for v in local if eps < v < span - eps])

    def _default_positions(span):
        n = int(span / spacing_mm) + 2
        return [(i - (n - 1) / 2) * spacing_mm for i in range(n)]

    x_world = list(col_x) if col_x is not None else _default_positions(grid_x_mm)
    z_world = list(col_z) if col_z is not None else _default_positions(grid_z_mm)

    X_struts = _to_local(x_world, half_x, grid_x_mm)
    Z_struts = _to_local(z_world, half_z, grid_z_mm)

    # Cell size (uniform spacing assumed)
    sx = sz = spacing_mm
    diag_norm = np.sqrt(sx ** 2 + sz ** 2)

    # Grid nodes for diagonal enumeration — only actual column positions
    x_local_all = sorted([x + half_x for x in x_world
                          if -half_x - eps <= x <= half_x + eps])
    z_local_all = sorted([z + half_z for z in z_world
                          if -half_z - eps <= z <= half_z + eps])

    diag_pos_b = np.unique([sz * x - sx * z
                            for x in x_local_all for z in z_local_all])
    diag_neg_b = np.unique([sz * x + sx * z
                            for x in x_local_all for z in z_local_all])

    # Precomputed pixel grids
    px_mm = np.arange(W) * PX_MM
    pz_mm = np.arange(H) * PZ_MM
    diag_pos_field = sz * px_mm[None, :] - sx * pz_mm[:, None]
    diag_neg_field = sz * px_mm[None, :] + sx * pz_mm[:, None]

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        if y_mm >= total_h:
            return np.zeros((H, W), np.uint8)

        # Semicircle region: 0 .. strut_d_mm  (circle centre at strut_r)
        # Square region:     strut_r .. strut_r + strut_d_mm
        in_circle = y_mm < strut_d_mm
        in_square = y_mm >= strut_r

        if in_circle:
            dy = y_mm - strut_r
            circ_r = np.sqrt(strut_r ** 2 - dy ** 2)
        else:
            circ_r = 0.0

        eff_r = max(circ_r, strut_r if in_square else 0.0)
        img = np.zeros((H, W), np.uint8)

        if in_circle and not in_square:
            # Pure semicircle — variable width
            for zc in Z_struts:
                img[np.abs(pz_mm - zc) <= circ_r, :] = 255
            for xc in X_struts:
                img[:, np.abs(px_mm - xc) <= circ_r] = 255
            diag_r = circ_r * diag_norm
        else:
            # Square (or overlap) — constant strut_r half-width
            for zc in Z_struts:
                img[np.abs(pz_mm - zc) <= strut_r, :] = 255
            for xc in X_struts:
                img[:, np.abs(px_mm - xc) <= strut_r] = 255
            diag_r = strut_r * diag_norm

        for b in diag_pos_b:
            img[np.abs(diag_pos_field - b) <= diag_r] = 255
        for b in diag_neg_b:
            img[np.abs(diag_neg_field - b) <= diag_r] = 255

        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=-half_x,
        OFFSET_Y_MM=0.0,
        OFFSET_Z_MM=-half_z,
        make_slice=make_slice,
    )


def solid_column(radius=1.5, wall_y_mm=90.0, y_offset=5.0,
                 offset_x_mm=0.0, offset_z_mm=0.0, margin=0.5):
    """Solid vertical cylinder along Y."""
    img_x = 2 * radius + 2 * margin
    img_z = 2 * radius + 2 * margin
    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cx, cz = img_x / 2, img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]
    disk = ((X2D - cx) ** 2 + (Z2D - cz) ** 2 <= radius ** 2).astype(np.uint8) * 255

    def make_slice(_layer):
        return disk

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - cx,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def horizontal_solid_column(y_center_mm, x_start=-57.5, x_end=57.5, radius=1.5,
                            y_offset=5.0, offset_z_mm=0.0, margin=0.5):
    """Solid horizontal cylinder along X at a fixed Y height."""
    x_lo = min(x_start, x_end) - margin
    x_hi = max(x_start, x_end) + margin
    img_x = x_hi - x_lo
    img_z = 2 * radius + 2 * margin

    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))

    y_band = radius + margin
    N_SLICES = int(np.ceil(2 * y_band / LY_MM))

    cz = img_z / 2
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    def make_slice(layer):
        y_local = (layer + 0.5) * LY_MM
        y_wall = (y_center_mm - y_band) + y_local
        dy = y_wall - y_center_mm
        r_slice_sq = radius ** 2 - dy ** 2
        if r_slice_sq <= 0:
            return np.zeros((H, W), np.uint8)
        # Cross-section is a horizontal band in Z at this Y cut
        band = ((Z2D - cz) ** 2 <= r_slice_sq)   # (H, 1) bool
        return np.where(np.broadcast_to(band, (H, W)), np.uint8(255), np.uint8(0))

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_lo,
        OFFSET_Y_MM=y_offset + y_center_mm - y_band,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def quad_spiral_column(helix_r=2.0, filament_r=0.35, core_r=0.35,
                       helix_angle=45.0, wall_y_mm=90.0, y_offset=5.0,
                       offset_x_mm=-60.0, offset_z_mm=0.0, margin=1.0,
                       start_angle=0.0, ref_wall_y_mm=None,
                       base_taper_mm=0.0, base_helix_r=None,
                       base_filament_r=None):
    """Four helical filaments around a solid core, two pairs CW+CCW.

    Args:
        helix_r: distance from column axis to filament center (mm)
        filament_r: filament radius (mm)
        core_r: solid core radius (mm)
        helix_angle: spiral angle in degrees (0=horizontal ring, 90=vertical)
        start_angle: base angle of first filament pair (radians)
        ref_wall_y_mm: reference height for k-snapping (defaults to wall_y_mm)
        base_taper_mm: taper zone height from bottom (0 = no taper)
        base_helix_r: helix radius at bottom (tapers to helix_r)
        base_filament_r: filament radius at bottom (tapers to filament_r)
    """
    if ref_wall_y_mm is None:
        ref_wall_y_mm = wall_y_mm
    img_side = 2 * (helix_r + filament_r) + 2 * margin
    W = int(np.ceil(img_side / PX_MM))
    H = int(np.ceil(img_side / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cx = img_side / 2
    cz = img_side / 2

    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    # k = angular rate (rad/mm of height), snapped to integer revolutions
    k_raw = 1.0 / (helix_r * np.tan(np.radians(helix_angle)))
    n_revs = max(1, round(k_raw * ref_wall_y_mm / (2 * np.pi)))
    k = 2 * np.pi * n_revs / ref_wall_y_mm

    filaments = [
        (start_angle, +k),
        (start_angle, -k),
        (start_angle + np.pi, +k),
        (start_angle + np.pi, -k),
    ]
    filament_r_sq = filament_r ** 2
    core_r_sq = core_r ** 2
    DIST_CENTER_SQ = (X2D - cx) ** 2 + (Z2D - cz) ** 2

    # Intersection geometry:
    # Pairs (1,2) & (3,4) meet at y = n*π/k, angles {-π/2, +π/2}
    # Pairs (1,4) & (2,3) meet at y = (2m-1)*π/(2k), angles {0, π}
    # Combined: every π/(2k), alternating angle sets
    int_step = np.pi / (2 * k)
    tan_angle = np.tan(np.radians(helix_angle))
    dr = helix_r - core_r
    dy_strut = dr * tan_angle
    seg_len_sq = dr ** 2 + dy_strut ** 2

    # Precompute pixel offsets from center
    DX = X2D - cx
    DZ = Z2D - cz

    _do_taper = (base_taper_mm > 0 and base_helix_r is not None
                 and base_filament_r is not None)

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        img = np.zeros((H, W), np.uint8)

        # Per-layer taper: helix_r and filament_r both interpolate
        if _do_taper and y_mm < base_taper_mm:
            frac = y_mm / base_taper_mm
            ehr = base_helix_r + (helix_r - base_helix_r) * frac
            efr = base_filament_r + (filament_r - base_filament_r) * frac
            efr_sq = efr ** 2
            edr = ehr - core_r
            edy = edr * tan_angle
            esl = edr ** 2 + edy ** 2
        else:
            ehr = helix_r
            efr = filament_r
            efr_sq = filament_r_sq
            edr = dr
            edy = dy_strut
            esl = seg_len_sq

        # Solid core
        img[DIST_CENTER_SQ <= core_r_sq] = 255
        # Spiral filaments
        for theta0, ki in filaments:
            theta = theta0 + ki * y_mm
            fx = cx + ehr * np.cos(theta)
            fz = cz + ehr * np.sin(theta)
            dist_sq = (X2D - fx) ** 2 + (Z2D - fz) ** 2
            img[dist_sq <= efr_sq] = 255
        # Struts: 3D distance to angled line segment
        m_lo = int(np.floor((y_mm - edy - efr) / int_step))
        m_hi = int(np.ceil((y_mm + edy + efr) / int_step))
        for m in range(max(0, m_lo), m_hi + 1):
            y_int = m * int_step
            if m % 2 == 0:
                angles = [start_angle, start_angle + np.pi]
            else:
                angles = [start_angle + np.pi / 2, start_angle - np.pi / 2]
            for a in angles:
                cos_a = np.cos(a)
                sin_a = np.sin(a)
                # Pixel radial/perp decomposition at this angle
                along = DX * cos_a + DZ * sin_a
                perp_sq = (DX * sin_a - DZ * cos_a) ** 2
                for sign in (+1, -1):
                    y_core = y_int + sign * edy
                    if y_mm < min(y_int, y_core) - efr:
                        continue
                    if y_mm > max(y_int, y_core) + efr:
                        continue
                    # Segment: (core_r, y_core) → (ehr, y_int) in (along, y)
                    D_y = y_int - y_core
                    # Project pixel onto segment (parameter t)
                    t = np.clip(((along - core_r) * edr +
                                 (y_mm - y_core) * D_y) / esl, 0.0, 1.0)
                    near_r = core_r + t * edr
                    near_y = y_core + t * D_y
                    dist_sq = (along - near_r) ** 2 + perp_sq + (y_mm - near_y) ** 2
                    img[dist_sq <= efr_sq] = 255
        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - cx,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def bridge_struts(col_a_z, col_b_z, helix_r=2.0, filament_r=0.35, core_r=0.35,
                  helix_angle=45.0, wall_y_mm=90.0, y_offset=5.0,
                  offset_x_mm=-60.0, margin=0.5, start_angle=0.0,
                  taper_mm=0.0, taper_r=None, taper_end=None):
    """Angled struts bridging two spiral columns along the Z axis.

    Connects the Z-axis intersection struts between adjacent columns
    with tubes at the same helix angle as the column's internal struts.

    taper_mm/taper_r/taper_end: one-sided taper at end 'a' (min-Z column)
        or 'b' (max-Z column).  Radius tapers from taper_r at the frame
        edge to filament_r over taper_mm in Z.
    """
    # Match the column's snapped k
    k_raw = 1.0 / (helix_r * np.tan(np.radians(helix_angle)))
    n_revs = max(1, round(k_raw * wall_y_mm / (2 * np.pi)))
    k = 2 * np.pi * n_revs / wall_y_mm
    int_step = np.pi / (2 * k)

    # Gap between column surfaces (helix_r to helix_r)
    gap = abs(col_b_z - col_a_z) - 2 * helix_r

    # Number of int_steps the bridge covers in Y
    horiz_per_step = np.pi * helix_r / 2
    n_z_steps = max(1, round(gap / horiz_per_step))
    dy_bridge = n_z_steps * int_step

    seg_len_sq = gap ** 2 + dy_bridge ** 2

    # Image padding per end (taper end needs more room)
    _do_taper = taper_mm > 0 and taper_r is not None and taper_end in ('a', 'b')
    pad_a = taper_r if (_do_taper and taper_end == 'a') else filament_r
    pad_b = taper_r if (_do_taper and taper_end == 'b') else filament_r
    max_r = max(pad_a, pad_b)

    img_x = 2 * max_r + 2 * margin
    img_z = gap + pad_a + pad_b
    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cx = img_x / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    DX_sq = (X2D - cx) ** 2
    z_local = Z2D - pad_a  # 0 = col A surface, gap = col B surface

    filament_r_sq = filament_r ** 2

    # Build segment list: (y_start, y_end) for each bridge strut
    # Z-axis struts (angles ±π/2) are at even m when start_angle≈±π/2,
    # odd m when start_angle≈0 or π.
    z_at_odd = abs(np.cos(start_angle)) > abs(np.sin(start_angle))
    m_start = 1 if z_at_odd else 0
    m_max = int(np.ceil(wall_y_mm / int_step))
    segments = []
    for m in range(m_start, m_max + 1, 2):
        y_int = m * int_step
        y_int = m * int_step
        # Skip intersections above the column height
        if y_int > wall_y_mm:
            break
        # "Up": from (z=0, y_int) to (z=gap, y_int + dy_bridge)
        if y_int + dy_bridge <= wall_y_mm + filament_r:
            segments.append((y_int, y_int + dy_bridge))
        # "Down": from (z=0, y_int) to (z=gap, y_int - dy_bridge)
        if y_int - dy_bridge >= -filament_r:
            segments.append((y_int, y_int - dy_bridge))

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        img = np.zeros((H, W), np.uint8)

        for (y_a, y_b) in segments:
            y_lo = min(y_a, y_b) - filament_r
            y_hi = max(y_a, y_b) + filament_r
            if y_mm < y_lo or y_mm > y_hi:
                continue

            dy = y_b - y_a
            t = np.clip(((y_mm - y_a) * dy + z_local * gap) / seg_len_sq,
                        0.0, 1.0)
            near_y = y_a + t * dy
            near_z = t * gap
            dist_sq = DX_sq + (y_mm - near_y) ** 2 + (z_local - near_z) ** 2

            if _do_taper:
                d = near_z if taper_end == 'a' else (gap - near_z)
                taper_frac = np.clip(d / taper_mm, 0, 1)
                eff_r_sq = (taper_r + (filament_r - taper_r) * taper_frac) ** 2
                img[dist_sq <= eff_r_sq] = 255
            else:
                img[dist_sq <= filament_r_sq] = 255

        return img

    z_min = min(col_a_z, col_b_z)
    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=offset_x_mm - cx,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=z_min + helix_r - pad_a,
        make_slice=make_slice,
    )


def bridge_struts_x(col_a_x, col_b_x, helix_r=2.0, filament_r=0.35, core_r=0.35,
                    helix_angle=45.0, wall_y_mm=90.0, y_offset=5.0,
                    offset_z_mm=0.0, margin=0.5, start_angle=0.0,
                    ref_wall_y_mm=None):
    """Angled struts bridging two spiral columns along the X axis.

    Same as bridge_struts but the gap spans X (columns tiled along X).
    Uses X-axis intersection points instead of Z-axis ones.
    """
    if ref_wall_y_mm is None:
        ref_wall_y_mm = wall_y_mm
    k_raw = 1.0 / (helix_r * np.tan(np.radians(helix_angle)))
    n_revs = max(1, round(k_raw * ref_wall_y_mm / (2 * np.pi)))
    k = 2 * np.pi * n_revs / ref_wall_y_mm
    int_step = np.pi / (2 * k)

    gap = abs(col_b_x - col_a_x) - 2 * helix_r

    horiz_per_step = np.pi * helix_r / 2
    n_x_steps = max(1, round(gap / horiz_per_step))
    dy_bridge = n_x_steps * int_step

    seg_len_sq = gap ** 2 + dy_bridge ** 2

    # Image: gap-sized in X, thin in Z
    img_x = gap + 2 * filament_r
    img_z = 2 * filament_r + 2 * margin
    W = int(np.ceil(img_x / PX_MM))
    H = int(np.ceil(img_z / PZ_MM))
    N_SLICES = int(np.ceil(wall_y_mm / LY_MM))

    cz = img_z / 2
    X2D = (np.arange(W) * PX_MM)[None, :]
    Z2D = (np.arange(H) * PZ_MM)[:, None]

    x_local = X2D - filament_r          # 0 = col A surface, gap = col B surface
    DZ_sq = (Z2D - cz) ** 2

    filament_r_sq = filament_r ** 2

    # X-axis intersections (angles 0, π): even m when start_angle≈0,
    # odd m when start_angle≈±π/2.
    z_at_odd = abs(np.cos(start_angle)) > abs(np.sin(start_angle))
    m_start = 0 if z_at_odd else 1      # flipped vs bridge_struts
    m_max = int(np.ceil(wall_y_mm / int_step))
    segments = []
    for m in range(m_start, m_max + 1, 2):
        y_int = m * int_step
        if y_int > wall_y_mm:
            break
        if y_int + dy_bridge <= wall_y_mm + filament_r:
            segments.append((y_int, y_int + dy_bridge))
        if y_int - dy_bridge >= -filament_r:
            segments.append((y_int, y_int - dy_bridge))

    def make_slice(layer):
        y_mm = (layer + 0.5) * LY_MM
        img = np.zeros((H, W), np.uint8)

        for (y_a, y_b) in segments:
            y_lo = min(y_a, y_b) - filament_r
            y_hi = max(y_a, y_b) + filament_r
            if y_mm < y_lo or y_mm > y_hi:
                continue

            dy = y_b - y_a
            t = np.clip(((y_mm - y_a) * dy + x_local * gap) / seg_len_sq,
                        0.0, 1.0)
            near_y = y_a + t * dy
            near_x = t * gap
            dist_sq = DZ_sq + (y_mm - near_y) ** 2 + (x_local - near_x) ** 2
            img[dist_sq <= filament_r_sq] = 255

        return img

    x_min = min(col_a_x, col_b_x)
    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x_min + helix_r - filament_r,
        OFFSET_Y_MM=y_offset,
        OFFSET_Z_MM=offset_z_mm - cz,
        make_slice=make_slice,
    )


def tpms_block(size_x=10.0, size_y=10.0, size_z=10.0, period_mm=5.0, threshold=0.2):
    """Rectangular block filled with a TPMS lattice (same field as supports.py).

    Solid where |f| < threshold. Measured solid fraction for this field:
    0.15 → 15%, 0.3 → 30%, 0.6 → 60%, 0.94 → 90%.
    """
    W = int(np.ceil(size_x / PX_MM))
    H = int(np.ceil(size_z / PZ_MM))
    N_SLICES = int(np.ceil(size_y / LY_MM))

    k = 2.0 * np.pi / period_mm
    sx = np.arange(W) * PX_MM * k
    sz = np.arange(H) * PZ_MM * k
    sin_x, cos_x, sin_2x = np.sin(sx), np.cos(sx), np.sin(2 * sx)
    sin_z, cos_z, sin_2z = np.sin(sz), np.cos(sz), np.sin(2 * sz)

    def make_slice(layer):
        y = (layer + 0.5) * LY_MM * k
        f = (sin_2x[None, :] * np.cos(y) * sin_z[:, None] +
             np.sin(2 * y) * cos_z[:, None] * sin_x[None, :] +
             sin_2z[:, None] * cos_x[None, :] * np.sin(y))
        return (np.abs(f) < threshold).astype(np.uint8) * 255

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=-size_x / 2,
        OFFSET_Y_MM=0.0,
        OFFSET_Z_MM=-size_z / 2,
        make_slice=make_slice,
    )


def implicit_piece(size_x, size_y, size_z, solid_fn, x0=0.0, y0=0.0, z0=0.0):
    """General piece from a solid test: solid_fn(x, y, z) → bool array.

    x is (1, W), z is (H, 1) — local mm from the piece's corner — and y is a
    scalar (mm, layer center). Piece occupies [x0, x0+size_x] × [y0, …] × [z0, …].
    """
    W = int(np.ceil(size_x / PX_MM))
    H = int(np.ceil(size_z / PZ_MM))
    N_SLICES = int(np.ceil(size_y / LY_MM))
    x = ((np.arange(W) + 0.5) * PX_MM)[None, :]
    z = ((np.arange(H) + 0.5) * PZ_MM)[:, None]

    def make_slice(layer):
        mask = np.broadcast_to(solid_fn(x, (layer + 0.5) * LY_MM, z), (H, W))
        return mask.astype(np.uint8) * 255

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=x0, OFFSET_Y_MM=y0, OFFSET_Z_MM=z0,
        make_slice=make_slice,
    )
