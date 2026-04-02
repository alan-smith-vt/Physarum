"""Generate tree-like support structures for 3D printing.

Analyzes a voxel model (from generate.py or generate_julia.py) to detect
overhanging regions, then builds branching tree supports down to the build
plate or the nearest solid surface below.

Two-pass algorithm (inspired by Cura tree supports):
  Pass 1 — Centerlines:
    1. Detect overhang pixels (solid with nothing directly below)
    2. Subsample overhang regions on a regular grid
    3. Drop branch centerlines from each sample point, descending layer by layer
    4. Attract nearby branches toward each other as they descend
    5. Merge branches that converge within a threshold distance
    6. Record center positions + merge weight per layer

  Pass 2 — Radius assignment:
    Walk the stored centerlines and assign radii from the topology:
    - weight (number of original tips merged) → base radius
    - tip taper near the overhang contact
    - base flare near the build plate for adhesion

Usage (standalone — re-encodes _slices.bin with supports included):
    cd slicer_preview && python supports.py

Usage (importable — returns support piece dicts):
    from supports import generate_supports
    pieces = generate_supports(make_slice_fn, W, H, N_SLICES, ...)
"""
import sys, os, struct, json, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from printer import PX_MM, PZ_MM, LY_MM, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX

# ── Support Parameters ──────────────────────────────────────────────
SUPPORT_GRAY = 128            # grayscale value (model uses 255)
TIP_RADIUS_MM = 0.25          # thin contact point at overhang
BRANCH_RADIUS_MM = 0.5        # single-branch radius after taper
TRUNK_RADIUS_MM = 1.2         # maximum radius (many merges)
SAMPLE_SPACING_MM = 3.0       # overhang sampling grid spacing
MERGE_DISTANCE_MM = 5.0       # branches merge when closer than this
ATTRACTION_FACTOR = 0.03      # per-layer lateral pull toward neighbors
GAP_LAYERS = 2                # empty layers between tip and model
TIP_TAPER_MM = 1.5            # length of thin→branch taper
BASE_FLARE_MM = 1.5           # flare height on build plate
BASE_FLARE_MULT = 1.6         # radius multiplier at the very base
MIN_SUPPORT_HEIGHT_MM = 0.5   # ignore overhangs with tiny gaps


# ── Overhang detection ──────────────────────────────────────────────

def _detect_overhangs_raw(make_slice, W, H, N_SLICES):
    """Single bottom-to-top pass detecting overhang sample points.

    Returns list of (x_local_mm, z_local_mm, start_layer, floor_layer).
    Coordinates are in local pixel-mm space (no global offset applied).
    """
    sample_x = max(1, int(round(SAMPLE_SPACING_MM / PX_MM)))
    sample_z = max(1, int(round(SAMPLE_SPACING_MM / PZ_MM)))
    min_gap_layers = max(1, int(round(MIN_SUPPORT_HEIGHT_MM / LY_MM)))

    solid_height = np.full((H, W), -1, dtype=np.int32)
    prev = np.zeros((H, W), dtype=np.uint8)
    points = []

    print("  Scanning layers for overhangs ...", flush=True)
    for layer in range(N_SLICES):
        curr = make_slice(layer)

        if layer > 0:
            overhang = (curr > 0) & (prev == 0)
            if overhang.any():
                sampled = np.zeros_like(overhang)
                sampled[::sample_z, ::sample_x] = True
                sampled &= overhang

                zs, xs = np.nonzero(sampled)
                if len(xs) > 0:
                    floors = solid_height[zs, xs]
                    floor_layers = np.where(floors >= 0, floors + 1, 0)
                    start_layer = max(0, layer - 1 - GAP_LAYERS)

                    for x, z, fl in zip(xs.tolist(), zs.tolist(),
                                        floor_layers.tolist()):
                        if start_layer - fl >= min_gap_layers:
                            points.append((x * PX_MM, z * PZ_MM,
                                           start_layer, fl))

        solid_height[curr > 0] = layer
        prev = curr

        if (layer + 1) % 200 == 0 or layer + 1 == N_SLICES:
            print(f"    {layer+1}/{N_SLICES}  ({len(points)} support points)",
                  flush=True)

    print(f"  {len(points)} overhang sample points", flush=True)
    return points


# ── Pass 1: build centerlines ───────────────────────────────────────

def _build_centerlines(overhang_points):
    """Top-to-bottom descent building branch centerlines.

    Only tracks positions and merge weight — no radius computation.

    Returns
    -------
    skeleton : dict  layer → (xs, zs, weights, birth_layers, floor_layers)
        All numpy arrays.  weight = number of original tips merged into
        this branch.
    """
    if not overhang_points:
        return {}

    starts_at = {}
    for x_mm, z_mm, start_layer, floor_layer in overhang_points:
        starts_at.setdefault(start_layer, []).append((x_mm, z_mm, floor_layer))

    max_layer = max(starts_at.keys())

    # Active branch state (no radius here)
    ax = np.empty(0, dtype=np.float64)
    az = np.empty(0, dtype=np.float64)
    aw = np.empty(0, dtype=np.float64)   # merge weight (starts at 1)
    af = np.empty(0, dtype=np.int32)     # floor layer
    ab = np.empty(0, dtype=np.int32)     # birth layer

    skeleton = {}

    print("  Pass 1: building centerlines (top → bottom) ...", flush=True)
    for layer in range(max_layer, -1, -1):
        # ── add new tips starting at this layer ──
        if layer in starts_at:
            pts = starts_at[layer]
            n = len(pts)
            ax = np.concatenate([ax, np.array([p[0] for p in pts])])
            az = np.concatenate([az, np.array([p[1] for p in pts])])
            aw = np.concatenate([aw, np.ones(n)])
            af = np.concatenate([af, np.array([p[2] for p in pts], dtype=np.int32)])
            ab = np.concatenate([ab, np.full(n, layer, dtype=np.int32)])

        if len(ax) == 0:
            continue

        # ── record this layer's centerlines ──
        skeleton[layer] = (ax.copy(), az.copy(), aw.copy(),
                           ab.copy(), af.copy())

        # ── attract toward nearest neighbor ──
        if len(ax) > 1:
            dx = ax[:, None] - ax[None, :]
            dz = az[:, None] - az[None, :]
            dist = np.sqrt(dx ** 2 + dz ** 2)
            np.fill_diagonal(dist, np.inf)

            nearest = np.argmin(dist, axis=1)
            nd = dist[np.arange(len(ax)), nearest]

            toward_x = ax[nearest] - ax
            toward_z = az[nearest] - az
            norm = np.maximum(nd, 1e-9)

            in_range = nd < MERGE_DISTANCE_MM * 3
            step = ATTRACTION_FACTOR * np.minimum(nd, 1.0)
            ax += np.where(in_range, step * toward_x / norm, 0.0)
            az += np.where(in_range, step * toward_z / norm, 0.0)

            # ── merge close branches ──
            merged = np.zeros(len(ax), dtype=bool)
            for i in range(len(ax)):
                if merged[i]:
                    continue
                close = (dist[i] < MERGE_DISTANCE_MM) & ~merged
                close[i] = False
                if not close.any():
                    continue
                partners = np.where(close)[0]
                idxs = np.concatenate([[i], partners])
                # weighted-average position (heavier branches pull more)
                w = aw[idxs]
                w_sum = w.sum()
                ax[i] = (ax[idxs] * w).sum() / w_sum
                az[i] = (az[idxs] * w).sum() / w_sum
                aw[i] = w_sum
                af[i] = min(af[i], af[partners].min())
                ab[i] = max(ab[i], ab[partners].max())
                merged[partners] = True

            if merged.any():
                keep = ~merged
                ax, az, aw = ax[keep], az[keep], aw[keep]
                af, ab = af[keep], ab[keep]

        # ── kill branches that reached their floor ──
        alive = layer > af
        ax, az, aw = ax[alive], az[alive], aw[alive]
        af, ab = af[alive], ab[alive]

        if (max_layer - layer + 1) % 200 == 0:
            print(f"    layer {layer}  ({len(ax)} active branches)", flush=True)

    total_nodes = sum(len(xs) for xs, _, _, _, _ in skeleton.values())
    print(f"  Centerlines: {len(skeleton)} layers, {total_nodes} nodes", flush=True)
    return skeleton


# ── Pass 2: assign radii from topology ──────────────────────────────

def _assign_radii(skeleton):
    """Convert skeleton (centerlines + weights) into renderable tree.

    Returns dict  layer → (xs, zs, rs)  with radii in mm.
    """
    if not skeleton:
        return {}

    taper_layers = max(1, int(round(TIP_TAPER_MM / LY_MM)))
    flare_layers = max(1, int(round(BASE_FLARE_MM / LY_MM)))

    tree = {}

    print("  Pass 2: assigning radii ...", flush=True)
    for layer, (xs, zs, weights, births, floors) in skeleton.items():
        # ── weight → base radius (area-preserving) ──
        # single tip = BRANCH_RADIUS_MM, more merges → wider, capped
        weight_r = BRANCH_RADIUS_MM * np.sqrt(weights)
        weight_r = np.minimum(weight_r, TRUNK_RADIUS_MM)

        # ── tip taper: thin contact that widens over taper_layers ──
        age = (births - layer).astype(np.float64)  # 0 at birth, grows
        taper_frac = np.clip(age / taper_layers, 0.0, 1.0)
        r = TIP_RADIUS_MM + (weight_r - TIP_RADIUS_MM) * taper_frac

        # ── base flare near build plate ──
        flare_frac = np.clip(1.0 - layer / flare_layers, 0.0, 1.0)
        r = r * (1.0 + (BASE_FLARE_MULT - 1.0) * flare_frac)

        tree[layer] = (xs.copy(), zs.copy(), r)

    return tree


# ── Piece generation ────────────────────────────────────────────────

def _make_support_piece(tree):
    """Create a piece dict from the final tree (layer → xs, zs, rs).

    Outputs SUPPORT_GRAY instead of 255 so supports are visually distinct.
    """
    if not tree:
        return None

    # tight bounding box (mm)
    all_x_lo, all_x_hi = [], []
    all_z_lo, all_z_hi = [], []
    for xs, zs, rs in tree.values():
        all_x_lo.append((xs - rs).min())
        all_x_hi.append((xs + rs).max())
        all_z_lo.append((zs - rs).min())
        all_z_hi.append((zs + rs).max())

    bb_x0 = min(all_x_lo) - 1.0
    bb_x1 = max(all_x_hi) + 1.0
    bb_z0 = min(all_z_lo) - 1.0
    bb_z1 = max(all_z_hi) + 1.0

    W = int(np.ceil((bb_x1 - bb_x0) / PX_MM))
    H = int(np.ceil((bb_z1 - bb_z0) / PZ_MM))

    min_layer = min(tree.keys())
    max_layer = max(tree.keys())
    N_SLICES = max_layer - min_layer + 1
    offset_y_mm = min_layer * LY_MM

    # Pre-convert to pixel coords relative to piece origin
    tree_px = {}
    gray = np.uint8(SUPPORT_GRAY)
    for layer, (xs, zs, rs) in tree.items():
        local_layer = layer - min_layer
        px_cx = (xs - bb_x0) / PX_MM
        px_cz = (zs - bb_z0) / PZ_MM
        px_rx = np.maximum(rs / PX_MM, 1.0)
        px_rz = np.maximum(rs / PZ_MM, 1.0)
        tree_px[local_layer] = (px_cx, px_cz, px_rx, px_rz)

    def make_slice(layer, _tree_px=tree_px, _H=H, _W=W, _gray=gray):
        img = np.zeros((_H, _W), dtype=np.uint8)
        if layer not in _tree_px:
            return img

        cxs, czs, rxs, rzs = _tree_px[layer]
        for i in range(len(cxs)):
            cx, cz = cxs[i], czs[i]
            rx, rz = rxs[i], rzs[i]
            irx, irz = int(np.ceil(rx)), int(np.ceil(rz))

            x0 = max(0, int(cx) - irx)
            x1 = min(_W, int(cx) + irx + 1)
            z0 = max(0, int(cz) - irz)
            z1 = min(_H, int(cz) + irz + 1)
            if x0 >= x1 or z0 >= z1:
                continue

            xx = np.arange(x0, x1, dtype=np.float64) - cx
            zz = np.arange(z0, z1, dtype=np.float64) - cz
            mask = (xx[None, :] ** 2 / (rx * rx) +
                    zz[:, None] ** 2 / (rz * rz)) <= 1.0

            img[z0:z1, x0:x1][mask] = _gray

        return img

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=bb_x0,
        OFFSET_Y_MM=offset_y_mm,
        OFFSET_Z_MM=bb_z0,
        make_slice=make_slice,
    )


# ── Public API ──────────────────────────────────────────────────────

def generate_supports(make_slice, W, H, N_SLICES,
                      offset_x_mm, offset_y_mm, offset_z_mm):
    """Analyze model overhangs and return support piece dicts.

    Parameters
    ----------
    make_slice : callable(layer) → uint8 (H, W)
    W, H, N_SLICES : int
    offset_x_mm, offset_y_mm, offset_z_mm : float

    Returns
    -------
    list of piece dicts (empty if no supports needed).
    """
    raw_points = _detect_overhangs_raw(make_slice, W, H, N_SLICES)
    if not raw_points:
        print("No overhangs detected — no supports needed.")
        return []

    # convert pixel-space mm to world mm
    world_points = [
        (offset_x_mm + x, offset_z_mm + z, sl, fl)
        for x, z, sl, fl in raw_points
    ]

    skeleton = _build_centerlines(world_points)
    if not skeleton:
        return []

    tree = _assign_radii(skeleton)
    piece = _make_support_piece(tree)
    return [piece] if piece else []


# ── Standalone entry point ──────────────────────────────────────────

def main():
    from generate import (make_global_slice, W, H, N_SLICES,
                          OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM,
                          PIECES, _compute_globals)

    SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
    META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')

    print("=" * 60)
    print("Tree Support Generator")
    print("=" * 60)
    print(f"Model: {W}×{H} px, {N_SLICES} layers")
    print(f"Pixel pitch: {PX_MM:.4f} × {PZ_MM:.4f} mm, layer: {LY_MM:.4f} mm")
    print()

    t0 = time.time()
    support_pieces = generate_supports(
        make_global_slice, W, H, N_SLICES,
        OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM)

    if not support_pieces:
        print("Nothing to do.")
        return

    PIECES.extend(support_pieces)
    new_ox, new_oy, new_oz, new_W, new_H, new_N = _compute_globals(PIECES)

    import generate as gen
    gen.OFFSET_X_MM = new_ox
    gen.OFFSET_Y_MM = new_oy
    gen.OFFSET_Z_MM = new_oz
    gen.W = new_W
    gen.H = new_H
    gen.N_SLICES = new_N

    print()
    print(f"Encoding {len(PIECES)} pieces (model + supports) ...")
    blob = gen.encode_all()

    with open(SLICES_FILE, 'wb') as f:
        f.write(blob)

    meta = {
        "width": new_W, "height": new_H, "num_slices": new_N,
        "pixel_x_um": PIXEL_X_UM, "pixel_y_um": PIXEL_Y_UM,
        "layer_um": LAYER_UM,
        "plate_w_mm": PLATE_W_PX * 14.0 / 1000,
        "plate_h_mm": PLATE_H_PX * 19.0 / 1000,
        "offset_x_mm": new_ox,
        "offset_y_mm": new_oy,
        "offset_z_mm": new_oz,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    sz = os.path.getsize(SLICES_FILE)
    print(f"\nDone in {elapsed:.1f}s — {sz:,} bytes ({sz/1024/1024:.1f} MB)")
    print(f"Output: {SLICES_FILE}")


if __name__ == '__main__':
    main()
