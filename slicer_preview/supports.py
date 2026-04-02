"""Generate tree-like support structures for 3D printing.

Analyzes a voxel model (from generate.py or generate_julia.py) to detect
overhanging regions, then builds branching tree supports down to the build
plate or the nearest solid surface below.

Algorithm (inspired by Cura tree supports):
  1. Detect overhang pixels (solid with nothing directly below)
  2. Subsample overhang regions on a regular grid
  3. Drop branch centerlines from each sample point, descending layer by layer
  4. Each layer, branches move laterally toward their nearest neighbor,
     limited by a maximum branch angle from vertical
  5. Branches merge only when they physically converge (< MERGE_DISTANCE_MM)
  6. The lateral movement over many layers produces smooth diagonal curves
     that join into shared trunks — no teleportation

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
MAX_BRANCH_ANGLE_DEG = 40.0   # max angle from vertical for branch movement
MERGE_DISTANCE_MM = 0.3       # branches merge only when essentially touching
ATTRACTION_FACTOR = 0.15      # fraction of distance moved per layer (before cap)
GAP_LAYERS = 0                # layers of gap between support tip and model
TIP_TAPER_MM = 1.5            # length of thin→branch taper
BASE_FLARE_MM = 1.5           # flare height on build plate
BASE_FLARE_MULT = 1.6         # radius multiplier at the very base
MIN_SUPPORT_HEIGHT_MM = 0.5   # ignore overhangs with tiny gaps

# Max lateral step per layer, derived from branch angle
MAX_STEP_MM = np.tan(np.radians(MAX_BRANCH_ANGLE_DEG)) * LY_MM


# ── Overhang detection ──────────────────────────────────────────────

ISLAND_GRAY = 5               # voxel value for island overlay (viewer maps 1-10 → blue)


def _detect_overhangs_raw(make_slice, W, H, N_SLICES):
    """Single bottom-to-top pass detecting overhang sample points.

    Returns (support_points, island_layers) where:
      support_points: list of (x_local_mm, z_local_mm, start_layer, floor_layer)
      island_layers:  dict  layer → bool mask (H, W) of ALL overhang pixels
    """
    sample_x = max(1, int(round(SAMPLE_SPACING_MM / PX_MM)))
    sample_z = max(1, int(round(SAMPLE_SPACING_MM / PZ_MM)))
    min_gap_layers = max(1, int(round(MIN_SUPPORT_HEIGHT_MM / LY_MM)))

    solid_height = np.full((H, W), -1, dtype=np.int32)
    prev = np.zeros((H, W), dtype=np.uint8)
    points = []
    island_layers = {}

    print("  Scanning layers for overhangs ...", flush=True)
    for layer in range(N_SLICES):
        curr = make_slice(layer)

        if layer > 0:
            overhang = (curr > 0) & (prev == 0)
            if overhang.any():
                # Store full overhang mask for island visualization
                island_layers[layer] = overhang

                # Subsample for support placement
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
            print(f"    {layer+1}/{N_SLICES}  ({len(points)} support points, "
                  f"{len(island_layers)} island layers)",
                  flush=True)

    print(f"  {len(points)} overhang sample points, "
          f"{len(island_layers)} layers with islands", flush=True)
    return points, island_layers


def _make_island_piece(island_layers, W, H, N_SLICES, offset_y_mm_global):
    """Create a piece that overlays island pixels on the model.

    Renders overhang pixels with ISLAND_GRAY (value 1-10) so the viewer
    shows them in blue.
    """
    if not island_layers:
        return None

    min_layer = min(island_layers.keys())
    max_layer = max(island_layers.keys())
    n_slices = max_layer - min_layer + 1
    piece_offset_y = offset_y_mm_global + min_layer * LY_MM

    # Pre-encode masks
    masks = {}
    gray = np.uint8(ISLAND_GRAY)
    for layer, mask in island_layers.items():
        local = layer - min_layer
        img = np.zeros((H, W), dtype=np.uint8)
        img[mask] = gray
        masks[local] = img

    def make_slice(layer, _masks=masks, _H=H, _W=W):
        if layer in _masks:
            return _masks[layer]
        return np.zeros((_H, _W), dtype=np.uint8)

    n_pixels = sum(m.sum() for m in island_layers.values())
    print(f"  Island overlay: {len(island_layers)} layers, "
          f"~{n_pixels:,} overhang pixels", flush=True)

    return dict(
        W=W, H=H, N_SLICES=n_slices,
        OFFSET_X_MM=0.0,   # same as model origin (pixel coords are model-local)
        OFFSET_Y_MM=piece_offset_y,
        OFFSET_Z_MM=0.0,
        make_slice=make_slice,
    )


# ── Build centerlines with continuous branch paths ──────────────────

def _build_centerlines(overhang_points, make_model_slice,
                       model_W, model_H, offset_x_mm, offset_z_mm):
    """Top-to-bottom descent building connected branch centerlines.

    Each branch moves laterally toward its nearest neighbor each layer,
    capped by MAX_STEP_MM (derived from the max branch angle).  Branches
    only merge when they physically converge to within MERGE_DISTANCE_MM,
    so every centerline traces a continuous, printable diagonal path.

    At each layer, branches whose current position falls inside solid
    model geometry are terminated — supports never penetrate the part.

    Returns
    -------
    skeleton : dict  layer → (xs, zs, weights, birth_layers, floor_layers)
        All numpy arrays.
    """
    if not overhang_points:
        return {}

    starts_at = {}
    for x_mm, z_mm, start_layer, floor_layer in overhang_points:
        starts_at.setdefault(start_layer, []).append((x_mm, z_mm, floor_layer))

    max_layer = max(starts_at.keys())

    # Active branch state
    ax = np.empty(0, dtype=np.float64)
    az = np.empty(0, dtype=np.float64)
    aw = np.empty(0, dtype=np.float64)   # merge weight
    af = np.empty(0, dtype=np.int32)     # floor layer
    ab = np.empty(0, dtype=np.int32)     # birth layer

    skeleton = {}

    print(f"  Building centerlines (top → bottom, "
          f"max_step={MAX_STEP_MM:.4f} mm/layer, "
          f"merge_dist={MERGE_DISTANCE_MM} mm) ...", flush=True)

    for layer in range(max_layer, -1, -1):
        # ── add new tips ──
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

        # ── collision check: kill branches inside solid model ──
        model_img = make_model_slice(layer)
        px_x = np.round((ax - offset_x_mm) / PX_MM).astype(np.int32)
        px_z = np.round((az - offset_z_mm) / PZ_MM).astype(np.int32)
        in_bounds = (px_x >= 0) & (px_x < model_W) & (px_z >= 0) & (px_z < model_H)
        inside_model = np.zeros(len(ax), dtype=bool)
        inside_model[in_bounds] = model_img[px_z[in_bounds], px_x[in_bounds]] > 0
        alive = ~inside_model
        if not alive.all():
            ax, az, aw = ax[alive], az[alive], aw[alive]
            af, ab = af[alive], ab[alive]

        if len(ax) == 0:
            continue

        # ── record this layer ──
        skeleton[layer] = (ax.copy(), az.copy(), aw.copy(),
                           ab.copy(), af.copy())

        # ── attract toward nearest neighbor (angle-limited) ──
        if len(ax) > 1:
            dx = ax[:, None] - ax[None, :]
            dz = az[:, None] - az[None, :]
            dist = np.sqrt(dx ** 2 + dz ** 2)
            np.fill_diagonal(dist, np.inf)

            nearest = np.argmin(dist, axis=1)
            nd = dist[np.arange(len(ax)), nearest]

            # direction toward nearest
            toward_x = ax[nearest] - ax
            toward_z = az[nearest] - az
            norm = np.maximum(nd, 1e-9)

            # step = fraction of distance, capped by max angle
            step = np.minimum(nd * ATTRACTION_FACTOR, MAX_STEP_MM)

            ax += step * toward_x / norm
            az += step * toward_z / norm

            # ── merge branches that have converged ──
            # recompute distances after movement
            dx2 = ax[:, None] - ax[None, :]
            dz2 = az[:, None] - az[None, :]
            dist2 = np.sqrt(dx2 ** 2 + dz2 ** 2)
            np.fill_diagonal(dist2, np.inf)

            merged = np.zeros(len(ax), dtype=bool)
            for i in range(len(ax)):
                if merged[i]:
                    continue
                close = (dist2[i] < MERGE_DISTANCE_MM) & ~merged
                close[i] = False
                if not close.any():
                    continue
                partners = np.where(close)[0]
                idxs = np.concatenate([[i], partners])
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


# ── Piece generation (1px centerlines for debugging) ────────────────

def _make_support_piece(skeleton, offset_y_mm_global):
    """Create a piece dict rendering single-pixel centerlines.

    Draws 1px dots at each branch center per layer.  Since branches
    move gradually each layer, the dots trace continuous diagonal paths
    in 3D.  Uses SUPPORT_GRAY so supports are visually distinct.
    """
    if not skeleton:
        return None

    all_xs, all_zs = [], []
    for xs, zs, _w, _b, _f in skeleton.values():
        all_xs.append(xs)
        all_zs.append(zs)

    cat_x = np.concatenate(all_xs)
    cat_z = np.concatenate(all_zs)
    bb_x0 = cat_x.min() - 2.0
    bb_x1 = cat_x.max() + 2.0
    bb_z0 = cat_z.min() - 2.0
    bb_z1 = cat_z.max() + 2.0

    W = int(np.ceil((bb_x1 - bb_x0) / PX_MM))
    H = int(np.ceil((bb_z1 - bb_z0) / PZ_MM))

    min_layer = min(skeleton.keys())
    max_layer = max(skeleton.keys())
    N_SLICES = max_layer - min_layer + 1
    piece_offset_y = offset_y_mm_global + min_layer * LY_MM

    # Pre-convert to pixel coords
    tree_px = {}
    gray = np.uint8(SUPPORT_GRAY)
    for layer, (xs, zs, _w, _b, _f) in skeleton.items():
        local_layer = layer - min_layer
        px_x = np.round((xs - bb_x0) / PX_MM).astype(np.int32)
        px_z = np.round((zs - bb_z0) / PZ_MM).astype(np.int32)
        tree_px[local_layer] = (px_x, px_z)

    def make_slice(layer, _tree_px=tree_px, _H=H, _W=W, _gray=gray):
        img = np.zeros((_H, _W), dtype=np.uint8)
        if layer not in _tree_px:
            return img
        pxs, pzs = _tree_px[layer]
        valid = (pxs >= 0) & (pxs < _W) & (pzs >= 0) & (pzs < _H)
        img[pzs[valid], pxs[valid]] = _gray
        return img

    print(f"  Support piece: {W}x{H} px, {N_SLICES} layers "
          f"(global layers {min_layer}..{max_layer})")
    print(f"  Bounding box: x=[{bb_x0:.1f}, {bb_x1:.1f}] "
          f"z=[{bb_z0:.1f}, {bb_z1:.1f}] mm")
    print(f"  OFFSET_Y_MM={piece_offset_y:.4f} "
          f"(global_oy={offset_y_mm_global:.4f}, min_layer={min_layer})")

    return dict(
        W=W, H=H, N_SLICES=N_SLICES,
        OFFSET_X_MM=bb_x0,
        OFFSET_Y_MM=piece_offset_y,
        OFFSET_Z_MM=bb_z0,
        make_slice=make_slice,
    )


# ── Public API ──────────────────────────────────────────────────────

def generate_supports(make_slice, W, H, N_SLICES,
                      offset_x_mm, offset_y_mm, offset_z_mm):
    """Analyze model overhangs and return support + island-overlay pieces."""
    raw_points, island_layers = _detect_overhangs_raw(make_slice, W, H, N_SLICES)
    pieces = []

    # Island overlay (blue in viewer) — always generated if overhangs exist
    if island_layers:
        island_piece = _make_island_piece(island_layers, W, H, N_SLICES,
                                          offset_y_mm)
        if island_piece:
            # Offsets must match the model's global origin
            island_piece['OFFSET_X_MM'] = offset_x_mm
            island_piece['OFFSET_Z_MM'] = offset_z_mm
            pieces.append(island_piece)

    if not raw_points:
        print("No overhangs detected — no supports needed.")
        return pieces

    world_points = [
        (offset_x_mm + x, offset_z_mm + z, sl, fl)
        for x, z, sl, fl in raw_points
    ]

    start_layers = [p[2] for p in world_points]
    floor_layers = [p[3] for p in world_points]
    print(f"  Overhang start layers: {min(start_layers)}..{max(start_layers)}")
    print(f"  Floor layers: {min(floor_layers)}..{max(floor_layers)}")
    print(f"  Global offset: x={offset_x_mm:.2f} y={offset_y_mm:.4f} "
          f"z={offset_z_mm:.2f} mm")

    skeleton = _build_centerlines(world_points, make_slice, W, H,
                                   offset_x_mm, offset_z_mm)
    if skeleton:
        piece = _make_support_piece(skeleton, offset_y_mm)
        if piece:
            pieces.append(piece)

    return pieces


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
