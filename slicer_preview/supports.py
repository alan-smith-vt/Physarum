"""Generate tree-like support structures for 3D printing.

Analyzes a voxel model (from generate.py) to detect overhanging regions,
then builds branching tree supports from overhang contact points down to
the build plate or the nearest solid surface below.

Algorithm (inspired by Cura's tree supports):
  1. Detect overhang pixels (solid with nothing directly below)
  2. Subsample overhang regions on a regular grid
  3. Drop branches from each sample point, descending layer by layer
  4. Attract nearby branches toward each other as they descend
  5. Merge branches that converge within a threshold distance
  6. Widen merged trunks proportional to the branches they absorb
  7. Flare the base on the build plate for adhesion
  8. Leave a small gap at the top for easy removal after printing

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
TIP_RADIUS_MM = 0.3           # thin contact point at overhang
BRANCH_RADIUS_MM = 0.6        # default branch radius
TRUNK_RADIUS_MM = 1.5         # maximum radius after merges
SAMPLE_SPACING_MM = 3.0       # overhang sampling grid spacing
MERGE_DISTANCE_MM = 5.0       # branches merge when closer than this
ATTRACTION_FACTOR = 0.05      # per-layer lateral pull toward neighbors
GAP_LAYERS = 2                # empty layers between tip and model
TIP_TAPER_LAYERS = None       # computed from TIP_TAPER_MM
TIP_TAPER_MM = 1.0            # length of thin→branch taper
BASE_FLARE_MM = 1.5           # flare height on build plate
BASE_FLARE_MULT = 1.8         # radius multiplier at the very base
MIN_SUPPORT_HEIGHT_MM = 0.5   # ignore overhangs with tiny gaps


# ── Phase 1: overhang detection ─────────────────────────────────────

def detect_overhangs(make_slice, W, H, N_SLICES):
    """Single bottom-to-top pass detecting overhang sample points.

    Returns list of (x_mm, z_mm, start_layer, floor_layer) where
    start_layer is the layer just below the overhang (support grows
    downward from there) and floor_layer is the layer to terminate at.
    """
    sample_x = max(1, int(round(SAMPLE_SPACING_MM / PX_MM)))
    sample_z = max(1, int(round(SAMPLE_SPACING_MM / PZ_MM)))
    min_gap_layers = max(1, int(round(MIN_SUPPORT_HEIGHT_MM / LY_MM)))

    # solid_height[z, x] = most recent layer where pixel was solid
    solid_height = np.full((H, W), -1, dtype=np.int32)
    prev = np.zeros((H, W), dtype=np.uint8)
    overhang_points = []

    print("  Scanning layers for overhangs ...", flush=True)
    for layer in range(N_SLICES):
        curr = make_slice(layer)

        if layer > 0:
            overhang = (curr > 0) & (prev == 0)
            if overhang.any():
                # subsample on a regular grid
                sampled = np.zeros_like(overhang)
                sampled[::sample_z, ::sample_x] = True
                sampled &= overhang

                zs, xs = np.nonzero(sampled)
                if len(xs) > 0:
                    floors = solid_height[zs, xs]  # -1 means build plate
                    floor_layers = np.where(floors >= 0, floors + 1, 0)
                    start_layers = layer - 1 - GAP_LAYERS

                    for x, z, sl, fl in zip(xs.tolist(), zs.tolist(),
                                            np.broadcast_to(start_layers, len(xs)).tolist(),
                                            floor_layers.tolist()):
                        if sl - fl >= min_gap_layers and sl >= 0:
                            overhang_points.append((x * PX_MM, z * PZ_MM, sl, fl))

        # update height map where currently solid
        solid_height[curr > 0] = layer
        prev = curr

        if (layer + 1) % 200 == 0 or layer + 1 == N_SLICES:
            print(f"    {layer+1}/{N_SLICES}  ({len(overhang_points)} support points)",
                  flush=True)

    print(f"  {len(overhang_points)} overhang sample points detected", flush=True)
    return overhang_points


# ── Phase 2: tree construction ──────────────────────────────────────

def build_tree_supports(overhang_points):
    """Build tree support structure from overhang points.

    Processes layers top-to-bottom.  Branches attract and merge as they
    descend, producing a dict  layer → (xs, zs, rs)  of numpy arrays.
    """
    if not overhang_points:
        return {}

    # Group by start layer
    starts_at = {}
    for x_mm, z_mm, start_layer, floor_layer in overhang_points:
        starts_at.setdefault(start_layer, []).append((x_mm, z_mm, floor_layer))

    max_layer = max(starts_at.keys())
    taper_layers = max(1, int(round(TIP_TAPER_MM / LY_MM)))

    # Active branch arrays
    ax = np.empty(0, dtype=np.float64)
    az = np.empty(0, dtype=np.float64)
    ar = np.empty(0, dtype=np.float64)
    af = np.empty(0, dtype=np.int32)     # floor layer per branch
    a_birth = np.empty(0, dtype=np.int32)  # layer where branch was born (for taper)

    tree = {}  # layer → (xs, zs, rs)

    print("  Building tree (top → bottom) ...", flush=True)
    for layer in range(max_layer, -1, -1):
        # ── add new branches starting at this layer ──
        if layer in starts_at:
            pts = starts_at[layer]
            n = len(pts)
            ax = np.concatenate([ax, np.array([p[0] for p in pts])])
            az = np.concatenate([az, np.array([p[1] for p in pts])])
            ar = np.concatenate([ar, np.full(n, TIP_RADIUS_MM)])
            af = np.concatenate([af, np.array([p[2] for p in pts], dtype=np.int32)])
            a_birth = np.concatenate([a_birth, np.full(n, layer, dtype=np.int32)])

        if len(ax) == 0:
            continue

        # ── tip taper: linearly grow from TIP_RADIUS to BRANCH_RADIUS ──
        age = (a_birth - layer).astype(np.float64)
        taper_frac = np.clip(age / taper_layers, 0.0, 1.0)
        base_r = TIP_RADIUS_MM + (BRANCH_RADIUS_MM - TIP_RADIUS_MM) * taper_frac

        # ── base flare ──
        flare_layers = max(1, int(round(BASE_FLARE_MM / LY_MM)))
        flare_frac = np.where(
            layer < flare_layers,
            1.0 - layer / flare_layers,
            0.0)
        display_r = np.maximum(ar, base_r) * (1.0 + (BASE_FLARE_MULT - 1.0) * flare_frac)

        tree[layer] = (ax.copy(), az.copy(), display_r.copy())

        # ── attract toward nearest neighbor ──
        if len(ax) > 1:
            # pairwise distances (vectorised, O(n²))
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
                w = ar[idxs]
                w_sum = w.sum()
                ax[i] = (ax[idxs] * w).sum() / w_sum
                az[i] = (az[idxs] * w).sum() / w_sum
                ar[i] = min(np.sqrt((ar[idxs] ** 2).sum()), TRUNK_RADIUS_MM)
                af[i] = min(af[i], af[partners].min())
                a_birth[i] = max(a_birth[i], a_birth[partners].max())
                merged[partners] = True

            if merged.any():
                keep = ~merged
                ax, az, ar = ax[keep], az[keep], ar[keep]
                af, a_birth = af[keep], a_birth[keep]

        # ── grow radius slightly as we descend ──
        ar = np.minimum(ar * 1.002, TRUNK_RADIUS_MM)

        # ── kill branches that reached their floor ──
        alive = layer > af
        ax, az, ar = ax[alive], az[alive], ar[alive]
        af, a_birth = af[alive], a_birth[alive]

        if (max_layer - layer + 1) % 200 == 0:
            print(f"    layer {layer}  ({len(ax)} active branches)", flush=True)

    total_nodes = sum(len(xs) for xs, _, _ in tree.values())
    print(f"  Tree built: {len(tree)} layers, {total_nodes} total nodes", flush=True)
    return tree


# ── Phase 3: piece generation ───────────────────────────────────────

def make_support_piece(tree, global_offset_x, global_offset_z):
    """Create a piece dict from tree support layer data.

    The piece covers just the bounding box of all support branches,
    keeping memory use proportional to the support footprint.
    """
    if not tree:
        return None

    # compute tight bounding box (mm) across all layers
    all_x_lo, all_x_hi = [], []
    all_z_lo, all_z_hi = [], []
    for xs, zs, rs in tree.values():
        all_x_lo.append((xs - rs).min())
        all_x_hi.append((xs + rs).max())
        all_z_lo.append((zs - rs).min())
        all_z_hi.append((zs + rs).max())

    bb_x0 = min(all_x_lo)
    bb_x1 = max(all_x_hi)
    bb_z0 = min(all_z_lo)
    bb_z1 = max(all_z_hi)

    # pad by 1 mm for safety
    bb_x0 -= 1.0;  bb_x1 += 1.0
    bb_z0 -= 1.0;  bb_z1 += 1.0

    W = int(np.ceil((bb_x1 - bb_x0) / PX_MM))
    H = int(np.ceil((bb_z1 - bb_z0) / PZ_MM))

    min_layer = min(tree.keys())
    max_layer = max(tree.keys())
    N_SLICES = max_layer - min_layer + 1
    offset_y_mm = min_layer * LY_MM

    # Pre-convert tree data to pixel coords relative to piece origin
    tree_px = {}
    for layer, (xs, zs, rs) in tree.items():
        local_layer = layer - min_layer
        px_cx = ((xs - bb_x0) / PX_MM).astype(np.float64)
        px_cz = ((zs - bb_z0) / PZ_MM).astype(np.float64)
        px_rx = np.maximum(rs / PX_MM, 1.0)
        px_rz = np.maximum(rs / PZ_MM, 1.0)
        tree_px[local_layer] = (px_cx, px_cz, px_rx, px_rz)

    def make_slice(layer):
        img = np.zeros((H, W), dtype=np.uint8)
        if layer not in tree_px:
            return img

        cxs, czs, rxs, rzs = tree_px[layer]
        for i in range(len(cxs)):
            cx, cz = cxs[i], czs[i]
            rx, rz = rxs[i], rzs[i]
            irx, irz = int(np.ceil(rx)), int(np.ceil(rz))

            x0 = max(0, int(cx) - irx)
            x1 = min(W, int(cx) + irx + 1)
            z0 = max(0, int(cz) - irz)
            z1 = min(H, int(cz) + irz + 1)
            if x0 >= x1 or z0 >= z1:
                continue

            xx = np.arange(x0, x1, dtype=np.float64) - cx
            zz = np.arange(z0, z1, dtype=np.float64) - cz
            mask = (xx[None, :] ** 2 / (rx * rx) +
                    zz[:, None] ** 2 / (rz * rz)) <= 1.0

            img[z0:z1, x0:x1][mask] = 255

        return img

    # Convert bounding box origin to global mm (matching piece convention)
    # bb_x0/z0 are already in the same world-mm space as the generate pieces
    # since overhang coords were computed as OFFSET + pixel * PX_MM
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
        The model's global slice function.
    W, H, N_SLICES : int
        Global model dimensions in pixels / layers.
    offset_x_mm, offset_y_mm, offset_z_mm : float
        World-space origin of the global bounding box.

    Returns
    -------
    list of piece dicts (empty if no supports needed).
    """
    # patch offset into overhang coords: pixel coords → mm
    # detect_overhangs returns mm coords already using pixel * PX_MM
    # but we need to add the global offset so pieces align
    raw_points = _detect_overhangs_raw(make_slice, W, H, N_SLICES)
    if not raw_points:
        print("No overhangs detected — no supports needed.")
        return []

    # convert pixel-space coords to world mm
    world_points = []
    for x_px_mm, z_px_mm, start_layer, floor_layer in raw_points:
        world_points.append((
            offset_x_mm + x_px_mm,
            offset_z_mm + z_px_mm,
            start_layer,
            floor_layer,
        ))

    tree = build_tree_supports(world_points)
    if not tree:
        return []

    piece = make_support_piece(tree, offset_x_mm, offset_z_mm)
    return [piece] if piece else []


def _detect_overhangs_raw(make_slice, W, H, N_SLICES):
    """Overhang detection returning coords in local pixel-mm space (no global offset)."""
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


# ── Standalone entry point ──────────────────────────────────────────

def main():
    from generate import (make_global_slice, W, H, N_SLICES,
                          OFFSET_X_MM, OFFSET_Y_MM, OFFSET_Z_MM,
                          PIECES, _compute_globals)
    from helpers import encode_surface_voxels

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

    # Add supports to PIECES and recompute globals
    PIECES.extend(support_pieces)
    new_ox, new_oy, new_oz, new_W, new_H, new_N = _compute_globals(PIECES)

    # Patch generate module globals so encode_all uses updated values
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
