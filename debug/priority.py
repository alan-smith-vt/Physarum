"""
Debug node priority: visualizes the Voronoi-style node classification
for multi-filament claim priority.

Shows nodes colored by their priority assignment (seed A exclusive,
seed B exclusive, or shared) at multiple Z levels. Simulates both
filament tips to show their XY positions at each plotted Z.

Run from Physarum folder: python -m debug.priority
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from scipy.spatial.distance import cdist

from analysis.analysis import AnalysisPipeline

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

PCO_FILE = '../chromosome.pco'
THRESHOLD_FACTOR = 1.5  # node_size multiplier for shared zone width

Z_STEP = 0.2
CLAIM_RADIUS = 0.5
MAX_XY_FACTOR = 2.0
MIN_POINTS = 3
HEATMAP_RES = 25

# Z levels to visualize (fractions of the Z range)
Z_FRACTIONS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85]

# Colors
COLOR_A = '#ff4444'       # seed A exclusive
COLOR_B = '#4488ff'       # seed B exclusive
COLOR_SHARED = '#ffcc00'  # shared zone
COLOR_EMPTY = '#1a1d26'   # empty nodes (no points)
COLOR_BG = '#0c0e12'

# ═══════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════

pipeline = AnalysisPipeline()
pipeline.load_pco(PCO_FILE)

points = pipeline.points
bounds_min = np.array(pipeline.bounds_min)
bounds_max = np.array(pipeline.bounds_max)

# Seeds: match the 3D viewer convention
seed_a = np.array([
    (bounds_min[0] + bounds_max[0]) / 2 - 2.25,
    (bounds_min[1] + bounds_max[1]) / 2,
    bounds_min[2] + 0.1,
], dtype=np.float32)
seed_b = np.array([
    (bounds_min[0] + bounds_max[0]) / 2 + 2.25,
    (bounds_min[1] + bounds_max[1]) / 2,
    bounds_min[2] + 0.1,
], dtype=np.float32)
seeds = [seed_a, seed_b]

print(f"Seed A: {seed_a}")
print(f"Seed B: {seed_b}")
print(f"Node size: {pipeline.node_size:.4f}")
print(f"Threshold: {THRESHOLD_FACTOR} x {pipeline.node_size:.4f} = {THRESHOLD_FACTOR * pipeline.node_size:.4f}")

# ═══════════════════════════════════════════
# CLASSIFY
# ═══════════════════════════════════════════

node_priority = pipeline.classify_node_priority(seeds, threshold_factor=THRESHOLD_FACTOR)

# ═══════════════════════════════════════════
# COMPUTE Z LEVELS
# ═══════════════════════════════════════════

z_min = bounds_min[2]
z_max = bounds_max[2]
z_range = z_max - z_min
z_levels = [z_min + f * z_range for f in Z_FRACTIONS]

print(f"\nZ range: {z_min:.2f} to {z_max:.2f}")
print(f"Plotting at Z = {[f'{z:.2f}' for z in z_levels]}")

# ═══════════════════════════════════════════
# SIMULATE BOTH TIPS
# Uses simplified trace logic (centroid following)
# to get approximate tip XY at each Z level.
# ═══════════════════════════════════════════

max_xy_step = Z_STEP * MAX_XY_FACTOR
half_slab = Z_STEP / 2.0
z_coords = points[:, 2]

def gather_all(region):
    idx = []
    for nid in region:
        idx.extend(pipeline.node_index[nid]['indices'])
    return np.array(idx) if idx else np.array([], dtype=int)

def z_slice_fn(indices, z):
    if len(indices) == 0:
        return indices
    return indices[np.abs(z_coords[indices] - z) <= half_slab]

def find_peak(center_xy, pts_xy, max_r):
    gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, HEATMAP_RES)
    gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, HEATMAP_RES)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_flat = np.column_stack([gxx.ravel(), gyy.ravel()])
    grid_dist_sq = (grid_flat[:, 0] - center_xy[0])**2 + \
                   (grid_flat[:, 1] - center_xy[1])**2
    disk_mask = grid_dist_sq <= max_r**2
    if len(pts_xy) == 0 or not disk_mask.any():
        return None
    dists = cdist(grid_flat[disk_mask], pts_xy)
    counts = np.sum(dists <= CLAIM_RADIUS, axis=1)
    max_count = int(counts.max())
    if max_count < MIN_POINTS:
        return None
    centroid = pts_xy.mean(axis=0)
    tied = counts == max_count
    tied_positions = grid_flat[disk_mask][tied]
    centroid_dists = (tied_positions[:, 0] - centroid[0])**2 + \
                    (tied_positions[:, 1] - centroid[1])**2
    return tied_positions[np.argmin(centroid_dists)]

def simulate_tip(seed, filament_id):
    """Simulate a single filament tip, returning {z: [x, y]} for each Z level."""
    center = seed[:2].copy()
    current_z = float(seed[2])
    claimed_points_local = np.zeros(len(points), dtype=bool)
    claimed_nodes_local = set()
    ref_density = None
    density_decay = 0.9
    prev_region = None

    priority_kwargs = dict(node_priority=node_priority, filament_id=filament_id)

    # Initial region
    region = pipeline._region_grow_nodes(center, current_z, claimed_nodes_local,
                                          **priority_kwargs)
    prev_region = region
    if region:
        ref_density = max(len(pipeline.node_index[nid]['indices']) for nid in region)
        all_idx = gather_all(region)
        sliced = z_slice_fn(all_idx, current_z)
        if len(sliced) > 0:
            dists = np.linalg.norm(points[sliced][:, :2] - center, axis=1)
            claim_idx = sliced[dists <= CLAIM_RADIUS]
            claimed_points_local[claim_idx] = True

    # Record tip positions at each target Z
    tip_at_z = {}
    target_z_max = max(z_levels) + Z_STEP

    while current_z + Z_STEP <= target_z_max:
        new_z = current_z + Z_STEP

        region = pipeline._region_grow_nodes(
            center, new_z, claimed_nodes_local,
            ref_density=ref_density, prev_region=prev_region,
            **priority_kwargs)
        if not region:
            break

        all_idx = gather_all(region)
        sliced = z_slice_fn(all_idx, new_z)
        unclaimed = sliced[~claimed_points_local[sliced]]
        if len(unclaimed) == 0:
            break

        step_xy = points[unclaimed][:, :2]
        best_xy = find_peak(center, step_xy, max_xy_step)
        if best_xy is None:
            break

        displacement = best_xy - center
        dist = np.linalg.norm(displacement)
        if dist > max_xy_step:
            best_xy = center + displacement * (max_xy_step / dist)

        # Claim
        claim_dists = np.linalg.norm(points[unclaimed][:, :2] - best_xy, axis=1)
        claim_idx = unclaimed[claim_dists <= CLAIM_RADIUS]
        if len(claim_idx) < MIN_POINTS:
            break
        claimed_points_local[claim_idx] = True

        for nid in region:
            ni = np.array(pipeline.node_index[nid]['indices'])
            if np.all(claimed_points_local[ni]):
                claimed_nodes_local.add(nid)

        if ref_density is not None and region:
            current_density = max(len(pipeline.node_index[nid]['indices']) for nid in region)
            ref_density = density_decay * ref_density + (1 - density_decay) * current_density

        center = best_xy.copy()
        current_z = new_z
        prev_region = region

        # Check if we've passed any target Z levels
        for zl in z_levels:
            if zl not in tip_at_z and current_z >= zl:
                tip_at_z[zl] = center.copy()

    return tip_at_z

print("\nSimulating tip A (filament 0)...")
tip_a = simulate_tip(seed_a, filament_id=0)
print(f"  Reached {len(tip_a)} of {len(z_levels)} Z levels")

print("Simulating tip B (filament 1)...")
tip_b = simulate_tip(seed_b, filament_id=1)
print(f"  Reached {len(tip_b)} of {len(z_levels)} Z levels")

# ═══════════════════════════════════════════
# HELPER: get nodes at a Z level
# ═══════════════════════════════════════════

def nodes_at_z(z):
    gz = int((z - pipeline.root_min[2]) / pipeline.node_size)
    gz = max(0, min(gz, 2**pipeline.node_depth - 1))
    result = {}
    for nid, info in pipeline.node_index.items():
        if info['grid'][2] == gz:
            result[nid] = info
    return result

# ═══════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════

n_levels = len(z_levels)
fig, axes = plt.subplots(2, n_levels, figsize=(5 * n_levels, 10), squeeze=False)

priority_colors = {
    0: COLOR_A,
    1: COLOR_B,
    -1: COLOR_SHARED,
}

for col, z in enumerate(z_levels):
    z_nodes = nodes_at_z(z)

    # Count priorities at this level
    level_counts = {0: 0, 1: 0, -1: 0}
    for nid in z_nodes:
        p = node_priority.get(nid, -1)
        level_counts[p] = level_counts.get(p, 0) + 1

    # Get tip positions at this Z (or None if tip didn't reach)
    tip_a_xy = tip_a.get(z)
    tip_b_xy = tip_b.get(z)

    # ── Row 0: Node priority map ──
    ax = axes[0][col]
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect('equal')
    ax.set_title(f'z={z:.2f}\nA={level_counts[0]} B={level_counts[1]} shared={level_counts[-1]}',
                 fontsize=9)

    for nid, info in z_nodes.items():
        gx, gy, _ = info['grid']
        x0 = pipeline.root_min[0] + gx * pipeline.node_size
        y0 = pipeline.root_min[1] + gy * pipeline.node_size

        p = node_priority.get(nid, -1)
        n_pts = len(info['indices'])

        if n_pts == 0:
            fc = COLOR_EMPTY
            ec = '#2a2d36'
            alpha = 0.3
        else:
            fc = priority_colors.get(p, COLOR_SHARED)
            ec = '#ffffff'
            alpha = 0.3 + 0.7 * min(n_pts / 40, 1.0)

        ax.add_patch(Rectangle((x0, y0), pipeline.node_size, pipeline.node_size,
                                linewidth=0.3, edgecolor=ec, facecolor=fc, alpha=alpha))

    # Mark tip positions
    if tip_a_xy is not None:
        ax.plot(tip_a_xy[0], tip_a_xy[1], '+', color=COLOR_A, markersize=14,
                markeredgewidth=2.5, zorder=10)
    if tip_b_xy is not None:
        ax.plot(tip_b_xy[0], tip_b_xy[1], '+', color=COLOR_B, markersize=14,
                markeredgewidth=2.5, zorder=10)

    margin = pipeline.node_size * 3
    ax.set_xlim(bounds_min[0] - margin, bounds_max[0] + margin)
    ax.set_ylim(bounds_min[1] - margin, bounds_max[1] + margin)
    ax.set_xlabel('X', fontsize=8)
    if col == 0:
        ax.set_ylabel('Y', fontsize=8)

    # ── Row 1: Points colored by priority ──
    ax = axes[1][col]
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect('equal')

    half_slab_plot = pipeline.node_size
    slab_mask = np.abs(points[:, 2] - z) <= half_slab_plot
    slab_pts = points[slab_mask]

    if len(slab_pts) > 0:
        slab_indices = np.where(slab_mask)[0]
        pt_colors = np.zeros((len(slab_pts), 4))

        for nid, info in z_nodes.items():
            p = node_priority.get(nid, -1)
            rgba = to_rgba(priority_colors.get(p, COLOR_SHARED))
            for idx in info['indices']:
                if slab_mask[idx]:
                    local = np.searchsorted(slab_indices, idx)
                    if local < len(slab_indices) and slab_indices[local] == idx:
                        pt_colors[local] = rgba

        unmapped = pt_colors[:, 3] == 0
        pt_colors[unmapped] = to_rgba('#555555')

        ax.scatter(slab_pts[:, 0], slab_pts[:, 1], s=0.5, c=pt_colors, alpha=0.7)

    ax.set_title(f'Points z={z:.2f} ({len(slab_pts)} pts)', fontsize=9)

    if tip_a_xy is not None:
        ax.plot(tip_a_xy[0], tip_a_xy[1], '+', color=COLOR_A, markersize=14,
                markeredgewidth=2.5, zorder=10)
    if tip_b_xy is not None:
        ax.plot(tip_b_xy[0], tip_b_xy[1], '+', color=COLOR_B, markersize=14,
                markeredgewidth=2.5, zorder=10)

    ax.set_xlim(bounds_min[0] - margin, bounds_max[0] + margin)
    ax.set_ylim(bounds_min[1] - margin, bounds_max[1] + margin)
    ax.set_xlabel('X', fontsize=8)
    if col == 0:
        ax.set_ylabel('Y', fontsize=8)

# ── Legend ──
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_A,
           markersize=12, label='Seed A exclusive', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_B,
           markersize=12, label='Seed B exclusive', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_SHARED,
           markersize=12, label='Shared', linestyle='None'),
    Line2D([0], [0], marker='+', color=COLOR_A, markeredgewidth=2.5,
           markersize=12, label='Tip A', linestyle='None'),
    Line2D([0], [0], marker='+', color=COLOR_B, markeredgewidth=2.5,
           markersize=12, label='Tip B', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
           framealpha=0.8, edgecolor='#555555')

fig.suptitle(f'Node Priority Classification — threshold={THRESHOLD_FACTOR}x node_size ({THRESHOLD_FACTOR * pipeline.node_size:.3f})',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
out_path = os.path.join(os.path.dirname(__file__), 'images', 'priority.png')
plt.savefig(out_path, dpi=150, facecolor=COLOR_BG)
print(f"\nSaved {out_path}")
plt.show()
