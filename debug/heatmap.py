"""
Debug heatmap: runs the actual trace from analysis.py up to PLOT_STEP,
then shows the heatmap at that position to debug what the tracer sees.

Run from Physarum folder: python -m debug.heatmap
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from analysis.analysis import AnalysisPipeline

# ═══════════════════════════════════════════
# CONFIG — matches 3D viewer defaults
# ═══════════════════════════════════════════

PCO_FILE = 'chromosome.pco'
Z_STEP = 0.2
CLAIM_RADIUS = 0.5
MAX_XY_FACTOR = 2.0
MIN_POINTS = 3
HEATMAP_RES = 30
MIN_BRANCH_POINTS = 100
MAX_PEAKS = 5

PLOT_STEP = 2  # which step to visualize (0 = first step from seed)

# ═══════════════════════════════════════════
# LOAD + SETUP
# ═══════════════════════════════════════════

pipeline = AnalysisPipeline()
pipeline.load_pco(PCO_FILE)

points = pipeline.points
bounds_min = np.array(pipeline.bounds_min)
bounds_max = np.array(pipeline.bounds_max)

# Match the 3D viewer seed
seed = [
    (bounds_min[0] + bounds_max[0]) / 2 - 2.25,
    (bounds_min[1] + bounds_max[1]) / 2,
    bounds_min[2] + 0.1,
]
print(f"Seed: {seed}")

# ═══════════════════════════════════════════
# RUN TRACE UP TO PLOT_STEP
# Uses the exact same code path as the 3D viewer
# but we intercept at the target step
# ═══════════════════════════════════════════

# We need to replicate the trace internals to capture mid-trace state.
# Import what we need from the pipeline.
seed_arr = np.array(seed, dtype=np.float32)
max_xy_step = Z_STEP * MAX_XY_FACTOR
half_slab = Z_STEP / 2.0
z_coords = points[:, 2]
z_max = float(z_coords.max())

claimed_points = np.zeros(len(points), dtype=bool)
claimed_nodes = set()
ref_density = None
density_decay = 0.9

def gather_all(region):
    idx = []
    for nid in region:
        idx.extend(pipeline.node_index[nid]['indices'])
    return np.array(idx) if idx else np.array([], dtype=int)

def z_slice_fn(indices, z):
    if len(indices) == 0:
        return indices
    return indices[np.abs(z_coords[indices] - z) <= half_slab]

def find_peaks(center_xy, pts_xy, max_r, max_peaks=MAX_PEAKS):
    gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, HEATMAP_RES)
    gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, HEATMAP_RES)
    gxx, gyy = np.meshgrid(gx, gy)

    remaining = pts_xy.copy()
    peaks = []

    for _ in range(max_peaks):
        if len(remaining) == 0:
            break
        best_count = 0
        best_dist_sq = float('inf')
        best_pos = center_xy.copy()
        for i in range(HEATMAP_RES):
            for j in range(HEATMAP_RES):
                cx, cy = gxx[i, j], gyy[i, j]
                dx, dy = cx - center_xy[0], cy - center_xy[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > max_r**2:
                    continue
                d = np.sqrt((remaining[:, 0] - cx)**2 + (remaining[:, 1] - cy)**2)
                c = int(np.sum(d <= CLAIM_RADIUS))
                if c > best_count or (c == best_count and dist_sq < best_dist_sq):
                    best_count = c
                    best_dist_sq = dist_sq
                    best_pos = np.array([cx, cy])
        if best_count < MIN_POINTS:
            break
        peaks.append({'xy': best_pos, 'count': best_count})
        d = np.sqrt((remaining[:, 0] - best_pos[0])**2 + (remaining[:, 1] - best_pos[1])**2)
        remaining = remaining[d > CLAIM_RADIUS]

    return peaks

def compute_heatmap(center_xy, pts_xy, max_r):
    """Return full heatmap grid for plotting."""
    gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, HEATMAP_RES)
    gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, HEATMAP_RES)
    gxx, gyy = np.meshgrid(gx, gy)
    heatmap = np.zeros((HEATMAP_RES, HEATMAP_RES), dtype=np.int32)
    for i in range(HEATMAP_RES):
        for j in range(HEATMAP_RES):
            cx, cy = gxx[i, j], gyy[i, j]
            if (cx - center_xy[0])**2 + (cy - center_xy[1])**2 > max_r**2:
                continue
            d = np.sqrt((pts_xy[:, 0] - cx)**2 + (pts_xy[:, 1] - cy)**2)
            heatmap[i, j] = int(np.sum(d <= CLAIM_RADIUS))
    dist_from_center = np.sqrt((gxx - center_xy[0])**2 + (gyy - center_xy[1])**2)
    heatmap_masked = np.ma.masked_where(dist_from_center > max_r, heatmap)
    return gxx, gyy, heatmap_masked

# Initial claim
center = seed_arr[:2].copy()
current_z = seed_arr[2]

region = pipeline._region_grow_nodes(center, current_z, claimed_nodes)
prev_region = region
if region:
    ref_density = max(len(pipeline.node_index[nid]['indices']) for nid in region)
    init_slab = np.where(np.abs(z_coords - current_z) <= half_slab)[0]
    if len(init_slab) > 0:
        dists = np.linalg.norm(points[init_slab][:, :2] - center, axis=1)
        claim_idx = init_slab[dists <= CLAIM_RADIUS]
        if len(claim_idx) > 0:
            claimed_points[claim_idx] = True

print(f"Init: {int(claimed_points.sum())} pts claimed, ref_density={ref_density}")

# Step loop
centerline = [[center[0], center[1], current_z]]

for step_i in range(PLOT_STEP + 1):
    new_z = current_z + Z_STEP
    if new_z > z_max:
        print(f"Step {step_i}: hit top of cloud")
        break

    region = pipeline._region_grow_nodes_seeded(
        center, new_z, claimed_nodes, prev_region, ref_density=ref_density)
    if not region:
        print(f"Step {step_i}: no region found")
        break

    all_idx = gather_all(region)
    sliced = z_slice_fn(all_idx, new_z)
    unclaimed = sliced[~claimed_points[sliced]]
    if len(unclaimed) == 0:
        print(f"Step {step_i}: no unclaimed points")
        break

    step_xy = points[unclaimed][:, :2]

    if step_i < PLOT_STEP:
        # Ghost simulate: run the same logic but don't plot
        peaks = find_peaks(center, step_xy, max_xy_step)
        if not peaks:
            print(f"Step {step_i}: no peaks, stopping")
            break
        best_xy = peaks[0]['xy']
        displacement = best_xy - center
        dist = np.linalg.norm(displacement)
        if dist > max_xy_step:
            displacement = displacement * (max_xy_step / dist)
            best_xy = center + displacement

        # Claim from all slab points
        all_slab = np.where(np.abs(z_coords - new_z) <= half_slab)[0]
        all_slab_unclaimed = all_slab[~claimed_points[all_slab]]
        claim_dists = np.linalg.norm(points[all_slab_unclaimed][:, :2] - best_xy, axis=1)
        claim_idx = all_slab_unclaimed[claim_dists <= CLAIM_RADIUS]
        if len(claim_idx) < MIN_POINTS:
            print(f"Step {step_i}: below min_points, stopping")
            break
        claimed_points[claim_idx] = True

        for nid in region:
            ni = np.array(pipeline.node_index[nid]['indices'])
            if np.all(claimed_points[ni]):
                claimed_nodes.add(nid)

        if ref_density is not None and region:
            current_density = max(len(pipeline.node_index[nid]['indices']) for nid in region)
            ref_density = density_decay * ref_density + (1 - density_decay) * current_density

        center = best_xy.copy()
        current_z = new_z
        prev_region = region
        centerline.append([center[0], center[1], current_z])
        print(f"  [skip] Step {step_i}: z={new_z:.2f} center=[{center[0]:.3f}, {center[1]:.3f}] claimed={int(claimed_points.sum())}")

    else:
        # THIS IS THE STEP WE PLOT
        print(f"\n--- PLOT STEP {step_i}: z={new_z:.2f} ---")
        print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}]")
        print(f"  Region: {len(region)} nodes, slab: {len(sliced)}, unclaimed: {len(unclaimed)}")
        print(f"  Ref density: {ref_density:.1f}")

        # Compute heatmap
        gxx, gyy, heatmap_masked = compute_heatmap(center, step_xy, max_xy_step)

        # Extract peaks
        peaks = find_peaks(center, step_xy, max_xy_step)
        for pi, p in enumerate(peaks):
            print(f"  Peak {pi}: [{p['xy'][0]:.3f}, {p['xy'][1]:.3f}] -> {p['count']} pts")

        # Centroid
        slab_centroid = step_xy.mean(axis=0)
        print(f"  Centroid: [{slab_centroid[0]:.3f}, {slab_centroid[1]:.3f}]")
        if peaks:
            print(f"  Peak0 vs centroid offset: [{peaks[0]['xy'][0]-slab_centroid[0]:.3f}, {peaks[0]['xy'][1]-slab_centroid[1]:.3f}]")

        # Also get the full slab (all points, not just region-filtered)
        full_slab_mask = np.abs(z_coords - new_z) <= half_slab
        full_slab_xy = points[full_slab_mask][:, :2]

        # ═══════════════════════════════════════════
        # PLOTS
        # ═══════════════════════════════════════════

        PEAK_COLORS = ['#00ff88', '#ff6644', '#44aaff', '#ffcc00', '#ff44cc']
        VIEW_MARGIN = 2.0

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
        axes = axes[0]

        # ── Plot 0: XZ Overview ──
        ax = axes[0]
        ax.set_facecolor('#0c0e12')
        ax.set_title(f'Overview (XZ) — step {step_i}')
        ax.set_aspect('equal')

        subsample = np.random.choice(len(points), min(5000, len(points)), replace=False)
        ax.scatter(points[subsample, 0], points[subsample, 2], s=0.3, c='#555555', alpha=0.3)

        # Claimed so far
        claimed_so_far = np.where(claimed_points)[0]
        if len(claimed_so_far) > 0:
            ax.scatter(points[claimed_so_far, 0], points[claimed_so_far, 2],
                       s=0.5, c='#00ff88', alpha=0.3)

        # Trace path
        cl = np.array(centerline)
        ax.plot(cl[:, 0], cl[:, 2], 'w-', linewidth=1, alpha=0.6)
        ax.plot(cl[:, 0], cl[:, 2], 'wo', markersize=2)

        ax.axhspan(new_z - half_slab, new_z + half_slab, color='#e8a44a', alpha=0.15)
        ax.axhline(new_z, color='#e8a44a', linewidth=1, alpha=0.6, label=f'z={new_z:.2f}')
        ax.plot(center[0], new_z, 'r+', markersize=15, markeredgewidth=2, label='current')
        ax.legend(fontsize=8)
        ax.set_xlabel('X'); ax.set_ylabel('Z')

        # ── Plot 1: Heatmap with peaks + centroid ──
        ax = axes[1]
        ax.set_facecolor('#0c0e12')
        ax.set_title(f'Heatmap step {step_i} z={new_z:.2f} ({len(unclaimed)} region pts)')
        ax.set_aspect('equal')

        im = ax.pcolormesh(gxx, gyy, heatmap_masked, cmap='inferno', shading='auto')
        plt.colorbar(im, ax=ax, label='claimable points', shrink=0.7)

        ax.add_patch(Circle(center, max_xy_step, fill=False,
                             edgecolor='#ffffff', linewidth=1, linestyle=':', alpha=0.4, label='max r'))
        ax.plot(center[0], center[1], 'r+', markersize=15, markeredgewidth=2, label='center', zorder=5)

        ax.plot(slab_centroid[0], slab_centroid[1], 'o', color='cyan', markersize=10,
                markeredgewidth=2, markerfacecolor='none', zorder=5, label='centroid')

        for pi, peak in enumerate(peaks):
            color = PEAK_COLORS[pi % len(PEAK_COLORS)]
            ax.plot(peak['xy'][0], peak['xy'][1], '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='white', zorder=5,
                    label=f'peak {pi} ({peak["count"]} pts)')
            ax.add_patch(Circle(peak['xy'], CLAIM_RADIUS, fill=False,
                                 edgecolor=color, linewidth=1.5, linestyle='--'))

        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_xlim(center[0] - VIEW_MARGIN, center[0] + VIEW_MARGIN)
        ax.set_ylim(center[1] - VIEW_MARGIN, center[1] + VIEW_MARGIN)

        # ── Plot 2: Points + peaks + centroid ──
        ax = axes[2]
        ax.set_facecolor('#0c0e12')
        ax.set_title(f'Points step {step_i} z={new_z:.2f}')
        ax.set_aspect('equal')

        # Full slab points (faint, shows what's actually there)
        ax.scatter(full_slab_xy[:, 0], full_slab_xy[:, 1], s=1, c='#444444', alpha=0.3,
                   zorder=1, label=f'all slab ({len(full_slab_xy)})')

        # Region-filtered unclaimed points (what heatmap sees)
        ax.scatter(step_xy[:, 0], step_xy[:, 1], s=1.5, c='#00ff88', alpha=0.5,
                   zorder=2, label=f'region unclaimed ({len(step_xy)})')

        ax.add_patch(Circle(center, max_xy_step, fill=False,
                             edgecolor='#ffffff', linewidth=1, linestyle=':', alpha=0.4, label='max r'))
        ax.plot(center[0], center[1], 'r+', markersize=15, markeredgewidth=2, label='center', zorder=5)

        ax.plot(slab_centroid[0], slab_centroid[1], 'o', color='cyan', markersize=10,
                markeredgewidth=2, markerfacecolor='none', zorder=5, label='centroid')

        for pi, peak in enumerate(peaks):
            color = PEAK_COLORS[pi % len(PEAK_COLORS)]
            ax.plot(peak['xy'][0], peak['xy'][1], '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='white', zorder=5,
                    label=f'peak {pi} ({peak["count"]} pts)')
            ax.add_patch(Circle(peak['xy'], CLAIM_RADIUS, fill=False,
                                 edgecolor=color, linewidth=1.5, linestyle='--'))

        ax.legend(fontsize=6, loc='upper right')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_xlim(center[0] - VIEW_MARGIN, center[0] + VIEW_MARGIN)
        ax.set_ylim(center[1] - VIEW_MARGIN, center[1] + VIEW_MARGIN)

        plt.tight_layout()
        plt.show()