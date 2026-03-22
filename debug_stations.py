"""
Debug stations: ghost-simulate tip progression to PLOT_STEP,
create a station at that Z-slab, run thickening to convergence,
and plot snapshots of the polar curve evolution.

Run from Physarum folder: python debug_stations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import deque

from analysis.analysis import AnalysisPipeline, PolarCrossSection, Station

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

PCO_FILE = 'chromosome.pco'
Z_STEP = 0.2
CLAIM_RADIUS = 0.5
MAX_XY_FACTOR = 2.0
MIN_POINTS = 3
HEATMAP_RES = 25
MIN_BRANCH_POINTS = 100

# Station thickening params
N_DIRS = 40        # max — will be scaled down for sparse slabs
NUDGE_BUDGET = 20
MAX_THICKEN_STEPS = 500

# Expansion params
MAX_NUDGES_PER_DIR = 3     # max expansion attempts per direction per tick (each validated)
MIN_DENSITY_RATIO = 0.3    # band density must be >= this * direction's EMA to accept
DENSITY_EMA_ALPHA = 0.3    # EMA blend for per-direction density history
LOOKAHEAD_STEPS = 2        # when a step fails, check this many bands ahead before giving up

# Retraction params
RETRACT_DIVISOR = 8        # retract_step = nudge_step / this (higher = smaller steps)
MAX_POINT_LOSS_PCT = 0.005  # max fraction of points a single retraction can drop (0.5%)
MIN_DENSITY_IMPROVEMENT = 0.001  # require 0.1% density improvement to accept

# Center mode
USE_TIP_CENTER = True  # True = keep center at tip position, False = drift to claimed CG

PLOT_STEP = 35  # which tip z-step to create the station at

# ═══════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════

pipeline = AnalysisPipeline()
pipeline.load_pco(PCO_FILE)

points = pipeline.points
bounds_min = np.array(pipeline.bounds_min)
bounds_max = np.array(pipeline.bounds_max)

seed = np.array([
    (bounds_min[0] + bounds_max[0]) / 2 - 2.25,
    (bounds_min[1] + bounds_max[1]) / 2,
    bounds_min[2] + 0.1,
], dtype=np.float32)
print(f"Seed: {seed}")

# ═══════════════════════════════════════════
# GHOST-SIMULATE TIP TO PLOT_STEP
# Replicates trace_with_stations tip logic
# ═══════════════════════════════════════════

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


def gather_expanded(region, z):
    gz_list = pipeline._nodes_in_z_slab(z)
    visited = set()
    queue = deque()
    expanded_nids = set()
    for nid in region:
        gx, gy, _ = pipeline.node_index[nid]['grid']
        for gz in gz_list:
            key = (gx, gy, gz)
            if key in pipeline.grid_to_node and key not in visited:
                visited.add(key)
                queue.append(key)
    while queue:
        cx, cy, cz = queue.popleft()
        nid = pipeline.grid_to_node.get((cx, cy, cz))
        if nid is not None:
            expanded_nids.add(nid)
            for dz_gz in gz_list:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz_gz == cz:
                            continue
                        nkey = (cx + dx, cy + dy, dz_gz)
                        if nkey not in visited and nkey in pipeline.grid_to_node:
                            visited.add(nkey)
                            queue.append(nkey)
    idx = []
    for nid in expanded_nids:
        idx.extend(pipeline.node_index[nid]['indices'])
    return np.array(idx) if idx else np.array([], dtype=int)


def z_slice_fn(indices, z):
    if len(indices) == 0:
        return indices
    return indices[np.abs(z_coords[indices] - z) <= half_slab]


def find_peaks(center_xy, pts_xy, max_r, max_peaks=5):
    from scipy.spatial.distance import cdist
    gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, HEATMAP_RES)
    gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, HEATMAP_RES)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_flat = np.column_stack([gxx.ravel(), gyy.ravel()])
    grid_dist_sq = (grid_flat[:, 0] - center_xy[0])**2 + \
                   (grid_flat[:, 1] - center_xy[1])**2
    disk_mask = grid_dist_sq <= max_r**2
    remaining = pts_xy.copy()
    peaks = []
    for _ in range(max_peaks):
        if len(remaining) == 0:
            break
        centroid = remaining.mean(axis=0)
        dists = cdist(grid_flat[disk_mask], remaining)
        counts = np.sum(dists <= CLAIM_RADIUS, axis=1)
        max_count = int(counts.max())
        if max_count < MIN_POINTS:
            break
        tied = counts == max_count
        tied_positions = grid_flat[disk_mask][tied]
        centroid_dists = (tied_positions[:, 0] - centroid[0])**2 + \
                        (tied_positions[:, 1] - centroid[1])**2
        best_pos = tied_positions[np.argmin(centroid_dists)]
        peaks.append((best_pos.copy(), max_count))
        d = np.sqrt((remaining[:, 0] - best_pos[0])**2 +
                   (remaining[:, 1] - best_pos[1])**2)
        remaining = remaining[d > CLAIM_RADIUS]
    return peaks


# ── Initial claim at seed ──

center = seed[:2].copy()
current_z = float(seed[2])

region = pipeline._region_grow_nodes(center, current_z, claimed_nodes)
prev_region = region
if region:
    ref_density = max(len(pipeline.node_index[nid]['indices']) for nid in region)
    expanded = gather_expanded(region, current_z)
    sliced = z_slice_fn(expanded, current_z)
    if len(sliced) > 0:
        dists = np.linalg.norm(points[sliced][:, :2] - center, axis=1)
        claim_idx = sliced[dists <= CLAIM_RADIUS]
        claimed_points[claim_idx] = True

print(f"Init: {int(claimed_points.sum())} pts claimed, ref_density={ref_density}")

# ── Ghost-simulate tip ──

centerline = [[center[0], center[1], current_z]]
target_region = None
target_center = None
target_z = None

for step_i in range(PLOT_STEP + 1):
    new_z = current_z + Z_STEP
    if new_z > z_max:
        print(f"Step {step_i}: hit top of cloud")
        break

    region = pipeline._region_grow_nodes(
        center, new_z, claimed_nodes,
        ref_density=ref_density, prev_region=prev_region)
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
    peaks = find_peaks(center, step_xy, max_xy_step)
    if not peaks:
        print(f"Step {step_i}: no peaks")
        break

    best_xy, best_count = peaks[0]
    displacement = best_xy - center
    dist = np.linalg.norm(displacement)
    if dist > max_xy_step:
        best_xy = center + displacement * (max_xy_step / dist)

    # Claim
    expanded_idx = gather_expanded(region, new_z)
    expanded_sliced = z_slice_fn(expanded_idx, new_z)
    expanded_unclaimed = expanded_sliced[~claimed_points[expanded_sliced]]
    claim_dists = np.linalg.norm(points[expanded_unclaimed][:, :2] - best_xy, axis=1)
    claim_idx = expanded_unclaimed[claim_dists <= CLAIM_RADIUS]
    if len(claim_idx) < MIN_POINTS:
        print(f"Step {step_i}: below min_points")
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

    print(f"  Step {step_i}: z={new_z:.2f} center=[{center[0]:.3f}, {center[1]:.3f}] "
          f"claimed={int(claimed_points.sum())} slab={len(sliced)} uncl={len(unclaimed)}")

    if step_i == PLOT_STEP:
        target_region = region
        target_center = np.array([center[0], center[1], current_z], dtype=np.float32)
        target_z = current_z

if target_center is None:
    print("ERROR: could not reach PLOT_STEP")
    exit(1)

# ═══════════════════════════════════════════
# CREATE STATION AT TARGET STEP
# ═══════════════════════════════════════════

slab_idx = gather_expanded(target_region, target_z)
slab_idx = z_slice_fn(slab_idx, target_z)
slab_xy = points[slab_idx][:, :2]

print(f"\n--- Station at step {PLOT_STEP}, z={target_z:.2f} ---")
print(f"  Center: [{target_center[0]:.3f}, {target_center[1]:.3f}]")
print(f"  Slab points: {len(slab_idx)}")

station = Station(
    target_center, slab_idx, slab_xy, age=PLOT_STEP,
    n_dirs=N_DIRS)

ss = station._slab_stats
print(f"  Slab diameter: {ss['slab_diameter']:.3f}")
print(f"  Avg spacing:   {ss['avg_spacing']:.4f}")
print(f"  -> nudge_step:    {ss['effective_nudge']:.4f}")
print(f"  -> initial_radius: {ss['effective_radius']:.4f}")
print(f"  -> n_dirs:         {ss['effective_dirs']}")
print(f"  Initial claim: {station.cross_section.claimed_count}")

# ═══════════════════════════════════════════
# RUN THICKENING TO CONVERGENCE
# ═══════════════════════════════════════════

expansion_kwargs = dict(
    max_nudges_per_dir=MAX_NUDGES_PER_DIR,
    min_density_ratio=MIN_DENSITY_RATIO,
    density_ema_alpha=DENSITY_EMA_ALPHA,
    lookahead_steps=LOOKAHEAD_STEPS,
)

print(f"\nThickening (budget={NUDGE_BUDGET}, max_steps={MAX_THICKEN_STEPS})...")
print(f"  Expansion: max_per_dir={MAX_NUDGES_PER_DIR}, "
      f"density_ratio={MIN_DENSITY_RATIO}, ema={DENSITY_EMA_ALPHA}, "
      f"lookahead={LOOKAHEAD_STEPS}")

history = []
for tick in range(MAX_THICKEN_STEPS):
    stats = station.thicken(nudge_budget=NUDGE_BUDGET, use_tip_center=USE_TIP_CENTER,
                            **expansion_kwargs)
    history.append(stats)

    if tick % 10 == 0 or station.converged:
        print(f"  Tick {tick:3d}: claimed={stats['total_claimed']:5d}  "
              f"new=+{stats['new_claimed']:4d}  "
              f"dirs={stats['n_dirs']:3d}  "
              f"nudges={stats['nudges_spent']}")

    if station.converged:
        print(f"  Converged at tick {tick}")
        break

n_ticks = len(history)

# Retraction pass
print(f"\nRetraction pass...")
pre_area = PolarCrossSection._polygon_area(station.cross_section.get_boundary_xy())
pre_claimed = station.cross_section.claimed_count
pre_density = pre_claimed / max(pre_area, 1e-6)

retract_stats = station.cross_section.retract_pass(slab_xy)
post_area = PolarCrossSection._polygon_area(station.cross_section.get_boundary_xy())
post_claimed = station.cross_section.claimed_count
post_density = post_claimed / max(post_area, 1e-6)

print(f"  Before: {pre_claimed} pts, area={pre_area:.2f}, density={pre_density:.1f}")
print(f"  After:  {post_claimed} pts, area={post_area:.2f}, density={post_density:.1f}")
print(f"  Retractions: {retract_stats['total_retractions']}, iters: {retract_stats['iterations']}")

# ═══════════════════════════════════════════
# PLOT: tick 0, pre-retract, then retraction iterations
# ═══════════════════════════════════════════

# Replay thickening to get tick 0 and pre-retract states
station2 = Station(
    target_center, slab_idx, slab_xy, age=PLOT_STEP,
    n_dirs=N_DIRS)

# Tick 0
tick0_stats = station2.thicken(nudge_budget=NUDGE_BUDGET, use_tip_center=USE_TIP_CENTER, **expansion_kwargs)
snap_tick0 = {
    'boundary': station2.cross_section.get_boundary_xy().copy(),
    'center': station2.cross_section.center.copy(),
    'n_dirs': station2.cross_section.n_dirs,
    'claimed': station2.cross_section.claimed.copy(),
}

# Run to convergence
for tick in range(1, MAX_THICKEN_STEPS):
    station2.thicken(nudge_budget=NUDGE_BUDGET, use_tip_center=USE_TIP_CENTER, **expansion_kwargs)
    if station2.converged:
        break

snap_pre = {
    'boundary': station2.cross_section.get_boundary_xy().copy(),
    'center': station2.cross_section.center.copy(),
    'n_dirs': station2.cross_section.n_dirs,
    'claimed': station2.cross_section.claimed.copy(),
}
pre_claimed = station2.cross_section.claimed_count
pre_area = PolarCrossSection._polygon_area(station2.cross_section.get_boundary_xy())

# Now do retraction manually, one iteration at a time, capturing snapshots
cs = station2.cross_section
retract_step = cs.nudge_step / RETRACT_DIVISOR
retract_snaps = []

print(f"\nRetraction params: divisor={RETRACT_DIVISOR} -> step={retract_step:.4f}, "
      f"max_loss={MAX_POINT_LOSS_PCT*100:.1f}%, min_improvement={MIN_DENSITY_IMPROVEMENT*100:.1f}%")

MAX_RETRACT_ITERS = 500
for r_iter in range(MAX_RETRACT_ITERS):
    improved = False
    current_count = cs.claimed_count
    if current_count == 0:
        break
    current_area = PolarCrossSection._polygon_area(cs.get_boundary_xy())
    current_density = current_count / max(current_area, 1e-6)
    max_loss = max(1, int(current_count * MAX_POINT_LOSS_PCT))
    density_threshold = current_density * (1 + MIN_DENSITY_IMPROVEMENT)
    n_accepted = 0
    total_lost = 0
    n_rejected_loss = 0
    n_rejected_density = 0

    for i in range(cs.n_dirs):
        old_r = cs.radii[i]
        new_r = max(0.05, old_r - retract_step)
        if new_r >= old_r:
            continue
        cs.radii[i] = new_r
        new_inside = cs.points_inside_mask(slab_xy)
        new_count = int(new_inside.sum())
        lost = current_count - new_count
        if lost > max_loss:
            cs.radii[i] = old_r
            n_rejected_loss += 1
            continue
        new_area = PolarCrossSection._polygon_area(cs.get_boundary_xy())
        new_density = new_count / max(new_area, 1e-6)
        if new_density >= density_threshold:
            cs.claimed = new_inside
            current_count = new_count
            current_area = new_area
            current_density = new_density
            density_threshold = current_density * (1 + MIN_DENSITY_IMPROVEMENT)
            improved = True
            n_accepted += 1
            total_lost += lost
        else:
            cs.radii[i] = old_r
            n_rejected_density += 1

    cs.claimed = cs.points_inside_mask(slab_xy)

    retract_snaps.append({
        'boundary': cs.get_boundary_xy().copy(),
        'center': cs.center.copy(),
        'n_dirs': cs.n_dirs,
        'claimed': cs.claimed.copy(),
        'accepted': n_accepted,
        'lost': total_lost,
        'density': current_density,
        'rejected_loss': n_rejected_loss,
        'rejected_density': n_rejected_density,
    })

    print(f"  Retract iter {r_iter}: {cs.claimed_count} pts, "
          f"area={current_area:.2f}, density={current_density:.1f}, "
          f"accepted={n_accepted}, lost=-{total_lost}, "
          f"rej_loss={n_rejected_loss}, rej_density={n_rejected_density}")

    if not improved:
        break

n_retract = len(retract_snaps)
print(f"\n  Total retraction iterations: {n_retract}")
print(f"  Pre:  {pre_claimed} pts, area={pre_area:.2f}")
print(f"  Post: {cs.claimed_count} pts, area={PolarCrossSection._polygon_area(cs.get_boundary_xy()):.2f}")

# Determine plot bounds
pad = 0.5
xmin, xmax = slab_xy[:, 0].min() - pad, slab_xy[:, 0].max() + pad
ymin, ymax = slab_xy[:, 1].min() - pad, slab_xy[:, 1].max() + pad


def plot_snapshot(ax, snap, title):
    claimed = snap['claimed']
    boundary = snap['boundary']
    ctr = snap['center']
    ax.scatter(slab_xy[~claimed, 0], slab_xy[~claimed, 1],
              s=0.5, c='#444444', alpha=0.3, rasterized=True)
    if claimed.sum() > 0:
        ax.scatter(slab_xy[claimed, 0], slab_xy[claimed, 1],
                  s=1.5, c='#e8a44a', alpha=0.7, rasterized=True)
    bnd = np.vstack([boundary, boundary[:1]])
    ax.plot(bnd[:, 0], bnd[:, 1], 'r-', linewidth=1.5, alpha=0.9)
    ax.plot(ctr[0], ctr[1], 'r+', markersize=8, markeredgewidth=2)
    for i in range(snap['n_dirs']):
        bx, by = boundary[i % len(boundary)]
        ax.plot([ctr[0], bx], [ctr[1], by], 'r-', linewidth=0.3, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.1)


# Layout: tick0 + pre-retract + up to 6 retract iterations
n_retract_to_show = min(6, n_retract)
# Pick evenly spaced retraction snapshots
if n_retract <= 6:
    retract_indices = list(range(n_retract))
else:
    retract_indices = [0]
    for i in range(1, 5):
        retract_indices.append(int(i * (n_retract - 1) / 5))
    retract_indices.append(n_retract - 1)
    retract_indices = sorted(set(retract_indices))

n_plots = 2 + len(retract_indices)  # tick0, pre-retract, retract snapshots
n_cols = 4
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]
fig.suptitle(f'Station Retraction Debug — Step {PLOT_STEP}, z={target_z:.2f}, '
             f'{len(slab_idx)} slab pts, nudge={ss["effective_nudge"]:.3f}, '
             f'retract_step={retract_step:.4f}',
             fontsize=13, fontweight='bold')

plot_idx = 0

# Tick 0
r, c = plot_idx // n_cols, plot_idx % n_cols
plot_snapshot(axes[r][c], snap_tick0,
             f'Tick 0: {tick0_stats["total_claimed"]} pts')
plot_idx += 1

# Pre-retract
r, c = plot_idx // n_cols, plot_idx % n_cols
plot_snapshot(axes[r][c], snap_pre,
             f'Pre-retract: {pre_claimed} pts, area={pre_area:.2f}')
plot_idx += 1

# Retraction iterations
for ri in retract_indices:
    r, c = plot_idx // n_cols, plot_idx % n_cols
    rs = retract_snaps[ri]
    plot_snapshot(axes[r][c], rs,
                 f'Retract {ri}: {int(rs["claimed"].sum())} pts, '
                 f'+{rs["accepted"]}ok -{rs["lost"]}lost '
                 f'({rs.get("rejected_loss",0)}rej_loss {rs.get("rejected_density",0)}rej_dens)')
    plot_idx += 1

# Hide unused
while plot_idx < n_rows * n_cols:
    r, c = plot_idx // n_cols, plot_idx % n_cols
    axes[r][c].axis('off')
    plot_idx += 1

plt.tight_layout()
out_path = 'debug_stations.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {out_path}")
plt.close()