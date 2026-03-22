"""
Debug script for trace advancement with:
- Temporal continuity (region grow seeded from previous step's nodes)
- Forward/lateral angle check (15° threshold)
- Branch flagging when angle > 15° and scores are similar
- Candidate sampling along displacement vector
- Segment data model (preparing for multi-agent thickening)

Run from Physarum folder: python debug_trace.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from collections import deque

from PCO import PCOReader
from PCO.pco_format import OctreeUtils

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

PCO_FILE = 'chromosome.pco'
SEED = None  # computed after loading to match 3D viewer
Z_STEP = 0.2
SLAB_THICKNESS = Z_STEP
CLAIM_RADIUS = 0.5
MAX_XY_FACTOR = 2.0
MIN_POINTS = 3
DENSITY_FRACTION = 0.5
N_PLOT_STEPS = 2             # how many steps to plot
PLOT_FROM_STEP = 40           # start plotting from this step (simulates all prior steps silently)
N_CANDIDATES = 10
TURN_ANGLE_DEG = 15.0
SHARP_TURN_RATIO = 1.5

# ═══════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════

reader = PCOReader(PCO_FILE)
reader.load_metadata()
header = reader.header
level = header['max_depth']
root_min = np.array(header['root_min'])
root_size = header['root_size']
node_size = root_size / (2 ** level)

all_data = reader.read_level(level)
parsed = reader.parse_binary_data(all_data)
points = parsed['xyz'].astype(np.float32)
print(f"Loaded {len(points):,} points, depth {level}, node size {node_size:.4f}")

# Compute seed to match 3D viewer: center_x - 2.25, center_y, z_min + 0.1
bounds_min = points.min(axis=0)
bounds_max = points.max(axis=0)
if SEED is None:
    SEED = [
        (bounds_min[0] + bounds_max[0]) / 2 - 2.25,
        (bounds_min[1] + bounds_max[1]) / 2,
        bounds_min[2] + 0.1,
    ]
print(f"Bounds: {bounds_min} to {bounds_max}")
print(f"Seed: {SEED}")

# ═══════════════════════════════════════════
# BUILD NODE INDEX
# ═══════════════════════════════════════════

node_index = {}
grid_to_node = {}
point_offset = 0

node_ids_in_order = [nid for nid in reader.index.keys() if len(nid) == level + 1]
for nid in node_ids_in_order:
    _, count = reader.index[nid]
    grid = tuple(OctreeUtils.node_id_to_grid_coords(nid))
    node_index[nid] = {'grid': grid, 'indices': list(range(point_offset, point_offset + count))}
    grid_to_node[grid] = nid
    point_offset += count

print(f"Nodes: {len(node_index)}")

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def nodes_at_z(z):
    gz = int((z - root_min[2]) / node_size)
    gz = max(0, min(gz, 2**level - 1))
    result = {}
    for nid, info in node_index.items():
        if info['grid'][2] == gz:
            result[nid] = info
    return result, gz

def region_grow_nodes(center_xy, z, claimed_nodes=set(), density_fraction=DENSITY_FRACTION,
                       seed_node_ids=None, ref_density=None):
    """
    BFS through XY-adjacent nodes across multiple Z layers (gz±1).
    If seed_node_ids is provided, seeds from those nodes (temporal continuity).
    Uses ref_density for threshold if provided, otherwise uses seed node density.
    """
    gz_center = int((z - root_min[2]) / node_size)
    mc = 2**level - 1
    gz_center = max(0, min(gz_center, mc))
    gz_list = [g for g in [gz_center - 1, gz_center, gz_center + 1] if 0 <= g <= mc]

    accepted, visited, queue = [], set(), deque()
    seed_density = 0

    if seed_node_ids is not None:
        # Temporal continuity: project previous nodes to new Z layers only
        for nid in seed_node_ids:
            gx, gy, _ = node_index[nid]['grid']
            for gz in gz_list:
                key = (gx, gy, gz)
                if key in grid_to_node and key not in visited:
                    nbr = grid_to_node[key]
                    if nbr not in claimed_nodes:
                        queue.append(key)
                        visited.add(key)
                        seed_density = max(seed_density, len(node_index[nbr]['indices']))

    if len(queue) == 0:
        # Fallback: seed from center_xy across Z layers
        gx = max(0, min(int((center_xy[0] - root_min[0]) / node_size), mc))
        gy = max(0, min(int((center_xy[1] - root_min[1]) / node_size), mc))

        for gz in gz_list:
            seed_key = (gx, gy, gz)
            if seed_key in grid_to_node and grid_to_node[seed_key] not in claimed_nodes:
                queue.append(seed_key)
                visited.add(seed_key)
                seed_density = max(seed_density, len(node_index[grid_to_node[seed_key]]['indices']))

        if len(queue) == 0:
            best_dist, best_key = float('inf'), None
            for gz in gz_list:
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        key = (gx + dx, gy + dy, gz)
                        if key in grid_to_node and grid_to_node[key] not in claimed_nodes:
                            d = dx*dx + dy*dy
                            if d < best_dist:
                                best_dist, best_key = d, key
            if best_key:
                queue.append(best_key)
                visited.add(best_key)
                seed_density = len(node_index[grid_to_node[best_key]]['indices'])

    if not queue:
        return []

    effective_density = ref_density if ref_density is not None else seed_density
    min_dens = max(1, int(effective_density * density_fraction))

    while queue:
        cx, cy, cz = queue.popleft()
        nid = grid_to_node.get((cx, cy, cz))
        if nid is None or nid in claimed_nodes:
            continue
        accepted.append(nid)
        # Expand XY neighbors across all Z layers
        for dz_gz in gz_list:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz_gz == cz:
                        continue
                    nkey = (cx + dx, cy + dy, dz_gz)
                    if nkey not in visited and nkey in grid_to_node:
                        nbr_nid = grid_to_node[nkey]
                        if nbr_nid not in claimed_nodes:
                            visited.add(nkey)
                            if len(node_index[nbr_nid]['indices']) >= min_dens:
                                queue.append(nkey)
    return accepted

def gather_all(region_nodes):
    idx = []
    for nid in region_nodes:
        idx.extend(node_index[nid]['indices'])
    return np.array(idx) if idx else np.array([], dtype=int)

def z_slice(indices, z, half_slab):
    if len(indices) == 0:
        return indices
    return indices[np.abs(points[indices, 2] - z) <= half_slab]

def count_claim(pts_xy, center_xy, radius):
    dists = np.linalg.norm(pts_xy - center_xy, axis=1)
    return int(np.sum(dists <= radius))

def angle_between(v1, v2):
    """Angle in degrees between two 2D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))

# ═══════════════════════════════════════════
# RUN MULTI-STEP TRACE
# ═══════════════════════════════════════════

seed = np.array(SEED, dtype=np.float32)
max_xy_step = Z_STEP * MAX_XY_FACTOR
half_slab = SLAB_THICKNESS / 2.0
turn_threshold = TURN_ANGLE_DEG

claimed_points = np.zeros(len(points), dtype=bool)
claimed_nodes_set = set()

center = seed[:2].copy()
current_z = seed[2]
velocity = None
ref_density = None
density_decay = 0.9

# Segment list (the multi-agent data model)
segments = []

# Initial claim
init_region = region_grow_nodes(center, current_z, claimed_nodes_set)
# Establish reference density from initial region
if init_region:
    ref_density = max(len(node_index[nid]['indices']) for nid in init_region)
    print(f"Reference density: {ref_density}")

init_all = gather_all(init_region)
init_sliced = z_slice(init_all, current_z, half_slab)
init_claimed = np.array([], dtype=int)
if len(init_sliced) > 0:
    dists = np.linalg.norm(points[init_sliced][:, :2] - center, axis=1)
    init_claimed = init_sliced[dists <= CLAIM_RADIUS]
    claimed_points[init_claimed] = True
    print(f"Init: {len(init_claimed)} pts claimed at z={current_z:.2f}")

segments.append({
    'center': [center[0], center[1], current_z],
    'radius': CLAIM_RADIUS,
    'claimed_indices': init_claimed.tolist(),
    'age': 0,
    'branch_flags': [],
})

trace_centers = [[center[0], center[1], current_z]]
prev_region_nodes = init_region
step_data = []

total_steps = PLOT_FROM_STEP + N_PLOT_STEPS

for step_i in range(total_steps):
    new_z = current_z + Z_STEP
    is_plotted = step_i >= PLOT_FROM_STEP
    prefix = f"  " if is_plotted else f"  [skip] "

    if is_plotted:
        print(f"\n--- Step {step_i}: z {current_z:.2f} -> {new_z:.2f} ---")
    else:
        print(f"  [skip] Step {step_i}: z {current_z:.2f} -> {new_z:.2f}")

    # 1. Region grow with temporal continuity and reference density
    region = region_grow_nodes(center, new_z, claimed_nodes_set,
                                seed_node_ids=prev_region_nodes,
                                ref_density=ref_density)
    if not region:
        print(f"{prefix}No region found, stopping")
        break

    # 2. Gather all points from region, Z-slice, filter claimed
    all_idx = gather_all(region)
    sliced_idx = z_slice(all_idx, new_z, half_slab)
    unclaimed_mask = ~claimed_points[sliced_idx]
    step_idx = sliced_idx[unclaimed_mask]

    if len(step_idx) == 0:
        print(f"{prefix}No unclaimed pts (region={len(all_idx)}, slab={len(sliced_idx)})")
        break

    step_pts = points[step_idx]
    step_xy = step_pts[:, :2]

    # 3. Compute centroid and displacement
    raw_centroid = step_xy.mean(axis=0)
    displacement = raw_centroid - center
    dist = np.linalg.norm(displacement)
    if dist > max_xy_step:
        displacement = displacement * (max_xy_step / dist)

    # 4. Candidate sampling along displacement vector
    candidates = []
    for ci in range(N_CANDIDATES + 1):
        t = ci / N_CANDIDATES
        cand_xy = center + displacement * t
        n_claim = count_claim(step_xy, cand_xy, CLAIM_RADIUS)
        candidates.append({'t': t, 'xy': cand_xy.copy(), 'count': n_claim})

    best = max(candidates, key=lambda c: c['count'])

    # 5. Angle check: compare best candidate direction to current velocity
    best_displacement = best['xy'] - center
    angle = 0.0
    branch_flag = None
    action = 'step'
    straight_score = 0
    straight_pos = None

    if velocity is not None:
        angle = angle_between(velocity, best_displacement)

        if angle > turn_threshold:
            # Score the "continue straight" position
            v_norm = velocity / np.linalg.norm(velocity)
            straight_pos = center + v_norm * np.linalg.norm(best_displacement)
            straight_score = count_claim(step_xy, straight_pos, CLAIM_RADIUS)

            if is_plotted:
                print(f"  Angle: {angle:.1f}° > {turn_threshold}°")
                print(f"  Best: {best['count']} pts, Straight: {straight_score} pts")

            if straight_score < MIN_POINTS:
                # Straight is dead, follow best (geometry curved)
                action = 'curve'
                if is_plotted:
                    print(f"  -> Straight dead ({straight_score} pts), following best")
            elif best['count'] < MIN_POINTS:
                # Best is dead, go straight
                action = 'forced_straight'
                best = {'t': -1, 'xy': straight_pos.copy(), 'count': straight_score}
                if is_plotted:
                    print(f"  -> Best dead, going straight")
            else:
                # Both viable — flag the weaker as branch, go with stronger
                action = 'branch_flag'
                if best['count'] >= straight_score:
                    # Best has more, go there, flag straight
                    branch_flag = {
                        'z': float(new_z),
                        'xy': straight_pos.tolist(),
                        'direction': (v_norm * np.linalg.norm(best_displacement)).tolist(),
                        'density': straight_score,
                        'angle': float(angle),
                    }
                    if is_plotted:
                        print(f"  -> Both viable, going best ({best['count']}), flagging straight ({straight_score})")
                else:
                    # Straight has more, go straight, flag best
                    branch_flag = {
                        'z': float(new_z),
                        'xy': best['xy'].tolist(),
                        'direction': best_displacement.tolist(),
                        'density': best['count'],
                        'angle': float(angle),
                    }
                    best = {'t': -1, 'xy': straight_pos.copy(), 'count': straight_score}
                    if is_plotted:
                        print(f"  -> Both viable, going straight ({straight_score}), flagging best ({branch_flag['density']})")

    new_center = best['xy']

    # 6. Claim
    claim_dists = np.linalg.norm(step_xy - new_center, axis=1)
    claim_idx = step_idx[claim_dists <= CLAIM_RADIUS]

    if is_plotted:
        print(f"  Region: {len(region)} nodes, {len(sliced_idx)} slab, {len(step_idx)} unclaimed")
        print(f"  Raw centroid: [{raw_centroid[0]:.3f}, {raw_centroid[1]:.3f}]")
        print(f"  Action: {action}, angle: {angle:.1f}°, claimed: {len(claim_idx)}")

    if len(claim_idx) < MIN_POINTS:
        print(f"{prefix}Below min_points, stopping")
        break

    claimed_points[claim_idx] = True

    for nid in region:
        ni = np.array(node_index[nid]['indices'])
        if np.all(claimed_points[ni]):
            claimed_nodes_set.add(nid)

    # Build segment
    seg = {
        'center': [float(new_center[0]), float(new_center[1]), float(new_z)],
        'radius': CLAIM_RADIUS,
        'claimed_indices': claim_idx.tolist(),
        'age': step_i + 1,
        'branch_flags': [branch_flag] if branch_flag else [],
    }
    segments.append(seg)

    # Update velocity
    new_disp = new_center - center
    if np.linalg.norm(new_disp) > 1e-8:
        velocity = new_disp.copy()

    # Update reference density (exponential decay toward current)
    if ref_density is not None and region:
        current_density = max(len(node_index[nid]['indices']) for nid in region)
        ref_density = density_decay * ref_density + (1 - density_decay) * current_density

    # Only store plot data for the plotted range
    if is_plotted:
        step_data.append({
            'step': step_i,
            'z': new_z,
            'region_nodes': region,
            'sliced_idx': sliced_idx,
            'step_idx': step_idx,
            'raw_centroid': raw_centroid.copy(),
            'displacement': displacement.copy(),
            'candidates': candidates,
            'best': best,
            'new_center': new_center.copy(),
            'claim_idx': claim_idx,
            'velocity': velocity.copy() if velocity is not None else None,
            'angle': angle,
            'action': action,
            'straight_pos': straight_pos,
            'straight_score': straight_score,
            'branch_flag': branch_flag,
        })

    center = new_center.copy()
    current_z = new_z
    prev_region_nodes = region
    trace_centers.append([center[0], center[1], current_z])

trace_centers = np.array(trace_centers)
print(f"\nTrace: {len(trace_centers)} centers, {int(claimed_points.sum())} total claimed")
print(f"Segments: {len(segments)}")
branch_flags = [f for s in segments for f in s['branch_flags']]
if branch_flags:
    print(f"Branch flags: {len(branch_flags)}")
    for bf in branch_flags:
        print(f"  z={bf['z']:.2f} xy=[{bf['xy'][0]:.2f},{bf['xy'][1]:.2f}] angle={bf['angle']:.1f}° density={bf['density']}")

# ═══════════════════════════════════════════
# PLOTS: 3 columns x (1 seed row + N step rows)
# ═══════════════════════════════════════════

n_rows = 1 + len(step_data)
fig, all_axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)

for row in all_axes:
    for ax in row:
        ax.set_facecolor('#0c0e12')

def zoom_to_z(ax, z, margin=2):
    pts_z = points[np.abs(points[:, 2] - z) < node_size * 2]
    if len(pts_z) > 0:
        ax.set_xlim(pts_z[:, 0].min() - margin, pts_z[:, 0].max() + margin)
        ax.set_ylim(pts_z[:, 1].min() - margin, pts_z[:, 1].max() + margin)

def draw_nodes(ax, z, highlight_nodes=None):
    z_nodes, gz = nodes_at_z(z)
    for nid, info in z_nodes.items():
        gx, gy, gzz = info['grid']
        x0 = root_min[0] + gx * node_size
        y0 = root_min[1] + gy * node_size
        if highlight_nodes and nid in highlight_nodes:
            ec, fc = '#00ff8880', '#00ff8810'
        else:
            n_pts = len(info['indices'])
            ec = '#3a4050'
            fc = plt.cm.viridis(min(n_pts / 50, 1.0)) if n_pts > 0 else '#1a1d26'
        ax.add_patch(Rectangle((x0, y0), node_size, node_size,
                                linewidth=0.5, edgecolor=ec, facecolor=fc))

# ═══════════════════════════════════════════
# ROW 0: state at PLOT_FROM_STEP
# ═══════════════════════════════════════════

# Use the center/z at the start of the plotted range
if PLOT_FROM_STEP > 0 and PLOT_FROM_STEP < len(trace_centers):
    row0_center = np.array(trace_centers[PLOT_FROM_STEP][:2])
    row0_z = trace_centers[PLOT_FROM_STEP][2]
else:
    row0_center = seed[:2]
    row0_z = seed[2]

# col 0: nodes at row0 Z
ax = all_axes[0][0]
ax.set_title(f'Nodes at z={row0_z:.2f} (step {PLOT_FROM_STEP})')
ax.set_aspect('equal')
draw_nodes(ax, row0_z)
ax.plot(row0_center[0], row0_center[1], 'r+', markersize=15, markeredgewidth=2, label='center')
ax.legend(fontsize=8)
ax.set_xlabel('X'); ax.set_ylabel('Y')
zoom_to_z(ax, row0_z)

# col 1: region grow at row0
ax = all_axes[0][1]
row0_region = region_grow_nodes(row0_center, row0_z)
ax.set_title(f'Region grow ({len(row0_region)} nodes)')
ax.set_aspect('equal')
draw_nodes(ax, row0_z, highlight_nodes=set(row0_region))

row0_region_idx = gather_all(row0_region)
row0_sliced = z_slice(row0_region_idx, row0_z, half_slab)
if len(row0_sliced) > 0:
    srpts = points[row0_sliced]
    ax.scatter(srpts[:, 0], srpts[:, 1], s=1, c='#00ff88', alpha=0.5, label='slab pts')

row0_centroid = srpts[:, :2].mean(axis=0) if len(row0_sliced) > 0 else row0_center
ax.plot(row0_center[0], row0_center[1], 'r+', markersize=15, markeredgewidth=2, label='center')
ax.plot(row0_centroid[0], row0_centroid[1], 'wo', markersize=8, markeredgewidth=2,
        markerfacecolor='none', label='centroid')
ax.add_patch(Circle(row0_center, CLAIM_RADIUS, fill=False,
                     edgecolor='#00ff88', linewidth=1, linestyle='--', label='claim r'))
ax.legend(fontsize=7)
ax.set_xlabel('X'); ax.set_ylabel('Y')
zoom_to_z(ax, row0_z)

# col 2: claimed state at this point
ax = all_axes[0][2]
# Show what's been claimed so far up to PLOT_FROM_STEP
claimed_so_far = np.where(claimed_points)[0] if PLOT_FROM_STEP > 0 else init_claimed
ax.set_title(f'Claimed so far ({len(claimed_so_far)} pts)')
ax.set_aspect('equal')
draw_nodes(ax, row0_z, highlight_nodes=set(row0_region))

if len(row0_sliced) > 0:
    ax.scatter(srpts[:, 0], srpts[:, 1], s=1, c='#555555', alpha=0.3, label='slab pts')

# Show claimed points in this slab
claimed_in_slab = row0_sliced[claimed_points[row0_sliced]]
if len(claimed_in_slab) > 0:
    ax.scatter(points[claimed_in_slab, 0], points[claimed_in_slab, 1], s=3, c='#00ff88', alpha=0.8, label='claimed')

ax.add_patch(Circle(row0_center, CLAIM_RADIUS, fill=False,
                     edgecolor='#00ff88', linewidth=1, linestyle='--', label='claim r'))
ax.plot(row0_center[0], row0_center[1], 'r+', markersize=15, markeredgewidth=2, label='center')
ax.legend(fontsize=7)
ax.set_xlabel('X'); ax.set_ylabel('Y')
zoom_to_z(ax, row0_z)

# ═══════════════════════════════════════════
# ROWS 1+: each step
# ═══════════════════════════════════════════

for si, sd in enumerate(step_data):
    row = si + 1
    # Use the global step index to find previous center in trace_centers
    global_step = sd['step']
    prev_center = np.array(trace_centers[global_step][:2])

    # col 0: nodes
    ax = all_axes[row][0]
    ax.set_title(f'Step {sd["step"]} nodes z={sd["z"]:.2f}')
    ax.set_aspect('equal')
    draw_nodes(ax, sd['z'])
    ax.plot(prev_center[0], prev_center[1], 'r+', markersize=12, markeredgewidth=2, label='prev center')
    ax.plot(sd['new_center'][0], sd['new_center'][1], 'wo', markersize=8, markeredgewidth=2,
            markerfacecolor='none', label='new center')
    # Velocity arrow
    if sd['velocity'] is not None:
        v = sd['velocity']
        ax.annotate('', xy=prev_center + v * 2, xytext=prev_center,
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1, alpha=0.5))
    ax.legend(fontsize=7)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    zoom_to_z(ax, sd['z'])

    # col 1: region + candidates
    ax = all_axes[row][1]
    title = f'Step {sd["step"]} [{sd["action"]}] angle={sd["angle"]:.1f}°'
    ax.set_title(title)
    ax.set_aspect('equal')
    draw_nodes(ax, sd['z'], highlight_nodes=set(sd['region_nodes']))

    rpts = points[sd['sliced_idx']]
    ax.scatter(rpts[:, 0], rpts[:, 1], s=1, c='#555555', alpha=0.3, label='slab pts')

    ax.plot(sd['raw_centroid'][0], sd['raw_centroid'][1], 'rx', markersize=10,
            markeredgewidth=2, label='raw centroid')

    for c in sd['candidates']:
        is_best = (c is sd['best'])
        color = '#ffcc00' if is_best else '#ffffff40'
        size = 7 if is_best else 2
        label = f'best (t={c["t"]:.1f}, n={c["count"]})' if is_best else None
        ax.plot(c['xy'][0], c['xy'][1], 'o', color=color, markersize=size, label=label)

    # Straight position if computed
    if sd['straight_pos'] is not None:
        ax.plot(sd['straight_pos'][0], sd['straight_pos'][1], 's', color='cyan',
                markersize=8, markeredgewidth=1.5, markerfacecolor='none',
                label=f'straight (n={sd["straight_score"]})')

    # Branch flag marker
    if sd['branch_flag']:
        bf = sd['branch_flag']
        ax.plot(bf['xy'][0], bf['xy'][1], '*', color='#ff4444', markersize=12,
                markeredgewidth=1, label=f'branch flag')

    ax.annotate('', xy=sd['best']['xy'], xytext=prev_center,
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))

    ax.plot(prev_center[0], prev_center[1], 'r+', markersize=12, markeredgewidth=2, label='prev center')
    ax.add_patch(Circle(sd['best']['xy'], CLAIM_RADIUS, fill=False,
                         edgecolor='#00ff88', linewidth=1, linestyle='--', label='claim r'))
    ax.legend(fontsize=5, loc='upper right')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    zoom_to_z(ax, sd['z'])

    # col 2: claimed
    ax = all_axes[row][2]
    ax.set_title(f'Step {sd["step"]} claimed: {len(sd["claim_idx"])} pts')
    ax.set_aspect('equal')
    draw_nodes(ax, sd['z'], highlight_nodes=set(sd['region_nodes']))

    ax.scatter(rpts[:, 0], rpts[:, 1], s=1, c='#555555', alpha=0.3, label='slab pts')

    if len(sd['claim_idx']) > 0:
        cpts = points[sd['claim_idx']]
        ax.scatter(cpts[:, 0], cpts[:, 1], s=3, c='#00ff88', alpha=0.8, label='claimed')

    ax.add_patch(Circle(sd['new_center'], CLAIM_RADIUS, fill=False,
                         edgecolor='#00ff88', linewidth=1, linestyle='--', label='claim r'))
    ax.plot(sd['new_center'][0], sd['new_center'][1], 'wo', markersize=8, markeredgewidth=2,
            markerfacecolor='none', label='new center')
    ax.legend(fontsize=7)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    zoom_to_z(ax, sd['z'])

plt.tight_layout()
plt.show()
