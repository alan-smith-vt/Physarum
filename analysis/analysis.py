"""
Analysis pipeline for filament extraction from point clouds.
Uses PCO octree node structure for efficient spatial queries.
"""

import numpy as np
from collections import deque
from PCO import PCOReader, PCOFormat
from PCO.pco_format import OctreeUtils


class AnalysisPipeline:

    def __init__(self):
        self.points = None       # Nx3 float32
        self.colors = None       # Nx3 uint8
        self.metadata = None
        self.results = {}

        # Node spatial index
        self.node_index = None   # {node_id: {'grid': (x,y,z), 'indices': [...]}}
        self.grid_to_node = None # {(gx,gy,gz): node_id}
        self.node_depth = 0
        self.node_size = 0.0     # world-space size of one node at max depth
        self.root_min = None
        self.root_size = 0.0

    # ── Data loading ──

    def load_pco(self, filename, level=None):
        """Load points and build node spatial index."""
        reader = PCOReader(filename)
        reader.load_metadata()
        self.metadata = reader.header
        self.root_min = np.array(reader.header['root_min'])
        self.root_size = reader.header['root_size']

        if level is None:
            level = reader.header['max_depth']
        self.node_depth = level
        self.node_size = self.root_size / (2 ** level)

        # Read all points in sorted node order (matches server's /level/ endpoint)
        node_ids_sorted = sorted([nid for nid in reader.index.keys() if len(nid) == level + 1])
        all_data = reader.read_nodes(node_ids_sorted)
        parsed = reader.parse_binary_data(all_data)
        self.points = parsed['xyz'].astype(np.float32)
        self.colors = parsed.get('rgb', None)

        self.bounds_min = self.points.min(axis=0).tolist()
        self.bounds_max = self.points.max(axis=0).tolist()

        # Build node index in same sorted order
        self.node_index = {}
        self.grid_to_node = {}
        point_offset = 0

        for nid in node_ids_sorted:
            _, count = reader.index[nid]
            grid = tuple(OctreeUtils.node_id_to_grid_coords(nid))
            indices = list(range(point_offset, point_offset + count))
            self.node_index[nid] = {
                'grid': grid,
                'indices': indices,
            }
            self.grid_to_node[grid] = nid
            point_offset += count

        print(f"Loaded {len(self.points):,} points from {filename} (level {level})")
        print(f"  Bounds: {self.bounds_min} to {self.bounds_max}")
        print(f"  Nodes: {len(self.node_index)}, node size: {self.node_size:.3f}")
        return self

    # ── Node utilities ──

    def _point_to_grid(self, point):
        """Convert a world-space point to grid coordinates at node_depth."""
        rel = (np.array(point[:3]) - self.root_min) / self.node_size
        return tuple(np.clip(rel.astype(int), 0, 2**self.node_depth - 1))

    def _grid_z_range(self, gz):
        """Get the Z bounds of a grid row."""
        z_min = self.root_min[2] + gz * self.node_size
        z_max = z_min + self.node_size
        return z_min, z_max

    def _nodes_in_z_slab(self, z):
        """
        Find grid Z indices that overlap the given Z value.
        Returns the primary gz and its neighbors to handle boundary cases.
        """
        gz = int((z - self.root_min[2]) / self.node_size)
        max_gz = 2**self.node_depth - 1
        gz = max(0, min(gz, max_gz))
        # Include neighbor layers to handle z_step smaller than node_size
        result = []
        for g in [gz - 1, gz, gz + 1]:
            if 0 <= g <= max_gz:
                result.append(g)
        return result

    def _region_grow_nodes(self, center_xy, z, claimed_nodes, density_fraction=0.5,
                            ref_density=None, prev_region=None):
        """
        BFS region grow through octree nodes at the given Z layer(s).

        If prev_region is provided, seeds from those nodes projected to new Z
        (temporal continuity). Otherwise seeds from center_xy.

        Uses ref_density for the density threshold if provided,
        otherwise uses the seed node's density.
        """
        gz_list = self._nodes_in_z_slab(z)

        accepted = []
        visited = set()
        queue = deque()
        seed_density = 0

        # Seed from previous region if available (temporal continuity)
        if prev_region:
            for nid in prev_region:
                gx, gy, _ = self.node_index[nid]['grid']
                for gz in gz_list:
                    key = (gx, gy, gz)
                    if key in self.grid_to_node and key not in visited:
                        nbr = self.grid_to_node[key]
                        if nbr not in claimed_nodes:
                            queue.append(key)
                            visited.add(key)
                            seed_density = max(seed_density,
                                len(self.node_index[nbr]['indices']))

        # Fallback: seed from center_xy
        if len(queue) == 0:
            gx = int((center_xy[0] - self.root_min[0]) / self.node_size)
            gy = int((center_xy[1] - self.root_min[1]) / self.node_size)
            max_coord = 2**self.node_depth - 1
            gx = max(0, min(gx, max_coord))
            gy = max(0, min(gy, max_coord))

            for gz in gz_list:
                key = (gx, gy, gz)
                if key in self.grid_to_node:
                    nid = self.grid_to_node[key]
                    if nid not in claimed_nodes:
                        queue.append(key)
                        visited.add(key)
                        seed_density = max(seed_density,
                            len(self.node_index[nid]['indices']))

            # If still nothing, search nearby
            if len(queue) == 0:
                best_dist = float('inf')
                best_key = None
                for gz in gz_list:
                    for dx in range(-5, 6):
                        for dy in range(-5, 6):
                            key = (gx + dx, gy + dy, gz)
                            if key in self.grid_to_node:
                                nid = self.grid_to_node[key]
                                if nid not in claimed_nodes:
                                    d = dx*dx + dy*dy
                                    if d < best_dist:
                                        best_dist = d
                                        best_key = key
                if best_key is not None:
                    queue.append(best_key)
                    visited.add(best_key)
                    seed_density = len(self.node_index[self.grid_to_node[best_key]]['indices'])

        if not queue:
            return []

        # Density threshold
        effective_density = ref_density if ref_density is not None else seed_density
        min_density = max(1, int(effective_density * density_fraction))

        # BFS
        while queue:
            cx, cy, cz = queue.popleft()
            nid = self.grid_to_node.get((cx, cy, cz))
            if nid is None or nid in claimed_nodes:
                continue
            accepted.append(nid)

            for dz_gz in gz_list:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz_gz == cz:
                            continue
                        nkey = (cx + dx, cy + dy, dz_gz)
                        if nkey not in visited and nkey in self.grid_to_node:
                            nbr_nid = self.grid_to_node[nkey]
                            if nbr_nid not in claimed_nodes:
                                nbr_density = len(self.node_index[nbr_nid]['indices'])
                                if nbr_density >= min_density:
                                    visited.add(nkey)
                                    queue.append(nkey)

        return accepted

    # ── Cross section ──

    def node_bounds_at_z(self, z):
        """
        Get bounding boxes of all nodes that overlap a given Z value.
        Returns list of {node_id, grid, min, max, point_count}.
        """
        gz_list = self._nodes_in_z_slab(z)
        results = []
        for nid, info in self.node_index.items():
            gx, gy, gz = info['grid']
            if gz in gz_list:
                node_min = self.root_min + np.array([gx, gy, gz]) * self.node_size
                node_max = node_min + self.node_size
                results.append({
                    'node_id': nid,
                    'grid': [gx, gy, gz],
                    'min': node_min.tolist(),
                    'max': node_max.tolist(),
                    'point_count': len(info['indices']),
                })
        return results

    def all_node_bounds(self):
        """
        Get bounding boxes of all nodes. Built once, used for the grid overlay.
        """
        results = []
        for nid, info in self.node_index.items():
            gx, gy, gz = info['grid']
            node_min = self.root_min + np.array([gx, gy, gz]) * self.node_size
            node_max = node_min + self.node_size
            results.append({
                'min': node_min.tolist(),
                'max': node_max.tolist(),
                'point_count': len(info['indices']),
            })
        return results

    def cross_section(self, axis, value, thickness):
        half = thickness / 2.0
        coords = self.points[:, axis]
        mask = np.abs(coords - value) <= half
        indices = np.where(mask)[0]

        axes_2d = [a for a in range(3) if a != axis]
        points_2d = self.points[indices][:, axes_2d]

        result = {
            'indices': indices,
            'points_2d': points_2d,
            'axis': axis,
            'value': value,
            'thickness': thickness,
        }
        if self.colors is not None:
            result['colors'] = self.colors[indices]

        return result

    # ── Seed-based operations ──

    def find_nearest(self, point):
        point = np.array(point, dtype=np.float32)
        dists = np.linalg.norm(self.points - point, axis=1)
        idx = np.argmin(dists)
        return {
            'index': int(idx),
            'position': self.points[idx].tolist(),
            'distance': float(dists[idx]),
        }

    # ── Centerline tracer ──

    def trace_centerline(self, seed, z_step=0.2, claim_radius=0.5,
                          max_xy_factor=2.0, min_points=3,
                          slab_thickness=None, heatmap_res=25,
                          min_branch_points=100):
        """
        Trace a centerline upward from a seed point with:
        - Octree node region growing (coarse spatial filter)
        - Z-slicing (precise slab)
        - Heatmap peak extraction for positioning and branch detection
        - Temporal continuity (seeds from previous step's nodes)
        - Reference density for region grow thresholding
        """
        if slab_thickness is None:
            slab_thickness = z_step

        seed = np.array(seed, dtype=np.float32)
        max_xy_step = z_step * max_xy_factor
        half_slab = slab_thickness / 2.0
        z_max = float(self.points[:, 2].max())
        z_coords = self.points[:, 2]

        claimed_points = np.zeros(len(self.points), dtype=bool)
        claimed_nodes = set()

        def gather_all(region):
            idx = []
            for nid in region:
                idx.extend(self.node_index[nid]['indices'])
            return np.array(idx) if idx else np.array([], dtype=int)

        def gather_expanded(region, z):
            """Gather points from region nodes expanded outward by BFS
            with no density filtering — stops only at missing nodes (void).
            This ensures claiming doesn't clip at density-filtered boundaries."""
            gz_list = self._nodes_in_z_slab(z)
            visited = set()
            queue = deque()
            expanded_nids = set()

            # Seed from region nodes across Z layers
            for nid in region:
                gx, gy, _ = self.node_index[nid]['grid']
                for gz in gz_list:
                    key = (gx, gy, gz)
                    if key in self.grid_to_node and key not in visited:
                        visited.add(key)
                        queue.append(key)

            # BFS: expand to any existing adjacent node (no density check)
            while queue:
                cx, cy, cz = queue.popleft()
                nid = self.grid_to_node.get((cx, cy, cz))
                if nid is not None:
                    expanded_nids.add(nid)
                    for dz_gz in gz_list:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0 and dz_gz == cz:
                                    continue
                                nkey = (cx + dx, cy + dy, dz_gz)
                                if nkey not in visited and nkey in self.grid_to_node:
                                    visited.add(nkey)
                                    queue.append(nkey)

            idx = []
            for nid in expanded_nids:
                idx.extend(self.node_index[nid]['indices'])
            return np.array(idx) if idx else np.array([], dtype=int)

        def z_slice_fn(indices, z):
            if len(indices) == 0:
                return indices
            return indices[np.abs(z_coords[indices] - z) <= half_slab]

        def find_peaks(center_xy, pts_xy, max_r, max_peaks=5):
            """
            Compute density heatmap and extract peaks greedily.
            Uses claim_radius as sensing kernel.
            Tie-breaks by distance to centroid of remaining points
            (so flat plateaus default to geometric center of structure).
            """
            from scipy.spatial.distance import cdist

            gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, heatmap_res)
            gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, heatmap_res)
            gxx, gyy = np.meshgrid(gx, gy)
            grid_flat = np.column_stack([gxx.ravel(), gyy.ravel()])

            # Mask grid cells outside movement disk
            grid_dist_sq = (grid_flat[:, 0] - center_xy[0])**2 + \
                           (grid_flat[:, 1] - center_xy[1])**2
            disk_mask = grid_dist_sq <= max_r**2

            remaining = pts_xy.copy()
            peaks = []

            for _ in range(max_peaks):
                if len(remaining) == 0:
                    break

                # Centroid of remaining points for tie-breaking
                centroid = remaining.mean(axis=0)

                # Density sensing with claim_radius
                dists = cdist(grid_flat[disk_mask], remaining)
                counts = np.sum(dists <= claim_radius, axis=1)

                max_count = int(counts.max())
                if max_count < min_points:
                    break

                # Tie-break: among cells with max count, pick closest to centroid
                tied = counts == max_count
                tied_positions = grid_flat[disk_mask][tied]
                centroid_dists = (tied_positions[:, 0] - centroid[0])**2 + \
                                (tied_positions[:, 1] - centroid[1])**2
                best_local = np.argmin(centroid_dists)
                best_pos = tied_positions[best_local]

                peaks.append((best_pos.copy(), max_count))

                # Remove points within claim_radius of peak
                d = np.sqrt((remaining[:, 0] - best_pos[0])**2 +
                           (remaining[:, 1] - best_pos[1])**2)
                remaining = remaining[d > claim_radius]

            return peaks

        center = seed.copy()
        centerline = [center.tolist()]
        claimed_per_step = []
        branch_flags = []
        velocity = None
        ref_density = None
        density_decay = 0.9

        # Initial claim from ALL points in slab (not just region nodes)
        region = self._region_grow_nodes(center[:2], center[2], claimed_nodes)
        prev_region = region
        if region:
            ref_density = max(len(self.node_index[nid]['indices']) for nid in region)

            init_expanded = gather_expanded(region, center[2])
            init_sliced = z_slice_fn(init_expanded, center[2])
            if len(init_sliced) > 0:
                dists = np.linalg.norm(self.points[init_sliced][:, :2] - center[:2], axis=1)
                claim_idx = init_sliced[dists <= claim_radius]
                if len(claim_idx) > 0:
                    claimed_points[claim_idx] = True
                    claimed_per_step.append(claim_idx.tolist())

        step_count = 0
        while center[2] + z_step <= z_max:
            new_z = center[2] + z_step

            # 1. Region grow with temporal continuity and reference density
            region = self._region_grow_nodes(
                center[:2], new_z, claimed_nodes,
                ref_density=ref_density, prev_region=prev_region)
            if not region:
                break

            # 2. Gather -> Z-slice -> filter claimed (for heatmap)
            all_idx = gather_all(region)
            sliced = z_slice_fn(all_idx, new_z)
            unclaimed = sliced[~claimed_points[sliced]]
            if len(unclaimed) == 0:
                break

            step_xy = self.points[unclaimed][:, :2]

            # 3. Heatmap peak extraction (uses region-filtered points)
            peaks = find_peaks(center[:2], step_xy, max_xy_step)

            if not peaks:
                break

            # Primary peak: where we move
            best_xy, best_count = peaks[0]

            # Clamp displacement
            displacement = best_xy - center[:2]
            dist = np.linalg.norm(displacement)
            if dist > max_xy_step:
                displacement = displacement * (max_xy_step / dist)
                best_xy = center[:2] + displacement

            new_center = np.array([best_xy[0], best_xy[1], new_z], dtype=np.float32)

            # 4. Claim from expanded region (region + 1 ring of neighbors)
            expanded_idx = gather_expanded(region, new_z)
            expanded_sliced = z_slice_fn(expanded_idx, new_z)
            expanded_unclaimed = expanded_sliced[~claimed_points[expanded_sliced]]
            claim_dists = np.linalg.norm(
                self.points[expanded_unclaimed][:, :2] - best_xy, axis=1)
            claim_idx = expanded_unclaimed[claim_dists <= claim_radius]

            if len(claim_idx) < min_points:
                break

            claimed_points[claim_idx] = True

            for nid in region:
                ni = np.array(self.node_index[nid]['indices'])
                if np.all(claimed_points[ni]):
                    claimed_nodes.add(nid)

            # 5. Branch flags from secondary peaks
            for peak_xy, peak_count in peaks[1:]:
                if peak_count >= min_branch_points:
                    branch_flags.append({
                        'z': float(new_z),
                        'xy': peak_xy.tolist(),
                        'density': int(peak_count),
                    })

            # Update velocity
            new_disp = best_xy - center[:2]
            if np.linalg.norm(new_disp) > 1e-8:
                velocity = new_disp.copy()

            # Update reference density
            if ref_density is not None and region:
                current_density = max(len(self.node_index[nid]['indices']) for nid in region)
                ref_density = density_decay * ref_density + (1 - density_decay) * current_density

            center = new_center
            centerline.append(center.tolist())
            claimed_per_step.append(claim_idx.tolist())
            prev_region = region
            step_count += 1

        all_claimed = np.where(claimed_points)[0]

        return {
            'centerline': centerline,
            'claimed_per_step': claimed_per_step,
            'all_claimed': all_claimed.tolist(),
            'branch_flags': branch_flags,
            'steps': int(step_count),
        }

    # ── Station-based trace ──

    def trace_with_stations(self, seed, z_step=0.2, claim_radius=0.5,
                             max_xy_factor=2.0, min_points=3,
                             slab_thickness=None, heatmap_res=25,
                             min_branch_points=100, nudge_budget=20,
                             n_dirs=40, use_tip_center=True,
                             max_nudges_per_dir=3, min_density_ratio=0.3,
                             density_ema_alpha=0.3, lookahead_steps=2,
                             retract_divisor=8, max_point_loss_pct=0.005,
                             min_density_improvement=0.001,
                             progress_callback=None, progress_interval=1):
        """
        Trace a centerline upward from seed, creating stations at each Z-step.
        Each timestep: tip advances, then all existing stations thicken once.
        Station nudge_step and initial_radius are auto-derived from slab geometry.

        progress_callback: if provided, called with (snapshot_dict) periodically.
        progress_interval: call progress_callback every N tip steps / thicken ticks.

        Returns dict with centerline, stations (with polar boundaries), branch_flags.
        """
        if slab_thickness is None:
            slab_thickness = z_step

        seed = np.array(seed, dtype=np.float32)
        max_xy_step = z_step * max_xy_factor
        half_slab = slab_thickness / 2.0
        z_max = float(self.points[:, 2].max())
        z_coords = self.points[:, 2]

        claimed_points = np.zeros(len(self.points), dtype=bool)
        claimed_nodes = set()
        stations = []

        def gather_all(region):
            idx = []
            for nid in region:
                idx.extend(self.node_index[nid]['indices'])
            return np.array(idx) if idx else np.array([], dtype=int)

        def gather_expanded(region, z):
            gz_list = self._nodes_in_z_slab(z)
            visited = set()
            queue = deque()
            expanded_nids = set()
            for nid in region:
                gx, gy, _ = self.node_index[nid]['grid']
                for gz in gz_list:
                    key = (gx, gy, gz)
                    if key in self.grid_to_node and key not in visited:
                        visited.add(key)
                        queue.append(key)
            while queue:
                cx, cy, cz = queue.popleft()
                nid = self.grid_to_node.get((cx, cy, cz))
                if nid is not None:
                    expanded_nids.add(nid)
                    for dz_gz in gz_list:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0 and dz_gz == cz:
                                    continue
                                nkey = (cx + dx, cy + dy, dz_gz)
                                if nkey not in visited and nkey in self.grid_to_node:
                                    visited.add(nkey)
                                    queue.append(nkey)
            idx = []
            for nid in expanded_nids:
                idx.extend(self.node_index[nid]['indices'])
            return np.array(idx) if idx else np.array([], dtype=int)

        def z_slice_fn(indices, z):
            if len(indices) == 0:
                return indices
            return indices[np.abs(z_coords[indices] - z) <= half_slab]

        def get_slab_points(center_xy, z, region):
            """Get slab indices and their XY coords for a station."""
            expanded = gather_expanded(region, z)
            sliced = z_slice_fn(expanded, z)
            if len(sliced) == 0:
                return sliced, np.empty((0, 2))
            return sliced, self.points[sliced][:, :2]

        def find_peaks(center_xy, pts_xy, max_r, max_peaks=5):
            from scipy.spatial.distance import cdist
            gx = np.linspace(center_xy[0] - max_r, center_xy[0] + max_r, heatmap_res)
            gy = np.linspace(center_xy[1] - max_r, center_xy[1] + max_r, heatmap_res)
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
                counts = np.sum(dists <= claim_radius, axis=1)
                max_count = int(counts.max())
                if max_count < min_points:
                    break
                tied = counts == max_count
                tied_positions = grid_flat[disk_mask][tied]
                centroid_dists = (tied_positions[:, 0] - centroid[0])**2 + \
                                (tied_positions[:, 1] - centroid[1])**2
                best_pos = tied_positions[np.argmin(centroid_dists)]
                peaks.append((best_pos.copy(), max_count))
                d = np.sqrt((remaining[:, 0] - best_pos[0])**2 +
                           (remaining[:, 1] - best_pos[1])**2)
                remaining = remaining[d > claim_radius]
            return peaks

        # ── Main loop ──

        center = seed.copy()
        centerline = [center.tolist()]
        branch_flags = []
        ref_density = None
        density_decay = 0.9
        prev_region = None

        # Initial region + station at seed
        region = self._region_grow_nodes(center[:2], center[2], claimed_nodes)
        prev_region = region
        if region:
            ref_density = max(len(self.node_index[nid]['indices']) for nid in region)
            slab_idx, slab_xy = get_slab_points(center[:2], center[2], region)
            if len(slab_idx) > 0:
                station = Station(
                    center, slab_idx, slab_xy, age=0, n_dirs=n_dirs)
                stations.append(station)

        def _build_snapshot(phase, step):
            """Build a lightweight progress snapshot — just claimed + centerline."""
            # Gather station-claimed via bool arrays (fast)
            combined_mask = claimed_points.copy()
            for s in stations:
                # Map local claimed back to global indices
                local_claimed = s.cross_section.claimed
                if local_claimed.any():
                    combined_mask[s.slab_indices[local_claimed]] = True
            return {
                'phase': phase,
                'step': int(step),
                'centerline': centerline.copy(),
                'all_claimed': np.where(combined_mask)[0].tolist(),
                'n_stations': len(stations),
                'done': False,
            }

        step_count = 0
        while center[2] + z_step <= z_max:
            new_z = center[2] + z_step

            # 1. Tip advancement (same logic as trace_centerline)
            region = self._region_grow_nodes(
                center[:2], new_z, claimed_nodes,
                ref_density=ref_density, prev_region=prev_region)
            if not region:
                break

            all_idx = gather_all(region)
            sliced = z_slice_fn(all_idx, new_z)
            unclaimed = sliced[~claimed_points[sliced]]
            if len(unclaimed) == 0:
                break

            step_xy = self.points[unclaimed][:, :2]
            peaks = find_peaks(center[:2], step_xy, max_xy_step)
            if not peaks:
                break

            best_xy, best_count = peaks[0]
            displacement = best_xy - center[:2]
            dist = np.linalg.norm(displacement)
            if dist > max_xy_step:
                best_xy = center[:2] + displacement * (max_xy_step / dist)

            new_center = np.array([best_xy[0], best_xy[1], new_z], dtype=np.float32)

            # Claim from expanded region
            expanded_idx = gather_expanded(region, new_z)
            expanded_sliced = z_slice_fn(expanded_idx, new_z)
            expanded_unclaimed = expanded_sliced[~claimed_points[expanded_sliced]]
            claim_dists = np.linalg.norm(
                self.points[expanded_unclaimed][:, :2] - best_xy, axis=1)
            claim_idx = expanded_unclaimed[claim_dists <= claim_radius]

            if len(claim_idx) < min_points:
                break

            claimed_points[claim_idx] = True

            for nid in region:
                ni = np.array(self.node_index[nid]['indices'])
                if np.all(claimed_points[ni]):
                    claimed_nodes.add(nid)

            # Branch flags from secondary peaks
            for peak_xy, peak_count in peaks[1:]:
                if peak_count >= min_branch_points:
                    branch_flags.append({
                        'z': float(new_z),
                        'xy': peak_xy.tolist(),
                        'density': int(peak_count),
                    })

            # 2. Create new station at tip position
            slab_idx, slab_xy = get_slab_points(best_xy, new_z, region)
            if len(slab_idx) > 0:
                station = Station(
                    new_center, slab_idx, slab_xy, age=step_count + 1,
                    n_dirs=n_dirs)
                stations.append(station)

            # 3. Thicken all existing stations (except the one just created)
            expansion_kwargs = dict(
                max_nudges_per_dir=max_nudges_per_dir,
                min_density_ratio=min_density_ratio,
                density_ema_alpha=density_ema_alpha,
                lookahead_steps=lookahead_steps,
            )
            for s in stations[:-1]:
                s.thicken(nudge_budget=nudge_budget, use_tip_center=use_tip_center,
                          **expansion_kwargs)

            # Update ref density
            if ref_density is not None and region:
                current_density = max(len(self.node_index[nid]['indices']) for nid in region)
                ref_density = density_decay * ref_density + (1 - density_decay) * current_density

            center = new_center
            centerline.append(center.tolist())
            prev_region = region
            step_count += 1

            # Progress callback
            if progress_callback and step_count % progress_interval == 0:
                progress_callback(_build_snapshot('tracing', step_count))

        # 4. Continue thickening all stations until convergence
        max_post_steps = 500
        for post_step in range(max_post_steps):
            any_active = False
            for s in stations:
                if not s.converged:
                    s.thicken(nudge_budget=nudge_budget, use_tip_center=use_tip_center,
                              **expansion_kwargs)
                    any_active = True
            if not any_active:
                break
            if progress_callback and post_step % max(progress_interval * 10, 10) == 0:
                progress_callback(_build_snapshot('thickening', post_step))

        # Final retraction pass on all stations
        if progress_callback:
            progress_callback(_build_snapshot('retracting', 0))

        for s in stations:
            s.cross_section.retract_pass(s.slab_xy,
                                          retract_divisor=retract_divisor,
                                          max_point_loss_pct=max_point_loss_pct,
                                          min_density_improvement=min_density_improvement)

        # Build final result
        all_station_claimed = set()
        for s in stations:
            all_station_claimed.update(s.get_global_claimed().tolist())
        all_claimed_combined = set(np.where(claimed_points)[0].tolist()) | all_station_claimed

        station_data = [s.to_dict() for s in stations]

        result = {
            'centerline': centerline,
            'stations': station_data,
            'all_claimed': list(all_claimed_combined),
            'branch_flags': branch_flags,
            'steps': int(step_count),
            'done': True,
        }

        if progress_callback:
            progress_callback(result)

        return result

    # ── Reset ──

    def reset(self):
        self.results = {}


# ═══════════════════════════════════════════
# Station-based tracing with polar cross-section thickening
# ═══════════════════════════════════════════

class PolarCrossSection:
    """
    Polar-curve cross-section that grows via greedy nudge allocation.
    Optimized: bool array for claimed, vectorized scoring.
    """

    def __init__(self, center, n_points, initial_radius=0.3, n_dirs=10, nudge_step=0.1):
        self.center = np.array(center, dtype=np.float64)
        self.n_dirs = n_dirs
        self.nudge_step = nudge_step
        self.angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        self.radii = np.full(n_dirs, initial_radius, dtype=np.float64)
        self.dir_density = np.full(n_dirs, -1.0, dtype=np.float64)
        self.claimed = np.zeros(n_points, dtype=bool)
        self.min_arc_length = 0.25

    @property
    def claimed_count(self):
        return int(self.claimed.sum())

    def get_boundary_xy(self):
        x = self.center[0] + self.radii * np.cos(self.angles)
        y = self.center[1] + self.radii * np.sin(self.angles)
        return np.column_stack([x, y])

    def get_perimeter(self):
        verts = self.get_boundary_xy()
        diffs = np.diff(np.vstack([verts, verts[:1]]), axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    @staticmethod
    def _polygon_area(verts):
        x, y = verts[:, 0], verts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _rescale_directions(self):
        perimeter = self.get_perimeter()
        new_n = max(10, int(np.round(perimeter / self.min_arc_length)))
        if new_n == self.n_dirs:
            return
        new_angles = np.linspace(0, 2 * np.pi, new_n, endpoint=False)
        old_ext = np.concatenate([self.angles - 2*np.pi, self.angles, self.angles + 2*np.pi])
        self.radii = np.interp(new_angles, old_ext, np.tile(self.radii, 3))
        self.dir_density = np.interp(new_angles, old_ext, np.tile(self.dir_density, 3))
        self.angles = new_angles
        self.n_dirs = new_n

    def points_inside_mask(self, points_xy):
        dx = points_xy[:, 0] - self.center[0]
        dy = points_xy[:, 1] - self.center[1]
        angles = np.arctan2(dy, dx) % (2 * np.pi)
        dists = np.sqrt(dx**2 + dy**2)
        idx = np.searchsorted(self.angles, angles) % self.n_dirs
        idx_prev = (idx - 1) % self.n_dirs
        a0, a1 = self.angles[idx_prev], self.angles[idx]
        da = (a1 - a0) % (2 * np.pi)
        safe_da = np.where(da > 1e-10, da, 1.0)
        t = np.clip(((angles - a0) % (2 * np.pi)) / safe_da, 0, 1)
        r_interp = self.radii[idx_prev] * (1 - t) + self.radii[idx] * t
        return dists <= r_interp

    def thicken_step(self, all_points_xy, nudge_budget=10, center_override=None,
                     max_nudges_per_dir=3, min_density_ratio=0.3,
                     density_ema_alpha=0.3, lookahead_steps=2):
        """
        One thickening tick.
        center_override: fixed [x,y] center, skips re-centering.
        max_nudges_per_dir: max expansion attempts per direction per tick.
            Each attempt is one nudge_step, validated independently.
        min_density_ratio: band density must be >= this * direction's EMA to accept.
        density_ema_alpha: EMA blend for per-direction density history.
        lookahead_steps: when a single step fails, check this many additional
            steps ahead. If any has good density, accept the current step
            (bridging a gap). 0 = no lookahead (strict).
        """
        n_claimed = self.claimed_count

        # 1. Recompute CG — skip entirely if center_override matches current
        if center_override is not None:
            # Fixed center mode: no re-centering, no radii re-expression
            pass
        elif n_claimed > 0:
            new_cg = all_points_xy[self.claimed].mean(axis=0)
            old_boundary = self.get_boundary_xy()
            self.center = new_cg
            dx = old_boundary[:, 0] - new_cg[0]
            dy = old_boundary[:, 1] - new_cg[1]
            self.radii = np.sqrt(dx**2 + dy**2)
            self.angles = np.arctan2(dy, dx) % (2 * np.pi)
            sort_idx = np.argsort(self.angles)
            self.angles = self.angles[sort_idx]
            self.radii = self.radii[sort_idx]
            self.dir_density = self.dir_density[sort_idx] if len(self.dir_density) == len(sort_idx) else self.dir_density

        # 2. Unclaimed points + polar coords
        unclaimed_idx = np.where(~self.claimed)[0]
        if len(unclaimed_idx) == 0:
            return {'nudges_spent': 0, 'new_claimed': 0, 'total_claimed': n_claimed,
                    'n_dirs': self.n_dirs}

        unc_pts = all_points_xy[unclaimed_idx]
        dx_u = unc_pts[:, 0] - self.center[0]
        dy_u = unc_pts[:, 1] - self.center[1]
        unc_angles = np.arctan2(dy_u, dx_u) % (2 * np.pi)
        unc_dists = np.sqrt(dx_u**2 + dy_u**2)

        # 3. Precompute wedge membership
        half_wedge = np.pi / self.n_dirs
        diff = unc_angles[:, None] - self.angles[None, :]
        ang_diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
        in_wedge = ang_diff < half_wedge

        # 4+5. Iterative nudge passes.
        #   Each pass: try one nudge per direction sequentially.
        #   Each direction gets up to max_nudges_per_dir independent attempts.
        #   Each attempt: expand by one nudge_step, check density in that
        #   single increment, accept or reject independently.
        nudges_allocated = np.zeros(self.n_dirs, dtype=int)
        failed = np.zeros(self.n_dirs, dtype=bool)  # failed last attempt = no more tries
        total_nudges_spent = 0

        for _pass in range(max_nudges_per_dir):
            for i in range(self.n_dirs):
                if nudges_allocated[i] >= max_nudges_per_dir or failed[i]:
                    continue

                wedge_mask = in_wedge[:, i]
                old_r = self.radii[i]
                new_r = old_r + self.nudge_step

                # Count points in this single increment band
                in_band = (unc_dists > old_r) & (unc_dists <= new_r)
                band_count = int(np.sum(wedge_mask & in_band))

                # Band area for this single increment
                band_area = half_wedge * (new_r**2 - old_r**2)
                band_density = band_count / max(band_area, 1e-6)

                # Check against this direction's density history
                ref = 0.0 if self.dir_density[i] < 0 else self.dir_density[i]

                if band_density >= ref * min_density_ratio and band_count > 0:
                    # Accept this nudge
                    self.radii[i] = new_r
                    nudges_allocated[i] += 1
                    total_nudges_spent += 1

                    # Update density EMA
                    if self.dir_density[i] < 0:
                        self.dir_density[i] = band_density
                    else:
                        self.dir_density[i] = (density_ema_alpha * band_density +
                                               (1 - density_ema_alpha) * self.dir_density[i])
                else:
                    # Current band failed — lookahead: check future bands
                    # If any band within lookahead_steps has good density,
                    # accept the current step (bridge the gap)
                    bridged = False
                    if lookahead_steps > 0:
                        for la in range(1, lookahead_steps + 1):
                            la_inner = old_r + self.nudge_step * la
                            la_outer = la_inner + self.nudge_step
                            la_band = (unc_dists > la_inner) & (unc_dists <= la_outer)
                            la_count = int(np.sum(wedge_mask & la_band))
                            la_area = half_wedge * (la_outer**2 - la_inner**2)
                            la_density = la_count / max(la_area, 1e-6)
                            if la_density >= ref * min_density_ratio and la_count > 0:
                                bridged = True
                                break

                    if bridged:
                        # Accept: bridge through the gap
                        self.radii[i] = new_r
                        nudges_allocated[i] += 1
                        total_nudges_spent += 1
                        # Don't update EMA — the gap band isn't representative
                    else:
                        # Truly empty ahead — stop this direction
                        failed[i] = True

        # 6. Claim
        old_count = n_claimed
        self.claimed = self.points_inside_mask(all_points_xy)
        self._rescale_directions()

        return {
            'nudges_spent': int(total_nudges_spent),
            'new_claimed': self.claimed_count - old_count,
            'total_claimed': self.claimed_count,
            'n_dirs': self.n_dirs,
        }

    def retract_pass(self, all_points_xy, max_iters=500,
                     retract_divisor=8, max_point_loss_pct=0.005,
                     min_density_improvement=0.001):
        """
        Refine boundary inward to remove empty overshoot.

        Args:
            retract_divisor: retract_step = nudge_step / retract_divisor.
                Higher = smaller steps = more conservative. Default 8.
            max_point_loss_pct: reject any single retraction that drops more
                than this fraction of current points. Default 0.5%.
            min_density_improvement: require density to improve by at least
                this fraction (e.g. 0.015 = 1.5%) to accept. Default 1.5%.
        """
        retract_step = self.nudge_step / retract_divisor
        total_retractions = 0

        for iteration in range(max_iters):
            improved = False
            current_count = self.claimed_count
            if current_count == 0:
                break
            current_area = self._polygon_area(self.get_boundary_xy())
            current_density = current_count / max(current_area, 1e-6)
            max_loss = max(1, int(current_count * max_point_loss_pct))
            density_threshold = current_density * (1 + min_density_improvement)

            for i in range(self.n_dirs):
                old_r = self.radii[i]
                new_r = max(0.05, old_r - retract_step)
                if new_r >= old_r:
                    continue
                self.radii[i] = new_r
                new_inside = self.points_inside_mask(all_points_xy)
                new_count = int(new_inside.sum())
                lost = current_count - new_count

                if lost > max_loss:
                    self.radii[i] = old_r
                    continue

                new_area = self._polygon_area(self.get_boundary_xy())
                new_density = new_count / max(new_area, 1e-6)

                if new_density >= density_threshold:
                    self.claimed = new_inside
                    current_count = new_count
                    current_area = new_area
                    current_density = new_density
                    density_threshold = current_density * (1 + min_density_improvement)
                    improved = True
                    total_retractions += 1
                else:
                    self.radii[i] = old_r

            if not improved:
                break

        self.claimed = self.points_inside_mask(all_points_xy)
        final_area = self._polygon_area(self.get_boundary_xy())
        return {'iterations': iteration + 1, 'total_retractions': total_retractions,
                'final_claimed': self.claimed_count,
                'final_area': float(final_area),
                'final_density': self.claimed_count / max(final_area, 1e-6)}

    def to_dict(self):
        """Serialize for JSON transport to the viewer."""
        boundary = self.get_boundary_xy()
        return {
            'center': self.center.tolist(),
            'n_dirs': self.n_dirs,
            'angles': self.angles.tolist(),
            'radii': self.radii.tolist(),
            'boundary': boundary.tolist(),
            'claimed_count': self.claimed_count,
        }


class Station:
    """
    A fixed point along a filament centerline that grows its cross-section
    over time via PolarCrossSection thickening.
    """

    def __init__(self, center_3d, slab_indices, all_points_xy, age=0,
                 initial_radius=None, n_dirs=10, nudge_step=None):
        """
        Args:
            center_3d: [x, y, z] position
            slab_indices: indices into the full point cloud for this Z-slab
            all_points_xy: Nx2 XY coords of slab points (indexed by slab_indices)
            age: creation timestep
            initial_radius: if None, derived from slab geometry
            n_dirs: maximum — actual count scales with slab point count
            nudge_step: if None, derived from slab geometry (avg inter-point spacing)
        """
        self.center = np.array(center_3d, dtype=np.float32)
        self.z = float(center_3d[2])
        self.age = age
        self.slab_indices = slab_indices
        self.slab_xy = all_points_xy

        n_pts = len(slab_indices)

        # Estimate slab geometry
        if n_pts > 1:
            spread = all_points_xy.max(axis=0) - all_points_xy.min(axis=0)
            slab_diameter = np.linalg.norm(spread)
            slab_area = spread[0] * spread[1] if spread[0] > 0 and spread[1] > 0 else slab_diameter**2
            avg_spacing = np.sqrt(slab_area / n_pts)
        else:
            slab_diameter = 1.0
            avg_spacing = 0.1

        # Derive nudge_step from avg inter-point spacing (2-3× spacing is a natural step)
        effective_nudge = nudge_step if nudge_step is not None else avg_spacing * 2.5

        # Derive initial_radius from slab diameter (~5% of cross-section)
        effective_radius = initial_radius if initial_radius is not None else slab_diameter * 0.05

        # Scale n_dirs: ~1 direction per 50 points, clamped to [6, n_dirs]
        effective_dirs = max(6, min(n_dirs, n_pts // 50))

        # Store for debug inspection
        self._slab_stats = {
            'n_pts': n_pts,
            'slab_diameter': float(slab_diameter),
            'avg_spacing': float(avg_spacing),
            'effective_nudge': float(effective_nudge),
            'effective_radius': float(effective_radius),
            'effective_dirs': effective_dirs,
        }

        self.cross_section = PolarCrossSection(
            center=center_3d[:2],
            n_points=n_pts,
            initial_radius=effective_radius,
            n_dirs=effective_dirs,
            nudge_step=effective_nudge,
        )

        # Initial claim
        self.cross_section.claimed = self.cross_section.points_inside_mask(self.slab_xy)

        self.branch_flags = []
        self.converged = False
        self._stall_count = 0
        # The tip center (XY) at creation — used if use_tip_center=True
        self.tip_center_xy = np.array(center_3d[:2], dtype=np.float64)

    def thicken(self, nudge_budget=10, use_tip_center=False, **expansion_kwargs):
        """Run one thickening step. Marks converged after repeated stalls.
        use_tip_center: if True, keep center fixed at the tip position.
        expansion_kwargs: passed through to thicken_step (lookahead_mult,
            max_nudges_per_dir, diminishing_factor, min_density_ratio,
            density_ema_alpha)."""
        if self.converged:
            return {'nudges_spent': 0, 'new_claimed': 0,
                    'total_claimed': self.cross_section.claimed_count,
                    'n_dirs': self.cross_section.n_dirs}

        override = self.tip_center_xy if use_tip_center else None
        stats = self.cross_section.thicken_step(self.slab_xy, nudge_budget=nudge_budget,
                                                 center_override=override,
                                                 **expansion_kwargs)

        if stats['new_claimed'] <= 0:
            self._stall_count += 1
            if self._stall_count >= 3:
                self.converged = True
        else:
            self._stall_count = 0

        return stats

    def get_global_claimed(self):
        """Map local claimed mask back to global point indices."""
        return self.slab_indices[self.cross_section.claimed]

    def to_dict(self):
        """Serialize for JSON transport."""
        cs = self.cross_section.to_dict()
        cs['z'] = self.z
        cs['age'] = self.age
        cs['branch_flags'] = self.branch_flags
        cs['slab_stats'] = self._slab_stats
        return cs