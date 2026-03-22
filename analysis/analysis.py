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

    # ── Reset ──

    def reset(self):
        self.results = {}