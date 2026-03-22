"""
Debug script: polar-curve thickening on a 2D tree cross-section.

Generates a slightly oval point cloud with bark ridge patterns,
saves it as PCO, then steps through the thickening algorithm
and plots each step.

Run from Physarum folder:
    python debug_thicken.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from PCO import PCOWriter, PCOReader


# ═══════════════════════════════════════════
# 1. GENERATE TREE CROSS-SECTION POINT CLOUD
# ═══════════════════════════════════════════

def generate_tree_cross_section(n_points=8000, center=(0, 0), seed=42):
    """
    Generate a 2D point cloud resembling a tree trunk cross-section.
    Slightly oval, with bark ridges on the outer edge and growth rings inside.
    
    Returns points as Nx3 (x, y, z=0) for PCO compatibility.
    """
    rng = np.random.default_rng(seed)
    
    # Base oval parameters
    a = 3.0   # semi-major (x)
    b = 2.5   # semi-minor (y)
    
    points = []
    colors = []
    
    # --- Interior: growth rings ---
    # Dense fill with slight ring density modulation
    n_interior = int(n_points * 0.6)
    angles = rng.uniform(0, 2 * np.pi, n_interior)
    # Radial distribution: sqrt for uniform area, then scale to oval
    radii = np.sqrt(rng.uniform(0, 1, n_interior))
    
    # Growth ring modulation: density peaks at certain radii
    n_rings = 8
    ring_positions = np.linspace(0.1, 0.85, n_rings)
    ring_boost = np.zeros(n_interior)
    for rp in ring_positions:
        ring_boost += np.exp(-((radii - rp) ** 2) / (2 * 0.015 ** 2))
    
    # Accept/reject based on ring boost (keep all, but add extra points at rings)
    x_int = radii * a * np.cos(angles)
    y_int = radii * b * np.sin(angles)
    
    # Color: darker toward center (heartwood), lighter rings
    gray = (radii * 120 + 60 + ring_boost * 40).clip(0, 255).astype(np.uint8)
    
    for i in range(n_interior):
        points.append([x_int[i], y_int[i], 0.0])
        colors.append([gray[i], int(gray[i] * 0.7), int(gray[i] * 0.4)])
    
    # --- Extra ring points for visible growth rings ---
    n_ring_extra = int(n_points * 0.1)
    for rp in ring_positions:
        n_per = n_ring_extra // n_rings
        ang = rng.uniform(0, 2 * np.pi, n_per)
        r = rng.normal(rp, 0.01, n_per).clip(0.05, 0.95)
        x_r = r * a * np.cos(ang)
        y_r = r * b * np.sin(ang)
        for i in range(n_per):
            points.append([x_r[i], y_r[i], 0.0])
            g = int(r[i] * 100 + 80)
            colors.append([g, int(g * 0.65), int(g * 0.35)])
    
    # --- Bark: outer edge with ridges ---
    n_bark = int(n_points * 0.3)
    
    # Bark ridges: radial bumps at certain angles
    n_ridges = 12
    ridge_angles = rng.uniform(0, 2 * np.pi, n_ridges)
    ridge_widths = rng.uniform(0.08, 0.2, n_ridges)
    ridge_heights = rng.uniform(0.05, 0.2, n_ridges)
    
    bark_angles = rng.uniform(0, 2 * np.pi, n_bark)
    bark_radii = rng.normal(0.95, 0.04, n_bark).clip(0.85, 1.15)
    
    # Add ridge bumps
    for i in range(n_ridges):
        angle_diff = np.abs(np.arctan2(
            np.sin(bark_angles - ridge_angles[i]),
            np.cos(bark_angles - ridge_angles[i])
        ))
        mask = angle_diff < ridge_widths[i]
        bump = ridge_heights[i] * np.exp(-(angle_diff ** 2) / (2 * (ridge_widths[i] / 2) ** 2))
        bark_radii += bump
    
    x_bark = bark_radii * a * np.cos(bark_angles)
    y_bark = bark_radii * b * np.sin(bark_angles)
    
    for i in range(n_bark):
        points.append([x_bark[i], y_bark[i], 0.0])
        g = int(rng.uniform(40, 80))
        colors.append([g, int(g * 0.6), int(g * 0.3)])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    return points, colors


# ═══════════════════════════════════════════
# 2. POLAR CURVE THICKENING ALGORITHM
# ═══════════════════════════════════════════

class PolarCrossSection:
    """
    Represents a filament cross-section as a polar curve centered on
    the CG of claimed points. The boundary is N radial samples that
    expand via greedy nudge allocation.
    
    Optimized: claimed is a bool array, scoring is fully vectorized,
    polar coords are precomputed once per tick.
    """
    
    def __init__(self, center, n_points, initial_radius=0.3, n_dirs=10, nudge_step=0.1):
        """
        Args:
            center: [x, y] center point
            n_points: total number of points in the cloud (for bool array sizing)
            initial_radius: starting radius in all directions
            n_dirs: number of angular samples
            nudge_step: how far one nudge pushes the boundary
        """
        self.center = np.array(center, dtype=np.float64)
        self.n_dirs = n_dirs
        self.nudge_step = nudge_step
        
        # Angular samples evenly spaced
        self.angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        # Radius at each angle
        self.radii = np.full(n_dirs, initial_radius, dtype=np.float64)
        
        # Per-direction density history (EMA, -1 = no history yet)
        self.dir_density = np.full(n_dirs, -1.0, dtype=np.float64)
        
        # Bool array for claimed points — no more set conversions
        self.claimed = np.zeros(n_points, dtype=bool)
        
        # Min arc length for perimeter-based N scaling
        self.min_arc_length = 0.25
    
    @property
    def claimed_count(self):
        return int(self.claimed.sum())
    
    def get_boundary_xy(self):
        """Get boundary vertices in world XY coordinates."""
        x = self.center[0] + self.radii * np.cos(self.angles)
        y = self.center[1] + self.radii * np.sin(self.angles)
        return np.column_stack([x, y])
    
    def get_perimeter(self):
        """Approximate perimeter of the boundary polygon."""
        verts = self.get_boundary_xy()
        diffs = np.diff(np.vstack([verts, verts[:1]]), axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    @staticmethod
    def _polygon_area(verts):
        """Shoelace formula for polygon area from Nx2 vertices."""
        x = verts[:, 0]
        y = verts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    def _rescale_directions(self):
        """
        Adjust N to maintain resolution as perimeter grows.
        Interpolates radii and density history to new angular sampling.
        """
        perimeter = self.get_perimeter()
        new_n = max(10, int(np.round(perimeter / self.min_arc_length)))
        
        if new_n == self.n_dirs:
            return
        
        new_angles = np.linspace(0, 2 * np.pi, new_n, endpoint=False)
        # Periodic interpolation
        old_angles_ext = np.concatenate([self.angles - 2*np.pi, self.angles, self.angles + 2*np.pi])
        old_radii_ext = np.tile(self.radii, 3)
        old_density_ext = np.tile(self.dir_density, 3)
        
        self.radii = np.interp(new_angles, old_angles_ext, old_radii_ext)
        self.dir_density = np.interp(new_angles, old_angles_ext, old_density_ext)
        self.angles = new_angles
        self.n_dirs = new_n
    
    def points_inside_mask(self, points_xy):
        """Vectorized inside check for Nx2 array of points."""
        dx = points_xy[:, 0] - self.center[0]
        dy = points_xy[:, 1] - self.center[1]
        angles = np.arctan2(dy, dx) % (2 * np.pi)
        dists = np.sqrt(dx**2 + dy**2)
        
        idx = np.searchsorted(self.angles, angles) % self.n_dirs
        idx_prev = (idx - 1) % self.n_dirs
        
        a0 = self.angles[idx_prev]
        a1 = self.angles[idx]
        da = (a1 - a0) % (2 * np.pi)
        safe_da = np.where(da > 1e-10, da, 1.0)
        t = np.clip(((angles - a0) % (2 * np.pi)) / safe_da, 0, 1)
        
        r_interp = self.radii[idx_prev] * (1 - t) + self.radii[idx] * t
        return dists <= r_interp
    
    def _compute_polar(self, points_xy):
        """Precompute polar coords relative to current center. Reused across scoring + contraction."""
        dx = points_xy[:, 0] - self.center[0]
        dy = points_xy[:, 1] - self.center[1]
        return np.arctan2(dy, dx) % (2 * np.pi), np.sqrt(dx**2 + dy**2)
    
    def _angular_diff_matrix(self, point_angles):
        """
        Compute angular difference between each point and each direction.
        Returns (n_points, n_dirs) matrix of absolute angular differences.
        """
        # point_angles: (n_points,), self.angles: (n_dirs,)
        diff = point_angles[:, None] - self.angles[None, :]  # (n_points, n_dirs)
        return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    
    def thicken_step(self, all_points_xy, nudge_budget=10):
        """
        One thickening tick, optimized:
        - Bool array for claimed (no set conversions)
        - Vectorized scoring via 2D angular diff matrix
        - Precomputed polar coords reused across phases
        """
        n_claimed = self.claimed_count
        
        # 1. Recompute CG from claimed points
        if n_claimed > 0:
            new_cg = all_points_xy[self.claimed].mean(axis=0)
            
            # 2. Re-express boundary from new center
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
        
        # 3. Get unclaimed points and precompute their polar coords
        unclaimed_mask = ~self.claimed
        unclaimed_idx = np.where(unclaimed_mask)[0]
        
        if len(unclaimed_idx) == 0:
            return {'nudges_spent': 0, 'new_claimed': 0, 'total_claimed': n_claimed,
                    'n_dirs': self.n_dirs, 'nudge_distribution': []}
        
        unc_pts = all_points_xy[unclaimed_idx]
        unc_angles, unc_dists = self._compute_polar(unc_pts)
        
        # Vectorized scoring: angular diff matrix (n_unclaimed, n_dirs)
        half_wedge = np.pi / self.n_dirs
        ang_diff = self._angular_diff_matrix(unc_angles)  # (n_unc, n_dirs)
        in_wedge = ang_diff < half_wedge  # (n_unc, n_dirs)
        
        # Radial band check: each direction has its own radius
        # unc_dists: (n_unc,), self.radii: (n_dirs,)
        in_radial = ((unc_dists[:, None] > self.radii[None, :]) &
                     (unc_dists[:, None] <= self.radii[None, :] + self.nudge_step * 3))
        
        scores = (in_wedge & in_radial).sum(axis=0).astype(np.float64)  # (n_dirs,)
        
        # 4. Greedy allocation
        nudges_allocated = np.zeros(self.n_dirs, dtype=int)
        scores_work = scores.copy()
        for _ in range(nudge_budget):
            best = np.argmax(scores_work)
            if scores_work[best] <= 0:
                break
            nudges_allocated[best] += 1
            scores_work[best] *= 0.6
        
        # 5. Expand + contract with per-direction density
        old_radii = self.radii.copy()
        self.radii += nudges_allocated * self.nudge_step
        
        min_density_ratio = 0.3
        density_ema_alpha = 0.3
        
        # Only process directions that got nudges
        nudged = np.where(nudges_allocated > 0)[0]
        if len(nudged) > 0:
            for i in nudged:
                wedge_mask = in_wedge[:, i]  # precomputed
                
                r_min = old_radii[i]
                r_max = self.radii[i]
                best_r = r_max
                best_density = 0.0
                
                ref_density = 0.0 if self.dir_density[i] < 0 else self.dir_density[i]
                
                for _ in range(5):
                    r_mid = (r_min + r_max) / 2
                    in_band = (unc_dists > old_radii[i]) & (unc_dists <= r_mid)
                    band_count = np.sum(wedge_mask & in_band)
                    band_area = half_wedge * (r_mid**2 - old_radii[i]**2)
                    band_density = band_count / max(band_area, 1e-6)
                    
                    if band_density >= ref_density * min_density_ratio:
                        best_r = r_mid
                        best_density = band_density
                        r_min = r_mid
                    else:
                        r_max = r_mid
                
                self.radii[i] = best_r
                
                if best_density > 0:
                    if self.dir_density[i] < 0:
                        self.dir_density[i] = best_density
                    else:
                        self.dir_density[i] = (density_ema_alpha * best_density +
                                               (1 - density_ema_alpha) * self.dir_density[i])
        
        # 6. Claim — update bool array in place
        old_claimed = n_claimed
        self.claimed = self.points_inside_mask(all_points_xy)
        new_claimed = self.claimed_count
        
        # 7. Rescale directions
        self._rescale_directions()
        
        return {
            'nudges_spent': int(nudges_allocated.sum()),
            'new_claimed': new_claimed - old_claimed,
            'total_claimed': new_claimed,
            'n_dirs': self.n_dirs,
            'nudge_distribution': nudges_allocated.tolist(),
        }
    
    def retract_pass(self, all_points_xy, retract_step=None, max_iters=50):
        """
        Global refinement: iteratively retract each direction inward if doing so
        improves the overall points-to-area ratio.
        """
        if retract_step is None:
            retract_step = self.nudge_step / 2
        
        total_retractions = 0
        
        for iteration in range(max_iters):
            improved = False
            
            current_area = self._polygon_area(self.get_boundary_xy())
            current_density = self.claimed_count / max(current_area, 1e-6)
            
            for i in range(self.n_dirs):
                old_r = self.radii[i]
                new_r = max(0.05, old_r - retract_step)
                if new_r >= old_r:
                    continue
                
                self.radii[i] = new_r
                
                new_inside = self.points_inside_mask(all_points_xy)
                new_count = new_inside.sum()
                new_area = self._polygon_area(self.get_boundary_xy())
                new_density = new_count / max(new_area, 1e-6)
                
                if new_density > current_density:
                    self.claimed = new_inside
                    current_area = new_area
                    current_density = new_density
                    improved = True
                    total_retractions += 1
                else:
                    self.radii[i] = old_r
            
            if not improved:
                break
        
        # Final sync
        self.claimed = self.points_inside_mask(all_points_xy)
        final_area = self._polygon_area(self.get_boundary_xy())
        
        return {
            'iterations': iteration + 1,
            'total_retractions': total_retractions,
            'final_claimed': self.claimed_count,
            'final_area': final_area,
            'final_density': self.claimed_count / max(final_area, 1e-6),
        }


# ═══════════════════════════════════════════
# 3. GENERATE, SAVE, AND RUN
# ═══════════════════════════════════════════

def main():
    # --- Generate tree cross-section ---
    print("Generating tree cross-section point cloud...")
    points_3d, colors = generate_tree_cross_section(n_points=8000)
    points_xy = points_3d[:, :2]
    
    print(f"  Points: {len(points_3d)}")
    print(f"  XY bounds: ({points_xy.min(0)}) to ({points_xy.max(0)})")
    
    # --- Save as PCO ---
    writer = PCOWriter()
    root = writer.calculate_root_bounds(points_3d, padding=1.5)
    pco_path = 'tree_cross_section.pco'
    writer.write(pco_path, points_3d, root, max_depth=5, colors=colors, use_lod=False)
    
    reader = PCOReader(pco_path)
    reader.load_metadata()
    print(f"  PCO: {reader.header['total_points']} points, {reader.header['num_nodes']} nodes")
    
    # --- Initialize thickening from center ---
    seed_center = [0.0, 0.0]  # center of the tree
    cross = PolarCrossSection(
        center=seed_center,
        n_points=len(points_xy),
        initial_radius=0.3,
        n_dirs=10,
        nudge_step=0.15,
    )
    
    # Initial claim
    cross.claimed = cross.points_inside_mask(points_xy)
    print(f"  Initial claim: {cross.claimed_count} points")
    
    # --- Step through thickening ---
    import time
    n_steps = 150
    nudge_budget = 10
    
    # Set up figure — 2 rows: top = growth snapshots, bottom = before/after retraction
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Polar Curve Thickening — Tree Cross-Section', fontsize=14, fontweight='bold')
    
    steps_to_plot = [0, 10, 30, 60, 100, 149]
    plot_idx = 0
    
    t0 = time.perf_counter()
    for step in range(n_steps):
        stats = cross.thicken_step(points_xy, nudge_budget=nudge_budget)
        
        if step in steps_to_plot and plot_idx < 4:
            ax = axes[0][plot_idx]
            _plot_state(ax, cross, points_xy, colors, step, stats)
            plot_idx += 1
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: claimed={stats['total_claimed']:5d}  "
                  f"new=+{stats['new_claimed']:4d}  "
                  f"dirs={stats['n_dirs']:3d}  "
                  f"nudges={stats['nudges_spent']}")
    
    t_grow = time.perf_counter() - t0
    print(f"\n  Growth: {t_grow:.3f}s ({t_grow/n_steps*1000:.1f}ms/step)")
    
    # Plot last two growth steps in bottom row
    ax = axes[1][0]
    _plot_state(ax, cross, points_xy, colors, n_steps-1, stats, title_prefix='Pre-retract')
    
    # Compute pre-retract density
    pre_area = PolarCrossSection._polygon_area(cross.get_boundary_xy())
    pre_claimed = cross.claimed_count
    pre_density = pre_claimed / max(pre_area, 1e-6)
    print(f"\n  Pre-retract: {pre_claimed} pts, area={pre_area:.2f}, density={pre_density:.2f}")
    
    # --- Retraction pass ---
    print("\n  Running retraction pass...")
    retract_stats = cross.retract_pass(points_xy)
    print(f"  Retraction: {retract_stats['iterations']} iters, "
          f"{retract_stats['total_retractions']} retractions, "
          f"{retract_stats['final_claimed']} pts, "
          f"area={retract_stats['final_area']:.2f}, "
          f"density={retract_stats['final_density']:.2f}")
    
    ax = axes[1][1]
    _plot_state(ax, cross, points_xy, colors, n_steps-1,
                {'total_claimed': retract_stats['final_claimed'], 'n_dirs': cross.n_dirs, 'new_claimed': 0},
                title_prefix='Post-retract')
    
    # Density comparison text
    ax_text = axes[1][2]
    ax_text.axis('off')
    ax_text.text(0.1, 0.7, f"Before retraction:", fontsize=12, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.1, 0.6, f"  Points: {pre_claimed}", transform=ax_text.transAxes)
    ax_text.text(0.1, 0.55, f"  Area: {pre_area:.2f}", transform=ax_text.transAxes)
    ax_text.text(0.1, 0.5, f"  Density: {pre_density:.2f} pts/unit²", transform=ax_text.transAxes)
    ax_text.text(0.1, 0.35, f"After retraction:", fontsize=12, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.1, 0.25, f"  Points: {retract_stats['final_claimed']}", transform=ax_text.transAxes)
    ax_text.text(0.1, 0.2, f"  Area: {retract_stats['final_area']:.2f}", transform=ax_text.transAxes)
    ax_text.text(0.1, 0.15, f"  Density: {retract_stats['final_density']:.2f} pts/unit²", transform=ax_text.transAxes)
    improvement = (retract_stats['final_density'] - pre_density) / pre_density * 100
    ax_text.text(0.1, 0.03, f"  Density improvement: {improvement:+.1f}%", 
                fontsize=12, fontweight='bold', color='green' if improvement > 0 else 'red',
                transform=ax_text.transAxes)
    
    axes[1][3].axis('off')
    
    plt.tight_layout()
    out_path = 'thicken_debug.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.close()


def _plot_state(ax, cross, points_xy, colors, step, stats, title_prefix=None):
    """Helper to plot a single thickening state."""
    unclaimed_mask = ~cross.claimed
    
    ax.scatter(points_xy[unclaimed_mask, 0], points_xy[unclaimed_mask, 1],
              s=0.5, c='#444444', alpha=0.3, rasterized=True)
    
    if cross.claimed_count > 0:
        claimed_colors = colors[cross.claimed].astype(float) / 255.0
        ax.scatter(points_xy[cross.claimed, 0], points_xy[cross.claimed, 1],
                  s=1.5, c=claimed_colors, alpha=0.8, rasterized=True)
    
    boundary = cross.get_boundary_xy()
    boundary_closed = np.vstack([boundary, boundary[:1]])
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'r-', linewidth=1.5, alpha=0.9)
    ax.plot(cross.center[0], cross.center[1], 'r+', markersize=8, markeredgewidth=2)
    
    for i in range(cross.n_dirs):
        dx = 0.15 * np.cos(cross.angles[i])
        dy = 0.15 * np.sin(cross.angles[i])
        bx, by = boundary[i]
        ax.plot([bx, bx + dx], [by, by + dy], 'r-', linewidth=0.5, alpha=0.4)
    
    ax.set_aspect('equal')
    prefix = title_prefix or f'Step {step}'
    ax.set_title(f'{prefix}: {stats["total_claimed"]} pts, '
                f'N={stats["n_dirs"]} dirs',
                fontsize=10)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.0, 4.0)
    ax.grid(True, alpha=0.15)


if __name__ == '__main__':
    main()