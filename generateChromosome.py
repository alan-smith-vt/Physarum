"""
Generate an X-shaped chromosome point cloud and save as .pco file.

Anatomy: Two sister chromatids joined at the centromere. Each chromatid
has a short arm (p) above and a long arm (q) below. The two chromatids
splay apart from each other, giving the classic X silhouette.
"""

import numpy as np
import sys
#sys.path.insert(0, '/home/claude')

from PCO import PCOWriter
from PCO import PCOReader


def make_arm(n_points, length, radius, taper=0.4):
    """
    Generate points along a tapered cylindrical arm (along +Z from origin).
    Returns points in local space where the arm runs from z=0 to z=length.
    """
    t = np.random.uniform(0, 1, n_points)  # 0=base (centromere end), 1=tip
    
    # Taper: full radius at base, narrows toward tip
    envelope = 1.0 - (t ** 1.5) * (1.0 - 0.3)  # tapers to 30% at tip
    
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    
    # 70% surface, 30% interior
    surface_mask = np.random.random(n_points) < 0.7
    r = np.where(
        surface_mask,
        radius * envelope * (1.0 + np.random.normal(0, 0.05, n_points)),
        radius * envelope * np.random.uniform(0, 0.8, n_points)
    )
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = t * length
    
    # Helical banding
    for h in range(2):
        phase = h * np.pi / 2
        helix_angle = 5.0 * np.pi * t + phase
        alignment = np.clip(np.cos(theta - helix_angle), 0, 1) ** 2
        radial_x = np.cos(theta) * alignment * 0.15
        radial_y = np.sin(theta) * alignment * 0.15
        x += radial_x
        y += radial_y
    
    # Surface noise
    noise_scale = 0.05
    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)
    z += np.random.normal(0, noise_scale, n_points)
    
    return np.column_stack([x, y, z])


def make_centromere(n_points, radius):
    """Oblate spheroid at origin."""
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_t = np.random.uniform(-1, 1, n_points)
    sin_t = np.sqrt(1 - cos_t ** 2)
    r = radius * np.cbrt(np.random.uniform(0, 1, n_points))
    
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * cos_t * 0.35  # squish vertically
    
    return np.column_stack([x, y, z])


def transform_arm(pts, splay_x, direction_z, arm_radius, direction_x):
    """
    Place an arm: shift it in X by splay_x, and point it in direction_z 
    (+1 = up for p-arm, -1 = down for q-arm). The arm base sits at z=0.
    """
    out = pts.copy()
    out[:, 2] *= direction_z  # flip if q-arm (goes down)
    
    # Slight outward lean: arms splay more at the tips
    tip_fraction = np.abs(out[:, 2]) / np.abs(out[:, 2]).max()
    #out[:, 0] += tip_fraction * splay_x * 0.6  # lean outward
    curve = tip_fraction - 0.3 * tip_fraction ** 2
    out[:, 0] += arm_radius * direction_x * 0.5 + splay_x * curve
    
    return out


def color_by_label(points_list, labels):
    """Assign colors with per-point variation."""
    color_map = {
        'chromatid_a_p': (200, 80, 140),   # magenta - short arm
        'chromatid_a_q': (170, 60, 120),   # darker magenta - long arm
        'chromatid_b_p': (80, 120, 200),   # blue - short arm
        'chromatid_b_q': (60, 100, 170),   # darker blue - long arm
        'centromere':    (220, 200, 120),   # gold
    }
    
    colors = []
    for pts, label in zip(points_list, labels):
        base = np.array(color_map[label], dtype=np.float64)
        noise = np.random.randint(-15, 15, size=(len(pts), 3))
        c = np.clip(base + noise, 0, 255).astype(np.uint8)
        colors.append(c)
    
    return np.vstack(colors)


def generate_chromosome(pts_per_arm=25000, centromere_pts=10000,
                         p_arm_length=4.0, q_arm_length=8.0,
                         arm_radius=0.4, splay=1.0):
    """
    Generate X-shaped chromosome.
    
    Two chromatids (A and B), each with:
      - p-arm (short, pointing up)
      - q-arm (long, pointing down)
    Joined at centromere, splayed apart in X.
    """
    # Chromatid A (left side, splay = -splay)
    a_p = transform_arm(make_arm(pts_per_arm, p_arm_length, arm_radius), -splay, +1, arm_radius, -1)
    a_q = transform_arm(make_arm(pts_per_arm, q_arm_length, arm_radius), -splay, -1, arm_radius, -1)
    
    # Chromatid B (right side, splay = +splay)
    b_p = transform_arm(make_arm(pts_per_arm, p_arm_length, arm_radius), +splay, +1, arm_radius, +1)
    b_q = transform_arm(make_arm(pts_per_arm, q_arm_length, arm_radius), +splay, -1, arm_radius, +1)
    
    # Centromere
    centro = make_centromere(centromere_pts, radius=arm_radius * 1.2)
    
    all_parts = [a_p, a_q, b_p, b_q, centro]
    all_labels = ['chromatid_a_p', 'chromatid_a_q', 'chromatid_b_p', 'chromatid_b_q', 'centromere']
    
    points = np.vstack(all_parts).astype(np.float32)
    colors = color_by_label(all_parts, all_labels)
    
    return points, colors


def save_as_ply(filename, points, colors=None):
    """Write points (Nx3 float32) and optional colors (Nx3 uint8) as binary PLY."""
    n = len(points)
    has_color = colors is not None
    
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    
    dtype = [('x','f4'),('y','f4'),('z','f4')]
    if has_color:
        dtype += [('r','u1'),('g','u1'),('b','u1')]
    
    data = np.zeros(n, dtype=np.dtype(dtype))
    data['x'], data['y'], data['z'] = points[:,0], points[:,1], points[:,2]
    if has_color:
        data['r'], data['g'], data['b'] = colors[:,0], colors[:,1], colors[:,2]
    
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    
    print(f"Wrote {n:,} points to {filename}")

def main():
    # Grid of chromosomes to bloat the point count
    N_ROWS = 1
    N_COLS = 1
    SPACING = 8.0  # distance between chromosome centers
    TARGET_NODE_SIZE = 0.3  # target node size for resolving arm structure

    print(f"Generating {N_ROWS}x{N_COLS} grid of chromosomes...")

    all_points = []
    all_colors = []

    for row in range(N_ROWS):
        for col in range(N_COLS):
            offset_x = (col - (N_COLS - 1) / 2) * SPACING
            offset_y = (row - (N_ROWS - 1) / 2) * SPACING

            pts, cols = generate_chromosome(
                pts_per_arm=25000,
                centromere_pts=10000,
                p_arm_length=4.0,
                q_arm_length=8.0,
                arm_radius=0.4,
                splay=3,
            )

            pts[:, 0] += offset_x
            pts[:, 1] += offset_y

            all_points.append(pts)
            all_colors.append(cols)
            print(f"  Chromosome ({row},{col}): offset=[{offset_x:.1f}, {offset_y:.1f}], {len(pts):,} pts")

    points = np.vstack(all_points).astype(np.float32)
    colors = np.vstack(all_colors)
    
    print(f"  Total points: {len(points):,}")
    print(f"  Bounds: {points.min(axis=0)} to {points.max(axis=0)}")
    
    writer = PCOWriter()
    root = writer.calculate_root_bounds(points, padding=1.5)
    
    # Auto-compute depth so node_size ≈ TARGET_NODE_SIZE, cap at 8
    import math
    max_depth = max(6, math.ceil(math.log2(root['root_size'] / TARGET_NODE_SIZE)))
    node_size = root['root_size'] / (2 ** max_depth)
    print(f"  Root min: {root['root_min']}, size: {root['root_size']:.2f}")
    print(f"  Auto depth: {max_depth} (node size: {node_size:.4f})")
    
    output_path = 'chromosome.pco'
    writer.write(
        output_path,
        points,
        root,
        max_depth=max_depth,
        colors=colors,
        use_lod=True,
    )
    
    reader = PCOReader(output_path)
    reader.load_metadata()
    print(f"\nVerification:")
    print(reader.get_info())

    save_as_ply('chromosome.ply', points, colors)


if __name__ == '__main__':
    main()