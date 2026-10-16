"""BATTERY variant of pi_box_lids.py (copy): floor only on this plate (100 × 140, turned 90°),
no Pi notch (the Pi's ports are inside the taller box), a notch under the bank instead.
Panels: designs/pi_box_bat_panel_my.py, pi_box_bat_panel_py.py (one per plate).

Lidar rig Pi box, v1 floor + -Y (Pi) panel, printed flat on one plate.

Floor: outside face on the build plate; collar (round the well tube), web rails
and the floor top face up. Dowels (3×6) press into Ø3.3 holes and stand 3 mm
proud into the wall feet. 4 × M3×6 button screws up into the core's glued M3 inserts,
heads in shallow counterbores. Open notch under the Pi so its cables stay plugged in.
Panel: slides up into the camera-wall grooves; intake grille in front of the cooler.
The +Y panel is the same blank without the grille: designs/pi_box_panel.py.

Rig -> print for the floor: (X, Yr, d) -> (x = X, y = DEPTH + FLOOR_T - d, z = -Yr).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import *      # noqa: F401,F403

SUPPORTS = False
EXPOSURE_S = 2.3   # ELEGOO ABS-like clear blue + Siraya Tenacious clear 75/25, exposure_test 2026-10-15 (90% cell, vat ~25 C)
BOTTOM_EXPOSURE_S = 20.0

FLOOR_T = 3.0
COLLAR_H = 4.0
COLLAR_R0 = WELL_R + TUBE_T + 0.15 + RESIN_HOLE_COMP / 2      # bore round the tube (prints undersize)
COLLAR_R1 = COLLAR_R0 + 1.8
RAIL_H, RAIL_W = 4.0, 1.5
RAIL_X = (WELL_R + 3.0, FOOT_X0 - 0.5)
NOTCH_X, NOTCH_Y = 29.0, PCB_YR - 0.5                        # open to the -Y edge
SCREW_CLEAR = 4.5                      # M4 through the floor and up the core boss into an M4 insert set from the top
NUB_D, NUB_H, NUB_TIP = 3.0, 3.0, 0.6  # alignment nubs in place of dowels; external faces print ~0.35 over → ~Ø3.35
                                       # in the core's loose PIN_SLIP holes (file down if tight)
ETH_CRADLE = False                     # UE300 cradle + RJ45 opening: off (2026-10-15, the adapter's home is undecided)
FILL_T = 0.27                          # floor-plate gyroid sheet (walls use GYROID_T 0.35): sparser for area margin

PANEL_W = 2 * (X_IN + GROOVE_D) - 0.8      # external faces print ~0.35 oversize
PANEL_H = DEPTH - RIM_D - 0.2
GRILLE = [(-20.0 + 5.5 * i, -20.0 + 5.5 * i + 2.6) for i in range(8)]
GRILLE_D = (25.0, 85.0)

GAP = 3.0



def _floor_solid(x, y, z):
    Yr = x + Y_MIN                         # BATTERY: turned 90° on the plate (x = Yr, z = X)
    X = z - X_OUT
    aX = np.abs(X)
    r2 = X ** 2 + Yr ** 2
    if y < FLOOR_T:
        inset = max(CHAMFER - y, 0.0)
        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)
        s &= ~hole(0, 0, 2 * WELL_R, X, Yr)
        wx0, wx1, wy0, wy1 = PORT_WINDOW                         # BATTERY: the bank's ports, from below
        s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
        for bx, by in BOSSES:
            s &= ~hole(bx, by, SCREW_CLEAR, X, Yr)
        if ETH_CRADLE:                                         # BATTERY: UE300 RJ45 opening
            a = ETH_ADAPTER
            s &= ~((np.abs(X - a['x']) <= (ADAPTER_JACK[0] + RESIN_HOLE_COMP) / 2) &
                   (np.abs(Yr - a['y']) <= (ADAPTER_JACK[1] + RESIN_HOLE_COMP) / 2))
        return s
    h = y - FLOOR_T
    s = (r2 >= COLLAR_R0 ** 2) & (r2 <= COLLAR_R1 ** 2) & (h <= COLLAR_H)
    gap = WEB_T / 2 + 0.15 + RESIN_HOLE_COMP / 2
    rails = (aX >= RAIL_X[0]) & (aX <= RAIL_X[1]) & (np.abs(Yr) >= gap) & (np.abs(Yr) <= gap + RAIL_W)
    if ETH_CRADLE:
        rails &= ~((X > 0) & (Yr < 0))                 # the UE300 stands there, flush on the web
    s |= rails & (h <= RAIL_H)
    # Cradle for the UE300: walls on its two open sides only (the web and the +X wall bound the other
    # two); the core's seat over its top end holds it down on the floor.
    a = ETH_ADAPTER
    ix, iy = (a['wx'] + CRADLE_CLEAR + RESIN_HOLE_COMP) / 2, (a['wy'] + CRADLE_CLEAR + RESIN_HOLE_COMP) / 2
    dx, dy = np.abs(X - a['x']), np.abs(Yr - a['y'])
    wall = (dx <= ix + CRADLE_WALL) & (dy <= iy + CRADLE_WALL) & ~((dx <= ix) & (dy <= iy))
    wall &= (X < a['x']) | (Yr < a['y'])               # tube-side and far-Yr walls
    wall &= (Yr < -WEB_T / 2 - 0.5) & (X < X_IN - 0.5) & (r2 > (COLLAR_R1 + 0.3) ** 2)
    if ETH_CRADLE:
        s |= wall & (h <= CRADLE_H)
    for px, py in PINS:                                # alignment nubs, chamfered tip
        r = NUB_D / 2 - max(h - (NUB_H - NUB_TIP), 0.0)
        s |= ((X - px) ** 2 + (Yr - py) ** 2 <= r * r) & (h <= NUB_H)
    return s


def _floor_keep(step=0.1):
    """Solid-plate footprint the gyroid fill must not open (rig X × Yr, 0.1 mm), the core flange's recipe:
    perimeter band, wall feet, M4 boss rings, collar, port and RJ45 frames, 1.5 mm under everything
    standing on the plate, spokes + diamond; Gaussian-rounded."""
    from scipy.ndimage import gaussian_filter, binary_dilation
    xs = np.arange(-X_OUT, X_OUT + step, step)
    ys = np.arange(Y_MIN, Y_MAX + step, step)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    above = _floor_solid(YY - Y_MIN, FLOOR_T + 0.5, XX + X_OUT)
    req = binary_dilation(above, iterations=int(round(1.5 / step)))
    b = 5.0
    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)
    req |= (np.abs(XX) >= FOOT_X0) & (np.abs(YY) <= FOOT_Y)
    for bx, by in BOSSES:
        req |= np.hypot(XX - bx, YY - by) <= 5.6
    req |= np.hypot(XX, YY) <= COLLAR_R1 + 2.0
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    req |= (XX >= wx0 - 2) & (XX <= wx1 + 2) & (YY >= wy0 - 2) & (YY <= wy1 + 2)
    a = ETH_ADAPTER
    req |= ETH_CRADLE & (np.abs(XX - a['x']) <= ADAPTER_JACK[0] / 2 + 2.5) & (np.abs(YY - a['y']) <= ADAPTER_JACK[1] / 2 + 2.5)

    def seg(ax, ay, bx, by, w=3.0):
        vx, vy = bx - ax, by - ay
        t = np.clip(((XX - ax) * vx + (YY - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        return np.hypot(XX - ax - t * vx, YY - ay - t * vy) <= w / 2
    ye = Y_MAX - b
    req |= seg(0, -ye, 0, -WELL_R) | seg(0, ye, 0, WELL_R)
    mids = [(0, -ye), (X_OUT - b, 0), (0, ye), (-(X_OUT - b), 0)]
    for i in range(4):
        req |= seg(*mids[i], *mids[(i + 1) % 4])
    return step, gaussian_filter(req.astype(float), 2.0 / step) > 0.3


_KEEP_STEP, _KEEP = _floor_keep()


def floor(x, y, z):
    """Solid floor with its plate opened to a sheet gyroid (the walls' period, rig coordinates) outside _KEEP."""
    s = _floor_solid(x, y, z)
    if y >= FLOOR_T:
        return s
    keep = np.asarray(_KEEP)                           # np may be cupy under to_goo --gpu
    i = np.clip((z / _KEEP_STEP).astype(int), 0, keep.shape[0] - 1)
    j = np.clip((x / _KEEP_STEP).astype(int), 0, keep.shape[1] - 1)
    X, Yr, d = z - X_OUT, x + Y_MIN, DEPTH + FLOOR_T - y
    k = 2 * np.pi / GYROID_PERIOD
    g = np.sin(k * X) * np.cos(k * Yr) + np.sin(k * Yr) * np.cos(k * d) + np.sin(k * d) * np.cos(k * X)
    return s & (keep[i, j] | (np.abs(g) <= FILL_T))


PANEL_BORDER = 6.0                         # solid frame round a gyroid panel (2 of it sits in the wall grooves)


def panel(x, y, z, grille=True, cable_notch=False, button=False, gyroid=None):
    """gyroid=(face_Yr, sign): open sheet gyroid (the walls' period and thickness, in rig coordinates so
    it lines up with the walls) inside a PANEL_BORDER solid frame; y = face_Yr + sign * y is the rig Yr."""
    X = x - PANEL_W / 2
    d = z + RIM_D + 0.1                    # print z runs down the panel
    s = np.ones(np.broadcast_shapes(X.shape, d.shape), bool)
    if gyroid is not None:
        k = 2 * np.pi / GYROID_PERIOD
        Y = gyroid[0] + gyroid[1] * y
        g = np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y) * np.cos(k * d) + np.sin(k * d) * np.cos(k * X)
        inner = (np.abs(X) <= PANEL_W / 2 - PANEL_BORDER) & (z >= PANEL_BORDER) & (z <= PANEL_H - PANEL_BORDER)
        s &= ~inner | (np.abs(g) <= GYROID_T)
    inset = max(CHAMFER * 0.5 - y, 0.0)
    s &= (np.abs(X) <= PANEL_W / 2 - inset)
    if grille:
        for x0, x1 in GRILLE:
            s &= ~((X >= x0) & (X <= x1) & (d >= GRILLE_D[0]) & (d <= GRILLE_D[1]))
    if button:                             # BATTERY: hole to press the bank's display button
        r = (BUTTON['hole'] + RESIN_HOLE_COMP) / 2
        s &= ~((X - BUTTON['x']) ** 2 + (d - BUTTON['d']) ** 2 <= r * r)
    if cable_notch:
        # U-notch at the top of one end for the lidar cable; flip the panel to put it on +X
        r = (PANEL_NOTCH_W + RESIN_HOLE_COMP) / 2
        u = (np.abs(X - PANEL_NOTCH_X) <= r) & (d <= PANEL_NOTCH_D)
        u |= (X - PANEL_NOTCH_X) ** 2 + (d - PANEL_NOTCH_D) ** 2 <= r * r
        s &= ~u
    return s


PIECES = [
    implicit_piece(Y_MAX - Y_MIN, FLOOR_T + max(COLLAR_H, RAIL_H, NUB_H, CRADLE_H if ETH_CRADLE else 0), 2 * X_OUT, floor, x0=-(Y_MAX - Y_MIN) / 2, z0=-X_OUT),
]
