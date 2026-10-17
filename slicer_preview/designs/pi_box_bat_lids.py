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
ETH_CRADLE = True                      # UE300 cradle + RJ45 opening: on (2026-10-17, v3 keeps the UE300 in the +X / web corner)
# v3 floor (2026-10-17, user's sketch): the floor splits at the battery rib. The fixed part (stick hole, collar,
# rails, nubs, UE300 cradle) is screwed up with 4 M3s: the two Pi-side bosses and two new ones on the -Y side of the
# battery rib. Only the strip under the bank is a door: hinged on the +X wall, one draw latch on the -X wall (ear up
# the wall's outer face). The fixed floor's top half laps DOOR_LIP over the door's edge: the door closes flush on it.
FLOOR_SPLIT = True
SPLIT_Y, DOOR_GAP, DOOR_LIP = 30.5, 0.4, 2.0   # fixed floor's bottom half ends at SPLIT_Y (the rib's +Y face)
DOOR_YC = (SPLIT_Y + Y_MAX) / 2
FLOOR_BOSSES = [(sx * 44.0, Y_MIN + 10.5) for sx in (-1, 1)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]
BOSS_PINS = False                      # (2026-10-17 interim idea: pins instead of screws; superseded)
BOSS_PIN_D, BOSS_PIN_H, BOSS_PIN_TIP = 3.0, 4.0, 0.6
LATCH_EARS = True
EARS = [(-1, DOOR_YC)]                 # (side, Yr centre): the door's latch ear on the -X wall
EAR_T, EAR_H, EAR_W, EAR_GAP = 3.0, 16.0, 22.0, 0.4      # ear up the wall's outer face (above the plate: EAR_GAP off it)
CATCH_HOLES = [(-4.0, 8.0), (4.0, 8.0)]                  # (Yr off the ear centre, height above the floor top) [MEASURE]
CATCH_PILOT = 1.8
HINGE_SIDE = 1
HINGE_HOLES = [(HINGE_SIDE * (X_OUT - 6.0), DOOR_YC + dy) for dy in (-5.0, 5.0)]   # M2 pilots up into the door [MEASURE]
EAR_OUT = EAR_T + EAR_GAP if LATCH_EARS else 0.0
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
        if not BOSS_PINS:
            for bx, by in FLOOR_BOSSES:
                s &= ~hole(bx, by, SCREW_CLEAR, X, Yr)
        for hx, hy in HINGE_HOLES:
            s &= ~hole(hx, hy, CATCH_PILOT, X, Yr)
        if ETH_CRADLE:                                         # BATTERY: UE300 RJ45 opening
            a = ETH_ADAPTER
            s &= ~((np.abs(X - a['x']) <= (ADAPTER_JACK[0] + RESIN_HOLE_COMP) / 2) &
                   (np.abs(Yr - a['y']) <= (ADAPTER_JACK[1] + RESIN_HOLE_COMP) / 2))
        return s | _ears(X, Yr, y - FLOOR_T)
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
    if BOSS_PINS:                                      # alignment pins in place of the M3 screws
        for bx, by in FLOOR_BOSSES:
            r = BOSS_PIN_D / 2 - max(h - (BOSS_PIN_H - BOSS_PIN_TIP), 0.0)
            s |= ((X - bx) ** 2 + (Yr - by) ** 2 <= r * r) & (h <= BOSS_PIN_H)
    return s | _ears(X, Yr, h)


def _ears(X, Yr, h):
    """Latch ears: up the walls' outer faces (EARS), joined to the plate's edge; M2 pilots for the catch."""
    if not LATCH_EARS:
        return False
    x0 = X_OUT if h < 0 else X_OUT + EAR_GAP
    out = False
    for side, yc in EARS:
        sX = side * X
        s = (sX >= x0 - 0.01) & (sX <= X_OUT + EAR_OUT) & (np.abs(Yr - yc) <= EAR_W / 2) & (h <= EAR_H)
        for cy, ch in CATCH_HOLES:
            s &= ~((Yr - yc - cy) ** 2 + (h - ch) ** 2 <= ((CATCH_PILOT + RESIN_HOLE_COMP) / 2) ** 2)
        out = out | s
    return out


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
    for bx, by in FLOOR_BOSSES + HINGE_HOLES:
        req |= np.hypot(XX - bx, YY - by) <= 5.6
    if FLOOR_SPLIT:                                    # solid bands either side of the split (lap + door edge)
        req |= np.abs(YY - SPLIT_Y - DOOR_LIP / 2) <= DOOR_LIP / 2 + 4.0
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
PANEL_BEVEL = 0.8                          # 45° bevel on both faces of the long (sliding) edges
PANEL_LEAD = 1.2                           # 45° lead-in on the top edge (both faces) and its two corners


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
    # sliding edges (2026-10-16: the first panels had to be sanded to go in): 45° bevel on both faces of the
    # long edges, plus a lead-in on both faces and the corners of the top edge, which enters the grooves first
    yy = min(y, PANEL_T - y)
    s &= np.abs(X) <= PANEL_W / 2 - max(PANEL_BEVEL - yy, 0.0)
    s &= z >= PANEL_LEAD - yy
    s &= np.abs(X) <= PANEL_W / 2 - np.maximum(PANEL_LEAD - z, 0.0)
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


def _part(x, y, z, door):
    """Fixed floor or door mask: the fixed part's top half laps DOOR_LIP over the door's edge."""
    Yr = x + Y_MIN
    if not FLOOR_SPLIT:
        return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)), bool) | (not door)
    edge = SPLIT_Y + (DOOR_LIP if FLOOR_T / 2 <= y < FLOOR_T else 0.0)
    return Yr >= edge + DOOR_GAP if door else Yr <= edge


def floor_fixed(x, y, z):
    return floor(x, y, z) & _part(x, y, z, False)


def floor_door(x, y, z):
    return floor(x, y, z) & _part(x, y, z, True)


_LF = SPLIT_Y + DOOR_LIP - Y_MIN                  # fixed piece length (Yr), door piece from SPLIT_Y
_LD = Y_MAX - SPLIT_Y
_FH = FLOOR_T + max(COLLAR_H, RAIL_H, NUB_H, EAR_H if LATCH_EARS else 0, CRADLE_H if ETH_CRADLE else 0)
_X0 = -(_LF + GAP + _LD) / 2

PIECES = [
    implicit_piece(_LF, _FH, 2 * (X_OUT + EAR_OUT), lambda x, y, z: floor_fixed(x, y, z - EAR_OUT),
                   x0=_X0, z0=-(X_OUT + EAR_OUT)),
    implicit_piece(_LD, _FH, 2 * (X_OUT + EAR_OUT), lambda x, y, z: floor_door(x + (SPLIT_Y - Y_MIN), y, z - EAR_OUT),
                   x0=_X0 + _LF + GAP, z0=-(X_OUT + EAR_OUT)),
]
