"""IR-cut filter clip for the OV9281 M12 lens (2026-10-29): holds a 6.5 × 1.0 mm blue IR-cut glass on the lens's
rear face. 12 variants on one plate; each is a usable clip.

Lens rear face (from filterClip STEP + calipers): 5.6 mm square core (rear element centred in it), 8 radial fins
1 mm thick (4 on-axis, 4 diagonal), barrel inner wall taken as Ø9.6 (conservative; the STEP's Ø11.0 disagrees
with the ~2 mm square-to-wall caliper reading). Pockets are wedges between the fins.

Clip, lens → sensor:  8 pins down into the pockets, gripping the core's flat faces (2 per face, either side of
the on-axis fin) | ring with the glass pocket (glass rests on the fin tops / core rim) | 0.3 mm lip toward the
sensor with a Ø5.6 aperture (sensor diagonal ~4.5). Protrudes ~1.3 mm past the rear face.

Printed lip-down (Y up = toward the lens): lip, ring, then pins pointing up, no supports.
  columns: pin grip offset T_LIST = distance from the axis to the pins' inner tangent (core half-width 2.8)
  rows:    glass pocket Ø D_LIST
ID tabs on the first layers: holes on the -Z tab = row, on the +X tab = column. Snip the tabs off after choosing.
"""
import numpy as np
from geometry import implicit_piece

SUPPORTS = False

T_LIST = [2.85, 2.80, 2.77]          # cols: 0.05 clearance / line / 0.03 interference per side
D_LIST = [6.55, 6.60, 6.65, 6.70]    # rows: glass pocket Ø (glass 6.5; resin holes print a little undersize)

LIP_H, POCKET_H = 0.3, 1.0           # lip, then glass pocket depth (= glass thickness)
RING_R = 4.6                         # ring OD 9.2: stays inside the Ø9.6 barrel wall
APERTURE_R = 2.8                     # lip opening Ø5.6
PIN_R, PIN_L, PIN_TIP = 0.55, 1.5, 0.3   # pin radius, length into the pocket, tip taper (lead-in)
TAB_H, TAB_L, TAB_W, ID_R = 0.5, 3.5, 2.4, 0.35

PITCH = 16.0
NX, NZ = len(T_LIST), len(D_LIST)
SIZE_X, SIZE_Z = NX * PITCH, NZ * PITCH
RING_TOP = LIP_H + POCKET_H
TOP = RING_TOP + PIN_L

_BIS = np.deg2rad(22.5 + 45.0 * np.arange(8))   # pocket bisectors


def _pin_centres(t):
    """Pin centres for grip offset t: each pin touches the core face nearest its bisector."""
    xc = t + PIN_R                                   # along that face's normal
    d = xc / np.cos(np.deg2rad(22.5))
    return [(d * np.cos(a), d * np.sin(a)) for a in _BIS]


def solid(x, y, z):
    col = np.floor(x / PITCH)
    row = np.floor(z / PITCH)
    u = x - (col + 0.5) * PITCH
    v = z - (row + 0.5) * PITCH
    r2 = u ** 2 + v ** 2
    s = np.zeros(np.broadcast(x, z).shape, bool)

    if y < RING_TOP:
        inner = np.full(s.shape, APERTURE_R)
        if y >= LIP_H:
            for i, d in enumerate(D_LIST):
                inner = np.where(row == i, d / 2, inner)
        s = (r2 <= RING_R ** 2) & (r2 >= inner ** 2)
        if y < TAB_H:
            tab_z = (np.abs(u) <= TAB_W / 2) & (v < 0) & (v >= -RING_R - TAB_L)
            tab_x = (np.abs(v) <= TAB_W / 2) & (u > 0) & (u <= RING_R + TAB_L)
            s |= (r2 > (RING_R - 0.1) ** 2) & (tab_z | tab_x)   # outside the ring: clear of the glass pocket
            for n in range(1, NZ + 1):               # row count on the -Z tab
                for k in range(n):
                    cz = -RING_R - 0.6 - k * 0.8
                    s &= ~((row + 1 == n) & (u ** 2 + (v - cz) ** 2 <= ID_R ** 2))
            for n in range(1, NX + 1):               # column count on the +X tab
                for k in range(n):
                    cx = RING_R + 0.6 + k * 0.8
                    s &= ~((col + 1 == n) & ((u - cx) ** 2 + v ** 2 <= ID_R ** 2))
    elif y < TOP:
        h = y - RING_TOP
        pr = PIN_R if h < PIN_L - PIN_TIP else PIN_R * (0.5 + 0.5 * (PIN_L - h) / PIN_TIP)
        for j, t in enumerate(T_LIST):
            for (pu, pv) in _pin_centres(t):
                s |= (col == j) & ((u - pu) ** 2 + (v - pv) ** 2 <= pr ** 2)

    return s & (col >= 0) & (col < NX) & (row >= 0) & (row < NZ)


PIECES = [implicit_piece(SIZE_X, TOP, SIZE_Z, solid, x0=-SIZE_X / 2, z0=-SIZE_Z / 2)]
