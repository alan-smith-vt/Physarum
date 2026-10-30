"""IR-cut filter clip for the OV9281 M12 lens (2026-10-29): holds a 6.5 × 1.0 mm blue IR-cut glass on the lens's
rear face. 12 variants on one plate; each is a usable clip.

Lens rear face (from filterClip STEP + calipers): 5.6 mm square core (rear element centred in it), 8 radial fins
1 mm thick (4 on-axis, 4 diagonal), barrel inner wall taken as Ø9.6 (conservative; the STEP's Ø11.0 disagrees
with the ~2 mm square-to-wall caliper reading). Pockets are wedges between the fins.

Clip, lens → sensor:  8 wedge keys down into the pockets, each gripping the SIDES of its two fins | ring with the
glass pocket (glass rests on the fin tops / core rim) | 0.3 mm lip toward the sensor with a Ø5.6 aperture (sensor
diagonal ~4.5). Protrudes ~1.3 mm past the rear face.
v3 (2026-10-29, user): keys 4.0 long with a crush rib on each face, body fit -0.06..-0.10; recess Ø6.40-6.50 with
a 0.25 lead-in chamfer; no crush bumps on the glass (debris under the glass would tilt it).
v2 (2026-10-29): v1 ignored resin shrink: its Ø6.55-6.70 pockets printed ~0.35 undersize (below the 6.5 glass) and
the pins grew ~0.35, reaching the glass edge; they pressed on the core faces and were loose. v2 draws every size
compensated (RESIN_HOLE_COMP holes, EXT_COMP externals; the pocket also sits near the plate, where holes close
~0.25 more) and grips the fin sides with wedge keys outside the glass.

Printed lip-down (Y up = toward the lens): lip, ring, then pins pointing up, no supports.
  columns: key body fit C_LIST per side to each fin face (negative = interference), + a crush rib RIB_PROUD further
  rows:    glass pocket Ø D_LIST
ID tabs on the first layers: holes on the -Z tab = row, on the +X tab = column. Snip the tabs off after choosing.
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP    # holes print ~0.35 undersize on Ø (cage v1 fit strip)
EXT_COMP = 0.35                                       # external features print ~0.35 oversize (cage_v2 NUB_D note)

SUPPORTS = False

C_LIST = [-0.06, -0.08, -0.10]       # cols: key body fit per side to the fin faces (v2: -0.06 left ~0.03 play)
D_LIST = [6.40, 6.45, 6.50]          # rows: glass pocket Ø AS PRINTED (v2: 6.55 left a little play round the glass)
RIB_R, RIB_PROUD, RIB_AT = 0.25, 0.08, 4.0   # crush rib on each key face: radius, proud of the key face, at radius

LIP_H, POCKET_H = 0.3, 1.0           # lip, then glass pocket depth (= glass thickness)
RING_R = 4.6 - EXT_COMP / 2           # ring OD 9.2 as printed
APERTURE_R = (5.6 + RESIN_HOLE_COMP) / 2   # lip opening Ø5.6 as printed
KEY_R0, KEY_R1, PIN_L, PIN_TIP = 3.65, 4.4, 4.0, 0.4   # key span as drawn (grows ~0.175: 3.48..4.58; pocket ≤ 3.43, wall 4.8)
FIN_HALF = 0.5
GUIDE = 0.25                         # 45° lead-in chamfer at the recess mouth (lens side), seats the glass square
TAB_H, TAB_L, TAB_W, ID_R = 0.5, 3.5, 2.4, 0.35

PITCH = 16.0
NX, NZ = len(C_LIST), len(D_LIST)
SIZE_X, SIZE_Z = NX * PITCH, NZ * PITCH
RING_TOP = LIP_H + POCKET_H
TOP = RING_TOP + PIN_L

def _keys(u, v, c, taper=0.0):
    """8 wedge keys: between each pair of adjacent fins (radial, every 45°), clear of the fin faces by c (+ taper
    for the lead-in), radially KEY_R0..KEY_R1."""
    rr = np.hypot(u, v)
    th = np.mod(np.arctan2(v, u), np.pi / 4)                   # angle inside the 45° pocket
    d0 = rr * np.sin(th)                                       # distance to the fin on the pocket's 0° side
    d1 = rr * np.sin(np.pi / 4 - th)                           # ... and to the one on its 45° side
    g = FIN_HALF + c + EXT_COMP / 2 + taper                   # drawn narrower: the key grows EXT_COMP/2 per side
    body = (rr >= KEY_R0) & (rr <= KEY_R1) & (d0 >= g) & (d1 >= g)
    gc = g - RIB_PROUD + RIB_R                                 # rib axis: RIB_PROUD past the key face toward the fin
    ac = np.sqrt(RIB_AT ** 2 - gc ** 2)
    rib = (((rr * np.cos(th) - ac) ** 2 + (d0 - gc) ** 2 <= RIB_R ** 2) |
           ((rr * np.cos(np.pi / 4 - th) - ac) ** 2 + (d1 - gc) ** 2 <= RIB_R ** 2))
    return body | rib


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
                inner = np.where(row == i, (d + RESIN_HOLE_COMP) / 2, inner)
            inner = inner + max(0.0, y - (RING_TOP - GUIDE))     # lead-in chamfer at the mouth
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
        taper = 0.0 if h < PIN_L - PIN_TIP else 0.25 * (h - (PIN_L - PIN_TIP)) / PIN_TIP
        for j, c in enumerate(C_LIST):
            s |= (col == j) & _keys(u, v, c, taper)

    return s & (col >= 0) & (col < NX) & (row >= 0) & (row < NZ)


PIECES = [implicit_piece(SIZE_X, TOP, SIZE_Z, solid, x0=-SIZE_X / 2, z0=-SIZE_Z / 2)]
