"""Lidar cage resin mock v3: reprint of the cross top + 4 standoffs (base plate from v2 was good).

v2 fit check (2026-10-11):
  - base plate: M3 pattern, nubs, M4 holes all good
  - cross: one arm split between layers about halfway up, DURING the print (not
    removal) — peel force beat the layer bond. v2 printed the whole 90 × 90 base
    on the same layers (~8000 mm² per peel). v3 has far less area per layer, and
    exposure goes 2.4 → 2.6 s for a stronger layer bond.
  - standoffs: horizontal M4 pilots far too tight (screw stopped ~5 mm in).
    Horizontal holes close from the roof on top of the ~0.35 mm XY shrink, so
    v3 bores are stretched vertically (BORE_Z_EXTRA) and only the middle
    GRIP_LEN mm grips: M4 × 35 from each end reaches ~28.6 (base) and ~30.2
    (cross) mm in, so both tips land inside the grip zone without meeting.
  - one standoff printed only its base layer: the lone one lying along Z at the
    +X end of the v2 layout, plate coords X 85…103, Z −56.5…+8.5 mm (centred
    plate, X = long axis). The 3 in the row along +Z were fine. Suspect a plate
    flatness or film defect there — v3 keeps clear of that area (X ≤ 80.5).
    If something fails there again, test that spot deliberately.

v3 result (2026-10-11): cross perfect (no delamination). Standoff bores now too LOOSE:
  v2 round Ø3.75 drawn → far too tight; v3 ellipse 3.85 w × 4.25 h → too loose, and
  M4×35 only caught ~2 mm of the grip zone. Next time: +0.35 comp and only ~+0.15–0.2
  vertical, grip where the screw tips actually land (or full length). Assembled the
  mock with fan screws biting the post ends instead — fine for the proof of concept.
"""
import numpy as np
from geometry import implicit_piece
from designs.cage_v2 import (CROSS_SPAN, CROSS_T, cross_top, _centered, CHAMFER,
                             RESIN_HOLE_COMP, M4_CLEAR, STANDOFF_LEN, CORE_D, FLANGE_T,
                             FLANGE_END, FLANGE_MID, AX, SO_W)

SUPPORTS = False
EXPOSURE_S = 2.6          # +8% over v2 for layer bond (v2 cross arm delaminated mid-print)
BOTTOM_EXPOSURE_S = 20.0
# Resin: vat blend from v2, topped up from a 400 ml stock batch (100 Tenacious + 300 Fast gray, 25%) mixed 2026-10-11

GRIP_D = 3.5               # M4 self-tap, before compensation
GRIP_LEN = 20.0            # centred on the standoff
BORE_Z_EXTRA = 0.4         # extra vertical size for horizontal bores (roof closure)


def _bore(a, b, d):
    """Horizontal bore, elliptical: wider vertically to cancel roof closure."""
    rh = (d + RESIN_HOLE_COMP) / 2
    rv = rh + BORE_Z_EXTRA / 2
    return ((a - AX) / rh) ** 2 + ((b - AX) / rv) ** 2 <= 1


def standoff(u, b, a):
    t = u / STANDOFF_LEN
    reach = AX + FLANGE_MID + (FLANGE_END - FLANGE_MID) * np.abs(2 * t - 1) ** 3
    inset = max(CHAMFER - b, 0)
    core = (a - AX) ** 2 + (b - AX) ** 2 <= AX ** 2
    corner = (a <= AX) & (b <= AX)
    flat = (b <= FLANGE_T) & (a <= reach)
    upright = (a <= FLANGE_T) & (b <= reach)
    solid = core | corner | flat | upright
    solid &= (a >= inset) & (a <= reach - inset) & (u >= inset) & (u <= STANDOFF_LEN - inset)
    grip = np.abs(u - STANDOFF_LEN / 2) <= GRIP_LEN / 2
    solid &= ~np.where(grip, _bore(a, b, GRIP_D), _bore(a, b, M4_CLEAR))
    return solid


# ── Layout: cross on the left, 4 standoffs stacked along Z on the right ──
GAP = 4.0
TOTAL_X = CROSS_SPAN + GAP + STANDOFF_LEN
X0 = -TOTAL_X / 2
SO_X0 = X0 + CROSS_SPAN + GAP
SO_Z0 = -(4 * SO_W + 3 * GAP) / 2

PIECES = [implicit_piece(CROSS_SPAN, CROSS_T, CROSS_SPAN, _centered(cross_top, CROSS_SPAN),
                         x0=X0, z0=-CROSS_SPAN / 2)]
for i in range(4):
    PIECES.append(implicit_piece(STANDOFF_LEN, SO_W, SO_W, standoff,
                                 x0=SO_X0, z0=SO_Z0 + i * (SO_W + GAP)))
