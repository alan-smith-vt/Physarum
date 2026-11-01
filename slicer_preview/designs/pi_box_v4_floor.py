"""v4 fixed floor, APPROVED by the user 2026-10-29 (with door A), for designs/pi_box_v4_core.py (flip variant is the main one).

The v3 fixed floor (pi_box_bat_lids FLOOR_SPLIT: stick hole, collar, web rails, nubs, UE300 cradle + RJ45 opening,
sheet-gyroid plate), patched for v4:
  - outline follows the core's R30 front rounds (centres (±20, -32));
  - screws at the core's FLOOR_BOSSES_V4 (front pair moved to (±31, -51));
  - ends at the battery rib (Yr <= RIB_Y - DOOR_GAP), no lap: the battery door is a separate, swappable module
    that hangs in the core's wall grooves (option A: designs/pi_box_v4_door_a.py);
  - no hinge holes, no latch ears.
Rig -> print: as pi_box_bat_lids (outside face down).
"""
import sys
from pathlib import Path
import numpy as np                                                     # noqa: F401  (used by the patched source)

R_FRONT, ROUND_C = 30.0, (20.0, -32.0)       # core v4 R_FRONT and round centres (|X|, Yr)
# wide front panel (user approved, home:2136): the front pair leaves the R30 rounds: (-42, -18) off the -X wall,
# (45, 6.5) on the +X web foot (the +X front quadrant is the UE300); the +X nub at (46, 5.5) goes. = core
FLOOR_BOSSES_V4 = [(-42.0, -18.0), (45.0, 6.5)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]
DROP_PINS = [(46.0, 5.5)]
RIB_Y, DOOR_GAP = 30.5, 0.4
SWITCH_XY, SWITCH_D, SWITCH_SEAT = (33.0, 14.0), 16.0, 24.0   # +X bank side (2026-11-01: board above -X)
# # hole() adds RESIN_HOLE_COMP: cut Ø16.35 [MEASURE nut]

_ROUND = '((np.abs({x}) <= _V4.ROUND_C[0]) | ({y} >= _V4.ROUND_C[1]) | ' \
         '(np.hypot(np.abs({x}) - _V4.ROUND_C[0], {y} - _V4.ROUND_C[1]) <= _V4.R_FRONT{m}))'
_PATCHES = [
    ('FLOOR_BOSSES = [(sx * 44.0, Y_MIN + 10.5) for sx in (-1, 1)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]',
     'FLOOR_BOSSES = _V4.FLOOR_BOSSES_V4\nPINS = [q for q in PINS if q not in _V4.DROP_PINS]'),
    ('LATCH_EARS = True', 'LATCH_EARS = False'),
    ('HINGE_HOLES = [(HINGE_SIDE * (X_OUT - 6.0), DOOR_YC + dy) for dy in (-5.0, 5.0)]', 'HINGE_HOLES = []'),
    ('SPLIT_Y, DOOR_GAP, DOOR_LIP = 30.5, 0.4, 2.0', 'SPLIT_Y, DOOR_GAP, DOOR_LIP = _V4.RIB_Y - _V4.DOOR_GAP, 0.0, 0.0'),
    ('        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)\n',
     '        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)\n'
     '        s &= ' + _ROUND.format(x='X', y='Yr', m=' - inset') + '\n'),
    ('    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)\n',
     '    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)\n'
     '    req |= ~' + _ROUND.format(x='XX', y='YY', m=' - b') + '\n'),
    # uni USB-A Gigabit adapter replaces the UE300 (user, home:2158; spec home/v4-layout home:2161) [MEASURE]: body
    # 16 (X) x 24 (Yr), centred (37.9, -15) (16 deep clears the flip Pi's +X post, home:2169), RJ45 down through the floor opening; cradle walls + opening follow it
    ('ETH_CRADLE = True ', 'ETH_ADAPTER = dict(x=37.9, y=-12.1, wx=15.4, wy=23.7, h=53.0)\nETH_CRADLE = True '),
    # nut sweep (2026-11-01): no raised feature (stick collar, web rails) within NUT_SWEEP_R of the button
    ('    rails = (aX >= RAIL_X[0]) & (aX <= RAIL_X[1]) & (np.abs(Yr) >= gap) & (np.abs(Yr) <= gap + RAIL_W)\n',
     '    _Yw = Yr - np.where(X > 0, _V4.WEB_DY, 0.0)\n'
     '    rails = (aX >= RAIL_X[0]) & (aX <= RAIL_X[1]) & (np.abs(_Yw) >= gap) & (np.abs(_Yw) <= gap + RAIL_W)\n'),
    ('    s |= rails & (h <= RAIL_H)\n',
     '    s |= rails & (h <= RAIL_H)\n'
     '    s &= ~((X - _V4.SWITCH_XY[0]) ** 2 + (Yr - _V4.SWITCH_XY[1]) ** 2 <= _V4.NUT_SWEEP_R ** 2)\n'),
    # cradle's tube-side wall lowered under the Pi's +X lower mounting boss gusset (2026-11-01)
    ('    if ETH_CRADLE:\n        s |= wall & (h <= CRADLE_H)\n',
     '    pass   # no GbE cradle: the adapter is glued to the core web (2026-11-01)\n'),
    # floor power button (user approved, home:2092; home/v4-power-button): Ø16 switch down through the plate at
    # BUTTON_XY, nut on the top face: plate solid (no gyroid) over a Ø24 seat
    ('                s &= ~hole(bx, by, SCREW_CLEAR, X, Yr)\n',
     '                s &= ~hole(bx, by, SCREW_CLEAR, X, Yr)\n'
     '        s &= ~hole(*_V4.SWITCH_XY, _V4.SWITCH_D, X, Yr)\n'),
    ('    for bx, by in FLOOR_BOSSES + HINGE_HOLES:\n',
     '    req |= np.hypot(XX - _V4.SWITCH_XY[0], YY - _V4.SWITCH_XY[1]) <= _V4.SWITCH_SEAT / 2\n'
     '    for bx, by in FLOOR_BOSSES + HINGE_HOLES:\n'),
    # stray gyroid slivers between the keep bands (user 2026-10-31: one by the -X rear boss): fill any open
    # patch smaller than _V4.MIN_PANE mm2 (2026-10-31: 150, the rear strips too)
    ('    return step, gaussian_filter(req.astype(float), 2.0 / step) > 0.3\n',
     '    keep = gaussian_filter(req.astype(float), 2.0 / step) > 0.3\n'
     '    from scipy.ndimage import label\n'
     '    lab, n = label(~keep)\n'
     '    area = np.bincount(lab.ravel()) * step * step\n'
     '    keep |= (lab > 0) & (area[lab] < _V4.MIN_PANE)\n'
     '    return step, keep\n'),
]
WEB_DY = 1.5                                # = pi_box_v4_core.WEB_DY: +X web half shifted toward the bank
NUT_SWEEP_R = 9.4 + 1.6                     # M16 nut 18.8 AC (measured): its corners sweep r 9.4, + 1.6 to turn it
#   (user 2026-11-01: on the printed floor the nut fit but couldn't rotate: collar ring at r 10.3, web rail ~10.7)
CRADLE_TUBE_H = 5.0                         # cradle tube-side wall: the Pi's +X lower mounting boss gusset reaches
#   d ~168 over it (assembly check 2026-11-01: 48.8 mm2 at d 168, the clash seen on the printed core)
MIN_PANE = 150.0                            # also the two thin strips by the rear boss: solid for looks (user)
# UE300 side swap (home:2065) REVERTED by the user (home:2102): cradle + RJ45 back on +X, as in v3. The mirror
# patches, if ever needed again: a90a30b.


def _load():
    src = Path(__file__).with_name('pi_box_bat_lids.py').read_text(encoding='utf-8')
    for a, b in _PATCHES:
        assert src.count(a) == 1, f'lids patch does not match exactly once: {a!r}'
        src = src.replace(a, b)
    src = src[:src.index('\nPIECES = [')]
    g = {'__name__': __name__ + '._lids', '__file__': __file__, '_V4': sys.modules[__name__]}
    exec(compile(src, 'pi_box_bat_lids.py (v4-patched)', 'exec'), g)
    return g


_g = _load()
SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
floor_fixed = _g['floor_fixed']
_LF = RIB_Y - DOOR_GAP - _g['Y_MIN']
_FH = _g['_FH']
_XO = _g['X_OUT']

from geometry import implicit_piece                                    # noqa: E402
PIECES = [implicit_piece(_LF, _FH, 2 * _XO, floor_fixed, x0=-_LF / 2, z0=-_XO)]
