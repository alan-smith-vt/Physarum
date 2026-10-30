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
FLOOR_BOSSES_V4 = [(sx * 31.0, -51.0) for sx in (-1, 1)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]   # = core
RIB_Y, DOOR_GAP = 30.5, 0.4

_ROUND = '((np.abs({x}) <= _V4.ROUND_C[0]) | ({y} >= _V4.ROUND_C[1]) | ' \
         '(np.hypot(np.abs({x}) - _V4.ROUND_C[0], {y} - _V4.ROUND_C[1]) <= _V4.R_FRONT{m}))'
_PATCHES = [
    ('FLOOR_BOSSES = [(sx * 44.0, Y_MIN + 10.5) for sx in (-1, 1)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]',
     'FLOOR_BOSSES = _V4.FLOOR_BOSSES_V4'),
    ('LATCH_EARS = True', 'LATCH_EARS = False'),
    ('HINGE_HOLES = [(HINGE_SIDE * (X_OUT - 6.0), DOOR_YC + dy) for dy in (-5.0, 5.0)]', 'HINGE_HOLES = []'),
    ('SPLIT_Y, DOOR_GAP, DOOR_LIP = 30.5, 0.4, 2.0', 'SPLIT_Y, DOOR_GAP, DOOR_LIP = _V4.RIB_Y - _V4.DOOR_GAP, 0.0, 0.0'),
    ('        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)\n',
     '        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)\n'
     '        s &= ' + _ROUND.format(x='X', y='Yr', m=' - inset') + '\n'),
    ('    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)\n',
     '    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)\n'
     '    req |= ~' + _ROUND.format(x='XX', y='YY', m=' - b') + '\n'),
    # UE300 swap (user approved, home:2065; home/v4-layout): cradle + RJ45 opening mirrored to -X
    # (X -45.9..-29.9, Yr -27.6..-1.6); the floor rail on that side opens for it instead of the +X one
    ('ETH_CRADLE = True ', 'ETH_ADAPTER = dict(ETH_ADAPTER, x=-ETH_ADAPTER["x"])\nETH_CRADLE = True '),
    ('        rails &= ~((X > 0) & (Yr < 0))', '        rails &= ~((X < 0) & (Yr < 0))'),
    ("    wall &= (X < a['x']) | (Yr < a['y'])", "    wall &= (X > a['x']) | (Yr < a['y'])"),
    ('    wall &= (Yr < -WEB_T / 2 - 0.5) & (X < X_IN - 0.5) &', '    wall &= (Yr < -WEB_T / 2 - 0.5) & (X > -X_IN + 0.5) &'),
]


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
