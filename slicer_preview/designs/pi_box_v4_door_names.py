"""Viewer-only review (user 2026-11-03): door A three times, flat (outside face up), one per shortlisted name font:
Old English, Monotype Corsiva, Algerian. Not a print plate.
View: PREVIEW_SCALE=8 python generate.py pi_box_v4_door_names  -> /v4doornames
"""
from geometry import implicit_piece
from designs import pi_box_v4_door_a as D

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, D.EXPOSURE_S, D.BOTTOM_EXPOSURE_S
F = 'C:/Windows/Fonts/'
FONTS = [('OLDENGL.TTF', 'Alnilam'), ('MTCORSVA.TTF', 'Alnilam'), ('ALGER.TTF', 'ALNILAM')]
GAP = 6.0
_L = D.Y_MAX - D.Y_LO
PIECES = []
for i, (font, text) in enumerate(FONTS):
    fn, _, _ = D.make_name_mask(F + font, text)
    PIECES.append(implicit_piece(2 * D.PLATE_X, D.GRIP['d1'] - D.D_TOP, _L,
                                 lambda x, y, z, fn=fn: D.door(x, y, z, name_fn=fn),
                                 x0=-D.PLATE_X, z0=(i - 1.5) * (_L + GAP) + GAP / 2))
