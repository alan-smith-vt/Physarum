"""Exposure test for the 75 % ELEGOO Fast smoky black + 25 % Siraya Tenacious clear blend (user 2026-10-31): the
standard exposure_test (6 cells, gray 100..50 %) with the 100 % cell at 3.5 s, so the ladder runs 3.5 / 3.15 / 2.8 /
2.45 / 2.1 / 1.75 s. The pigmented blend needs more than the clear-blue blend's 2.3 s (its 90 % of 2.5 s won), and
the ladder still brackets 2.3. Same vat temperature as the real prints (the 10-15 test ran unheated, ~22-25 °C).
Export on the CPU (to_goo --gpu fails on this design's constant base piece)."""
from designs import exposure_test as _E

SUPPORTS = False
EXPOSURE_S = 3.5
BOTTOM_EXPOSURE_S = getattr(_E, 'BOTTOM_EXPOSURE_S', None)
if BOTTOM_EXPOSURE_S is None:
    del BOTTOM_EXPOSURE_S
PIECES = _E.PIECES
