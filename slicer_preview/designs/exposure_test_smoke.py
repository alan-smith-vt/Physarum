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

# RESULT (user, 2026-10-31): winner = 80 % cell (2.8 s), top right. Pins on the 4 longest (3.5 .. 2.45 s), barely at
# 70 %; 50 % failed outright. Set 2.8 s in pi_box_v4_core / _front_panel / _back_panel.
# RESULT 2 (user, 2026-11-02, NEW FEP film, first export after the mirror fix): winner again #3 = 80 % (2.8 s).
#   #1-#3 (3.5/3.15/2.8 s): all pins; #3: 4 holes + 4 slits open, bridge clean. #4 (2.45 s): one pin lost, last slit
#   maybe open, bridge sags a hint. #5 (2.1 s): 3 pins lost, all slits/holes open, bridge struggling. #3 chosen over
#   #4 for margin on fine features. 2.8 s unchanged everywhere.
