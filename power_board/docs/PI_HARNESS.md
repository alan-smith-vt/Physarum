# Pi harness (J5 -> Pi 5 GPIO): colours LOCKED 2026-10-31

The lead is a KS-PH-MM 10P PH (far plug cut off), each wire spliced to a female Dupont, ~10 cm PH housing to Dupont
housing (v4 box: J5 sits ~15 mm from the Pi header). **J5 pin 1 = the left end, with the board's USB-C sockets up.**
Diagram: `pi_harness.png` (regenerate with `python pi_harness_diagram.py`).

| J5 | Signal | Pi pin | GPIO | Colour |
|---|---|---|---|---|
| 1 | 3V3 | 1 | | red: **leave unconnected for the first boot** |
| 2 | GND | 9 | | black |
| 3 | SDA | 3 | GPIO2 | blue |
| 4 | SCL | 5 | GPIO3 | yellow |
| 5 | LIDAR_EN | 11 | GPIO17 | orange |
| 6 | SW_SENSE | 13 | GPIO27 | green |
| 7 | KEEP_ALIVE | 15 | GPIO22 | purple |
| 8 | LIDAR_FLT (active low) | 16 | GPIO23 | brown |
| 9 | PD_PG (active low) | 18 | GPIO24 | white |
| 10 | BUCK_PG | 22 | GPIO25 | white with a blue stripe |

GPIO5/6 (pins 29/31) stay free for a future tracking-camera trigger.

Power button lead (J4, JST-XH 4, pin 1 = bottom with USB-C up), ~15 cm to the switch pigtail (floor removal slack):
green = C (1), yellow = NO (2), red = LED+ (3), black = LED- (4), white = NC (unused, taped).
