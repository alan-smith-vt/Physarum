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

## Pi config (lidarpi, Debian 13, verified 2026-10-31)

```
dtparam=i2c_arm=on
usb_max_current_enable=1
dtoverlay=gpio-shutdown,gpio_pin=27,gpio_pull=off
dtoverlay=gpio-poweroff,gpiopin=22,active_low=1
```
plus `i2c-dev` in /etc/modules-load.d/i2c-dev.conf (no /dev/i2c-1 without it).
**gpio_pull=off, not down:** with the Pi 5's pull-down on, SW_SENSE (through 10 k from the 100 k / 27 k tap) read LOW with
the switch latched, so gpio-shutdown would never see the switch-off edge. With no pull it reads high on, and the 27 k
holds it low off.
First readings (board A, bank, Pi idle, red on): INA226 at 0x40 (TI, die 0x2260), bus 15.10 V, shunt 2.48 mV = 0.25 A;
EXT5V 5.00 V; throttled 0x0; KEEP_ALIVE (GPIO22) high; BUCK_PG high; PD_PG low (good); LIDAR_EN low.
