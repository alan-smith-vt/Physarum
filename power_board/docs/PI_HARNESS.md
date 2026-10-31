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

## Bring-up results, board A (2026-10-31)
- Soft power: 4 of 4 switch-off cycles logged a clean shutdown (logind "System is powering down" -> sync -> board
  cuts 5 V after halt). An unlatch in the first ~8 s of boot (before the kernel's shutdown_button input, ~3.4 s
  kernel time, plus firmware) is a hard cut, by design. Boot to multi-user ~12 s.
- Buck under load: `stress-ng --cpu 4` for 5 min: throttled 0x0 throughout; bank side 15.06 V, 0.48 A, ~7.2 W;
  Pi EXT5V 4.91 V under load (5.00 V idle); SoC 60 °C. The ~90 mV droop is worth splitting into buck vs cable with a
  meter on J2 VBUS before adding heavy USB loads (Pi under-voltage warns below ~4.63 V).
- Tools on lidarpi: i2c-tools, stress-ng; logger ~/pb_log.py (INA226 + EXT5V + throttled + temp every 10 s).
