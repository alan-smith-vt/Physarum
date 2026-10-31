"""Pi 5 GPIO header with the J5 harness colours (locked 2026-10-31). Run: python pi_harness_diagram.py -> pi_harness.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Pi pin -> (signal, J5 pin, colour name, face colour, stripe colour or None)
HARNESS = {
    1: ('3V3', 1, 'red', '#d62728', None),
    9: ('GND', 2, 'black', '#111111', None),
    3: ('SDA', 3, 'blue', '#1f5fd6', None),
    5: ('SCL', 4, 'yellow', '#f2d21b', None),
    11: ('LIDAR_EN  GPIO17', 5, 'orange', '#ff7f0e', None),
    13: ('SW_SENSE  GPIO27', 6, 'green', '#2ca02c', None),
    15: ('KEEP_ALIVE  GPIO22', 7, 'purple', '#8e44ad', None),
    16: ('LIDAR_FLT  GPIO23', 8, 'brown', '#8c564b', None),
    18: ('PD_PG  GPIO24', 9, 'white', '#ffffff', None),
    22: ('BUCK_PG  GPIO25', 10, 'white / blue stripe', '#ffffff', '#1f5fd6'),
}
OTHER = {2: '5V', 4: '5V', 6: 'GND', 14: 'GND', 20: 'GND', 25: 'GND', 30: 'GND', 34: 'GND', 39: 'GND', 17: '3V3',
         29: 'GPIO5 (spare: cam trigger)', 31: 'GPIO6 (spare: cam trigger)'}

f, ax = plt.subplots(figsize=(11, 13))
ax.add_patch(Rectangle((-0.8, -20.6), 3.6, 21.4, fc='#1b1b1b', ec='k'))
for p in range(1, 41):
    col, row = (p - 1) % 2, (p - 1) // 2
    x, y = col * 2.0, -row
    if p in HARNESS:
        sig, j5, cname, fc, stripe = HARNESS[p]
        ax.add_patch(Circle((x, y), 0.42, fc=fc, ec='#888', lw=1.5))
        if stripe:
            ax.add_patch(Rectangle((x - 0.42, y - 0.1), 0.84, 0.2, fc=stripe, ec='none'))
        txt = f'pin {p}  {sig}  ({cname}, J5-{j5})' + ('  LEAVE OFF FIRST BOOT' if p == 1 else '')
        bold = 'bold'
    else:
        ax.add_patch(Circle((x, y), 0.3, fc='#d4af37', ec='#555'))
        txt, bold = f'pin {p}  {OTHER.get(p, "")}', 'normal'
    if col == 0:
        ax.text(x - 1.2, y, txt, ha='right', va='center', fontsize=9, weight=bold, color='k' if p in HARNESS else '#777')
    else:
        ax.text(x + 1.2, y, txt, ha='left', va='center', fontsize=9, weight=bold, color='k' if p in HARNESS else '#777')
ax.text(1.0, 1.6, 'Pi 5 GPIO header, seen from above (components up)\npin 1 at the SD-card end; odd pins on the inner row, '
        'even pins on the board edge', ha='center', fontsize=11)
ax.text(1.0, -21.6, 'USB / Ethernet end of the Pi ↓', ha='center', fontsize=10)
ax.set_xlim(-13, 15); ax.set_ylim(-22.5, 3)
ax.set_aspect('equal'); ax.axis('off')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('pi_harness.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
