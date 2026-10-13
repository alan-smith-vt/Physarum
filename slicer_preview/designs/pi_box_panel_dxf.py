"""Side panel cut file for ordering 1/8" polycarbonate (e.g. SendCutSend): out/pi_box_panel.dxf.

    python designs/pi_box_panel_dxf.py

A plain rectangle with R1 corners, drawn as LINE/ARC entities (as cage_dxf.py, after the
faceted-outline rejection). Order quantity 2. Size from the battery-in-box design.
"""
import math
import sys
from pathlib import Path
import ezdxf

sys.path.insert(0, str(Path(__file__).parent.parent))
from designs.pi_box_bat_lids import PANEL_W, PANEL_H     # noqa: E402

R = 1.0


def main():
    doc = ezdxf.new('R2010', setup=False)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    w, h = PANEL_W, PANEL_H
    corners = [(w - R, R, 270), (w - R, h - R, 0), (R, h - R, 90), (R, R, 180)]
    msp.add_line((R, 0), (w - R, 0))
    msp.add_line((w, R), (w, h - R))
    msp.add_line((w - R, h), (R, h))
    msp.add_line((0, h - R), (0, R))
    for cx, cy, a0 in corners:
        msp.add_arc((cx, cy), R, a0, a0 + 90)
    out = Path(__file__).parent.parent / 'out' / 'pi_box_panel.dxf'
    out.parent.mkdir(exist_ok=True)
    doc.saveas(out)
    print(f"{out.name}: {w:.2f} × {h:.2f} mm, R{R:g} corners, 1/8\" polycarbonate, qty 2")


if __name__ == '__main__':
    main()
