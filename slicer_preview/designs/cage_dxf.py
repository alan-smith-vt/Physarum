"""Aluminum cage plates → DXF for SendCutSend (2D profile, mm, one part per file).

    python designs/cage_dxf.py      → out/cage_base_plate.dxf, out/cage_cross_top.dxf

Same numbers as the resin prototypes (cage_v2), drawn at true size — no resin
compensation. Holes are drawn at the size SendCutSend should cut; countersink
and tapping are ordered as secondary operations on the website, not drawn.

Base plate (1/4" 6061-T6):
  - 4 × M3 clearance Ø3.4 at (±24, ±18), lidar button heads from below (no csk —
    SendCutSend countersinks one face, and the centre needs the top face)
  - 4 × M4 clearance Ø4.5 at (±40, ±40), button heads from below. A countersink
    here would leave ~0.8 mm of wall at the R5 corners.
  - 2 × dowel Ø3.2 at (0, -16) and (0, +23): Ø3 pins set with retaining compound
    (Loctite 638). Laser holes aren't press-fit accurate, and Ø3.0 is below the
    ~50%-of-thickness minimum hole size for 1/4" plate.
  - centre Ø6.6 1/4-20 clearance → order a 1/4-20 82° countersink on the TOP face
Cross top (3/16" 6061-T6):
  - 4 × M4 clearance Ø4.5 at (±40, ±40)
  - centre Ø5.1 → order 1/4-20 tapping (confirm the drill size in SendCutSend's tap guide)
  - no anti-twist pin hole: the GoPro cage has no matching pin feature
X = connector direction.
"""
import sys
from pathlib import Path
import ezdxf
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
from shapely import affinity

sys.path.insert(0, str(Path(__file__).parent.parent))
from designs.cage_v2 import (BASE, BASE_R, STANDOFF, LIDAR_M3, DOWEL_Z, ARM_W, HUB_D,
                             PAD_R, M3_CLEAR, M4_CLEAR, QTR_CLEAR)

DOWEL_HOLE = 3.2
QTR_TAP_DRILL = 5.1
FILLET = 2.0          # inside corners of the cross outline
RES = 64              # points per quarter circle in the outline


def base_plate():
    half = BASE / 2 - BASE_R
    outline = box(-half, -half, half, half).buffer(BASE_R, quad_segs=RES)
    holes = [(sx * STANDOFF, sz * STANDOFF, M4_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(sx * LIDAR_M3[0], sz * LIDAR_M3[1], M3_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(0, z, DOWEL_HOLE) for z in DOWEL_Z]
    holes += [(0, 0, QTR_CLEAR)]
    return outline, holes


def cross_top():
    arm_len = 2 * STANDOFF * 2 ** 0.5
    arm = box(-arm_len / 2, -ARM_W / 2, arm_len / 2, ARM_W / 2)
    parts = [affinity.rotate(arm, a, origin=(0, 0)) for a in (45, -45)]
    parts.append(Point(0, 0).buffer(HUB_D / 2, quad_segs=RES))
    parts += [Point(sx * STANDOFF, sz * STANDOFF).buffer(PAD_R, quad_segs=RES)
              for sx in (-1, 1) for sz in (-1, 1)]
    outline = unary_union(parts)
    outline = outline.buffer(FILLET, quad_segs=RES).buffer(-FILLET, quad_segs=RES)  # fillet inside corners
    holes = [(sx * STANDOFF, sz * STANDOFF, M4_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(0, 0, QTR_TAP_DRILL)]
    return outline, holes


def write_dxf(path, outline, holes):
    assert isinstance(outline, Polygon) and not outline.interiors
    for x, y, d in holes:
        assert outline.buffer(-1.0).contains(Point(x, y).buffer(d / 2)), f"hole at {x},{y} within 1 mm of edge"
    doc = ezdxf.new('R2010', setup=False)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    msp.add_lwpolyline(list(outline.exterior.coords)[:-1], close=True)
    for x, y, d in holes:
        msp.add_circle((x, y), d / 2)
    doc.saveas(path)
    minx, miny, maxx, maxy = outline.bounds
    print(f"{path.name}: {maxx - minx:.2f} × {maxy - miny:.2f} mm, {len(holes)} holes")


if __name__ == '__main__':
    out = Path(__file__).parent.parent / 'out'
    write_dxf(out / 'cage_base_plate.dxf', *base_plate())
    write_dxf(out / 'cage_cross_top.dxf', *cross_top())
