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
import math
import sys
from pathlib import Path
import ezdxf
import ezdxf.path
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
from shapely import affinity

sys.path.insert(0, str(Path(__file__).parent.parent))
from designs.cage_v2 import (BASE, BASE_R, STANDOFF, LIDAR_M3, DOWEL_Z, ARM_W, HUB_D,
                             PAD_R, M3_CLEAR, M4_CLEAR, QTR_CLEAR)

DOWEL_HOLE = 3.2
QTR_TAP_DRILL = 5.1
FILLET = 2.0          # inside corners of the cross outline
RES = 64              # points per quarter circle in the shapely check outline only

# The DXF outline is drawn analytically as straight segments and true arcs (an
# LWPOLYLINE with bulges), not a faceted polyline: SendCutSend rejected the
# faceted version for its thousands of nodes. The shapely outline is kept as an
# independent check that the analytic one matches the intended shape.


def _arc(start, centre, end, ccw):
    """Polyline vertex (x, y, bulge) for the arc start → end about centre."""
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    sweep = (a1 - a0) % (2 * math.pi) if ccw else -((a0 - a1) % (2 * math.pi))
    return (*start, math.tan(sweep / 4))


def _rot(p, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def base_plate():
    half = BASE / 2 - BASE_R
    verts = []
    for k in range(4):   # each side's line, then the CCW corner arc
        verts.append((*_rot((half + BASE_R, -half), 90 * k), 0.0))
        verts.append(_arc(_rot((half + BASE_R, half), 90 * k), _rot((half, half), 90 * k),
                          _rot((half, half + BASE_R), 90 * k), True))
    check = box(-half, -half, half, half).buffer(BASE_R, quad_segs=RES)
    holes = [(sx * STANDOFF, sz * STANDOFF, M4_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(sx * LIDAR_M3[0], sz * LIDAR_M3[1], M3_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(0, z, DOWEL_HOLE) for z in DOWEL_Z]
    holes += [(0, 0, QTR_CLEAR)]
    return verts, check, holes


def cross_top():
    hub_r, w, f = HUB_D / 2, ARM_W / 2, FILLET
    d = STANDOFF * 2 ** 0.5                     # pad centre distance along the arm
    u_hub = math.sqrt((hub_r + f) ** 2 - (w + f) ** 2)    # hub fillet centre, arm-local u
    u_pad = d - math.sqrt((PAD_R + f) ** 2 - (w + f) ** 2)  # pad fillet centre
    s_hub, s_pad = hub_r / (hub_r + f), PAD_R / (PAD_R + f)
    pc = (d, 0.0)
    verts = []
    for k in range(4):   # arm at 45° + 90k, traversed CCW: lower edge out, round the pad, upper edge in
        local = [
            _arc((u_hub * s_hub, -(w + f) * s_hub), (u_hub, -(w + f)), (u_hub, -w), False),
            ((u_hub, -w, 0.0)),
            _arc((u_pad, -w), (u_pad, -(w + f)), (d + (u_pad - d) * s_pad, -(w + f) * s_pad), False),
            _arc((d + (u_pad - d) * s_pad, -(w + f) * s_pad), pc, (d + (u_pad - d) * s_pad, (w + f) * s_pad), True),
            _arc((d + (u_pad - d) * s_pad, (w + f) * s_pad), (u_pad, w + f), (u_pad, w), False),
            ((u_pad, w, 0.0)),
            _arc((u_hub, w), (u_hub, w + f), (u_hub * s_hub, (w + f) * s_hub), False),
            _arc((u_hub * s_hub, (w + f) * s_hub), (0, 0), _rot((u_hub * s_hub, -(w + f) * s_hub), 90), True),
        ]
        a = 45 + 90 * k
        verts += [(*_rot(v[:2], a), v[2]) for v in local]

    arm_len = 2 * STANDOFF * 2 ** 0.5
    arm = box(-arm_len / 2, -ARM_W / 2, arm_len / 2, ARM_W / 2)
    parts = [affinity.rotate(arm, a, origin=(0, 0)) for a in (45, -45)]
    parts.append(Point(0, 0).buffer(HUB_D / 2, quad_segs=RES))
    parts += [Point(sx * STANDOFF, sz * STANDOFF).buffer(PAD_R, quad_segs=RES)
              for sx in (-1, 1) for sz in (-1, 1)]
    check = unary_union(parts).buffer(FILLET, quad_segs=RES).buffer(-FILLET, quad_segs=RES)
    holes = [(sx * STANDOFF, sz * STANDOFF, M4_CLEAR) for sx in (-1, 1) for sz in (-1, 1)]
    holes += [(0, 0, QTR_TAP_DRILL)]
    return verts, check, holes


def write_dxf(path, verts, check, holes):
    assert isinstance(check, Polygon) and not check.interiors
    for x, y, d in holes:
        assert check.buffer(-1.0).contains(Point(x, y).buffer(d / 2)), f"hole at {x},{y} within 1 mm of edge"
    doc = ezdxf.new('R2010', setup=False)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    pl = msp.add_lwpolyline(verts, format='xyb', close=True)
    for x, y, d in holes:
        msp.add_circle((x, y), d / 2)
    doc.saveas(path)

    # The analytic outline must match the shapely one to within facet error.
    drawn = Polygon([(v.x, v.y) for v in ezdxf.path.make_path(pl).flattening(0.001)])
    assert abs(drawn.area - check.area) / check.area < 1e-3, (drawn.area, check.area)
    assert drawn.hausdorff_distance(check) < 0.02, drawn.hausdorff_distance(check)
    minx, miny, maxx, maxy = drawn.bounds
    n_arcs = sum(1 for _, _, b in verts if b)
    print(f"{path.name}: {maxx - minx:.2f} × {maxy - miny:.2f} mm, "
          f"{len(verts) - n_arcs} lines + {n_arcs} arcs, {len(holes)} holes")


if __name__ == '__main__':
    out = Path(__file__).parent.parent / 'out'
    write_dxf(out / 'cage_base_plate.dxf', *base_plate())
    write_dxf(out / 'cage_cross_top.dxf', *cross_top())
