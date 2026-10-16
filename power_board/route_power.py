"""Power routing and planes for power_board.kicad_pcb (run after gen_board.py and route_buck.py).

  - Planes: In1 GND (from gen_board), In2 V15 (here), B.Cu GND (gen_board), F.Cu GND fill (here, lowest priority).
  - Input: J1 VBUS -> F1 -> R1 shunt (VIN15, with D1 TVS), Kelvin sense lines from R1's pads to U5
    (IN+ from the top pad, IN-/VBUS from the bottom pad); the load current leaves R1's bottom pad
    through three vias to the In2 V15 plane, so the sense lines carry no current.
  - Lidar eFuse: IN pins to the V15 plane, OUT pins to C12 and on to J3 (0.8 mm), lidar LED by J3.
  - Fan-out: every remaining surface-mount GND / V15 pad gets a short stub and a via to its plane.
Everything here is locked, so the autorouter (autoroute.py) routes around it.
"""
import math
from pathlib import Path

import pcbnew

PCB = Path(__file__).with_name("power_board.kicad_pcb")
board = pcbnew.LoadBoard(str(PCB))
GROUP = "power-route"
W, H = 46.0, 60.0
CLR = 0.2


def mm(v):
    return pcbnew.FromMM(v)


def to_mm(v):
    return pcbnew.ToMM(v)


def net(name):
    n = board.FindNet(f"/{name}")
    if n is None:
        raise SystemExit(f"no net {name}")
    return n


for g in list(board.Groups()):
    if g.GetName() == GROUP:
        for item in list(g.GetItems()):
            board.Remove(item)
        board.Remove(g)
group = pcbnew.PCB_GROUP(board)
group.SetName(GROUP)
board.Add(group)


def track(netname, pts, width, layer=pcbnew.F_Cu):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        t.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        t.SetWidth(mm(width))
        t.SetLayer(layer)
        t.SetNet(net(netname))
        t.SetLocked(True)
        board.Add(t)
        group.AddItem(t)


def via(netname, x, y, d=0.6, drill=0.3):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
    v.SetWidth(mm(d))
    v.SetDrill(mm(drill))
    v.SetNet(net(netname))
    v.SetLocked(True)
    board.Add(v)
    group.AddItem(v)


def plane(netname, layer, priority, inset=0.6):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    z.SetNet(net(netname))
    ol = z.Outline()
    ol.NewOutline()
    for x, y in [(inset, inset), (W - inset, inset), (W - inset, H - inset), (inset, H - inset)]:
        ol.Append(mm(x), mm(y))
    z.SetLocalClearance(mm(0.25))
    z.SetMinThickness(mm(0.25))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_THT_THERMAL)   # SMD pads solid, through-hole pads relieved for hand soldering
    z.SetAssignedPriority(priority)
    board.Add(z)
    group.AddItem(z)


# ── planes ──
plane("V15", pcbnew.In2_Cu, 0)
plane("GND", pcbnew.F_Cu, 0)

# ── J1 input: GND pins to the shell, VBUS pairs down into F1 ──
track("GND", [(39.2, 7.68), (40.32, 7.11)], 0.3)
track("GND", [(32.8, 7.68), (31.68, 7.11)], 0.3)
track("VBUS_RAW", [(38.4, 7.68), (38.4, 9.4), (37.4, 10.5)], 0.6)
track("VBUS_RAW", [(33.6, 7.68), (33.6, 9.8), (34.4, 10.6)], 0.6)

# ── F1 -> R1 (VIN15) with the TVS, Kelvin sense to U5 ──
track("VIN15", [(36.0, 15.14), (37.2, 17.6)], 1.0)
track("VIN15", [(41.5, 15.15), (39.4, 17.6)], 0.8)
track("VIN15", [(37.0, 25.65), (37.0, 25.4), (36.0, 24.9), (35.6, 24.3), (35.6, 18.4), (36.4, 18.04)], 0.2)  # IN+
track("V15", [(37.5, 25.9), (37.5, 24.4)], 0.2)                # IN- from R1's bottom pad
track("V15", [(38.0, 25.9), (38.0, 24.4)], 0.2)                # VBUS sense
track("V15", [(38.0, 22.7), (38.0, 23.6)], 1.0)                # load current: R1 bottom pad -> vias
track("V15", [(36.9, 22.7), (39.1, 22.7)], 0.6)
for x in (36.9, 38.0, 39.1):
    via("V15", x, 22.7)
track("GND", [(38.5, 27.15), (37.5, 29.85)], 0.2)              # U5 pin 7 GND under the body to pin 2
track("3V3_PI", [(39.0, 25.9), (39.6, 25.1), (40.6, 25.1), (40.9, 26.38)], 0.25)

# ── lidar eFuse (U4 turned 270: IN on top row, OUT on bottom row) ──
track("V15", [(19.68, 37.4), (19.68, 36.9), (22.93, 36.9)], 0.3)
for x in (21.63, 22.28, 22.93):
    track("V15", [(x, 37.4), (x, 36.9)], 0.4)
track("V15", [(22.65, 35.92), (22.65, 36.9)], 0.4)             # C10
via("V15", 20.6, 36.1)
via("V15", 21.7, 36.1)
track("V15", [(20.6, 36.1), (20.6, 36.9)], 0.3)
track("V15", [(21.7, 36.1), (21.7, 36.9)], 0.3)
track("GND", [(17.73, 38.9), (17.73, 39.5)], 0.3)              # pin 9 GND into the thermal pad
track("GND", [(20.33, 43.1), (20.33, 42.5)], 0.3)              # pin 16 PGTH GND into the thermal pad
track("VLIDAR", [(21.63, 44.6), (21.63, 45.1), (22.93, 45.1)], 0.5)
for x in (21.63, 22.28, 22.93):
    track("VLIDAR", [(x, 44.4), (x, 45.1)], 0.4)
track("VLIDAR", [(22.05, 45.1), (22.05, 45.9)], 0.5)           # C12
track("VLIDAR", [(22.93, 45.1), (35.6, 45.1), (35.6, 53.6), (35.0, 54.4)], 0.8)   # to J3 pin 1
track("VLIDAR", [(35.6, 50.77), (42.6, 50.77)], 0.3)           # lidar LED R15
track("LEDL_A", [(42.6, 52.43), (42.6, 54.2)], 0.25)           # R15 -> D3 anode
# U1 (CH224A) exposed pad and U5 GND pins
via("GND", 29.0, 47.8)
via("GND", 29.0, 49.4)
track("GND", [(37.0, 31.1), (37.5, 31.1)], 0.25)


# ── fan-out: remaining SMD GND / V15 pads get a stub + via ──
def pad_rect(p):
    bb = p.GetBoundingBox()
    return to_mm(bb.GetLeft()), to_mm(bb.GetTop()), to_mm(bb.GetRight()), to_mm(bb.GetBottom())


def d_point_rect(x, y, r):
    dx = max(r[0] - x, 0, x - r[2])
    dy = max(r[1] - y, 0, y - r[3])
    return math.hypot(dx, dy)


def d_point_seg(x, y, a, b):
    (ax, ay), (bx, by) = a, b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = 0 if L2 == 0 else max(0, min(1, ((x - ax) * vx + (y - ay) * vy) / L2))
    return math.hypot(x - ax - t * vx, y - ay - t * vy)


def obstacles():
    pads = [(p.GetNetname(), pad_rect(p), p) for fp in board.GetFootprints() for p in fp.Pads()]
    segs, vias = [], []
    for t in board.Tracks():
        if t.GetClass() == "PCB_VIA":
            vias.append((t.GetNetname(), to_mm(t.GetPosition().x), to_mm(t.GetPosition().y), to_mm(t.GetWidth(pcbnew.F_Cu)) / 2))
        else:
            segs.append((t.GetNetname(), (to_mm(t.GetStart().x), to_mm(t.GetStart().y)),
                         (to_mm(t.GetEnd().x), to_mm(t.GetEnd().y)), to_mm(t.GetWidth()) / 2, t.GetLayer()))
    return pads, segs, vias


def clear_point(x, y, r, netname, pads, segs, vias, layer_any=True):
    if not (0.8 < x < W - 0.8 and 0.8 < y < H - 0.8):
        return False
    for n, rect, p in pads:
        if n != netname and d_point_rect(x, y, rect) < r + CLR:
            return False
        if p.GetDrillSize().x > 0 and d_point_rect(x, y, rect) < r + 0.3:   # any hole
            if n != netname or p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH:
                return False
    for n, a, b, hw, layer in segs:
        if n != netname and d_point_seg(x, y, a, b) < r + hw + CLR:
            return False
    for n, vx, vy, vr in vias:
        if math.hypot(x - vx, y - vy) < r + vr + CLR:
            return False
    return True


def clear_stub(a, b, hw, netname, pads, segs, vias):
    n = 12
    for i in range(n + 1):
        x = a[0] + (b[0] - a[0]) * i / n
        y = a[1] + (b[1] - a[1]) * i / n
        for nn, rect, p in pads:
            if nn != netname and d_point_rect(x, y, rect) < hw + CLR:
                return False
        for nn, s0, s1, shw, layer in segs:
            if nn != netname and layer == pcbnew.F_Cu and d_point_seg(x, y, s0, s1) < hw + shw + CLR:
                return False
    return True


SKIP = {"U3", "C2", "C3", "C4", "C7", "C8", "C9", "R4", "J2", "J1", "U4", "C10", "R1", "U5"}
EXTRA = {("U5", "1"), ("U5", "2"), ("C13", "2")}
done = 0
failed = []
for fp in board.GetFootprints():
    ref = fp.GetReference()
    for p in fp.Pads():
        n = p.GetNetname()
        if n not in ("/GND", "/V15") or p.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
            continue
        if ref in SKIP and (ref, p.GetNumber()) not in EXTRA:
            continue
        pads, segs, vias = obstacles()
        cx, cy = to_mm(p.GetPosition().x), to_mm(p.GetPosition().y)
        r = pad_rect(p)
        hx, hy = (r[2] - r[0]) / 2, (r[3] - r[1]) / 2
        placed = False
        for dist in (0.45, 0.7, 1.0, 1.4, 1.9):
            for ang in range(0, 360, 45):
                ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                reach = min(hx / abs(ux) if abs(ux) > 1e-6 else 1e9, hy / abs(uy) if abs(uy) > 1e-6 else 1e9)
                vx, vy = cx + ux * (reach + dist), cy + uy * (reach + dist)
                if clear_point(vx, vy, 0.3, n, pads, segs, vias) and clear_stub((cx, cy), (vx, vy), 0.15, n, pads, segs, vias):
                    name = n.lstrip("/")
                    track(name, [(cx, cy), (vx, vy)], 0.3)
                    via(name, round(vx, 3), round(vy, 3))
                    placed = placed or True
                    break
            if placed:
                break
        if placed:
            done += 1
        else:
            failed.append(f"{ref}.{p.GetNumber()}")

board.Save(str(PCB))
print(f"power routed; fan-out vias {done}; no room for: {failed or 'none'}")
