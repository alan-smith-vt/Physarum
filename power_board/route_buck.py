"""Route the TPS56637 buck on power_board.kicad_pcb (run after gen_board.py, with KiCad's Python).

Follows the datasheet layout guide (SLVSEG1A 10.1):
  - VIN loop: F.Cu V15 pour joining C3/C2/C4 to the pin-8 bar and fingers; C2 and C4 span VIN-PGND
    directly under the chip, their GND side in an F.Cu GND pour with stitching vias to In1/B.Cu.
  - SW: short 0.4 mm stub from pin 6 into L1, boot cap C5 beside pin 7.
  - VOUT: F.Cu V5 pour from L1 across the output caps to J2's VBUS pins.
  - FB: pin 2 straight to the R3/R4 column on the quiet side; separate V5 sense from the output pour
    to the top of R3/R5 on In2; AGND (pin 3) gets its own via to the ground planes.
  - EN and PG escape to In2 vias for routing later with the rest of the board.
Coordinates are board mm (origin top-left), matching gen_board.PLACE.
"""
from pathlib import Path

import pcbnew

PCB = Path(__file__).with_name("power_board.kicad_pcb")
board = pcbnew.LoadBoard(str(PCB))
GROUP = "buck-route"


def mm(v):
    return pcbnew.FromMM(v)


def net(name):
    n = board.FindNet(f"/{name}")
    if n is None:
        raise SystemExit(f"no net {name}")
    return n


# start clean: drop anything this script made before
for item in list(board.Tracks()) + list(board.Zones()):
    if item.GetParentGroup() and item.GetParentGroup().GetName() == GROUP:
        board.Remove(item)
for g in list(board.Groups()):
    if g.GetName() == GROUP:
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


def pour(netname, pts, layer=pcbnew.F_Cu, clearance=0.2, priority=10):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    z.SetNet(net(netname))
    ol = z.Outline()
    ol.NewOutline()
    for x, y in pts:
        ol.Append(mm(x), mm(y))
    z.SetLocalClearance(mm(clearance))
    z.SetMinThickness(mm(0.2))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    z.SetAssignedPriority(priority)
    board.Add(z)
    group.AddItem(z)


# ── VIN (V15) loop ──
pour("V15", [(7.85, 24.05), (11.3, 24.05), (11.3, 32.0), (7.85, 32.0)])
for x, y in [(9.65, 26.6), (9.65, 27.6), (8.5, 29.5), (8.5, 30.7)]:   # VIN pour down to the In2 V15 plane
    via("V15", x, y)
pour("GND", [(11.85, 24.6), (12.62, 24.6), (12.62, 25.95), (13.3, 25.95), (13.3, 26.8), (14.8, 26.8),
             (14.8, 32.2), (11.55, 32.2), (11.55, 26.75), (11.85, 26.75)], clearance=0.15)
for y in (27.5, 28.6, 29.7, 30.8, 31.7):
    via("GND", 14.2, y)
# bulk cap C3's GND side
pour("GND", [(3.8, 25.4), (6.4, 25.4), (6.4, 28.6), (3.8, 28.6)])
via("GND", 4.4, 26.1)
via("GND", 4.4, 27.9)

# ── SW node and boot cap (C5 between L1 and the chip: pad 2 = SW on the left, pad 1 = BOOT on the right) ──
track("BUCK_SW", [(11.63, 23.6), (11.63, 22.4)], 0.4)          # pin 6 up into L1 pad 1
track("BUCK_SW", [(7.82, 24.2), (7.82, 22.4), (9.6, 22.4)], 0.4)   # C5 SW side up into L1 pad 1
track("BUCK_BOOT", [(9.38, 24.2), (10.1, 23.5), (11.13, 23.5)], 0.25)  # C5 boot side to pin 7

# ── VOUT (V5) ──
pour("V5", [(2.8, 8.45), (17.0, 8.45), (17.0, 15.8), (2.8, 15.8)])
track("V5", [(12.4, 7.68), (12.4, 8.9)], 0.5)                  # J2 A4/B9
track("V5", [(7.6, 7.68), (7.6, 8.9)], 0.5)                    # J2 A9/B4
track("GND", [(6.47, 10.55), (8.53, 10.55)], 0.6)              # C7 + C8 GND pads
via("GND", 7.5, 10.55)
track("GND", [(16.48, 10.55), (17.75, 10.55)], 0.6)            # C9 GND
via("GND", 17.75, 10.55)

# ── feedback ──
track("BUCK_FB", [(13.4, 25.15), (17.2, 25.15)], 0.2)          # pin 2 to the divider column
track("BUCK_FB", [(17.2, 23.93), (17.2, 26.57)], 0.2)          # R3 bottom - R4 top
track("BUCK_FB", [(17.2, 26.57), (17.9, 27.17), (20.5, 27.17)], 0.2)   # R4 top to C6 bottom (feed-forward)
track("BUCK_FF", [(20.5, 23.93), (20.5, 25.63)], 0.2)          # R5 bottom to C6 top
track("GND", [(17.2, 28.23), (17.2, 29.1)], 0.3)               # R4 bottom
via("GND", 17.2, 29.1)
track("GND", [(13.4, 24.65), (14.2, 24.65)], 0.2)              # AGND (pin 3) to its own via
via("GND", 14.2, 24.65, d=0.45, drill=0.2)
# V5 sense: from the output pour on In2 to the top of R3 and R5
via("V5", 15.4, 14.0)
track("V5", [(15.4, 14.0), (18.85, 14.0), (18.85, 21.3)], 0.25, pcbnew.In2_Cu)
via("V5", 18.85, 21.3)
track("V5", [(17.2, 22.27), (17.2, 21.3), (20.5, 21.3), (20.5, 22.27)], 0.25)

# ── J2 (5 V to the Pi): GND pins to the shell, CC pull-ups (R7 = CC1 lower, R8 = CC2 upper) ──
track("GND", [(13.2, 7.68), (14.32, 7.11)], 0.3)
track("GND", [(6.8, 7.68), (5.68, 7.11)], 0.3)
track("OUT_CC1", [(11.25, 7.68), (11.25, 5.6), (16.6, 5.6), (16.6, 6.17), (17.5, 6.17)], 0.2)
track("OUT_CC2", [(8.25, 7.68), (8.25, 4.8), (16.3, 4.8), (16.3, 2.17), (17.5, 2.17)], 0.2)
track("V5", [(17.5, 3.83), (18.5, 3.83), (18.5, 7.83), (17.5, 7.83)], 0.3)   # Rp tops together
track("V5", [(17.5, 7.83), (16.6, 8.0), (15.3, 8.9)], 0.3)                    # into the V5 pour

# ── PG pull-up R6 (V5 top, PG bottom) ──
track("V5", [(20.5, 21.3), (22.8, 21.3), (22.8, 26.67)], 0.25)

# ── EN / PG escapes (EN and PG to J5 are finished with the rest of the board) ──
track("BUCK_PG", [(13.4, 24.15), (13.9, 24.15), (14.3, 23.75)], 0.2)
via("BUCK_PG", 14.3, 23.75, d=0.45, drill=0.2)
track("BUCK_PG", [(14.3, 23.75), (14.9, 23.75), (14.9, 29.9), (22.8, 29.9)], 0.2, pcbnew.In2_Cu)
via("BUCK_PG", 22.8, 29.9)
track("BUCK_PG", [(22.8, 29.9), (22.8, 28.33)], 0.2)

board.Save(str(PCB))
print("buck routed:", sum(1 for i in group.GetItems()), "items")
