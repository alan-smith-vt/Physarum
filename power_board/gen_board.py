"""Build power_board.kicad_pcb from out.net: load footprints, assign nets, place parts, outline, GND planes.

Run with KiCad's own Python (it has pcbnew):
    "%LOCALAPPDATA%/Programs/KiCad/10.0/bin/python.exe" gen_board.py
Placement is a first pass for routing by hand in KiCad; re-running overwrites the board.
"""
import os
import re
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
KICAD_FP = Path(os.environ["LOCALAPPDATA"]) / "Programs/KiCad/10.0/share/kicad/footprints"
LIBS = {"power_board": HERE / "power_board.pretty"}

# Board stands upright against the -Y panel in the old buck/WAGO bay (pi_box_bat_rig: BUCK, WAGOS):
# X -2..44 (clear of the power switch body, which reaches X -6 at depth 147-163), depth ~116..176
# (below the Pi's ports, above the floor). Board x = box X + 2, board y = depth - 116.
# Top edge (y=0) faces the Pi, bottom edge faces the floor, left edge faces the switch.
W, H = 46.0, 60.0
LAYERS = 4                 # TI asks for 4 layers on the buck; inner layers are GND

# ref: (x, y, rotation deg), mm from the top-left corner
PLACE = {
    # top edge (toward the Pi): both USB-C sockets mouth up (5 V out left, bank in right), Rp, status LEDs
    "J2": (10.0, 4.0, 180), "R8": (17.5, 3.0, 90), "R7": (17.5, 7.0, 90), "J1": (36.0, 4.0, 180),
    "R14": (24.0, 3.0, 0), "D2": (24.0, 5.5, 0), "R15": (24.0, 8.0, 0), "D3": (24.0, 10.5, 0),
    # buck, upper left, per TPS56637 datasheet 10.1: U3 turned 180 so SW (pin 6) faces up into the
    # inductor and the 5 V caps sit just under J2; Cin hugs VIN/PGND on the left; the FB divider sits
    # on the right, the quiet side away from SW; boot cap between pin 7 and the SW node.
    # C2 sits under U3 spanning VIN (pin 8) and PGND (pin 9) with C4; C3 is bulk on the left.
    # C8 is turned 180 so its GND pad pairs with C7's on one via. FB column at x 17.2 (R3 top, R4 bottom).
    "U3": (12.0, 24.9, 180), "C2": (11.2, 30.5, 0), "C3": (7.0, 27.0, 180), "C4": (11.6, 27.8, 0),
    "C5": (8.6, 24.2, 180), "L1": (12.0, 17.9, 90),
    "C7": (5.0, 10.55, 0), "C8": (10.0, 10.55, 180), "C9": (15.0, 10.55, 0),
    "R3": (17.2, 23.1, 270), "R4": (17.2, 27.4, 270), "R5": (20.5, 23.1, 270), "C6": (20.5, 26.4, 270),
    "R6": (22.8, 27.5, 270),
    # input chain down the right side, in signal order: J1 -> F1 -> D1 (TVS) -> R1 shunt -> U5 monitor
    "F1": (36.0, 13.0, 90), "D1": (41.5, 13.0, 90), "R1": (38.0, 21.0, 90),
    "U5": (38.0, 28.5, 0), "C13": (43.0, 28.5, 90),
    # button connector on the left edge (switch side) + switch logic
    "J4": (2.5, 44.0, 90), "R17": (10.5, 34.0, 0), "R18": (10.5, 36.0, 0), "R19": (10.5, 38.0, 0),
    "R21": (10.5, 40.0, 0), "R16": (10.5, 42.0, 0), "D4": (10.0, 44.5, 0), "D5": (10.0, 47.0, 0),
    "R20": (6.5, 49.0, 0), "R22": (6.5, 51.0, 0),
    # lidar eFuse, middle
    "U4": (20.0, 41.0, 90), "C10": (15.0, 34.0, 0), "R9": (15.0, 37.0, 0), "R10": (15.0, 39.0, 0),
    "R11": (15.0, 41.0, 0), "R13": (15.0, 44.0, 0), "R12": (27.0, 38.0, 0), "C11": (27.0, 40.0, 0),
    "C12": (27.0, 44.0, 0),
    # bottom (floor side): top-entry Pi header and lidar terminal (plugs go in from the board face), PD sink
    "J5": (8.5, 56.0, 0), "C1": (20.0, 48.0, 0), "U1": (26.0, 49.0, 0), "R2": (31.0, 46.0, 90),
    "J3": (34.0, 55.0, 0),
}


def mm(v):
    return pcbnew.FromMM(v)


def pt(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def parse_netlist(path):
    text = path.read_text(encoding="utf-8")
    comps = {}
    for block in re.split(r'\(comp\s+(?=\(ref)', text[:text.index("(libparts")])[1:]:
        ref = re.search(r'\(ref\s+"([^"]+)"\)', block).group(1)
        comps[ref] = {
            "value": re.search(r'\(value\s+"([^"]*)"\)', block).group(1),
            "footprint": re.search(r'\(footprint\s+"([^"]*)"\)', block).group(1),
            "uuid": re.search(r'\(tstamps\s+"([0-9a-f-]{36})"\)', block).group(1),
            "description": (re.search(r'\(name\s+"Description"\)\s+"([^"]*)"\)', block) or [None, ""])[1],
        }
    nets = {}
    nets_block = text[text.index("(nets"):]
    for m in re.finditer(r'\(net\s+\(code\s+"?\d+"?\)\s+\(name\s+"([^"]*)"\)(.*?)(?=\(net\s+\(code|\Z)', nets_block, re.S):
        name = m.group(1)
        for ref, pin in re.findall(r'\(node\s+\(ref\s+"([^"]+)"\)\s+\(pin\s+"([^"]+)"\)', m.group(2)):
            nets[(ref, pin)] = name
    return comps, nets


def load_fp(fpid):
    lib, name = fpid.split(":", 1)
    libpath = LIBS.get(lib, KICAD_FP / f"{lib}.pretty")
    fp = pcbnew.FootprintLoad(str(libpath), name)
    if fp is None:
        raise SystemExit(f"footprint not found: {fpid}")
    fp.SetFPIDAsString(fpid)
    return fp


def add_outline(board):
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:] + corners[:1]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pt(x1, y1))
        seg.SetEnd(pt(x2, y2))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(mm(0.1))
        board.Add(seg)
    for x, y in [(3, H - 3), (W - 3, 47)]:  # M2.5 holes; the core needs matching bosses
        hole = load_fp("MountingHole:MountingHole_2.7mm_M2.5")
        hole.SetPosition(pt(x, y))
        hole.SetReference(f"H{1 if x < 10 else 2}")
        hole.SetBoardOnly(True)
        board.Add(hole)


def add_plane(board, layer, net):
    zone = pcbnew.ZONE(board)
    zone.SetLayer(layer)
    zone.SetNet(net)
    ol = zone.Outline()
    ol.NewOutline()
    for x, y in [(0.6, 0.6), (W - 0.6, 0.6), (W - 0.6, H - 0.6), (0.6, H - 0.6)]:
        ol.Append(mm(x), mm(y))
    zone.SetLocalClearance(mm(0.3))
    zone.SetMinThickness(mm(0.25))
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    board.Add(zone)


def main():
    comps, pin_nets = parse_netlist(HERE / "out.net")
    board = pcbnew.NewBoard(str(HERE / "power_board.kicad_pcb"))
    board.SetCopperLayerCount(LAYERS)

    netinfo = {}
    for name in sorted(set(pin_nets.values())):
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        netinfo[name] = ni

    ds = board.GetDesignSettings()
    ds.m_MinClearance = mm(0.15)            # 0.5 mm-pitch MSOP pads sit 0.15 apart (JLC min 0.1)
    ds.m_NetSettings.GetDefaultNetclass().SetClearance(mm(0.15))
    ds.m_HoleClearance = mm(0.15)           # GCT USB4105 pads sit 0.19 from its peg holes
    ds.m_MinThroughDrill = mm(0.2)          # 0.45/0.2 vias escape the buck's 0.5 mm-pitch pins (JLC 4-layer min 0.15)
    ds.m_ViasMinSize = mm(0.45)

    missing = []
    for ref, comp in sorted(comps.items()):
        fp = load_fp(comp["footprint"])
        fp.SetReference(ref)
        fp.SetValue(comp["value"])
        fp.SetPath(pcbnew.KIID_PATH(f"/{comp['uuid']}"))
        fp.SetField("Description", comp["description"])
        board.Add(fp)
        x, y, rot = PLACE.get(ref, (W + 10, 5 + 4 * len(missing), 0))
        if ref not in PLACE:
            missing.append(ref)
        fp.SetPosition(pt(x, y))
        fp.SetOrientationDegrees(rot)
        for pad in fp.Pads():
            name = pin_nets.get((ref, pad.GetNumber()))
            if name in netinfo:
                pad.SetNet(netinfo[name])

    add_outline(board)
    gnd = netinfo["/GND"]
    add_plane(board, pcbnew.In1_Cu, gnd)
    add_plane(board, pcbnew.B_Cu, gnd)   # zones are filled by KiCad (DRC --refill-zones, or B in the editor)

    board.Save(str(HERE / "power_board.kicad_pcb"))
    print("placed", len(comps), "parts; not in PLACE:", missing or "none")


if __name__ == "__main__":
    main()
