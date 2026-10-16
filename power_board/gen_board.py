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

W, H = 60.0, 46.0          # board outline, mm (buck bay size still [MEASURE])
LAYERS = 4                 # TI asks for 4 layers on the buck; inner layers are GND

# ref: (x, y, rotation deg), mm from the top-left corner
PLACE = {
    # input: USB-C on the left edge, PD sink, fuse, TVS, shunt, monitor
    "J1": (4.0, 14.0, 270), "F1": (12.0, 5.0, 0), "D1": (13.0, 11.0, 90), "C1": (9.0, 24.0, 0),
    "U1": (13.0, 30.0, 90), "R2": (8.0, 34.0, 90),
    "R1": (22.0, 5.0, 0), "U5": (29.0, 9.0, 0), "C13": (29.0, 4.5, 0),
    # buck
    "C2": (21.0, 15.0, 90), "C3": (21.0, 20.0, 90), "C4": (24.5, 17.5, 90), "U3": (28.0, 19.0, 0),
    "C5": (29.0, 15.0, 0), "L1": (35.5, 19.0, 0), "C7": (43.0, 14.0, 90), "C8": (43.0, 19.0, 90),
    "C9": (43.0, 24.0, 90), "R3": (26.0, 24.0, 0), "R4": (26.0, 26.0, 0), "R5": (29.5, 24.0, 0),
    "C6": (29.5, 26.0, 0), "R6": (35.0, 25.0, 0),
    # 5 V USB-C out on the right edge, Rp
    "J2": (56.0, 19.0, 90), "R7": (49.0, 14.0, 90), "R8": (49.0, 24.0, 90),
    # lidar eFuse, bottom
    "U4": (30.0, 37.0, 90), "C10": (23.0, 31.0, 0), "R9": (23.0, 34.0, 0), "R10": (23.0, 36.0, 0),
    "R11": (23.0, 38.0, 0), "R12": (37.0, 34.0, 0), "C11": (37.0, 36.0, 0), "R13": (23.0, 40.5, 0),
    "C12": (37.0, 40.0, 0), "J3": (44.0, 40.0, 0),
    # LEDs
    "R14": (51.0, 30.0, 0), "D2": (55.0, 30.0, 0), "R15": (51.0, 33.0, 0), "D3": (55.0, 33.0, 0),
    # Pi header (top edge) and button connector + switch logic (bottom left)
    "J5": (37.0, 4.0, 0), "J4": (5.0, 41.0, 0), "R16": (19.0, 44.0, 0),
    "R17": (19.0, 36.0, 0), "R18": (19.0, 38.0, 0), "R19": (19.0, 40.0, 0), "D4": (19.0, 30.0, 90),
    "D5": (17.0, 30.0, 90), "R20": (19.0, 26.5, 90), "R21": (19.0, 42.0, 0), "R22": (17.0, 26.5, 90),
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
    for x, y in [(3, 3), (W - 3, H - 3)]:  # M2.5 mounting holes, two corners [MEASURE against the bay]
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
