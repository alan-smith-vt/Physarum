"""Autoroute the signal nets with Freerouting, then import the result (KiCad's Python).

    autoroute.py export   -> power_board.dsn (patched: inner layers are planes, GND not routed,
                             power nets get 0.4 mm tracks, existing locked copper kept as fixed)
    (run Freerouting on the .dsn -> power_board.ses)
    autoroute.py import   -> merge power_board.ses into power_board.kicad_pcb

Freerouting: java -jar D:/Tools/freerouting/freerouting-1.9.0.jar -de power_board.dsn -do power_board.ses -mp 40
"""
import re
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
PCB, DSN, SES = HERE / "power_board.kicad_pcb", HERE / "power_board.dsn", HERE / "power_board.ses"
POWER = ("VBUS_RAW", "V5", "V15", "SW_FEED", "SW_ON")
NOROUTE = ("GND",)


def export():
    board = pcbnew.LoadBoard(str(PCB))
    if not pcbnew.ExportSpecctraDSN(board, str(DSN)):
        raise SystemExit("DSN export failed")
    text = DSN.read_text(encoding="utf-8")
    # inner layers are planes: keep the router off them
    for layer in ("In1.Cu", "In2.Cu"):
        text = re.sub(rf'(\(layer "?{re.escape(layer)}"?\s*\(type )signal\)', r"\1power)", text)
    # net classes: power nets wider; GND left to the planes and fan-out vias
    cls = (f'    (class POWER {" ".join("/" + n for n in POWER)}\n'
           f'      (circuit (use_via "Via[0-3]_600:300_um"))\n      (rule (width 400) (clearance 200)))\n'
           f'    (class NOROUTE {" ".join("/" + n for n in NOROUTE)}\n'
           f'      (circuit (use_via "Via[0-3]_600:300_um"))\n      (rule (width 300) (clearance 200)))\n')
    m = re.search(r"\(class kicad_default([^()]*)", text)
    names = [n for n in m.group(1).split() if n.strip('"').lstrip("/") not in POWER + NOROUTE]
    text = text[:m.start()] + cls + "    (class kicad_default " + "\n      ".join(
        " ".join(names[i:i + 6]) for i in range(0, len(names), 6)) + text[m.end():]
    DSN.write_text(text, encoding="utf-8")
    print("exported", DSN)


def do_import():
    board = pcbnew.LoadBoard(str(PCB))
    if not pcbnew.ImportSpecctraSES(board, str(SES)):
        raise SystemExit("SES import failed")
    board.Save(str(PCB))
    board = pcbnew.LoadBoard(str(PCB))   # fresh load: the track list is unreliable right after an import
    removed = drop_dangling(board)
    board.Save(str(PCB))
    print("removed dangling stubs:", removed)


def drop_dangling(board):
    """Remove track segments with a free end (left behind when the optimiser re-cut a locked track)."""
    removed = 0
    zones = [board.GetArea(i) for i in range(board.GetAreaCount())]
    for _ in range(10):
        tracks = [t for t in list(board.GetTracks()) if t.GetClass() == "PCB_TRACK"]
        vias = [t for t in list(board.GetTracks()) if t.GetClass() == "PCB_VIA"]
        pads = [p for fp in board.GetFootprints() for p in fp.Pads()]
        dead = []
        for t in tracks:   # fragments lying wholly on another track of the same net are redundant
            if any(o is not t and o.GetNetCode() == t.GetNetCode() and o.GetLayer() == t.GetLayer()
                   and o.GetLength() > t.GetLength() and o.HitTest(t.GetStart(), 0) and o.HitTest(t.GetEnd(), 0)
                   for o in tracks):
                dead.append(t)
        for t in tracks:
            if t in dead:
                continue
            for end in (t.GetStart(), t.GetEnd()):
                hit = any(o is not t and o.GetNetCode() == t.GetNetCode() and o.GetLayer() == t.GetLayer()
                          and (o.GetStart() == end or o.GetEnd() == end or o.HitTest(end, 0)) for o in tracks)
                hit = hit or any(v.GetNetCode() == t.GetNetCode() and v.HitTest(end, 0) for v in vias)
                hit = hit or any(p.GetNetCode() == t.GetNetCode() and p.IsOnLayer(t.GetLayer()) and p.HitTest(end, 0)
                                 for p in pads)
                hit = hit or any(z.GetNetCode() == t.GetNetCode() and z.IsOnLayer(t.GetLayer()) and
                                 z.HitTestFilledArea(t.GetLayer(), end, 0) for z in zones)
                if not hit:
                    dead.append(t)
                    break
        if not dead:
            break
        for t in dead:
            board.Remove(t)
            removed += 1
    return removed
    print("imported", SES)


if __name__ == "__main__":
    {"export": export, "import": do_import}[sys.argv[1]]()
