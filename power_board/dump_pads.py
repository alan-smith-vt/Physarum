"""Print pad positions/sizes/nets for the given refs (KiCad python): dump_pads.py U3 C2 ..."""
import sys
from pathlib import Path

import pcbnew

board = pcbnew.LoadBoard(str(Path(__file__).with_name("power_board.kicad_pcb")))
for ref in sys.argv[1:]:
    fp = board.FindFootprintByReference(ref)
    for pad in fp.Pads():
        p, s = pad.GetPosition(), pad.GetBoundingBox()
        print(f"{ref:4s} {pad.GetNumber():4s} {pad.GetNetname():14s} at ({pcbnew.ToMM(p.x):6.2f},{pcbnew.ToMM(p.y):6.2f})"
              f"  bbox x {pcbnew.ToMM(s.GetLeft()):6.2f}..{pcbnew.ToMM(s.GetRight()):6.2f}"
              f" y {pcbnew.ToMM(s.GetTop()):6.2f}..{pcbnew.ToMM(s.GetBottom()):6.2f}")
