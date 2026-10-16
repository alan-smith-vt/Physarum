"""Print each net in out.net (kicad-cli sexpr netlist) with its pins; flag single-pin nets."""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("out.net")).read_text(encoding="utf-8")
nets_block = text[text.index("(nets"):]
for m in re.finditer(r'\(net\s+\(code\s+"?\d+"?\)\s+\(name\s+"([^"]*)"\)(.*?)(?=\(net\s+\(code|\Z)', nets_block, re.S):
    name, body = m.group(1), m.group(2)
    nodes = re.findall(r'\(node\s+\(ref\s+"([^"]+)"\)\s+\(pin\s+"([^"]+)"\)', body)
    flag = "  <-- single pin" if len(nodes) < 2 else ""
    print(f"{name:14s} {len(nodes):2d}  " + " ".join(f"{r}.{p}" for r, p in nodes) + flag)
