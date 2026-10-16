"""Turn the copper polygons in TI's Ultra Librarian RPA0010A footprint into pad-8 custom pads.

UL exports pin 8's three fingers as bare F.Cu polygons, which carry no net and short against pad 8
in DRC. Each F.Cu fp_poly becomes `(pad "8" smd custom ...)` with the polygon as its primitive.
Run once after copying a fresh UL export into power_board.pretty/.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("power_board.pretty") / "RPA0010A.kicad_mod")
text = path.read_text(encoding="utf-8")


def blocks(src, head):
    """Yield (start, end) of balanced s-expressions starting with `head`."""
    i = 0
    while (i := src.find(head, i)) >= 0:
        depth = 0
        for j in range(i, len(src)):
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if depth == 0:
                yield i, j + 1
                break
        i = j + 1


out, last, n = [], 0, 0
for s, e in blocks(text, "(fp_poly"):
    blk = text[s:e]
    if '(layer "F.Cu")' not in blk:
        continue
    pts = re.findall(r"\(xy\s+(-?[\d.]+)\s+(-?[\d.]+)\)", blk)
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    rel = " ".join(f"(xy {float(x) - cx:.6f} {float(y) - cy:.6f})" for x, y in pts)
    pad = (f'(pad "8" smd custom (at {cx:.6f} {cy:.6f}) (size 0.1 0.1) (layers "F.Cu") '
           f'(options (clearance outline) (anchor rect)) '
           f'(primitives (gr_poly (pts {rel}) (width 0) (fill yes))))')
    out.append(text[last:s] + pad)
    last = e
    n += 1
out.append(text[last:])
path.write_text("".join(out), encoding="utf-8")
print(f"converted {n} copper polygons to pad 8")
