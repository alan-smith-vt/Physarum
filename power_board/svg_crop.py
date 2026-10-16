"""Rasterise a kicad-cli board SVG (page = board outline) and crop a board-mm window to PNG.

    python svg_crop.py in.svg out.png x0 y0 x1 y1 [px_per_mm]
"""
import sys

import fitz

src, out = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = map(float, sys.argv[3:7])
ppm = float(sys.argv[7]) if len(sys.argv) > 7 else 40.0
BOARD_W, BOARD_H = 46.0, 60.0

doc = fitz.open(src)
page = doc[0]
sx, sy = page.rect.width / BOARD_W, page.rect.height / BOARD_H   # points per board mm
clip = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
zoom = ppm / sx
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
pix.save(out)
print(out, pix.width, "x", pix.height)
