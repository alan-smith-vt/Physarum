"""Build a design at preview resolution and write _slices.bin + _meta.json for the viewer.

Usage:  python generate.py <design> [<design> ...]        (designs/<design>.py)

Each design is also kept under out/preview/<design>/, so server.py can show several
at once (http://localhost:5555/<design> or an alias, list at /designs). The root page
shows the last one generated, as before.
"""
import sys, time, shutil
from pathlib import Path
from compose import load_design, compute_bounds, add_supports, write_preview, SLICES_FILE, META_FILE

PREVIEW_DIR = Path(__file__).parent / 'out' / 'preview'


def main(name):
    design = load_design(name)
    pieces = list(design.PIECES)
    bounds = add_supports(pieces) if getattr(design, 'SUPPORTS', False) else compute_bounds(pieces)
    blob = write_preview(pieces, bounds, getattr(design, 'OVERLAYS', ()))
    keep = PREVIEW_DIR / name
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SLICES_FILE, keep / 'slices.bin')
    shutil.copyfile(META_FILE, keep / 'meta.json')
    return blob


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for name in sys.argv[1:]:
        print(f"Generating slices for '{name}'...", flush=True)
        t0 = time.time()
        blob = main(name)
        print(f"Done in {time.time() - t0:.1f}s  ({len(blob):,} bytes)")
