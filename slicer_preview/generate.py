"""Build a design at preview resolution and write _slices.bin + _meta.json for the viewer.

Usage:  python generate.py <design>        (designs/<design>.py)
"""
import sys, time
from compose import load_design, compute_bounds, add_supports, write_preview


def main(name):
    design = load_design(name)
    pieces = list(design.PIECES)
    bounds = add_supports(pieces) if getattr(design, 'SUPPORTS', False) else compute_bounds(pieces)
    return write_preview(pieces, bounds)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"Generating slices for '{sys.argv[1]}'...", flush=True)
    t0 = time.time()
    blob = main(sys.argv[1])
    print(f"Done in {time.time() - t0:.1f}s  ({len(blob):,} bytes)")
