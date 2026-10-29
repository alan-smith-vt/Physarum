# Physarum

Code-first 3D drafting for an ELEGOO Saturn 4 Ultra 16K (resin, 14 × 19 µm pixels).
Designs are Python; output is a printable `.goo`.

## Layout (`slicer_preview/`)

| File | Role |
|---|---|
| `designs/*.py` | One file per design; each defines `PIECES` (and optional `SUPPORTS = True`) |
| `geometry.py` | Primitives: frames, solid columns, base lattice, quad-spiral columns, bridge struts |
| `modifiers.py` | Wrappers that carve existing pieces: chamfer, arches, vaults, grayscale, ID dots |
| `stl_slicer.py` | `stl_piece(path)` — bring in an external mesh |
| `compose.py` | Composites pieces into one volume (shared by preview and export) |
| `supports.py` | TPMS (lidinoid) lattice supports — slow, still experimental |
| `printer.py` | Printer specs + preview downsample factor |

## Workflow

```
cd slicer_preview
cp designs/template.py designs/my_part.py   # edit PIECES
python generate.py my_part                  # preview data (8× downsampled)
python server.py                            # → http://localhost:5555
python to_goo.py my_part                    # full-res → out/my_part.goo
python send_to_printer.py out/my_part.goo   # upload over the network + start the print
```

**Getting a print to the printer:** always `send_to_printer.py`. The printer's USB port is dead and we don't use
the Elegoo software; the network (SDCP V3) is the only path. It finds the printer by broadcast (last seen at
192.168.0.4), refuses to upload while it's busy, and starts the print. `--status` shows progress, `--list` the
files on it, `--no-start` uploads only.

Coordinates are mm; X/Z = plate plane, Y = up.

## Notes

- `designs/whiteboard_box.py` printed great (Mar 2026) — good reference for lattice walls.
- `to_goo.py` keeps the template `.goo`'s bottom exposure on purpose. The Julia print's
  custom 30 s base fused it to the plate.
- `archive/`: the failed Julia print (generation takes ~16 h — don't rerun casually),
  decorative geometry (ropes/braids/textured columns), old outputs.
