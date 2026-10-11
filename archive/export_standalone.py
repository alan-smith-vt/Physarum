"""Export the current view as a standalone HTML with embedded gzipped data."""
import base64, gzip, json, time
from pathlib import Path
from generate import W, H, N_SLICES, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX, encode_all

OUTPUT = Path(__file__).parent / 'standalone.html'
VIEWER = Path(__file__).parent / 'viewer_instanced.html'

print("Generating slices...", flush=True)
t0 = time.time()
blob = encode_all()
print(f"Done in {time.time() - t0:.1f}s  ({len(blob):,} bytes raw)")

compressed = gzip.compress(blob, compresslevel=9)
print(f"Gzipped: {len(compressed):,} bytes ({len(compressed)/len(blob)*100:.0f}% of raw)")

b64 = base64.b64encode(compressed).decode('ascii')
print(f"Base64:  {len(b64):,} chars")

meta = json.dumps({
    "width": W, "height": H, "num_slices": N_SLICES,
    "pixel_x_um": PIXEL_X_UM, "pixel_y_um": PIXEL_Y_UM, "layer_um": LAYER_UM,
    "plate_w_mm": PLATE_W_PX * PIXEL_X_UM / 1000,
    "plate_h_mm": PLATE_H_PX * PIXEL_Y_UM / 1000,
})

# Read the viewer and replace the fetch calls with embedded data
html = VIEWER.read_text(encoding='utf-8')

fetch_block = """  /* ── Fetch metadata + binary in parallel ── */
  const [meta, buf] = await Promise.all([
    fetch('/meta').then(r => r.json()),
    fetch('/slices').then(r => r.arrayBuffer())
  ]);"""

inline_block = f"""  /* ── Embedded gzipped data (no server needed) ── */
  const meta = {meta};
  const _gz = Uint8Array.from(atob('{b64}'), c => c.charCodeAt(0));
  const _ds = new DecompressionStream('gzip');
  const _wr = _ds.writable.getWriter();
  _wr.write(_gz); _wr.close();
  const _chunks = [];
  const _rd = _ds.readable.getReader();
  while (true) {{
    const {{done, value}} = await _rd.read();
    if (done) break;
    _chunks.push(value);
  }}
  const buf = await new Blob(_chunks).arrayBuffer();"""

html = html.replace(fetch_block, inline_block)

OUTPUT.write_text(html, encoding='utf-8')
size = OUTPUT.stat().st_size
print(f"Wrote {OUTPUT}  ({size/1024/1024:.1f} MB)")
