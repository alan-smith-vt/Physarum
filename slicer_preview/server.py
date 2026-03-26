"""Stream diamond lattice slices to the Three.js viewer."""
import time, json
from flask import Flask, Response, send_from_directory
from generate import W, H, N_SLICES, PIXEL_X_UM, PIXEL_Y_UM, LAYER_UM, PLATE_W_PX, PLATE_H_PX, encode_all

app = Flask(__name__)

print("Generating slices...", flush=True)
t0 = time.time()
BLOB = encode_all()
print(f"Done in {time.time() - t0:.1f}s  ({len(BLOB):,} bytes)")


@app.route("/")
def index():
    return send_from_directory(".", "viewer.html")


@app.route("/instanced")
def instanced():
    return send_from_directory(".", "viewer_instanced.html")


@app.route("/meta")
def meta():
    return Response(json.dumps({
        "width": W, "height": H, "num_slices": N_SLICES,
        "pixel_x_um": PIXEL_X_UM, "pixel_y_um": PIXEL_Y_UM, "layer_um": LAYER_UM,
        "plate_w_mm": PLATE_W_PX * PIXEL_X_UM / 1000,
        "plate_h_mm": PLATE_H_PX * PIXEL_Y_UM / 1000
    }), mimetype="application/json")


@app.route("/slices")
def slices():
    return Response(BLOB, mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(port=5555, debug=True)
