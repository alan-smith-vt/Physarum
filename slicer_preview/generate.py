"""Generate test slices: diamond + orthogonal grid lattice."""
import numpy as np
import struct

# Physical dimensions (from goo_punch_hole_v2)
PIXEL_X_UM = 14.0                    # µm per pixel, horizontal
PIXEL_Y_UM = 19.0                    # µm per pixel, vertical (non-square!)
LAYER_UM   = 10.0                    # µm per slice layer
PLATE_W_PX = 15120                   # printer plate resolution
PLATE_H_PX = 6230

W, H, N_SLICES = 800, 50, 1500      # slice dims (px) and layer count
SPACING  = 200                       # between parallel struts (px)
RADIUS   = 10                        # strut cross-section radius (px)
SHIFT    = LAYER_UM / PIXEL_X_UM    # px/slice drift for 45° struts in physical space

# Precompute
_Y, _X = np.ogrid[:H, :W]
_YSQ = (_Y - H / 2) ** 2

# Auto-fill strut positions from bounds
_max_shift = N_SLICES * SHIFT
_i_lo = int(np.floor(-_max_shift / SPACING)) - 1
_i_hi = int(np.ceil((W + _max_shift) / SPACING)) + 1
_DIAG  = [i * SPACING for i in range(_i_lo, _i_hi + 1)]          # diagonal base x
_VERT  = [i * SPACING for i in range(int(W / SPACING) + 2)]      # vertical x
_H_STEP = SPACING * PIXEL_X_UM / LAYER_UM                        # horiz spacing in slices
_HORIZ = [j * _H_STEP for j in range(int(N_SLICES / _H_STEP) + 2)]  # horizontal z


def make_slice(z):
    """One antialiased XY cross-section at build height z."""
    img = np.zeros((H, W), np.uint8)

    # Diagonal struts (two families, ±45°)
    for bx in _DIAG:
        for s in (1, -1):
            cx = bx + s * z * SHIFT
            if cx < -RADIUS or cx > W + RADIUS:
                continue
            dist = np.sqrt((_X - cx) ** 2 + _YSQ)
            val = np.clip(RADIUS + 0.5 - dist, 0, 1) * 255
            img = np.maximum(img, val.astype(np.uint8))

    # Vertical struts (fixed x, every slice)
    for cx in _VERT:
        if cx < -RADIUS or cx > W + RADIUS:
            continue
        dist = np.sqrt((_X - cx) ** 2 + _YSQ)
        val = np.clip(RADIUS + 0.5 - dist, 0, 1) * 255
        img = np.maximum(img, val.astype(np.uint8))

    # Horizontal struts (full-width bands at certain z heights)
    for zc in _HORIZ:
        dz = (z - zc) * LAYER_UM / PIXEL_X_UM       # z-distance in X-pixel units
        if abs(dz) > RADIUS + 1:
            continue
        dist = np.sqrt(_YSQ + dz ** 2)               # shape (H, 1)
        val = np.clip(RADIUS + 0.5 - dist, 0, 1) * 255
        img = np.maximum(img, val.astype(np.uint8))   # broadcasts to (H, W)

    return img


def encode_all():
    """Generate every slice, return packed binary of non-zero pixels.

    Wire format:  u32 total_pixels
                  then per pixel: u16 x | u16 y | u16 z | u8 intensity   (7 bytes)
    """
    dt = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('v', 'u1')])
    chunks, total = [], 0
    for z in range(N_SLICES):
        img = make_slice(z)
        ys, xs = np.nonzero(img)
        if len(xs) == 0:
            continue
        n = len(xs)
        rec = np.empty(n, dtype=dt)
        rec['x'] = xs.astype(np.uint16)
        rec['y'] = ys.astype(np.uint16)
        rec['z'] = np.uint16(z)
        rec['v'] = img[ys, xs]
        chunks.append(rec.tobytes())
        total += n
    return struct.pack('<I', total) + b''.join(chunks)
