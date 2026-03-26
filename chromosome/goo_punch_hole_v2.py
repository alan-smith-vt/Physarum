import struct
import numpy as np
from pathlib import Path

INPUT = Path(r"original.goo")
OUTPUT = Path(r"modified.goo")
RADIUS_MM = 3.0
PIXEL_X_UM = 14.0
PIXEL_Y_UM = 19.0

def goo_decode(rle, width, height):
    total = width * height
    pixels = np.zeros(total, dtype=np.uint8)
    last_idx = len(rle) - 1
    color = 0
    pixel = 0
    i = 1
    while i < last_idx:
        chunk_type = rle[i] >> 6
        stride = 0
        si0, si1, si2, si3 = i, i+1, i+2, i+3
        if chunk_type == 0:
            color = 0
        elif chunk_type == 1:
            i += 1; color = rle[i]; si1 += 1; si2 += 1; si3 += 1
        elif chunk_type == 2:
            dt = (rle[i] >> 4) & 3
            dv = rle[i] & 0xF
            if dt == 0: color = (color + dv) & 0xFF; stride = 1
            elif dt == 1: color = (color + dv) & 0xFF; i += 1; stride = rle[i]
            elif dt == 2: color = (color - dv) & 0xFF; stride = 1
            elif dt == 3: color = (color - dv) & 0xFF; i += 1; stride = rle[i]
        elif chunk_type == 3:
            color = 255
        if chunk_type != 2:
            cl = (rle[si0] >> 4) & 3
            if cl == 0: stride = rle[si0] & 0xF
            elif cl == 1: stride = (rle[si1] << 4) + (rle[si0] & 0xF); i += 1
            elif cl == 2: stride = (rle[si1] << 12) + (rle[si2] << 4) + (rle[si0] & 0xF); i += 2
            elif cl == 3: stride = (rle[si1] << 20) + (rle[si2] << 12) + (rle[si3] << 4) + (rle[si0] & 0xF); i += 3
        end = min(pixel + stride, total)
        pixels[pixel:end] = color
        pixel = end
        i += 1
    return pixels.reshape(height, width)

def goo_encode(img):
    flat = img.ravel()
    n = len(flat)
    # Find run boundaries using numpy
    changes = np.where(flat[1:] != flat[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    lengths = np.diff(np.concatenate([starts, [n]]))
    colors = flat[starts]

    rle = bytearray([0x55])
    prev_color = 0

    for idx in range(len(colors)):
        cur = int(colors[idx])
        stride = int(lengths[idx])

        first = len(rle)
        rle.append(0)

        diff = abs(cur - prev_color)
        if diff <= 0xF and stride <= 255 and 0 < cur < 255:
            rle[first] = (0b10 << 6) | (diff & 0xF)
            if stride > 1:
                rle[first] |= 0x1 << 4
                rle.append(stride & 0xFF)
            if cur < prev_color:
                rle[first] |= 0x1 << 5
        else:
            if cur == 255:
                rle[first] |= 0b11 << 6
            elif cur > 0:
                rle[first] |= 0b01 << 6
                rle.append(cur)
            rle[first] |= stride & 0xF
            if stride <= 0xF:
                prev_color = cur; continue
            if stride <= 0xFFF:
                rle[first] |= 0b01 << 4
                rle.append((stride >> 4) & 0xFF)
                prev_color = cur; continue
            if stride <= 0xFFFFF:
                rle[first] |= 0b10 << 4
                rle.append((stride >> 12) & 0xFF)
                rle.append((stride >> 4) & 0xFF)
                prev_color = cur; continue
            if stride <= 0xFFFFFFF:
                rle[first] |= 0b11 << 4
                rle.append((stride >> 20) & 0xFF)
                rle.append((stride >> 12) & 0xFF)
                rle.append((stride >> 4) & 0xFF)
        prev_color = cur

    checksum = 0
    for b in rle[1:]:
        checksum = (checksum + b) & 0xFF
    rle.append((~checksum) & 0xFF)
    return bytes(rle)

def parse_goo(data):
    LAYER_DEF_SIZE = 70
    # Find LayerDefAddress from header
    # Offset calculated from GOO header struct field order
    lda_offset = (195310 + 4+2+2+1+1 + 4+4+4 + 4+4+1 + 4+4+4+4 +
                  4+4+4 + 4+4 + 4+4+4+4 + 4+4+4+4 + 4+4+4+4 + 4+4+4+4 +
                  2+2+1+4 + 4+4+4+8)
    layer_def_addr = struct.unpack_from('>I', data, lda_offset)[0]

    pos = 195310
    num_layers = struct.unpack_from('>I', data, pos)[0]
    res_x = struct.unpack_from('>H', data, pos+4)[0]
    res_y = struct.unpack_from('>H', data, pos+6)[0]

    print(f"LayerDefAddress: {layer_def_addr}")
    print(f"Layers: {num_layers}, Resolution: {res_x}x{res_y}")

    layers = []
    pos = layer_def_addr
    for n in range(num_layers):
        layer_def = data[pos:pos+LAYER_DEF_SIZE]
        data_length = struct.unpack_from('>I', data, pos + 66)[0]
        rle_data = data[pos+LAYER_DEF_SIZE:pos+LAYER_DEF_SIZE+data_length]
        layers.append((layer_def, rle_data))
        pos += LAYER_DEF_SIZE + data_length + 2
        if n < 3 or n == num_layers - 1:
            print(f"  Layer {n}: def={LAYER_DEF_SIZE}b rle={data_length}b tag=0x{rle_data[0]:02x}")

    return data[:layer_def_addr], layers, data[pos:], res_x, res_y

def main():
    print(f"Reading {INPUT}")
    data = INPUT.read_bytes()
    header, layers, footer, W, H = parse_goo(data)

    img0 = goo_decode(layers[0][1], W, H)
    coords = np.argwhere(img0 > 0)
    cy, cx = (coords.min(axis=0) + coords.max(axis=0)) // 2
    print(f"Geometry center: ({cx}, {cy})")

    rx = int(RADIUS_MM * 1000 / PIXEL_X_UM)
    ry = int(RADIUS_MM * 1000 / PIXEL_Y_UM)
    yy, xx = np.ogrid[:H, :W]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    print(f"Hole: {RADIUS_MM}mm -> {rx}x{ry}px, {mask.sum():,} pixels")

    new_layers = []
    for i, (layer_def, rle_data) in enumerate(layers):
        img = goo_decode(rle_data, W, H)
        img[mask] = 0
        new_rle = goo_encode(img)
        new_def = bytearray(layer_def)
        struct.pack_into('>I', new_def, 66, len(new_rle))
        new_layers.append((bytes(new_def), new_rle))
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{len(layers)} layers")

    print(f"Writing {OUTPUT}")
    with open(OUTPUT, 'wb') as f:
        f.write(header)
        for ld, rle in new_layers:
            f.write(ld)
            f.write(rle)
            f.write(b'\x0d\x0a')
        f.write(footer)
    print("Done!")

if __name__ == "__main__":
    main()