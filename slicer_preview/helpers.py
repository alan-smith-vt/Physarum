"""Shared helpers: surface extraction and voxel encoding."""
import numpy as np


def is_interior_inplane(mask):
    """True where pixel AND all 4 in-plane neighbors are non-zero."""
    result = mask.copy()
    result[1:, :]  &= mask[:-1, :]
    result[:-1, :] &= mask[1:, :]
    result[:, 1:]  &= mask[:, :-1]
    result[:, :-1] &= mask[:, 1:]
    result[0, :] = False
    result[-1, :] = False
    result[:, 0] = False
    result[:, -1] = False
    return result


def encode_surface_voxels(make_slice, w, h, n_slices, dx=0, dz=0, dy=0):
    """Encode surface-only voxels from a make_slice callable.

    Args:
        make_slice: callable(layer) → uint8 image (h, w)
        w, h, n_slices: piece dimensions
        dx, dz, dy: pixel offsets into global coordinate system

    Returns:
        (chunks, total): list of record byte chunks + voxel count
    """
    dt = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('v', 'u1')])
    chunks, total = [], 0

    prev = np.zeros((h, w), np.uint8)
    curr = make_slice(0)

    for z in range(n_slices):
        nxt = make_slice(z + 1) if z + 1 < n_slices else np.zeros((h, w), np.uint8)

        nonzero = curr > 0
        interior = is_interior_inplane(nonzero) & (prev > 0) & (nxt > 0)
        surface = nonzero & ~interior

        ys, xs = np.nonzero(surface)
        if len(xs) > 0:
            n = len(xs)
            rec = np.empty(n, dtype=dt)
            rec['x'] = (xs + dx).astype(np.uint16)
            rec['y'] = (ys + dz).astype(np.uint16)
            rec['z'] = np.uint16(z + dy)
            rec['v'] = curr[ys, xs]
            chunks.append(rec.tobytes())
            total += n

        prev = curr
        curr = nxt

    return chunks, total
