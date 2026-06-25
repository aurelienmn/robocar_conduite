"""Vectorized 2D raycast over a binary line mask.

Port of the Unity C# Raycast.cs algorithm:
- origin: bottom-center of the mask (x = W/2, y = H-1)
- N rays span an FOV cone symmetric around the up direction (90 deg image-frame)
- each ray walks pixel-by-pixel; distance = number of steps until it hits a
  foreground pixel (line) OR exits the image bounds
- output: ndarray of shape (n_rays,) with int distances in pixels
"""

import math
from typing import Optional, Tuple

import numpy as np


def cast_rays(
    mask,
    n_rays=50,
    fov=180.0,
    origin=None,
):
    if not 1 <= n_rays:
        raise ValueError("n_rays must be >= 1, got {}".format(n_rays))
    if not 1.0 <= fov <= 180.0:
        raise ValueError("fov must be in [1, 180] degrees, got {}".format(fov))

    if mask.ndim == 3:
        mask_bin = mask.min(axis=-1) > 127
    else:
        mask_bin = mask.astype(bool)

    h, w = mask_bin.shape
    if origin is None:
        ox, oy = w / 2.0, h - 1.0
    else:
        ox, oy = float(origin[0]), float(origin[1])

    if n_rays == 1:
        angles_deg = np.array([90.0], dtype=np.float64)
    else:
        step = fov / (n_rays - 1)
        angles_deg = np.arange(n_rays, dtype=np.float64) * step + (180.0 - fov) / 2.0
    angles_rad = np.deg2rad(angles_deg)

    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    max_steps = int(math.ceil(math.hypot(w, h)))
    distances = np.full(n_rays, max_steps, dtype=np.int32)
    active = np.ones(n_rays, dtype=bool)

    for step_idx in range(1, max_steps + 1):
        xs = ox + step_idx * cos_a
        ys = oy - step_idx * sin_a

        ix = np.floor(xs).astype(np.int32)
        iy = np.floor(ys).astype(np.int32)

        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        exited = active & ~in_bounds
        if exited.any():
            distances[exited] = step_idx - 1
            active &= ~exited

        if not active.any():
            break

        sample_idx = active & in_bounds
        if sample_idx.any():
            ix_a = ix[sample_idx]
            iy_a = iy[sample_idx]
            hits_in_active = mask_bin[iy_a, ix_a]
            global_idx = np.flatnonzero(sample_idx)[hits_in_active]
            if global_idx.size:
                distances[global_idx] = step_idx
                active[global_idx] = False
                if not active.any():
                    break

    return distances
