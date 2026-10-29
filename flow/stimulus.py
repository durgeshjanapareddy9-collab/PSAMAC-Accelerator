"""Realistic (image) stimulus for the power flow (M2.4).

Power depends on how often the wires toggle, which depends on the data. In
Milestone 1 we used uniform random inputs. Here the inputs come from the real
workload: filtering the 'camera' image (eval/image_conv.py).

  multiplier (mul_wrap): pairs "a b", a = pixel, b = filter weight, in the order
      ONE processing element sees them: the 25 (pixel, weight) pairs of one
      output pixel's 5x5 window, then the next output pixel along the row.
      -> build/stim_<design>_image.txt

  array (systolic_array): tiles of 4 output pixels x 4 filters (the same
      im2col tiles as the array test, filter_bank() kernels), written as the
      cycle-by-cycle schedule of model/systolic.py, including the clear and
      drain cycles a real tile needs.
      -> build/stim_array_N<n>_image.txt   (+ the tiles, for checking the outputs)

Both are deterministic, so every design sees exactly the same data.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eval.image_conv import camera, filter_bank, gaussian_kernel, patches   # noqa: E402
from model.systolic import ROWS, schedule                                      # noqa: E402

BUILD = REPO / "build"
# A band of the image with sky, coat and camera, so both flat and detailed areas.
ROW0, COL0 = 120, 100


def mul_image_pairs(n):
    """First n (pixel, weight) pairs of the 5x5 Gaussian filter, PE order."""
    w = gaussian_kernel(5, 1.0).ravel()
    pairs, r, c = [], ROW0, COL0
    while len(pairs) < n:
        p = patches(camera(), [(r, c)], 5)[0]
        pairs += list(zip(p, w))
        c += 1
        if c >= 512 - 2:
            r, c = r + 1, COL0
    return pairs[:n]


def array_image_tiles(n_tiles):
    """n_tiles tiles: A = 4 neighbouring 5x5 patches (4 x 25), B = 4 kernels (25 x 4)."""
    B = np.stack([k.ravel() for k in filter_bank()], axis=1)
    tiles = []
    for t in range(n_tiles):
        coords = [(ROW0 + (t * ROWS) // 400, COL0 + (t * ROWS) % 400 + i) for i in range(ROWS)]
        tiles.append(("im2col", patches(camera(), coords, 5), B, coords))
    return tiles


def write_mul(n, tag="mul"):
    """Write the multiplier stimulus; each design gets its own copy (tag), so
    designs can be simulated in parallel without sharing a file."""
    path = BUILD / f"stim_{tag}_image.txt"
    BUILD.mkdir(exist_ok=True)
    path.write_text("".join(f"{a} {b}\n" for a, b in mul_image_pairs(n)))
    return path


def write_array(n_tiles, tag="array"):
    path = BUILD / f"stim_{tag}_image.txt"
    BUILD.mkdir(exist_ok=True)
    tiles = array_image_tiles(n_tiles)
    path.write_text("\n".join(schedule(tiles)) + "\n")
    return path, tiles
