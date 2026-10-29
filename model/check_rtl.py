"""Compare an RTL simulation dump with the Python model.

Usage:  python -m model.check_rtl <N or design name> [...]
        e.g.  python -m model.check_rtl 0 1 2      (our multiplier)
              python -m model.check_rtl evo_1446    (a baseline)

Reads results/rtl_dump_N{N}.txt or results/rtl_dump_<name>.txt (lines "a b p"
written by tb/tb_approx_mul_exhaustive.sv) and checks every line against a
reference model:
    our designs  -> lut(N), the Python model (model/approx_mul.py)
    evo_*        -> EvoApproxLib's own C model (model/designs.py)
    trumd_<k>    -> TruMD ships no software model; we compare with its
                    description, "the k lowest columns truncated", which is
                    exactly our formula lut(k)
Prints PASS or FAIL, the number of mismatches and the first few mismatches.
Exits with status 1 if any check fails, so `make test` stops.
"""

import sys

import numpy as np

from model import designs
from model.approx_mul import RESULTS, lut

SHOW = 5   # how many mismatches to print


def reference(key):
    """(label, dump file, reference LUT) for an N or a design name."""
    if key.isdigit():
        n = int(key)
        return f"N={n:2d}", RESULTS / f"rtl_dump_N{n}.txt", lut(n)
    d = designs.get(key)
    if d.family == "evo":
        ref = designs.get_lut(key)               # built from the C model
    elif d.family == "trumd":
        ref = lut(int(key.split("_")[1]))        # "k lowest columns truncated"
    else:
        raise ValueError(f"no RTL reference for {key}")
    return key, RESULTS / f"rtl_dump_{key}.txt", ref


def check(key):
    name, path, ref = reference(str(key))
    if not path.exists():
        print(f"{name}: FAIL  missing {path}")
        return False
    data = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if data.shape != (65536, 3):
        print(f"{name}: FAIL  expected 65536 lines of 'a b p', got shape {data.shape}")
        return False
    a, b, p = data[:, 0], data[:, 1], data[:, 2]

    # Every pair must appear exactly once.
    if len(np.unique(a * 256 + b)) != 65536:
        print(f"{name}: FAIL  dump does not cover all 65,536 distinct pairs")
        return False

    expected = ref[a, b]
    bad = np.flatnonzero(p != expected)
    if len(bad) == 0:
        print(f"{name}: PASS  65536/65536 match")
        return True
    print(f"{name}: FAIL  {len(bad)} mismatches out of 65536")
    for k in bad[:SHOW]:
        print(f"        a={a[k]:3d} b={b[k]:3d}  rtl={p[k]:5d}  model={expected[k]:5d}")
    return False


def main():
    keys = sys.argv[1:] or ["0"]
    ok = all([check(k) for k in keys])   # list, so every design is checked and reported
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
