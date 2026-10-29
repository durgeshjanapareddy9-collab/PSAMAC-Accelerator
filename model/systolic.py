"""Golden models, test vectors and checkers for mac_pe.sv and systolic_array.sv.

The testbenches (tb/tb_mac_pe.sv, tb/tb_systolic_array.sv) are simple
"players": they read a vector file written here, drive the RTL cycle by
cycle, and dump what the RTL produced. This file then compares the dump with
a golden model built on the same LUT as everything else, lut(N)[a, b]. Every
product the RTL makes must match bit for bit.

Usage (normally via `make test-mac`):
    python -m model.systolic gen-pe <N>       -> build/pe_vectors_N<N>.txt
    python -m model.systolic check-pe <N>     <- results/pe_dump_N<N>.txt
    python -m model.systolic gen-array <N>    -> build/array_vectors_N<N>.txt
    python -m model.systolic check-array <N>  <- results/array_dump_N<N>.txt

Array tests (each one is C = A x B with A: 4 x K, B: K x 4):
    * random matrices with K = 1, 2, 4, 16, 64, including all-255 and all-0
      matrices, and random stalls (en = 0) in the middle of the data;
    * K = 256 with all-255 inputs: the largest sum, 256 * 65,025 = 16,646,400,
      still fits the 24-bit accumulator (no overflow);
    * image patches (im2col): rows of A are 5x5 patches of the cameraman
      image, columns of B are four 5x5 unsigned filters (Gaussian, box, and
      two more). Column 0 of C is then the raw (before >> 8) Gaussian-filter
      output of eval/image_conv.py for those pixels: this is task M2.11.
"""

import sys

import numpy as np

from model.approx_mul import REPO, RESULTS, lut

BUILD = REPO / "build"
ROWS = COLS = 4
ACC_W = 24
MASK = (1 << ACC_W) - 1


# ---- MAC PE --------------------------------------------------------------------

def pe_golden(ops, n):
    """Accumulator value after each clock edge, for a list of (clear, en, a, b).

    Mirrors mac_pe.sv: clear wins, then en adds lut(N)[a, b]; ACC_W-bit wrap.
    """
    table = lut(n)
    acc, out = 0, []
    for clear, en, a, b in ops:
        if clear:
            acc = 0
        elif en:
            acc = (acc + int(table[a, b])) & MASK
        out.append(acc)
    return out


def pe_vectors(n, seed=0):
    rng = np.random.default_rng(seed + 100 * n)
    ops = [(1, 1, 0, 0)]                                   # start from a known 0
    for _ in range(3000):                                  # random mix
        ops.append((int(rng.random() < 0.02), int(rng.random() < 0.8),
                    int(rng.integers(256)), int(rng.integers(256))))
    ops.append((1, 0, 0, 0))
    ops += [(0, 1, 255, 255)] * 300                        # long run: wraps past 2^24
    ops.append((1, 1, 0, 0))
    ops += [(0, 1, a, b) for a in (0, 1, 128, 255) for b in (0, 1, 127, 255)]   # corners
    return ops


# ---- Systolic array ---------------------------------------------------------------

def array_golden(A, B, n):
    """C[i][j] = sum_k lut(N)[A[i][k], B[k][j]]  (mod 2^ACC_W)."""
    table = lut(n)
    return table[A[:, :, None], B[None, :, :]].sum(axis=1) & MASK


def image_patch_tests():
    """im2col tests: 4 image patches x 4 filters per tile, K = 25."""
    from eval.image_conv import camera, filter_bank, patches
    img = camera()
    kernels = filter_bank()                                # 4 kernels, 5x5 each
    B = np.stack([k.ravel() for k in kernels], axis=1)     # 25 x 4
    tests = []
    # 16 tiles = 64 output pixels along a row of the image with lots of detail.
    coords = [(200, 180 + c) for c in range(64)]
    P = patches(img, coords, 5)                            # 64 x 25
    for t in range(0, len(coords), ROWS):
        tests.append(("im2col", P[t:t + ROWS], B, coords[t:t + ROWS]))
    return tests


def array_tests(n, seed=0):
    rng = np.random.default_rng(seed + 100 * n)
    tests = []
    for k in (1, 2, 4, 16, 64):
        for _ in range(3):
            tests.append(("random", rng.integers(0, 256, (ROWS, k)),
                          rng.integers(0, 256, (k, COLS)), None))
    tests.append(("all255", np.full((ROWS, 16), 255), np.full((16, COLS), 255), None))
    tests.append(("zeros", np.zeros((ROWS, 16), int), rng.integers(0, 256, (16, COLS)), None))
    tests.append(("max_k256", np.full((ROWS, 256), 255), np.full((256, COLS), 255), None))
    tests += image_patch_tests()
    return tests, rng


def schedule(tests, rng=None, stall_prob=0.0):
    """Cycle-by-cycle schedule (list of lines) for the array testbenches.

    Line formats:  'C clear en a0 a1 a2 a3 b0 b1 b2 b3'  one clock cycle
                   'D'                                    dump c_out (all 16 sums)
    Protocol per test (see rtl/systolic_array.sv):
        ROWS+COLS cycles with clear = 1, en = 1, zero data (empty the pipeline)
        K data cycles: column k of A and row k of B; with stall_prob > 0, random
            stall cycles (en = 0, random junk on the inputs, which must be
            ignored) are inserted in between
        ROWS+COLS-2 cycles of zeros (let the last values reach PE(3,3))
        dump
    """
    lines = []
    z = [0] * (ROWS + COLS)
    for _, A, B, _ in tests:
        lines += ["C 1 1 " + " ".join(map(str, z))] * (ROWS + COLS)
        for k in range(A.shape[1]):
            while stall_prob and rng.random() < stall_prob:
                junk = rng.integers(0, 256, ROWS + COLS)
                lines.append("C 0 0 " + " ".join(map(str, junk)))
            lines.append("C 0 1 " + " ".join(map(str, [*A[:, k], *B[k, :]])))
        lines += ["C 0 1 " + " ".join(map(str, z))] * (ROWS + COLS - 2)
        lines.append("D")
    return lines


def write_array_vectors(n, path):
    """Test schedule for tb_systolic_array.sv, with 10% random stalls."""
    tests, rng = array_tests(n)
    path.write_text("\n".join(schedule(tests, rng, stall_prob=0.1)) + "\n")
    return tests


# ---- Command line ----------------------------------------------------------------

def main():
    cmd, n = sys.argv[1], int(sys.argv[2])
    BUILD.mkdir(exist_ok=True)
    if cmd == "gen-pe":
        ops = pe_vectors(n)
        path = BUILD / f"pe_vectors_N{n}.txt"
        path.write_text("".join(f"{c} {e} {a} {b}\n" for c, e, a, b in ops))
        print(f"wrote {path.relative_to(REPO)} ({len(ops)} cycles)")
    elif cmd == "check-pe":
        expected = pe_golden(pe_vectors(n), n)
        got = [int(x) for x in (RESULTS / f"pe_dump_N{n}.txt").read_text().split()]
        bad = [i for i, (g, e) in enumerate(zip(got, expected)) if g != e]
        if len(got) != len(expected):
            bad.append(-1)
        wraps = sum(1 for x, y in zip(expected, expected[1:]) if y < x and y != 0)
        status = "PASS" if not bad else "FAIL"
        print(f"mac_pe N={n:2d}: {status}  {len(expected) - len(bad)}/{len(expected)} cycles "
              f"match (accumulator wrapped {wraps} times)")
        for i in bad[:5]:
            print(f"    cycle {i}: rtl={got[i] if 0 <= i < len(got) else '-'} model={expected[i]}")
        sys.exit(0 if not bad else 1)
    elif cmd == "gen-array":
        tests = write_array_vectors(n, BUILD / f"array_vectors_N{n}.txt")
        print(f"wrote build/array_vectors_N{n}.txt ({len(tests)} tests)")
    elif cmd == "check-array":
        tests, _ = array_tests(n)
        rows = (RESULTS / f"array_dump_N{n}.txt").read_text().split("\n")
        rows = [r for r in rows if r.strip()]
        ok, n_im, im_ok = True, 0, True
        if len(rows) != len(tests):
            print(f"array N={n:2d}: FAIL  {len(rows)} dumps for {len(tests)} tests")
            sys.exit(1)
        for t, ((kind, A, B, coords), row) in enumerate(zip(tests, rows)):
            got = np.array(row.split(), dtype=np.int64).reshape(ROWS, COLS)
            exp = array_golden(np.asarray(A), np.asarray(B), n)
            if not np.array_equal(got, exp):
                ok = False
                print(f"  test {t} ({kind}, K={A.shape[1]}): mismatch\n  rtl=\n{got}\n  model=\n{exp}")
            if kind == "im2col":
                # Same numbers through eval/image_conv.py (the image pipeline itself).
                from eval.image_conv import camera, conv_raw, filter_bank
                if n_im == 0:
                    raw = conv_raw(camera(), filter_bank()[0], lut(n))
                n_im += 1
                ref = np.array([raw[r, c] for r, c in coords])
                if not np.array_equal(got[:, 0], ref):
                    im_ok = ok = False
                    print(f"  test {t}: array column 0 != image_conv raw output {got[:, 0]} vs {ref}")
        print(f"array  N={n:2d}: {'PASS' if ok else 'FAIL'}  {len(tests)} tests "
              f"({len(tests) - n_im} matrix, {n_im} image-patch tiles = {n_im * ROWS} pixels; "
              f"image pixels {'match' if im_ok else 'DO NOT match'} eval/image_conv.py)")
        sys.exit(0 if ok else 1)
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
