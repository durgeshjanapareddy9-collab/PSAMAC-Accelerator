"""Registry of every multiplier design the project evaluates.

One place lists all designs, so the error metrics, image evaluation, CNN,
hardware flow and analysis all use the same set with the same names.

Each design has
    name     short identifier used in every CSV (e.g. trunc_N4, evo_1446)
    family   "trunc" (ours), "evo" (EvoApproxLib), "trumd" (TruMD), "recon"
    label    human-readable name for plots and tables
    module   RTL module with the uniform interface (a, b -> p, parameter N),
             or None if the design exists only as a software model
    N        truncation depth for our designs (passed to the RTL), else 0
    srcs     RTL files to compile (besides rtl/mul_wrap.sv for the flow)
    defines  Verilog macros needed to compile it
    lut()    256x256 table, lut[a, b] = design output for inputs a, b

Where each LUT comes from:
    trunc_N*  model/approx_mul.py (formula; verified against our RTL)
    evo_*     EvoApproxLib's own C model, compiled with gcc and run on all
              65,536 pairs (and verified against their Verilog: make baselines)
    trumd_*   TruMD has no software model, so its LUT is the exhaustive
              simulation of its own Verilog (results/rtl_dump_trumd_<k>.txt)
    recon_rather_d1
              functional-only reconstruction: N = 4 truncation plus a
              constant +12. It reproduces EDmax = 37 for 256 pairs, the values
              Rather et al. 2025 report for their Design 1. This is a sanity
              check, NOT a claim that it is their circuit; it has no RTL.
"""

import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from model.approx_mul import N_MAX, REPO, RESULTS, lut as trunc_lut

BUILD = REPO / "build"
EVO_DIR = REPO / "baselines" / "evoapprox"
TRUMD_DIR = REPO / "baselines" / "trumd" / "upstream"
WRAPPERS = "baselines/baseline_wrappers.sv"
MUL_SRCS = ["rtl/full_adder.sv", "rtl/half_adder.sv", "rtl/dadda8_reduce.sv", "rtl/approx_mul.sv"]

# EvoApproxLib designs we picked (see baselines/README.md for why).
EVO_NAMES = ["1446", "GS2", "185Q", "JQQ"]
# TruMD truncation depths available upstream, and their top-module names.
TRUMD_TOPS = {2: "Dadda_8bits_Trunc_2.v", 4: "Dadda_8bits_Trunc_4.v",
              6: "Dadda_8bits_Trunc_6.v", 8: "Dadda_8bits_Trunc_8.v"}


@dataclass
class Design:
    name: str
    family: str
    label: str
    module: str | None = None
    N: int = 0
    srcs: list = field(default_factory=list)
    defines: list = field(default_factory=list)

    @property
    def has_rtl(self):
        return self.module is not None

    def lut(self):
        return get_lut(self.name)


def _designs():
    ds = []
    for n in range(N_MAX + 1):
        ds.append(Design(f"trunc_N{n}", "trunc", f"Trunc N={n}", "approx_mul", n, MUL_SRCS))
    for e in EVO_NAMES:
        ds.append(Design(f"evo_{e}", "evo", f"EvoApprox mul8u_{e}", f"evo_{e}", 0,
                         [f"baselines/evoapprox/mul8u_{e}.v", WRAPPERS], [f"BASE_evo_{e}"]))
    for k, top in TRUMD_TOPS.items():
        folder = f"baselines/trumd/upstream/Dadda_Exact_8b_Trunc_{k}/OriginalFiles"
        srcs = [f"{folder}/{f}" for f in ("AND2.v", "HA.v", "FA.v", "COMP4to2.v", top)]
        ds.append(Design(f"trumd_{k}", "trumd", f"TruMD trunc {k}", f"trumd_{k}", 0,
                         srcs + [WRAPPERS], [f"BASE_trumd_{k}"]))
    ds.append(Design("recon_rather_d1", "recon", "Recon. Rather D1 (N=4, +12)"))
    return ds


DESIGNS = _designs()
BY_NAME = {d.name: d for d in DESIGNS}


def get(name):
    return BY_NAME[name]


def baselines():
    """The published designs (EvoApproxLib, TruMD) plus the reconstruction."""
    return [d for d in DESIGNS if d.family != "trunc"]


# ---- LUTs ---------------------------------------------------------------------

def _evo_lut_from_c(e):
    """Compile EvoApproxLib's C model and evaluate it on all 65,536 pairs."""
    BUILD.mkdir(exist_ok=True)
    driver = BUILD / f"evo_{e}_lut.c"
    exe = BUILD / f"evo_{e}_lut"
    # The C function is declared  uint64_t mul8u_X(const uint64_t B, const uint64_t A)
    # and uses B and A exactly as the Verilog ports B and A, so f(b, a) = output
    # for (A = a, B = b). The exhaustive RTL check confirms this mapping.
    driver.write_text(
        f'#include <stdio.h>\n#include <stdint.h>\n'
        f'#include "{EVO_DIR / f"mul8u_{e}.c"}"\n'
        f'int main(void) {{\n'
        f'    for (int a = 0; a < 256; a++)\n'
        f'        for (int b = 0; b < 256; b++)\n'
        f'            printf("%llu\\n", (unsigned long long) mul8u_{e}(b, a));\n'
        f'    return 0;\n}}\n')
    subprocess.run(["gcc", "-O1", "-w", "-o", str(exe), str(driver)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return np.array(out.split(), dtype=np.int64).reshape(256, 256)


def _lut_from_dump(name):
    """LUT from an exhaustive RTL simulation dump (lines 'a b p')."""
    path = RESULTS / f"rtl_dump_{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make baselines` first")
    data = np.loadtxt(path, dtype=np.int64)
    table = np.zeros((256, 256), dtype=np.int64)
    table[data[:, 0], data[:, 1]] = data[:, 2]
    return table


@cache
def get_lut(name):
    """256x256 LUT for a design, cached in results/lut_<name>.npy (except ours,
    which model/approx_mul.py caches as results/lut_N<n>.npy)."""
    d = BY_NAME[name]
    if d.family == "trunc":
        return trunc_lut(d.N)
    path = RESULTS / f"lut_{name}.npy"
    if path.exists():
        return np.load(path)
    if d.family == "evo":
        table = _evo_lut_from_c(name.removeprefix("evo_"))
    elif d.family == "trumd":
        table = _lut_from_dump(name)
    elif d.family == "recon":
        table = trunc_lut(4) + 12
    else:
        raise ValueError(name)
    RESULTS.mkdir(exist_ok=True)
    np.save(path, table)
    return table
