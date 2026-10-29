"""Verify every baseline multiplier and build its LUT (M2.6).

For each published baseline with RTL (EvoApproxLib, TruMD):
  1. simulate its own Verilog on all 65,536 input pairs, through our uniform
     wrapper (baselines/baseline_wrappers.sv) and the same exhaustive
     testbench as our multiplier -> results/rtl_dump_<name>.txt;
  2. compare with its reference model (model/check_rtl.py):
       EvoApproxLib -> the library's own C model,
       TruMD        -> its description "k lowest columns truncated" (= our lut(k));
  3. save its LUT to results/lut_<name>.npy (model/designs.py).
The functional-only reconstruction (recon_rather_d1) just gets its LUT.

Writes results/baseline_verification.csv. Exits with status 1 if an
EvoApproxLib design does not match its C model (that would mean the Verilog
or our port mapping is wrong). A TruMD mismatch is reported, not fatal: it
would mean TruMD is not plain truncation, which is a finding, not a bug.

Usage:  python baselines/verify.py --iverilog iverilog --vvp vvp   (via `make baselines`)
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model import designs                 # noqa: E402
from model.check_rtl import reference      # noqa: E402

import numpy as np                         # noqa: E402


def simulate(d, args):
    """Exhaustive RTL simulation of design d -> results/rtl_dump_<name>.txt."""
    build = REPO / "build"
    build.mkdir(exist_ok=True)
    vvp = build / f"tb_mul_{d.name}.vvp"
    cmd = [args.iverilog, "-g2012", "-Wall", "-Wno-timescale", "-Wno-implicit",
           f"-DMUL={d.module}", *[f"-D{x}" for x in d.defines],
           f'-Ptb_approx_mul_exhaustive.TAG="{d.name}"', "-o", str(vvp),
           "tb/tb_approx_mul_exhaustive.sv", *d.srcs]
    for c in (cmd, [args.vvp, "-n", str(vvp)]):
        r = subprocess.run(c, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"ERROR: {' '.join(c)}\n{r.stdout}{r.stderr}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iverilog", default="iverilog")
    p.add_argument("--vvp", default="vvp")
    args = p.parse_args()

    subprocess.run(["bash", "baselines/trumd/fetch.sh"], cwd=REPO, check=True)

    rows, fatal = [], False
    for d in designs.baselines():
        if d.has_rtl:
            simulate(d, args)
            name, dump, ref = reference(d.name)
            data = np.loadtxt(dump, dtype=np.int64)
            got = np.zeros((256, 256), dtype=np.int64)
            got[data[:, 0], data[:, 1]] = data[:, 2]
            bad = int(np.count_nonzero(got != ref))
            ok = bad == 0 and len(data) == 65536
            refname = "own C model" if d.family == "evo" else \
                f"k-column truncation formula (lut({d.name.split('_')[1]}))"
            print(f"{d.name:16s} RTL vs {refname}: {'PASS' if ok else 'FAIL'} "
                  f"({65536 - bad}/65536 match)")
            if d.family == "evo" and not ok:
                fatal = True
        else:
            refname, bad, ok = "none (software-only model)", "", ""
            print(f"{d.name:16s} software-only model, no RTL")
        table = designs.get_lut(d.name)           # builds and caches the LUT
        rows.append({"design": d.name, "family": d.family, "label": d.label,
                     "has_rtl": d.has_rtl, "reference_model": refname,
                     "mismatches": bad, "pass": ok,
                     "lut_min": int(table.min()), "lut_max": int(table.max())})

    out = REPO / "results" / "baseline_verification.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(REPO)}")
    sys.exit(1 if fatal else 0)


if __name__ == "__main__":
    main()
