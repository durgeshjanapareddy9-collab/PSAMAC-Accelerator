"""Power of synthesized designs: gate-level simulation + OpenSTA report_power.

For each design (whose netlist flow/ppa.py already produced):
  1. simulate the gate-level netlist with the sky130 cell models and record a
     VCD (every wire's switching):
       multipliers (mul_wrap): tb/tb_power.sv, one new (a, b) pair per cycle
       the array (systolic_array): tb/tb_array_power.sv, image tiles
  2. check the netlist's outputs: every product (multipliers) or every sum
     (array) must equal the design's LUT model -- so the synthesized gates are
     verified for EVERY design, exact or approximate;
  3. run OpenSTA (flow/power.tcl): read_vcd, then report_power;
  4. write one CSV row per design, and optionally add the power columns to a
     PPA table (--merge-into).

Stimulus (--stimulus):
  random  uniform random 8-bit a and b (Milestone 1; multipliers only)
  image   real filter data from flow/stimulus.py: a = pixel, b = weight (Milestone 2)

Every design uses the same clock (10 ns), number of cycles, data and corner,
so the numbers can be compared with each other.

Energy per operation = total power x clock period / operations per cycle:
  multiplier: one multiply per cycle;
  array: useful MACs only (16 PEs x 25 taps per tile), so the clear and drain
         cycles a tile needs are charged to it.

Usage (normally via `make power` / `make sweep` / `make baselines`):
  python flow/power.py --out results/power_n0.csv --design approx_mul:0 --design mul_behav:0 \\
      --lib <lib> --sta <sta> --cell-models <sky130 verilog dir> [--stimulus image]
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppa import REPO, SYNTH_OUT, resolve, run   # noqa: E402  (same helpers as the area/delay flow)
import stimulus                                  # noqa: E402

from model import designs                        # noqa: E402
from model.approx_mul import exact_lut, lut      # noqa: E402
from model.systolic import ROWS, COLS, array_golden   # noqa: E402

BUILD = REPO / "build"
# Array stimulus length: 50 tiles x 39 cycles = 1,950 cycles. The gate-level
# simulation of the 6,800-cell array writes a large VCD and runs ~5 s per tile,
# so this is a compromise between run time and statistics (the multiplier gets
# 5,000 cycles).
ARRAY_TILES = 50


def design_lut(r):
    """The LUT a mul_wrap design must reproduce."""
    if r["module"] == "approx_mul":
        return lut(r["N"])
    if r["module"] == "mul_behav":
        return exact_lut()
    return designs.get_lut(r["design"])


def simulate_mul(r, netlist, vcd, args):
    """Gate-level simulation of mul_wrap; returns (cycles, ops/cycle, check text)."""
    tag = f"{r['tag']}_{args.stimulus}"
    vvp, dump = BUILD / f"{tag}_power.vvp", BUILD / f"{tag}_out.txt"
    # Only exact designs can be checked against a*b inside the testbench.
    exact = r["module"] == "mul_behav" or (r["module"] == "approx_mul" and r["N"] == 0)
    stim = ""
    if args.stimulus == "image":
        stim = str(stimulus.write_mul(args.cycles + 20, r["tag"]).relative_to(REPO))
    models = Path(args.cell_models)
    run([args.iverilog, "-g2012", "-DFUNCTIONAL", "-DUNIT_DELAY=#1",
         f"-Ptb_power.CYCLES={args.cycles}", f"-Ptb_power.CLK_NS={args.clk_ns}",
         f"-Ptb_power.SEED={args.seed}", f"-Ptb_power.CHECK_EXACT={int(exact)}",
         f'-Ptb_power.VCD_FILE="{vcd.relative_to(REPO)}"',
         f'-Ptb_power.STIM_FILE="{stim}"',
         f'-Ptb_power.DUMP_FILE="{dump.relative_to(REPO)}"',
         "-o", str(vvp), "tb/tb_power.sv", str(netlist),
         str(models / "primitives.v"), str(models / "sky130_fd_sc_hd.v")],
        {}, BUILD / f"{tag}_iverilog.log")
    sim = run([args.vvp, "-n", str(vvp)], {}, BUILD / f"{tag}_sim.log")
    if exact:
        m = re.search(r"gate-level check (\d+)/(\d+) outputs correct", sim)
        if not m or m.group(1) != m.group(2):
            sys.exit(f"ERROR: gate-level simulation of {tag} gives wrong products:\n{sim}")
    # Every output vs the design's own LUT.
    d = np.loadtxt(dump, dtype=np.int64)
    bad = int(np.count_nonzero(design_lut(r)[d[:, 0], d[:, 1]] != d[:, 2]))
    if bad:
        sys.exit(f"ERROR: gate-level netlist of {tag}: {bad}/{len(d)} products differ from its LUT")
    return args.cycles, 1, f"{len(d)}/{len(d)} products match LUT"


def simulate_array(r, netlist, vcd, args):
    """Gate-level simulation of the array on image tiles."""
    tag = f"{r['tag']}_{args.stimulus}"
    vvp, dump = BUILD / f"{tag}_power.vvp", BUILD / f"{tag}_out.txt"
    stim, tiles = stimulus.write_array(ARRAY_TILES, r["tag"])
    models = Path(args.cell_models)
    run([args.iverilog, "-g2012", "-DFUNCTIONAL", "-DUNIT_DELAY=#1",
         f"-Ptb_array_power.CLK_NS={args.clk_ns}",
         f'-Ptb_array_power.STIM_FILE="{stim.relative_to(REPO)}"',
         f'-Ptb_array_power.DUMP_FILE="{dump.relative_to(REPO)}"',
         f'-Ptb_array_power.VCD_FILE="{vcd.relative_to(REPO)}"',
         "-o", str(vvp), "tb/tb_array_power.sv", str(netlist),
         str(models / "primitives.v"), str(models / "sky130_fd_sc_hd.v")],
        {}, BUILD / f"{tag}_iverilog.log")
    sim = run([args.vvp, "-n", str(vvp)], {}, BUILD / f"{tag}_sim.log")
    cycles = int(re.search(r"recorded (\d+) cycles", sim).group(1))
    got = np.loadtxt(dump, dtype=np.int64, ndmin=2)
    exp = np.stack([array_golden(A, B, r["N"]).ravel() for _, A, B, _ in tiles])
    if got.shape != exp.shape or not np.array_equal(got, exp):
        sys.exit(f"ERROR: gate-level array netlist {tag} sums differ from the LUT golden model")
    macs = ARRAY_TILES * ROWS * COLS * tiles[0][1].shape[1]
    return cycles, macs / cycles, f"{exp.size}/{exp.size} sums match LUT golden model"


def power_one(spec, args):
    r = resolve(spec)
    netlist = SYNTH_OUT / f"{r['tag']}.v"
    if not netlist.exists():
        sys.exit(f"ERROR: {netlist} not found -- run the synthesis step first")
    BUILD.mkdir(exist_ok=True)
    vcd = BUILD / f"{r['tag']}_{args.stimulus}.vcd"
    sim = simulate_array if r["top"] == "systolic_array" else simulate_mul
    cycles, ops_per_cycle, check = sim(r, netlist, vcd, args)

    # Power analysis.
    out = run([args.sta, "-no_init", "-no_splash", "-exit", "flow/power.tcl"],
              {"NETLIST": str(netlist), "LIB": args.lib, "VCD": str(vcd),
               "SCOPE": "tb_array_power/dut" if r["top"] == "systolic_array" else "tb_power/dut",
               "CLK_NS": str(args.clk_ns), "TOP": r["top"],
               "OUT": str(SYNTH_OUT / f"{r['tag']}_{args.stimulus}")},
              SYNTH_OUT / f"{r['tag']}_{args.stimulus}_power.log")
    ann = re.search(r"^vcd\s+(\d+)\s*\nunannotated\s+(\d+)", out, re.M)
    if not ann or int(ann.group(2)) != 0:
        sys.exit(f"ERROR: not every pin got simulation activity for {r['tag']}:\n{out}")

    # Rows look like: "Combinational  7.1e-04  4.9e-04  8.4e-10  1.2e-03  83.5%"
    def row(group):
        m = re.search(rf"^{group}\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", out, re.M)
        if not m:
            sys.exit(f"ERROR: no '{group}' row in report_power output for {r['tag']}")
        return [float(x) for x in m.groups()]      # internal, switching, leakage, total (W)

    seq, comb, tot = row("Sequential"), row("Combinational"), row("Total")
    uw = 1e6                                          # watts -> microwatts
    vcd.unlink()                                      # VCDs are large; the report is kept
    return {
        "design": r["design"],
        "module": r["module"],
        "N": r["N"],
        "clk_ns": args.clk_ns,
        "cycles": cycles,
        "stimulus": "uniform_random" if args.stimulus == "random" else "image",
        "annotated_pins": int(ann.group(1)),
        "power_comb_uW": round(comb[3] * uw, 3),
        "power_seq_uW": round(seq[3] * uw, 3),
        "power_total_uW": round(tot[3] * uw, 3),
        "switching_uW": round(tot[1] * uw, 3),
        "internal_uW": round(tot[0] * uw, 3),
        "leakage_uW": round(tot[2] * uw, 6),
        # power (W) x period (s) / operations = joules per operation -> pJ
        "energy_pJ_per_op": round(tot[3] * args.clk_ns * 1e-9 / ops_per_cycle * 1e12, 4),
        "ops_per_cycle": round(ops_per_cycle, 4),
        "gate_level_check": check,
    }


def merge_into(path, rows, stim):
    """Add this stimulus's power columns to a PPA CSV (matched by design name)."""
    ppa = pd.read_csv(path)
    p = pd.DataFrame(rows).set_index("design")
    for col, src in [(f"power_{stim}_total_uW", "power_total_uW"),
                     (f"power_{stim}_comb_uW", "power_comb_uW"),
                     (f"power_{stim}_seq_uW", "power_seq_uW"),
                     (f"energy_{stim}_pJ_per_op", "energy_pJ_per_op")]:
        ppa[col] = ppa["design"].map(p[src])
    ppa.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--design", action="append", required=True,
                   help="approx_mul:N, mul_behav:0, a baseline name, or array:N (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--merge-into", help="PPA CSV to add the power columns to")
    p.add_argument("--lib", required=True)
    p.add_argument("--sta", required=True)
    p.add_argument("--cell-models", required=True, help="sky130_fd_sc_hd/verilog directory")
    p.add_argument("--iverilog", default="iverilog")
    p.add_argument("--vvp", default="vvp")
    p.add_argument("--stimulus", choices=["random", "image"], default="random")
    p.add_argument("--clk-ns", type=float, default=10.0)
    p.add_argument("--cycles", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--jobs", type=int, default=1, help="designs simulated in parallel")
    args = p.parse_args()

    # Each design runs its own simulator and OpenSTA processes with its own
    # files, so several can run at once. Rows keep the order of --design.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda d: power_one(d, args), args.design))
    for r in rows:
        print(", ".join(f"{k}={v}" for k, v in r.items()))
    with open(REPO / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    if args.merge_into:
        merge_into(REPO / args.merge_into, rows, args.stimulus)
        print(f"added {args.stimulus} power columns to {args.merge_into}")


if __name__ == "__main__":
    main()
