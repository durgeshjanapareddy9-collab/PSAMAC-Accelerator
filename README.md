# AxMAC: a precision-scalable approximate MAC accelerator

An 8x8 unsigned Dadda multiplier whose amount of approximation is a single
compile-time parameter **N**: the partial products in the lowest N columns
are dropped (N = 0 is exact). We verify it exhaustively against a Python
model, measure area, delay and power on SkyWater 130 nm with open-source
tools only (Yosys, OpenSTA), put it in a MAC unit and a 4x4 systolic
array, evaluate it on image filtering and a small CNN, run published
approximate multipliers (EvoApproxLib, TruMD) through the identical flow, and
test which error metric best predicts application quality.

The multiplier architecture, a Dadda tree with truncated low-order columns,
is existing prior art. Our contribution is the parameterization, the
open-flow characterization and the evaluation framework.

Status: Milestone 2 complete. Repository: https://github.com/durgeshjanapareddy9-collab/PSAMAC-Accelerator-

- **Final report:** [docs/report.md](docs/report.md)
- **Complete project guide** (every step, every result, code line by line,
  technologies, AI-accelerator Q&A): [docs/project_guide.md](docs/project_guide.md)
- Milestone 1 submission: [docs/milestone1.md](docs/milestone1.md);
  task list: [PROGRESS.md](PROGRESS.md)

## Quick start: reproduce every result and figure

```bash
make venv venv-cnn   # Python packages (+ CPU PyTorch for the CNN)
make all             # everything below, in order (about 20 minutes on a 16-core laptop, plus package downloads)
```

Or step by step:

```bash
make test       # Python model tests + RTL vs model, all 65,536 pairs, N = 0..12
make test-mac   # MAC PE and 4x4 systolic array vs the LUT golden model, N = 0..12
make metrics    # error metrics vs N            -> results/error_metrics.csv
make synth      # area + delay, N = 0 vs a*b     -> results/ppa_n0.csv
make power      # power, N = 0 vs a*b (random)   -> results/power_n0.csv
make sweep      # area, delay, power (random + image data) for every N -> results/ppa_mul.csv
make array      # the 4x4 array: area, delay, power, energy/MAC per N  -> results/ppa_array.csv
make report     # Milestone 1 flow results       -> results/flow_report.md
make baselines  # verify EvoApproxLib + TruMD, metrics, identical flow -> results/ppa_baselines.csv
make eval       # image PSNR/SSIM, every design  -> results/image_quality.csv
make cnn        # MNIST top-1, every design      -> results/cnn_accuracy.csv
make analysis   # which metric predicts quality  -> results/metric_correlation.csv
make figures    # every figure (plots + diagrams) -> figures/  (needs make sweep netlists)
make diagrams   # circuit diagrams (Yosys show)  -> figures/yosys_*.svg/.png
make versions   # tool versions                  -> results/tool_versions.txt
make gen        # regenerate rtl/dadda8_reduce.sv
```

`make sim-N4` / `make check-N4` / `make pe-N4` / `make array-N4` run a single N.
`python scripts/m2_tables.py` prints every Milestone 2 table from the CSVs.
The array power step simulates 13 large netlists; `JOBS=8` (default) sets how
many run in parallel.

## Requirements

Versions we used are in `results/tool_versions.txt`.

| Tool | Used for | Notes |
|---|---|---|
| Yosys (0.33) | synthesis | `apt install yosys` |
| OpenSTA | timing and power | built from source, see below |
| Icarus Verilog (12) | RTL and gate-level simulation | `apt install iverilog` |
| sky130A PDK | standard cells `sky130_fd_sc_hd` | installed with [volare](https://github.com/efabless/volare) into `~/.volare` |
| Python 3.10+ | model, metrics, flow scripts, evaluation | `make venv` |
| PyTorch + torchvision (CPU) | training the CNN only | `make venv-cnn` (requirements-cnn.txt) |
| gcc | compiling EvoApproxLib's C models | `apt install gcc` |
| git + network | `make baselines` downloads TruMD; `make cnn` downloads MNIST | |

All tool and library paths are set in one place, **`flow/config.mk`**. If
yours differ, edit that file or override on the command line
(`make synth PDK_ROOT=/foss/pdks`).

The IIC-OSIC-TOOLS Docker image also contains all of these tools.

### Building OpenSTA (Ubuntu 24.04)

There is no prebuilt OpenSTA/OpenROAD package for Ubuntu 24.04, so we build
it from source into `~/tools` (no sudo needed after the first line):

```bash
sudo apt install cmake g++ bison flex swig tcl-dev libeigen3-dev zlib1g-dev

mkdir -p ~/tools/src && cd ~/tools/src

# CUDD (BDD library OpenSTA needs)
git clone https://github.com/cuddorg/cudd.git && cd cudd && git checkout 3.0.0
touch aclocal.m4 && sleep 1 && touch configure Makefile.in config.h.in   # stop make from re-running automake
./configure --prefix=$HOME/tools/cudd && make -j && make install && cd ..

# OpenSTA
git clone https://github.com/parallaxsw/OpenSTA.git && cd OpenSTA
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/tools/opensta -DCUDD_DIR=$HOME/tools/cudd -DCMAKE_BUILD_TYPE=Release
make -j && make install          # -> ~/tools/opensta/bin/sta
```

## How the flow works

1. `scripts/gen_dadda.py` plans the Dadda reduction tree, checks it in Python
   on all 65,536 inputs for every N, and writes `rtl/dadda8_reduce.sv`.
2. `rtl/approx_mul.sv` ties the dropped partial products to 0 and feeds the
   rest into the tree. The tree is the same for every N; synthesis removes the
   adders whose inputs are constant.
3. `tb/tb_approx_mul_exhaustive.sv` dumps all 65,536 outputs;
   `model/check_rtl.py` compares them with `model/approx_mul.py`.
4. `flow/ppa.py` synthesizes `rtl/mul_wrap.sv` (flip-flops around the
   multiplier) with `flow/synth.tcl` and times it with `flow/sta.tcl`
   (constraints in `flow/mul_wrap.sdc`): area, cell counts, Yosys `check`,
   setup and hold slack, WNS/TNS and the max/min path delays.
5. `flow/power.py` simulates the gate-level netlist (`tb/tb_power.sv`) and runs
   `flow/power.tcl` (OpenSTA `read_vcd` + `report_power`).
6. `flow/report.py` collects the results of steps 4 and 5 into
   `results/flow_report.md` (setup, summary, tables, limitations).
7. `rtl/mac_pe.sv` and `rtl/systolic_array.sv` are tested by `model/systolic.py`
   (vectors, golden model, checker) through `tb/tb_mac_pe.sv` and
   `tb/tb_systolic_array.sv`.
8. `baselines/verify.py` checks each baseline against its own model;
   `model/designs.py` is the registry of all designs and their LUTs.
9. `eval/image_conv.py`, `eval/cnn.py` and `eval/analysis.py` run the
   applications and the metric analysis; `eval/figures.py` draws every plot.

`docs/project_guide.md` explains each step in detail.

Corner: `sky130_fd_sc_hd__tt_025C_1v80` (typical, 25 °C, 1.80 V), pre-layout.

## Repository layout

```
rtl/        full_adder.sv, half_adder.sv, dadda8_reduce.sv (generated), approx_mul.sv,
            mul_behav.sv (a*b), mul_wrap.sv (registered wrapper), mac_pe.sv, systolic_array.sv
scripts/    gen_dadda.py, tool_versions.sh, md_tables.py, m2_tables.py
tb/         tb_approx_mul_exhaustive.sv, tb_mac_pe.sv, tb_systolic_array.sv,
            tb_power.sv, tb_array_power.sv
model/      approx_mul.py, designs.py, check_rtl.py, metrics.py, systolic.py, tests/
flow/       config.mk, synth.tcl, sta.tcl, power.tcl, mul_wrap.sdc, ppa.py, power.py,
            stimulus.py, make_synth_lib.py, report.py, diagrams.py
baselines/  evoapprox/ (4 designs, MIT), trumd/fetch.sh, baseline_wrappers.sv, verify.py, README.md
eval/       image_conv.py, cnn.py, analysis.py, figures.py
results/    generated CSVs and reports (never hand-edited); cnn_lenet_mnist.pt = trained weights
figures/    generated plots
docs/       report.md, project_guide.md, milestone1.md, REFERENCES.md
```

Large regenerable files (LUT `.npy`, RTL dumps, VCDs, netlists) are not
committed; `make` recreates them.

## Credits

References are listed in [docs/REFERENCES.md](docs/REFERENCES.md).

Baselines (details in [baselines/README.md](baselines/README.md)):

- **EvoApproxLib** (https://github.com/ehw-fit/evoapproxlib), MIT licence:
  `baselines/evoapprox/mul8u_{1446,GS2,185Q,JQQ}.{v,c}` are copied unchanged,
  with the licence in `baselines/evoapprox/LICENSE.md`. Cite Mrazek et al.,
  DATE 2017.
- **TruMD** (https://github.com/Hassan313/Approximate-Multiplier): the
  repository has no licence, so its code is **not** copied;
  `baselines/trumd/fetch.sh` downloads it at a pinned commit. The repository
  asks to cite Afzali-Kusha et al., IEEE TVLSI 2020.

Test images come from scikit-image (`skimage.data`); MNIST is downloaded by
torchvision.
