# AxMAC: The Complete Project Guide

*A Precision-Scalable Approximate MAC Accelerator with Open-Source ASIC Characterization*
AI Accelerators course project, IIIT Dharwad, 2026.

This is the one document that explains the whole project: what we built, why,
every step of the flow, every result and how it was obtained, the structure of
the repository, the important code line by line, the technologies used, and
the questions an AI-accelerator examiner is likely to ask, with answers.

Every number in this guide is copied from a CSV in `results/` (printed by
`python scripts/m2_tables.py`, or `scripts/md_tables.py` for Milestone 1).
None was typed by hand. The formal course report is `docs/report.md`.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background you need (beginner level)](#2-background-you-need-beginner-level)
3. [Technologies used](#3-technologies-used)
4. [Structure of the repository](#4-structure-of-the-repository)
5. [The flow, step by step](#5-the-flow-step-by-step)
6. [Results and how each was obtained](#6-results-and-how-each-was-obtained)
7. [Important code, line by line](#7-important-code-line-by-line)
8. [AI-accelerator questions and answers](#8-ai-accelerator-questions-and-answers)
9. [Glossary](#9-glossary)
10. [What we did, in order](#10-what-we-did-in-order)

---

## 1. The project in one page

**The problem.** Running a neural network or an image filter is mostly
multiply-accumulate (MAC): `acc = acc + a * b`, billions of times. The
multiplier is the biggest and most power-hungry part of a MAC unit. Many AI
and image tasks tolerate small arithmetic errors, so a slightly *wrong* but
much *cheaper* multiplier (an "approximate multiplier") can save area and
energy.

**What we built.**

1. An 8x8 unsigned **Dadda-tree multiplier** whose approximation is one
   compile-time knob **N**: the lowest N columns of partial products are dropped
   (N = 0 is exact, N = 12 is very rough).
2. A **MAC processing element** (multiplier + 24-bit accumulator) and a **4x4
   output-stationary systolic array** of 16 such PEs, built from scratch.
3. A **bit-exact Python model** (a 256x256 look-up table per N) that every other
   part is checked against.
4. A fully **open-source ASIC flow** on SkyWater 130 nm: Yosys (synthesis),
   OpenSTA (timing and power), Icarus Verilog (simulation).
5. **Application evaluation**: image filtering (PSNR, SSIM) and a small CNN on
   MNIST (top-1 accuracy), every multiply going through the design's LUT.
6. **Baselines**: 4 EvoApproxLib multipliers and 4 TruMD multipliers through the
   identical flow.
7. An **analysis** of which error metric best predicts application quality.

**Contribution (exact wording we use).** The multiplier architecture, a Dadda
tree with truncated low-order columns, is existing prior art. Our contribution
is the parameterization, the open-flow characterization and the evaluation
framework.

**Headline results** (sky130 tt corner, pre-layout, relative to our exact N = 0):

| | N = 4 | N = 8 |
|---|---|---|
| multiplier area | -12.1 % | -54.8 % |
| multiplier logic power (image data) | -26.9 % | -83.2 % |
| 4x4 array area | -13.0 % | -42.2 % |
| 4x4 array energy per MAC | 3.643 pJ (-16 %) | 2.405 pJ (-44.6 %) |
| camera image PSNR / SSIM | 44.8 dB / 0.9960 | 19.5 dB / 0.8418 |
| MNIST CNN top-1 change | -0.01 pp | -0.22 pp |

(N = 0 array: 4.340 pJ per MAC.) The CNN tolerates N = 8 almost for free; image
filtering needs N <= 5 to stay visually lossless.

**Main analysis finding.** Over all 65,536 inputs, the *error rate* ranked the
designs' image quality best; "EDmax and its frequency" did not. But the best
predictor by far was the mean error *on the inputs the workload actually uses*
(rank correlation >= 0.99), because real workloads use only a small part of
the input space.

---

## 2. Background you need (beginner level)

### 2.1 How a multiplier adds things up

Multiplying two 8-bit numbers a and b is like long multiplication in school,
but in binary. Each bit of b selects a shifted copy of a:

```
                a7 a6 a5 a4 a3 a2 a1 a0        (a)
              x b7 b6 b5 b4 b3 b2 b1 b0        (b)
   ---------------------------------------
                         a7b0 ... a1b0 a0b0     row 0 (b0 selects a)
                    a7b1 ... a1b1 a0b1          row 1, shifted left by 1
                ...
   a7b7 ... a0b7                                row 7, shifted left by 7
```

Each little product `a[j] AND b[i]` is a **partial product** pp(i, j). It is
one bit, and it is worth 2^(i+j). All partial products with the same i + j sit
in the same **column** and have the same weight. There are 64 partial products
in 15 columns (0..14); column heights are 1, 2, 3, ..., 8, ..., 3, 2, 1. The
product is the sum of all of them.

### 2.2 Full adders, half adders and the Dadda tree

You cannot add a column of 8 bits in one step. A **full adder (FA)** takes 3
bits of a column and outputs their sum as 2 bits: a *sum* bit that stays in the
column and a *carry* bit that moves to the next column. So one FA makes a
column 2 bits shorter. A **half adder (HA)** takes 2 bits and makes the column
1 bit shorter.

A **Dadda tree** [5] reduces all columns to height 2 in a few stages, using as
few adders as possible: each stage only shrinks each column to the next target
height in the sequence 2, 3, 4, 6, 9, ... For an 8x8 multiplier the targets are
6, 4, 3, 2 (four stages, 35 FAs and 7 HAs). After that, two 16-bit rows
remain, and one ordinary adder (the "final carry-propagate adder") produces the
16-bit product. Wallace trees [6] do the same job but reduce as early as
possible; Dadda uses fewer adders.

### 2.3 Truncation: our approximation knob N

We simply never generate the partial products in the lowest N columns (those
with i + j < N): they are tied to 0. That removes the ANDs and, after
synthesis, every adder that only had zeros as inputs.

- N = 0: nothing dropped, exact.
- The result can only get *smaller*, never larger (all dropped bits had positive
  weight). Errors are one-sided: the mean error (bias) equals -MED.
- The worst case (EDmax) happens when every dropped bit is 1, so
  EDmax = sum of 2^(i+j) over the dropped positions; e.g. N = 4: 1 + 2x2 + 3x4 +
  4x8 = 49.
- On uniformly random inputs each partial product is 1 with probability 1/4, so
  MED = EDmax / 4 (the closed form our tests check).

### 2.4 Error metrics

Computed over all 256 x 256 = 65,536 input pairs, with error = approx - exact
and ED = |error| [9, 10, 11]:

| metric | meaning |
|---|---|
| ER | error rate: fraction of pairs with any error |
| ME | mean (signed) error: the bias |
| MED | mean error distance: average of ED |
| NMED | MED / (255 x 255): MED relative to the largest product |
| MRED | mean relative ED (ED / exact), skipping exact = 0 |
| MSE | mean squared error |
| EDmax | the largest ED |
| EDmax count | how many of the 65,536 pairs hit EDmax |

### 2.5 MAC, systolic arrays and dataflow

A **MAC** unit computes `acc <- acc + a x b` every clock cycle. A dot product
of length K takes K cycles on one MAC. A matrix multiply C = A x B (A is M x K,
B is K x P) needs M x K x P MACs.

A **systolic array** is a grid of MAC processing elements (PEs) where data
moves one step per cycle between neighbours, like blood pumped through the
body (hence "systolic"). Each value read from memory is reused by every PE it
passes, so the array does many MACs per memory access. Google's TPU uses a
256 x 256 systolic array [15].

**Dataflows** say which value stays inside the PE:
- *Output-stationary* (ours): each PE keeps its own output C[i][j] in its
  accumulator; A and B stream through.
- *Weight-stationary* (TPU): each PE keeps one weight; activations stream
  through and partial sums flow down.
- *Input-stationary*: each PE keeps an input.

**Skew.** In our array A enters from the left and B from the top. PE(i, j) must
multiply A[i][k] by B[k][j] *in the same cycle*. Because data takes one cycle
per hop, row i must start i cycles late and column j must start j cycles late.
That staggered start is the **skew**; our array does it internally with small
shift registers.

### 2.6 Quantization (for the CNN)

Neural networks are trained with 32-bit floating point. Hardware accelerators
run them with 8-bit integers ("int8") because integer MACs are much cheaper.
**Post-training quantization** [19] maps each float tensor to integers with a
scale factor: value ≈ scale x integer. Our choices:
- activations (pixels and ReLU outputs, never negative): unsigned 8-bit, 0..255;
- weights: signed int8 in -127..127, one scale per layer ("symmetric");
- the multiplier is unsigned, so a signed weight is handled as
  `sign(w) x LUT[activation, |w|]`;
- after each layer, the 32-bit sum is **requantized** back to 8 bits for the next
  layer.

### 2.7 Image quality: PSNR and SSIM

- **PSNR** (peak signal-to-noise ratio) = 10 log10(255^2 / MSE between two
  images), in dB. Higher is better; identical images give infinity. Above
  about 40 dB differences are invisible; around 30 dB they start to show.
- **SSIM** (structural similarity) [18] compares local brightness, contrast and
  structure; 1 means identical, 0 means unrelated.

We compare each approximate output with the *exact multiplier's* output from
the same pipeline, so the numbers measure only the multiplier's damage.

### 2.8 The ASIC flow in plain words

| term | meaning |
|---|---|
| RTL | Register-transfer level: the hardware described in a language (SystemVerilog) |
| synthesis | turning RTL into a netlist of real logic gates (Yosys) |
| standard cells | the pre-designed gates of a chip process (AND2, full adder, flip-flop...); ours: `sky130_fd_sc_hd` |
| liberty file (.lib) | the cell library's data sheet: area, delay tables, power tables per cell |
| technology mapping | choosing which standard cells implement the logic (ABC inside Yosys) |
| netlist | the list of cells and the wires between them |
| static timing analysis (STA) | computing every path's delay from the .lib tables without simulation (OpenSTA) |
| setup check | the slowest signal must arrive before the next clock edge (minus the flop's setup time) |
| hold check | the fastest signal must not arrive so early that it corrupts the value being captured |
| slack | margin left; negative = violation. WNS/TNS = worst / total negative slack |
| min period, fmax | the shortest clock period that meets setup; fmax = 1 / min period |
| power: internal, switching, leakage | inside the cell / charging the wires it drives / leaking when idle |
| VCD | a file recording every signal change during simulation; gives switching activity |
| corner | process/temperature/voltage condition; ours: tt (typical), 25 °C, 1.80 V |
| pre-layout | before placement and routing: no real wire lengths, ideal clock |

---

## 3. Technologies used

| technology | version | used for | why this one |
|---|---|---|---|
| SystemVerilog (synthesizable subset) | IEEE 1800 | all RTL and testbenches | standard HDL; we use a simple subset that both Icarus and Yosys parse |
| Icarus Verilog (`iverilog`, `vvp`) | 12.0 | RTL simulation, gate-level simulation for power | free, simple, reads both RTL and sky130 cell models |
| Yosys | 0.33 | synthesis, area (`stat -liberty`), structural `check`, circuit diagrams (`show`) | the standard open-source synthesis tool |
| ABC (inside Yosys) | bundled | technology mapping to sky130 cells | Yosys's mapper |
| OpenSTA | 3.1.0 (built from source) | timing (setup/hold, min period) and power (`read_vcd`, `report_power`) | the timing engine of OpenROAD/OpenLane |
| SkyWater SKY130 PDK, `sky130_fd_sc_hd` | open_pdks 0fe599b (via volare) | standard cells, liberty file, cell simulation models | the open, manufacturable 130 nm process |
| Python | 3.12 | model, LUTs, metrics, flow drivers, evaluation, figures | glue language for everything |
| NumPy, pandas, SciPy | 2.5 / 3.0 / 1.18 | LUT arithmetic, CSVs, sparse matrices, Spearman correlation | standard scientific stack |
| scikit-image | 0.26 | test images, noise, PSNR, SSIM | reference implementations of image metrics |
| PyTorch, torchvision (CPU) | 2.14 / 0.29 | training the float CNN, MNIST download | only for training; inference is our own numpy code |
| matplotlib | 3.11 | every figure | — |
| pytest | 9.1 | model unit tests | — |
| GNU Make | — | one command per step, `make all` | reproducibility |
| gcc | — | compiling EvoApproxLib's C models into LUTs | the library ships C models |
| Graphviz `dot` | — | drawing the Yosys `show` diagrams | — |
| Git | — | version control | — |

Exact versions are recorded by `make versions` in `results/tool_versions.txt`.

---

## 4. Structure of the repository

```
PSMAC/
├── Makefile                 one target per step; `make all` runs everything
├── README.md                quick start and requirements
├── PROGRESS.md              task list and session log
├── requirements.txt         Python packages; requirements-cnn.txt adds CPU PyTorch
├── rtl/                     the hardware (SystemVerilog)
│   ├── half_adder.sv        2 bits in -> sum + carry
│   ├── full_adder.sv        3 bits in -> sum + carry
│   ├── dadda8_reduce.sv     the Dadda tree, GENERATED by scripts/gen_dadda.py
│   ├── approx_mul.sv        the multiplier: partial products, N mask, tree, final add
│   ├── mul_behav.sv         reference: plain a*b, Yosys picks the circuit
│   ├── mul_wrap.sv          flip-flops around any multiplier, for timing and power
│   ├── mac_pe.sv            processing element: multiplier + 24-bit accumulator
│   └── systolic_array.sv    4x4 output-stationary array of mac_pe, with skew
├── tb/                      testbenches
│   ├── tb_approx_mul_exhaustive.sv   all 65,536 pairs -> results/rtl_dump_*.txt
│   ├── tb_mac_pe.sv                  plays a vector file, dumps the accumulator
│   ├── tb_systolic_array.sv          plays a cycle schedule, dumps the 16 sums
│   ├── tb_power.sv                   gate-level multiplier sim, records a VCD
│   └── tb_array_power.sv             gate-level array sim, records a VCD
├── model/                   the bit-exact reference (Python)
│   ├── approx_mul.py        formula, 256x256 LUT per N (cached .npy)
│   ├── designs.py           registry of ALL designs (ours, baselines) and their LUTs
│   ├── check_rtl.py         compares an RTL dump with the reference model
│   ├── metrics.py           ER, ME, MED, NMED, MRED, MSE, EDmax (+count)
│   ├── systolic.py          golden models, vectors and checkers for PE and array
│   └── tests/               pytest checks of the model
├── scripts/
│   ├── gen_dadda.py         plans + self-checks the Dadda tree, writes rtl/dadda8_reduce.sv
│   ├── md_tables.py         Milestone 1 tables from the CSVs
│   ├── m2_tables.py         Milestone 2 tables from the CSVs
│   └── tool_versions.sh     writes results/tool_versions.txt
├── flow/                    the ASIC flow
│   ├── config.mk            the ONE place with tool and PDK paths
│   ├── synth.tcl            Yosys script
│   ├── mul_wrap.sdc         clock and I/O constraints
│   ├── sta.tcl              OpenSTA timing script
│   ├── power.tcl            OpenSTA power script
│   ├── ppa.py               drives synthesis + STA for many designs -> CSV
│   ├── power.py             drives gate-level sim + power for many designs -> CSV
│   ├── stimulus.py          image stimulus for the power flow
│   ├── make_synth_lib.py    liberty copy without "do not use" cells (for Yosys)
│   ├── report.py            results/flow_report.md (Milestone 1 flow report)
│   └── diagrams.py          Yosys `show` circuit diagrams
├── baselines/
│   ├── README.md            sources, licences, why these designs
│   ├── baseline_wrappers.sv gives every baseline our interface (a, b -> p)
│   ├── verify.py            RTL vs own model, all pairs; builds LUTs
│   ├── evoapprox/           4 EvoApproxLib designs (.v + .c, MIT licence)
│   └── trumd/fetch.sh       downloads TruMD at a pinned commit (no licence -> not copied)
├── eval/                    applications and analysis
│   ├── image_conv.py        image filtering through each LUT -> PSNR, SSIM
│   ├── cnn.py               LeNet on MNIST, int8 inference through each LUT
│   ├── analysis.py          Spearman correlation of metrics vs quality loss
│   └── figures.py           regenerates every plot
├── results/                 generated CSVs and reports (never hand-edited)
├── figures/                 generated plots
└── docs/
    ├── REFERENCES.md        every source we cite
    ├── milestone1.md        Milestone 1 submission
    ├── report.md            final report
    └── project_guide.md     this document
```

**Golden rule of the repository:** `results/` and `figures/` are only written
by scripts. If a number is in a document, a script produced it.

---

## 5. The flow, step by step

```
                 scripts/gen_dadda.py ──► rtl/dadda8_reduce.sv
                                              │
 model/approx_mul.py ──► LUT(N) ◄──compare──  rtl/approx_mul.sv ──► mac_pe.sv ──► systolic_array.sv
        │  (bit-exact                  (make test)                  (make test-mac: vs LUT golden model)
        │   reference)                       │
        │                                    ▼
        │                       Yosys (synth.tcl) ──► netlist ──► OpenSTA: area, delay  (make synth/sweep/array)
        │                                    │                    │
        │                                    ▼                    ▼
        │                    gate-level sim (image data) ──► VCD ──► OpenSTA power  (+ netlist output vs LUT)
        │
        ├──► model/metrics.py ──► MED, ER, EDmax, ...                         (make metrics)
        ├──► eval/image_conv.py ──► PSNR, SSIM                                (make eval)
        ├──► eval/cnn.py ──► MNIST top-1                                      (make cnn)
        │
 baselines (EvoApproxLib, TruMD) ──► same LUT / same flow / same evaluation   (make baselines)
        │
        └──► eval/analysis.py ──► which metric predicts quality               (make analysis)
                                     └──► eval/figures.py ──► figures/        (make figures)
```

Each step below lists: what it does, the command, what it reads and writes,
and how we know it is right.

### Step 1: Generate the Dadda tree (`make gen`)
- **What:** `scripts/gen_dadda.py` runs Dadda's algorithm on signal names
  (which bits go into which adder at each stage), then evaluates the planned
  tree in numpy on all 65,536 inputs for every N and compares with the model,
  then writes `rtl/dadda8_reduce.sv`.
- **Check:** the self-check must pass before any Verilog is written. The
  generated file is committed, so nobody needs to run this step.

### Step 2: Python model and tests (`make test-model`)
- **What:** `model/approx_mul.py` computes the truncated product straight from
  the formula `sum of a_j b_i 2^(i+j) over i + j >= N` (not from a*b, so it is
  an independent reference) and caches a 256x256 LUT per N.
- **Check:** 42 pytest tests: N = 0 equals a*b everywhere; error never
  negative; MED equals the closed form; N = 4 gives EDmax 49 for exactly 256
  pairs; N = 4 plus 12 gives EDmax 37 for 256 pairs.

### Step 3: RTL vs model, exhaustively (`make test`)
- **What:** for each N = 0..12, Icarus simulates `approx_mul` on all 65,536
  pairs and dumps `a b p` lines; `model/check_rtl.py` compares every line with
  the LUT.
- **Result:** 13/13 PASS (65,536/65,536 each).

### Step 4: MAC PE and systolic array (`make test-mac`)
- **What:** `model/systolic.py` writes a vector file (PE) or a cycle-by-cycle
  schedule (array); the testbench plays it and dumps what the RTL produced; the
  same script checks the dump against a golden model built on `lut(N)`.
- **PE test:** 3,319 cycles per N with random clears, stalls (`en = 0`), a long
  full-scale run that overflows and wraps the 24-bit accumulator, and corner
  values.
- **Array test:** 34 tests per N: random matrices (K = 1, 2, 4, 16, 64),
  all-255, zeros, K = 256 at full scale (largest possible sum, no overflow),
  random stall cycles with junk on the inputs, and 16 image tiles (im2col).
- **Result:** 13/13 PASS for both. The image tiles produce the same 64 pixel
  sums as `eval/image_conv.py` (the RTL array really computes the image filter).

### Step 5: Error metrics (`make metrics`)
- **What:** `model/metrics.py` computes ER, ME, MED, NMED, MRED, MSE, EDmax and
  its count from the LUT of each design -> `results/error_metrics.csv`,
  `figures/error_vs_N.png`.

### Step 6: Synthesis and timing (`make synth`, `make sweep`, `make array`)
- **What:** `flow/ppa.py` calls Yosys (`flow/synth.tcl`) to synthesize
  `mul_wrap` around a multiplier (or the whole array) to sky130 cells, reads
  area from `stat -liberty`, splits off the flip-flop area, then calls OpenSTA
  (`flow/sta.tcl`) for the minimum clock period, setup/hold slacks, WNS/TNS and
  path details.
- **Why the wrapper:** a combinational multiplier has no clock, so its delay is
  measured flop to flop: input flops -> multiplier -> output flops.
- **Checks:** Yosys `check` reports 0 problems; OpenSTA reports 0 setup and 0
  hold violations at 10 ns and no unconstrained endpoints, for every design.

### Step 7: Power (`make power`, part of `sweep`, `array`, `baselines`)
- **What:** `flow/power.py` simulates the *synthesized netlist* (real sky130
  cells, via their Verilog models) with a stimulus, records a VCD, and runs
  OpenSTA `read_vcd` + `report_power` (`flow/power.tcl`).
- **Stimulus:** uniform random (Milestone 1) and image data (Milestone 2): pixel
  x Gaussian weight pairs in the order one PE consumes them
  (`flow/stimulus.py`). The array gets 50 image tiles.
- **Checks:** every cell pin must get activity from the VCD (0 unannotated); and
  the netlist's outputs are compared with the design's LUT (5,008 products per
  multiplier netlist, 800 sums per array netlist). All match, which also proves
  synthesis did not change the function.

### Step 8: Baselines (`make baselines`)
- **What:** `baselines/verify.py` fetches TruMD, simulates each baseline's own
  Verilog on all 65,536 pairs through our wrapper and testbench, compares with
  its own model (EvoApproxLib: their C code compiled with gcc; TruMD: "k
  columns truncated"), builds its LUT; then metrics, synthesis, timing and power
  run through exactly the same scripts as our design.
- **Result:** 8/8 PASS. TruMD trunc-k = our N = k exactly.

### Step 9: Image evaluation (`make eval`)
- **What:** `eval/image_conv.py` filters three image workloads with each
  design's LUT and measures PSNR/SSIM against the exact pipeline ->
  `results/image_quality.csv`, `figures/image_samples.png`.

### Step 10: CNN evaluation (`make cnn`)
- **What:** `eval/cnn.py` trains LeNet in float (cached weights), quantizes it,
  and runs integer inference on 10,000 test images with each design's LUT ->
  `results/cnn_accuracy.csv`.
- **Check:** for N = 0 every layer's LUT result is asserted equal to a plain
  integer matrix multiply.

### Step 11: Analysis (`make analysis`) and figures (`make figures`)
- `eval/analysis.py` correlates error metrics with quality loss (Spearman) and
  computes workload-aware metrics; `eval/figures.py` draws every plot.

### Step 12: Reproduce everything (`make all`)
Runs every step in order (about 20 minutes on a 16-core laptop, measured). We
checked this on a fresh copy of the repository with all generated results
deleted.

---

## 6. Results and how each was obtained

### 6.1 Verification summary

| what | how | result |
|---|---|---|
| multiplier RTL | 65,536 pairs x 13 N vs formula model | 13/13 PASS |
| MAC PE | 3,319 cycles x 13 N vs LUT golden model | 13/13 PASS |
| systolic array | 34 tests x 13 N, incl. stalls and image tiles | 13/13 PASS |
| array computes the image filter | 64 pixels vs `image_conv.py` | match |
| baselines | own RTL vs own model, 65,536 pairs | 8/8 PASS |
| synthesized netlists | gate-level outputs vs LUT | 22 multiplier + 13 array netlists match |
| checker can fail | N = 4 dump vs N = 3 model | FAIL as expected |

### 6.2 Error metrics (`results/error_metrics*.csv`, from the LUTs)

| design | ER (%) | ME | MED | MSE | EDmax | pairs at EDmax |
|---|---|---|---|---|---|---|
| trunc_N2 | 50.0 | -1.25 | 1.25 | 3.75 | 5 | 4096 |
| trunc_N4 | 81.2 | -12.25 | 12.25 | 248.2 | 49 | 256 |
| trunc_N6 | 93.8 | -80.25 | 80.25 | 9158 | 321 | 16 |
| trunc_N8 | 98.0 | -448.25 | 448.25 | 2.633e+05 | 1793 | 1 |
| trunc_N10 | 99.0 | -1664.25 | 1664.25 | 3.672e+06 | 6657 | 1 |
| trunc_N12 | 99.2 | -4992.25 | 4992.25 | 3.527e+07 | 19969 | 1 |
| evo_1446 | 9.4 | 12.00 | 12.00 | 1792 | 192 | 2048 |
| evo_GS2 | 29.9 | -23.56 | 37.04 | 1.268e+04 | 748 | 16 |
| evo_185Q | 98.0 | -24.27 | 118.72 | 2.229e+04 | 518 | 4 |
| evo_JQQ | 19.8 | -249.00 | 731.44 | 5.577e+06 | 10176 | 32 |
| recon_rather_d1 | 90.6 | -0.25 | 7.85 | 98.25 | 37 | 256 |

*How:* the LUT minus the exact a*b table gives the error of every pair; the
metrics are averages and maxima over those 65,536 numbers. Cross-checks:
MED(N) equals the closed form; evo_1446's MED (12) and MSE (1792) equal the
values printed in the EvoApproxLib file header.

### 6.3 Multiplier hardware (`results/ppa_mul.csv`)

| design | area (um²) | area vs N=0 | min period (ns) | logic power, image (uW) | vs N=0 | logic power, random (uW) |
|---|---|---|---|---|---|---|
| N0 | 1774 | +0.0% | 4.19 | 82.8 | +0.0% | 169.2 |
| N4 | 1560 | -12.1% | 4.31 | 60.5 | -26.9% | 147.9 |
| N6 | 1277 | -28.0% | 3.46 | 36.0 | -56.5% | 119.3 |
| N8 | 802 | -54.8% | 3.28 | 13.9 | -83.2% | 74.2 |
| N10 | 415 | -76.6% | 2.34 | 2.6 | -96.9% | 36.0 |
| N12 | 120 | -93.2% | 1.25 | 0.1 | -99.9% | 8.8 |
| a*b | 1919 | +8.2% | 4.09 | 84.3 | +1.8% | 179.1 |

*How:* area = Yosys `stat -liberty` total minus the area of the 32 wrapper
flip-flops; min period = OpenSTA `find_clk_min_period` (clock-to-Q + logic +
setup); logic power = the "Combinational" row of OpenSTA `report_power` with the
VCD activity, at a 10 ns clock. All N in `figures/ppa_vs_N.png`.

*Why:* fewer partial products -> fewer ANDs and adders (area); less logic and
less toggling (power); delay changes little until the tall middle columns go,
because the critical path runs through the high columns and the final adder
(see Q30).

### 6.4 Array hardware (`results/ppa_array.csv`)

| N | area (um²) | area vs N=0 | flops | min period (ns) | power, image (uW) | power vs N=0 | energy/MAC (pJ) |
|---|---|---|---|---|---|---|---|
| 0 | 58,417 | +0.0% | 672 | 5.46 | 4,470 | +0.0% | 4.340 |
| 4 | 50,851 | -13.0% | 608 | 6.29 | 3,752 | -16.1% | 3.643 |
| 8 | 33,745 | -42.2% | 508 | 5.12 | 2,477 | -44.6% | 2.405 |
| 12 | 13,478 | -76.9% | 300 | 3.02 | 1,239 | -72.3% | 1.203 |

*How:* the whole `systolic_array` is synthesized; power from a gate-level
simulation of 50 image tiles (1,942 cycles); energy per MAC = power x 10 ns /
(useful MACs per cycle = 10.3). *Why smaller savings than the multiplier
alone:* accumulators, skew/pass-through registers and the clock load are not
approximated.

### 6.5 Image quality (`results/image_quality.csv`)

| N | camera PSNR (dB) / SSIM | Rather et al. case PSNR / SSIM |
|---|---|---|
| 2 | 55.6 / 0.9984 | identical |
| 4 | 44.8 / 0.9960 | 53.2 / 0.9990 |
| 6 | 32.0 / 0.9683 | 39.4 / 0.9969 |
| 8 | 19.5 / 0.8418 | 26.5 / 0.9529 |
| 10 | 11.3 / 0.5546 | 13.8 / 0.6158 |

*How:* `conv_raw` (Sec. 7.11) sums `LUT[pixel, weight]` over the window, `>> 8`,
and skimage's PSNR/SSIM compare it with the exact-LUT output. *Why the images
get darker:* every truncation error is negative and 25 of them add up per
pixel.

### 6.6 CNN (`results/cnn_accuracy.csv`)

Float 98.42 %, exact int8 98.45 %. N = 0..7: 98.43 to 98.48 %; N = 8: 98.23 %;
N = 9: 97.31 %; N = 10: 75.64 %; N = 11, 12: 10.28 % (it predicts "7" for
every image; 7s are 10.28 % of the test set). EvoApproxLib designs: 98.44 to 98.51 %.

*How:* Sec. 7.12. *Why so tolerant:* errors in a few products are averaged
over hundreds of MACs per output, then ReLU, pooling and argmax only care
about which class is largest, not the exact value.

### 6.7 Baselines through the identical flow

| design | area vs N=0 | logic power, image, vs N=0 | camera PSNR | CNN top-1 change |
|---|---|---|---|---|
| ours N4 / trumd_4 | -12.1% / -10.2% | -26.9% / -24.6% | 44.8 / 44.8 | -0.01 / -0.01 |
| ours N7 / evo_185Q | -41.3% / -41.3% | -69.9% / -73.8% | 26.5 / 33.5 | +0.01 / +0.06 |
| evo_1446 | +9.0% | -1.2% | identical | +0.01 |
| evo_JQQ | -6.3% | -10.2% | identical | +0.03 |

Same function in a different circuit (TruMD) costs about the same. evo_185Q
beats our curve at its point (`figures/tradeoff.png`). evo_1446/JQQ lose their
published savings after re-synthesis to sky130, and are exact on our images
because their errors only occur for weights >= 96 / >= 48.

### 6.8 The metric analysis (`results/metric_correlation.csv`)

Spearman rho, 17 designs (higher = metric ranks designs like the quality loss):

| metric | camera PSNR loss | Rather case PSNR loss | CNN top-1 drop |
|---|---|---|---|
| ER | +0.97 | +0.94 | +0.61 |
| MED | +0.76 | +0.83 | +0.46 |
| MSE | +0.65 | +0.74 | +0.40 |
| EDmax | +0.64 | +0.73 | +0.42 |
| EDmax_freq | -0.59 | -0.60 | -0.08 |
| wMED (workload-aware) | +1.00 | +1.00 | +0.62 |

*How:* `scipy.stats.spearmanr` between each metric column and each
quality-loss column, over the 17 designs. wMED = mean |error| weighted by how
often each (a, b) pair occurs in that workload (the image filters use only 0.4
to 2.3 % of all pairs). Interpretation in Q35-Q37.

---

## 7. Important code, line by line

Line numbers refer to the files as they are in the repository.

### 7.1 `rtl/half_adder.sv` and `rtl/full_adder.sv`

```systemverilog
13    assign sum  = a ^ b;                       // half adder
14    assign cout = a & b;
```
- Line 13: the sum bit of two bits is 1 when exactly one is 1, i.e. XOR.
- Line 14: the carry is 1 only when both are 1, i.e. AND.

```systemverilog
15    assign sum  = a ^ b ^ cin;                 // full adder
16    assign cout = (a & b) | (cin & (a ^ b));
```
- Line 15: the sum of three bits is odd when an odd number of them are 1:
  three-input XOR.
- Line 16: carry when at least two are 1: either a and b both, or exactly one of
  them plus cin.

### 7.2 `rtl/approx_mul.sv` (the multiplier and the N knob)

```systemverilog
20  module approx_mul #(
21      parameter int N = 0            // truncation depth: number of dropped columns, 0..12
22  ) (
23      input  logic [7:0]  a,
24      input  logic [7:0]  b,
25      output logic [15:0] p
26  );
28      logic [63:0] pp;
31      genvar i, j;
32      generate
33          for (i = 0; i < 8; i++) begin : g_row          // i = bit of b
34              for (j = 0; j < 8; j++) begin : g_col      // j = bit of a
35                  if (i + j >= N) begin : g_keep
36                      assign pp[i*8 + j] = a[j] & b[i];  // kept term
37                  end else begin : g_drop
38                      assign pp[i*8 + j] = 1'b0;         // dropped (truncated) term
39                  end
40              end
41          end
42      endgenerate
45      logic [15:0] row0, row1;
46      dadda8_reduce u_tree (.pp(pp), .row0(row0), .row1(row1));
53      assign p = row0 + row1;
54  endmodule
```
- Lines 20-21: `N` is a **parameter**, a compile-time constant. Each value of N
  produces a different circuit; it is not a run-time input.
- Lines 23-25: unsigned 8-bit inputs and a 16-bit product (255 x 255 = 65,025 <
  2^16).
- Line 28: all 64 partial products in one packed vector; bit `i*8 + j` holds
  pp(i, j).
- Lines 31-42: a **generate** loop creates hardware at compile time, 64 times.
  `i` walks the bits of b, `j` the bits of a.
- Line 35: the truncation rule. Column `i + j` is kept if it is >= N.
- Line 36: a kept term is a real AND gate.
- Line 38: a dropped term is the constant 0. No gate is built, and every adder
  fed only by zeros is later removed by synthesis (constant propagation).
- Lines 45-46: the Dadda tree (generated file) compresses the 64 bits into two
  16-bit rows.
- Line 53: the final carry-propagate adder. We write `+` and let Yosys build
  the adder (allowed by the spec, and documented).

### 7.3 `rtl/dadda8_reduce.sv` (generated) and `scripts/gen_dadda.py`

The generated file contains lines like:

```systemverilog
full_adder u_s2_c6_fa0 (.a(s1_c6_ha0_sum), .b(pp[20]), .cin(pp[27]),
                        .sum(s2_c6_fa0_sum), .cout(s2_c6_fa0_cout));  // column 6
```
Name = stage 2, column 6, first full adder; it adds a sum from stage 1 and two
partial products; its sum stays in column 6 and its carry goes to column 7 of
the next stage. The file header lists the column heights before each stage
(e.g. stage 1: `1 2 3 4 5 6 7 8 7 6 5 4 3 2 1 0`) and the adder count
(35 FA, 7 HA).

The planner, `scripts/gen_dadda.py`:

```python
46    cols = [[] for _ in range(COLS)]
47    for i in range(BITS):
48        for j in range(BITS):
49            cols[i + j].append(f"pp[{i * BITS + j}]")
...
52    for s, d in enumerate(TARGETS, start=1):          # targets 6, 4, 3, 2
...
57        for c in range(COLS):
58            bits = list(cols[c])
61            excess = len(bits) + carries_in - d
62            n_fa, n_ha = (excess // 2, excess % 2) if excess > 0 else (0, 0)
...
77            new[c].extend(bits)            # bits not used by any adder pass straight through
...
83                    new[c + 1].append(cout)
84            carries_in = carries_out
85        assert max(len(c) for c in new) <= d, f"stage {s} missed target {d}"
```
- Lines 46-49: put every partial product's name into its column.
- Line 52: one stage per Dadda target height.
- Line 61: how many bits too many this column will have after this stage: its
  own bits plus the carries arriving from the column to its right, minus the
  target.
- Line 62: Dadda's rule. Each FA removes 2 bits, each HA 1 bit, so use
  `excess // 2` FAs and one HA if the excess is odd. No more than necessary.
- Line 77: bits not used by an adder pass to the next stage unchanged.
- Line 83: each adder's carry goes into the next column of the next stage.
- Line 85: guard that the stage really reached its target.
- Lines 91-120 (`self_check`): evaluate the planned tree in numpy on all
  65,536 pairs for every N and compare with the model, before writing any
  Verilog.

### 7.4 `rtl/mul_wrap.sv` (the timing wrapper)

```systemverilog
35      always_ff @(posedge clk) begin
36          a_q <= a_in;
37          b_q <= b_in;
38      end
41      `MUL #(.N(N)) u_mul (.a(a_q), .b(b_q), .p(p_d));
48      always_ff @(posedge clk) begin
49          p_out <= p_d;
50      end
```
- Lines 35-38: input flip-flops launch the operands on a clock edge.
- Line 41: the multiplier under test. `` `MUL`` is a macro, so the same wrapper
  holds `approx_mul`, `mul_behav` or any baseline (all have ports a, b, p).
- Lines 48-50: the output flip-flops capture the product one edge later. The
  register-to-register path through the multiplier is what STA times.

### 7.5 `rtl/mac_pe.sv` (one processing element)

```systemverilog
28  module mac_pe #(
29      parameter int N     = 0,
30      parameter int ACC_W = 24
31  ) ( clk, clear, en, a_in, b_in, a_out, b_out, acc );
42      logic [15:0] p;
43      approx_mul #(.N(N)) u_mul (.a(a_in), .b(b_in), .p(p));
49      always_ff @(posedge clk) begin
50          if (clear) begin
51              acc <= '0;                              // start a new sum
52          end else if (en) begin
53              acc <= acc + ACC_W'(p);                 // zero-extend p to ACC_W bits and add
54          end
55          if (en) begin
56              a_out <= a_in;                          // pass the activation to the right
57              b_out <= b_in;                          // pass the weight down
58          end
59      end
```
- Lines 29-30: N goes straight to the multiplier; the accumulator is 24 bits
  (enough for 258 full-scale products).
- Line 43: the approximate multiplier; this is the only approximate part.
- Lines 50-51: `clear` (synchronous) starts a new dot product. It has priority.
- Lines 52-53: `en` adds this cycle's product. `ACC_W'(p)` widens the 16-bit
  product to 24 bits with zeros (it is unsigned). The sum wraps around past
  2^24, like any fixed-width adder.
- Lines 55-58: the PE forwards its inputs one cycle later to its right and
  lower neighbours: the "systolic" movement. With `en = 0` everything freezes
  (a stall).

### 7.6 `rtl/systolic_array.sv` (4x4 output-stationary array)

```systemverilog
58      logic [ROWS*(COLS+1)*8-1:0] a_h;      // A values between PEs, row by row
59      logic [(ROWS+1)*COLS*8-1:0] b_v;      // B values between PEs, column by column
64          for (i = 0; i < ROWS; i++) begin : g_skew_a
65              if (i == 0) begin : g_none
66                  assign a_h[0 +: 8] = a_left[0 +: 8];
67              end else begin : g_delay
70                  logic [i*8-1:0] sr;
71                  always_ff @(posedge clk) begin
72                      if (en) sr <= (i == 1) ? a_left[i*8 +: 8]
73                                             : {sr, a_left[i*8 +: 8]};
74                  end
75                  assign a_h[(i*(COLS+1))*8 +: 8] = sr[(i-1)*8 +: 8];
...
94          for (i = 0; i < ROWS; i++) begin : g_row
95              for (j = 0; j < COLS; j++) begin : g_col
96                  mac_pe #(.N(N), .ACC_W(ACC_W)) u_pe (
100                     .a_in  (a_h[(i*(COLS+1) + j)*8 +: 8]),
101                     .b_in  (b_v[(i*COLS + j)*8 +: 8]),
102                     .a_out (a_h[(i*(COLS+1) + j + 1)*8 +: 8]),
103                     .b_out (b_v[((i+1)*COLS + j)*8 +: 8]),
104                     .acc   (c_out[(i*COLS + j)*ACC_W +: ACC_W])
```
- Lines 58-59: the wires between PEs, kept as flat packed vectors. (Yosys 0.33
  crashed on 2-D unpacked arrays here, so we use this simpler form, which both
  tools accept.) `x[k*8 +: 8]` means "8 bits starting at bit k*8", i.e. byte k.
- Lines 64-66: row 0 of A needs no delay.
- Lines 70-75: row i goes through an i-byte shift register: each cycle the new
  byte enters at the bottom (`{sr, new}` shifts everything up by one byte) and
  the oldest byte, i cycles old, leaves at the top (line 75).
- Lines 80-91 (not shown): the same for the columns of B.
- Lines 94-104: the 16 PEs. PE(i, j) takes its A from position j of row i and
  passes it to position j + 1 (line 102); B from row i of column j to row i + 1
  (line 103). Its accumulator is slice (i*COLS + j) of `c_out` (line 104).
- Result of the skew: PE(i, j) sees A[i][t - i - j] and B[t - i - j][j] at
  cycle t, the same k on both, so it multiplies the right pair. The last pair
  reaches PE(3, 3) after K + 6 cycles.

### 7.7 `model/approx_mul.py` (the reference)

```python
45  def compute_lut(n):
51      a = np.arange(256, dtype=np.int64).reshape(256, 1)   # rows    = a
52      b = np.arange(256, dtype=np.int64).reshape(1, 256)   # columns = b
53      table = np.zeros((256, 256), dtype=np.int64)
54      for i in range(BITS):
55          for j in range(BITS):
56              if i + j >= n:
57                  table += ((a >> j) & 1) * ((b >> i) & 1) << (i + j)
58      return table
```
- Lines 51-52: a column vector of all a values and a row vector of all b values;
  numpy "broadcasting" makes every expression below a 256x256 table.
- Line 56: the same rule as the RTL (keep i + j >= N).
- Line 57: bit j of a times bit i of b, shifted to weight 2^(i+j), added to all
  65,536 entries at once.
- `lut(n)` (line 60) caches the table as `results/lut_N{n}.npy`. Every other
  script (metrics, image, CNN, golden models) uses this table, so they all agree
  bit for bit with the verified RTL.

### 7.8 `flow/synth.tcl` (Yosys)

```tcl
35  foreach f $::env(SRCS) { read_verilog -sv {*}$defs $f }
36  hierarchy -check -top $top -chparam N $n
41  synth -flatten -top $top
45  dfflibmap -liberty $lib
46  abc -liberty $lib
51  setundef -zero
52  hilomap -singleton -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO
53  splitnets
54  opt_clean -purge
57  tee -o $out/${tag}_stat.txt stat -liberty $lib
58  write_verilog -noattr -noexpr -nohex -nodec $out/${tag}.v
```
- Line 35: read every RTL file (with macros such as `-DMUL=evo_1446`).
- Line 36: choose the top module and **set its parameter N** (`-chparam`). This
  is how one RTL file gives 13 circuits.
- Line 41: generic synthesis: elaborate, optimize, propagate constants (this
  removes the dropped columns' adders), flatten into one module.
- Line 45: map flip-flops to sky130 flip-flop cells.
- Line 46: ABC maps the logic onto sky130 gates using the liberty file.
- Lines 51-54: tidy up for OpenSTA: constant outputs come from a tie cell, buses
  are split into single bits, unused wires removed.
- Line 57: area report: cell list and total area in um² from the liberty file.
- Line 58: the gate-level netlist for STA and gate-level simulation.

### 7.9 `flow/mul_wrap.sdc` and `flow/sta.tcl` (OpenSTA timing)

```tcl
13  create_clock -name clk -period $clk_ns [get_ports clk]
14  set_input_delay  0.0 -clock clk [delete_from_list [all_inputs] [get_ports clk]]
15  set_output_delay 0.0 -clock clk [all_outputs]
```
- One 10 ns clock; inputs change right at the edge; outputs are needed at the
  next edge.

```tcl
27  read_liberty $::env(LIB)
28  read_verilog $::env(NETLIST)
30  link_design  [expr {[info exists ::env(TOP)] ? $::env(TOP) : "mul_wrap"}]
33  read_sdc     [file join [file dirname [info script]] mul_wrap.sdc]
48  out MIN_PERIOD_NS [expr {[sta::find_clk_min_period [get_clocks clk] 0] * 1e9}]
51  out SETUP_SLACK_NS [worst_slack -max]
54  out HOLD_SLACK_NS  [worst_slack -min]
63  puts "SETUP_VIOLATIONS [n_violations max]"
72  proc split_delay {path prefix} { ... }
```
- Lines 27-33: load the cell timing data, the netlist, pick the top module
  (`mul_wrap` or `systolic_array`) and the constraints.
- Line 48: the minimum clock period: clock-to-Q of the launching flop + logic
  delay + setup time of the capturing flop. This is our "delay".
- Lines 51-56: worst slacks, WNS and TNS for setup ("max" paths) and hold
  ("min" paths) at 10 ns.
- Line 63: count endpoints with negative slack (0 for every design).
- Line 72 onward: walk the critical path and split its delay into cell delay
  and wire delay (wire delay is 0 pre-layout, as expected).

### 7.10 `flow/power.tcl` and `tb/tb_power.sv` (power)

```tcl
25  read_vcd -scope $::env(SCOPE) $::env(VCD)
29  report_activity_annotation
31  report_power -digits 6 > $::env(OUT).power.txt
```
- Line 25: attach the simulated switching activity (how often each pin toggles,
  how long it is 1) to every cell pin.
- Line 29: report how many pins got activity; `flow/power.py` requires 0
  unannotated pins.
- Line 31: per cell, internal + switching + leakage power from the liberty
  power tables, summed into Sequential / Combinational / Total.

`tb/tb_power.sv` drives the gate-level netlist, one new (a, b) pair per cycle
from a file (image data) or `$random`, and dumps the VCD with depth 2 (the
netlist top plus every cell's pins, which is what `read_vcd` matches). It also
writes every product so `flow/power.py` can check the netlist against the LUT.

### 7.11 `eval/image_conv.py` (image filtering through a LUT)

```python
105 def conv_raw(img, kernel, table):
107     k = kernel.shape[0]
108     pad = np.pad(img, k // 2, mode="reflect")
109     h, w = img.shape
110     raw = np.zeros((h, w), dtype=np.int64)
111     for u in range(k):
112         for v in range(k):
114             raw += table[pad[u:u + h, v:v + w], kernel[u, v]]
115     return raw
```
- Line 108: mirror the image at the borders so every output pixel has a full
  window.
- Lines 111-112: one loop iteration per kernel tap (25 for 5x5).
- Line 114: the key line. `pad[u:u+h, v:v+w]` is the whole image shifted by
  (u, v): the pixel that tap (u, v) sees for every output pixel at once.
  `table[pixels, weight]` looks up `LUT[pixel, weight]`, exactly what the
  hardware multiplier would output, and the products are accumulated. This is
  the same MAC the array does; `model/systolic.py` checks that the RTL array
  gives identical sums.
- `filter_image` then takes `raw >> 8` (weights sum to about 256, so >> 8
  divides by 256) and clips to 0..255.

### 7.12 `eval/cnn.py` (quantized CNN through a LUT)

```python
127     s_w = float(np.abs(w).max()) / 127
128     wq = np.clip(np.round(w / s_w), -127, 127).astype(np.int64)
129     bq = np.round(mod.bias.detach().numpy() / (s_in * s_w)).astype(np.int64)
131     mult = None if s_out is None else s_in * s_w / s_out
```
- Line 127: symmetric per-layer weight scale: the largest |w| maps to 127.
- Line 128: integer weights in -127..127.
- Line 129: biases in the accumulator's scale (input scale x weight scale).
- Line 131: the requantization factor from the accumulator scale to the next
  layer's uint8 scale.

```python
149 def lut_matmul(acts, wq, table):
151     n, K = acts.shape
153     G = (np.sign(wq).T[:, None, :] * table[:, np.abs(wq).T].transpose(1, 0, 2))
154     G = G.reshape(K * 256, -1).astype(np.float64)
155     cols = (np.arange(K) * 256 + acts).ravel()
156     X = sp.csr_matrix((np.ones(n * K), cols, np.arange(0, n * K + 1, K)), shape=(n, K * 256))
157     return np.rint(X @ G).astype(np.int64)
```
- Goal: `acc[n, o] = sum over k of sign(w[o,k]) x LUT[act[n,k], |w[o,k]|]`.
- Line 153: precompute, for every input position k, every possible activation
  value v (0..255) and every output o, the signed product
  `G[k, v, o] = sign(w) x LUT[v, |w|]`.
- Line 155: for each sample n and position k, the row of G to use is
  `k*256 + act[n, k]`.
- Line 156: a sparse matrix with one 1 per (n, k) at that column ("one-hot").
- Line 157: multiplying the one-hot matrix by G adds up exactly the selected
  products. All values are integers below 2^53, so float64 arithmetic is exact;
  `rint` converts back. For N = 0 the code asserts this equals a plain integer
  matrix multiply.

```python
160 def requant(acc, mult):
162     return np.clip(np.rint(acc * mult), 0, 255).astype(np.int64)
```
- Scale the 32-bit sum to the next layer's 8-bit range; clipping at 0 is the
  ReLU, clipping at 255 is saturation.

### 7.13 `eval/analysis.py` (the metric test)

The core is three lines inside `correlations()`:

```python
y = q.loc[names, qcol]                   # quality loss of each design
for m, x in cand.items():                # each candidate error metric
    rho, pval = spearmanr(x, y)          # rank correlation over the 17 designs
```
Spearman rho compares the *rankings*: +1 if the metric orders the designs
exactly like their quality loss. `workload_metrics()` builds a 256x256
histogram of the (a, b) pairs a workload really uses and weights the error
table by it (wMED).

---

## 8. AI-accelerator questions and answers

### A. Fundamentals

**Q1. Why are MAC operations the focus of AI accelerators?**
Convolution and fully connected layers are matrix multiplications, i.e. long
chains of multiply-accumulates. In our LeNet, one 28x28 image needs about
0.4 million MACs (conv1 117,600, conv2 240,000, fc1 48,000, fc2 1,200); large
CNNs need billions. Nearly all compute energy goes into MACs and moving their
operands.

**Q2. Why is the multiplier the target, not the adder?**
An n x n multiplier is roughly n² gates (64 partial products plus their
reduction tree); an adder is roughly n. In our array the 16 multipliers are
most of the logic area: at N = 0 the array's logic is 44,964 um², of which the
multipliers are about 16 x 1,774.

**Q3. What is approximate computing, and why is it acceptable for AI?**
Deliberately computing a slightly wrong result to save area, energy or time.
Neural networks are trained to be robust to noise, are already quantized to 8
bits, and decide by argmax; images are judged by human eyes. Our CNN keeps its
accuracy within 0.03 pp up to N = 7.

**Q4. Why 8-bit unsigned operands?**
int8 is the standard inference precision (TPU, mobile NPUs) [15, 19].
Activations after ReLU are non-negative, so unsigned 8-bit fits them; signed
weights are handled with a sign bit outside the multiplier.

**Q5. What does "precision-scalable" mean here?**
One RTL design with one parameter N covers every precision from exact (N = 0)
to very rough (N = 12). A designer picks N for the application. It is a
*compile-time* choice: each N is a different chip, not a run-time mode.

### B. The multiplier design

**Q6. Why a Dadda tree and not an array multiplier or Wallace tree?**
Tree multipliers have logarithmic depth (4 stages for 8 bits) instead of linear
depth, so they are faster. Dadda uses fewer adders than Wallace (it reduces only
as much as each stage requires) [5, 6].

**Q7. Why truncate the low columns rather than other bits?**
Low columns have the smallest weights (2^0, 2^1, ...), so dropping them costs
the least accuracy per removed gate. The error is bounded and known in closed
form (EDmax, MED).

**Q8. Why does truncation make the result always too small, and does it matter?**
Every dropped bit had positive weight, so the result can only lose value
(ME = -MED). In image filtering the 25 errors per pixel add up with the same
sign: the image gets darker (`figures/image_samples.png`). A correction
constant fixes most of this: N = 4 + 12 (our reconstruction of Rather et al.'s
Design 1) has ME -0.25 instead of -12.25 and 58.6 dB instead of 53.2 dB on
their smoothing case.

**Q9. Is the tree re-designed for every N?**
No. One tree is generated; truncated partial products are tied to 0 and Yosys
removes all logic driven only by constants. A tree planned specially for each
N's column heights might save a little more; we state this as a limitation.

**Q10. How does a compile-time parameter become 13 circuits?**
`hierarchy -chparam N <n>` in Yosys sets N before elaboration; the generate
loop then builds a different set of AND gates and constants, and synthesis
optimizes each version separately.

**Q11. Why is the final addition a behavioral `+`?**
The two final rows need a carry-propagate adder; writing `+` lets Yosys/ABC
choose and map a good adder. The spec allows it and the docs state it.

### C. Verification

**Q12. How do you know the RTL is correct?**
Exhaustive simulation: all 65,536 input pairs for every N, compared with a
Python model written from the formula, not from a*b. 13/13 pass. The checker
was also shown to fail on a deliberately wrong comparison.

**Q13. Why not just test random inputs?**
For an 8x8 multiplier exhaustive testing is cheap (65,536 cases, seconds), so
there is no reason to settle for less. Random testing could miss rare cases
like the single pair that hits EDmax at N = 8.

**Q14. How was the systolic array verified?**
Against a golden model C[i][j] = sum_k LUT[A[i][k], B[k][j]] (mod 2^24), with
random and corner-case matrices, K from 1 to 256, random stall cycles carrying
junk data, and real image patches. For the image patches the array's sums were
also compared with the image-filter script itself.

**Q15. How do you know synthesis did not break the design?**
We simulate the synthesized *gate-level netlist* (sky130 cell models) and
compare its outputs with the LUT: every multiplier netlist (5,008 products) and
every array netlist (800 sums) matches.

**Q16. How were the baselines verified?**
Each baseline's own Verilog was simulated on all 65,536 pairs through our
wrapper and compared with its own model: EvoApproxLib's C code compiled with
gcc; TruMD's description "k columns truncated" (TruMD has no software model).
All 8 pass.

### D. Systolic array and dataflow

**Q17. Why a systolic array?**
Each operand read from memory is reused by every PE it passes (4 times here,
256 times in a TPU), so memory traffic per MAC drops and many MACs run in
parallel with only neighbour-to-neighbour wires [15].

**Q18. Why output-stationary?**
Each PE accumulates its own output in place, so partial sums never move; it is
the simplest dataflow to build and verify, and suits convolution where each
output needs many MACs (25 for a 5x5 filter). Weight-stationary (TPU) suits
reusing the same weights across many inputs.

**Q19. What is the skew and why is it needed?**
PE(i, j) must multiply A[i][k] by B[k][j] in the same cycle. A takes j cycles to
reach column j and B takes i cycles to reach row i, so row i of A and column j
of B are delayed by i and j cycles at the edge (shift registers inside the
array). Then PE(i, j) sees index k = t - i - j on both inputs.

**Q20. How many cycles does one 4x4 tile take, and what is the utilization?**
K data cycles + 6 drain cycles (+ 8 clear cycles in our test protocol). For
K = 25: 39 cycles for 16 x 25 = 400 useful MACs, i.e. 10.3 MACs per cycle out
of 16 (64 %). Real accelerators overlap the drain of one tile with the fill of
the next (double-buffered accumulators) to approach 100 %.

**Q21. How would you map a convolution onto this array?**
im2col: each 5x5 window becomes a row of A (25 values), each filter a column of
B. The array then computes 4 output pixels x 4 filters per tile. Our array test
does exactly this on the cameraman image.

**Q22. Why a 24-bit accumulator?**
Each product is below 2^16; 24 bits hold 258 full-scale products. Our CNN's
largest dot product is 400 (fc1), but real values are far below full scale;
for longer sums or full-scale data, use 32 bits or saturate.

**Q23. How does the array's throughput scale?**
At N = 0 the array runs at 183 MHz (5.46 ns): 16 MACs x 183 MHz = 2.9 GMAC/s
peak (5.9 GOPS counting multiply and add separately), about 64 % of that with
our tile protocol. This is pre-layout at the typical corner.

### E. Power, performance, area (PPA)

**Q24. How is area measured?**
Yosys `stat -liberty` sums the liberty area of every cell in the netlist (um²).
For the multiplier we subtract the 32 wrapper flip-flops. Pre-layout: no
routing, no placement density, so a real layout would be larger.

**Q25. How is delay measured?**
OpenSTA's minimum clock period for the register-to-register path through the
multiplier: clock-to-Q + logic + setup, with the liberty delay tables, no wire
load, ideal clock.

**Q26. How is power measured, and why simulate the netlist?**
Dynamic power depends on how often each wire toggles. We simulate the
gate-level netlist with real stimulus, record every toggle in a VCD, and
OpenSTA's `report_power` combines it with the liberty power tables. Simulating
the netlist (not the RTL) gives activity for every internal wire.

**Q27. Why measure power with image data and not only random inputs?**
Power depends on data. Filter weights are small, so the high bits of b rarely
toggle. At N = 8 the logic power saving is -83.2 % with image data but -56.1 %
with random data. Random inputs misrepresent the real workload.

**Q28. What dominates the multiplier's total power in the wrapper?**
The flip-flops and their clock pins: with image data, N = 0 logic power is
82.8 uW of 248.9 uW total. We therefore report the multiplier's own
(combinational) power separately.

**Q29. What is energy per MAC and why report it?**
Energy per operation = power x clock period / operations per cycle. It is the
fair way to compare accelerators (it is the inverse of TOPS/W). Our array:
4.340 pJ/MAC at N = 0, 2.405 pJ/MAC at N = 8 (-44.6 %).

**Q30. Why is the delay not monotonic in N?**
Truncation first removes the low columns, which are not on the critical path;
the critical path goes through the tall middle columns, the high columns and
the final carry-propagate adder. Small netlist changes also make the
area-oriented ABC mapping pick slightly different gates, which moves the
critical path by a few tenths of a nanosecond (4.19 ns at N = 0, 4.69 ns at
N = 3, 3.46 ns at N = 6). Delay falls clearly only from N = 8 (3.28 ns) to
N = 12 (1.25 ns).

**Q31. Why does the array save a smaller fraction than the multiplier?**
Accumulators, skew and pass-through registers and the clock network are exact
and do not shrink with N (except accumulator bits that become constant): at
N = 8, array area -42.2 % vs multiplier -54.8 %.

**Q32. Why is our exact Dadda tree smaller than Yosys's a*b?**
1774 vs 1919 um² (7.6 % smaller) at a similar delay (4.19 vs 4.09 ns). Yosys's
generic multiplier is a reasonable but not specially optimized structure; our
explicit Dadda tree gives ABC a better starting point. This makes N = 0 a fair
exact reference.

### F. Approximate multipliers and baselines

**Q33. Why compare with EvoApproxLib and TruMD?**
They are published, open approximate multipliers with very different error
profiles. Running them through the *identical* flow, wrapper, clock, corner
and stimulus makes the comparison fair; we never compare our absolute numbers
with numbers from other tools or processes.

**Q34. Which design is best?**
None everywhere. Truncation gives a full, smooth sweep from one parameter, and
TruMD (same function, different circuit) costs about the same. But evo_185Q
beats our curve at its point (same area as N = 7, lower power, 33.5 vs 26.5 dB).
evo_1446 and evo_JQQ lose their published savings after re-synthesis to sky130
(their circuits were optimized for a 45 nm library).

### G. Error metrics and the analysis

**Q35. What hypothesis did you test, and what did you find?**
Rather et al. [1] observed that image quality tracks EDmax and its frequency,
not just MED/NMED. Over 17 designs, for image smoothing, the error rate ranked
designs best (rho 0.94-0.99), MED/NMED 0.76-0.83, EDmax 0.64-0.73, and
EDmax x frequency was *negatively* correlated. So in our pool, "EDmax and its
frequency" is not a good ranking rule for smoothing. Rather et al. made their
observation for sharpening and edge detection (and for smoothing they found the
lowest-MED design best), which we did not test.

**Q36. Why is EDmax frequency negatively correlated?**
For truncation, a larger N gives a larger EDmax hit by *fewer* pairs (16,384
pairs at N = 1, one pair from N = 8), while quality gets worse. Count and
quality loss move in opposite directions across most of the pool.

**Q37. What predicts application quality best?**
Error on the inputs the workload actually uses. The image filters use only 0.4
to 2.3 % of the 65,536 (a, b) pairs (small weights). Weighting MED by the
workload's pair histogram (wMED) gives rho >= 0.99. evo_1446 and evo_JQQ are
the extreme case: large uniform-input errors, zero error on the pairs used.
Lesson: choose an approximate multiplier with the target workload's operand
distribution, not only uniform-input metrics.

**Q38. Why are the CNN correlations weak?**
For 12 of the 17 designs the top-1 change is within ±0.06 pp, which is noise at
10,000 test images, so their ranks carry no information. Only the large-N
designs separate.

### H. Quantization and the CNN

**Q39. How is a signed weight handled by an unsigned multiplier?**
product = sign(w) x LUT[activation, |w|]: the magnitude goes through the
unsigned multiplier and the sign is applied afterwards (in hardware: negate
before accumulating). |w| <= 127, so only half of the b range is used.

**Q40. Why does the CNN collapse at N >= 10?**
Many products in the network are small (activations and weights well below
full scale), and truncating 10 or more columns removes products below about
2^10 almost entirely. Whole feature maps go to zero after requantization and
ReLU, and the network outputs the digit 7 for every image (checked); 7s are
1,028 of the 10,000 test images, hence 10.28 %.

**Q41. Why is exact int8 accuracy (98.45 %) slightly above float (98.42 %)?**
Quantization adds a little noise that happens to flip 3 test images the right
way; the difference is within noise.

**Q42. How do you know the LUT-based CNN is bit-exact hardware arithmetic?**
Every multiply is a LUT lookup of the verified multiplier, sums are exact
integers, and for N = 0 every layer is asserted equal to a plain integer matrix
multiply.

### I. Tools and methodology

**Q43. Why open-source tools?**
Anyone can rerun every number (`make all`), with no licences. The flow is the
same family OpenLane/LibreLane use for real sky130 tape-outs [21, 22].

**Q44. What are the main limitations?**
Pre-layout and one corner; delay from area-oriented mapping; unit-delay power
simulation (glitches not modelled); short array power simulation; tree not
re-planned per N; a small CNN on an easy dataset; smoothing filters only; 17
designs in the correlation. See `docs/report.md` Sec. 9.

**Q45. What would you do next?**
Place-and-route (LibreLane) for real wire and clock effects; a harder CNN
(CIFAR-10); sharpening/edge detection; a second approximation family
(partial-product perforation or a logarithmic multiplier); and integrating the
multiplier into a RISC-V core with an exact/approximate mode.

---

## 9. Glossary

| term | meaning |
|---|---|
| partial product | one bit a[j] AND b[i], worth 2^(i+j) |
| column | all partial products with the same weight (same i + j) |
| truncation depth N | number of lowest columns dropped |
| LUT | look-up table: here, the 256x256 table of a design's outputs |
| golden model | the trusted reference implementation tests compare against |
| exhaustive test | testing every possible input (65,536 for 8x8) |
| PE | processing element: one MAC unit of the array |
| im2col | turning convolution windows into matrix rows so convolution becomes matrix multiply |
| PPA | power, performance (delay), area |
| pp | percentage points (98.45 % -> 98.23 % is -0.22 pp) |
| Spearman rho | correlation of rankings, from -1 to +1 |
| wMED | MED weighted by how often each input pair occurs in a workload |

---

## 10. What we did, in order

**Milestone 1 (Oct 1):** environment and tool versions; Yosys smoke test;
Python model and tests; Dadda generator and adders; `approx_mul` with the N
mask; exhaustive tests for N = 0..12; error metrics; `mul_wrap`, synthesis,
timing and power for N = 0 vs a*b; area and delay for every N; README and
`docs/milestone1.md`.

**Milestone 2 (Oct 29):**
1. `mac_pe.sv` and `systolic_array.sv` with vector-file testbenches and
   Python golden models (M2.1, M2.2).
2. Image filtering evaluation, including the Rather et al. case (M2.3, M2.8).
3. Realistic image stimulus for power; power for every N; the array through the
   flow (M2.4, M2.5).
4. EvoApproxLib (4) and TruMD (4) fetched, verified against their own models,
   and run through the identical flow (M2.6, M2.7); the TruMD cross-check also
   completes the optional M1.8.
5. LeNet on MNIST with int8 LUT inference for every design (M2.9).
6. The metric analysis, including workload-aware metrics (M2.10).
7. Image patches through the RTL array match the Python model (M2.11).
8. Figures, report, this guide; `make all` from a clean copy (M2.12-M2.14).

Not done: the optional LibreLane layout (M2.15) and the optional signed-kernel
image cases.
