# Baselines

Published approximate 8x8 unsigned multipliers, run through exactly the same
verification, flow (`rtl/mul_wrap.sv`, `flow/synth.tcl`, `flow/sta.tcl`,
`flow/power.tcl`, same clock, corner and stimulus) and application evaluation
as our truncated multiplier. `make baselines` does all of it.

Every baseline gets our uniform interface (`a`, `b` -> `p`, parameter `N`
ignored) from a thin wrapper in `baseline_wrappers.sv`; the circuits
themselves are not modified. Port mapping: our `a` -> their `A`, our `b` ->
their `B`. In every LUT, `lut[a, b]` is the output for `A = a, B = b`. In the
array, image filter and CNN, `a` is always the activation (pixel) and `b` the
weight.

## EvoApproxLib (`evoapprox/`)

- Source: https://github.com/ehw-fit/evoapproxlib, commit
  `ec28be83bfce1b8e8b92bd456be520d323a568b5`, folder `multiplers/8x8_unsigned/`.
- Licence: MIT (`evoapprox/LICENSE.md`, copied with the files). The `.v` and `.c`
  files are copied unchanged, headers included.
- Cite: Mrazek et al., DATE 2017 (docs/REFERENCES.md ref. 12) and JETCAS 2020 (ref. 13).
- Verification: each design's Verilog is simulated on all 65,536 pairs and compared
  with the library's own C model (compiled with gcc). All four match 65,536/65,536.

Why these four: their mean error falls inside our N range, but their error
profiles are very different from plain truncation (and from each other). That
difference is what makes the error-metric analysis meaningful (all values from
`results/error_metrics_baselines.csv`):

| design | MED | closest our N (MED) | EDmax (pairs) | error rate |
|---|---|---|---|---|
| mul8u_1446 | 12.0 | N=4 (12.25) | 192 (2,048) | 9.4 % |
| mul8u_GS2 | 37.0 | N=5 (32.25) | 748 (16) | 29.9 % |
| mul8u_185Q | 118.7 | N=6 (80.25) | 518 (4) | 98.0 % |
| mul8u_JQQ | 731.4 | N=9 (896.25) | 10,176 (32) | 19.8 % |

For comparison our N=4 has EDmax 49 (256 pairs) and error rate 81 %.

## TruMD (`trumd/`)

- Source: https://github.com/Hassan313/Approximate-Multiplier, commit
  `ba73ab939e80501bc1ba83ce1ee4c8bca485ad2d`, folders
  `Dadda_Exact_8b_Trunc_{2,4,6,8}/OriginalFiles/` (technology-independent RTL:
  AND gates, half/full adders and 4:2 compressors).
- Licence: **none** in the repository, so we do not copy its code.
  `trumd/fetch.sh` clones it at the pinned commit into `trumd/upstream/`
  (ignored by git).
- Cite: the repository asks for Afzali-Kusha et al., IEEE TVLSI 2020 (X-Dadda,
  docs/REFERENCES.md ref. 2).
- Verification: TruMD ships no software model, so its Verilog is compared with
  its description, "the k lowest columns truncated", which is our formula
  `lut(k)`. **trumd_2/4/6/8 match our N = 2/4/6/8 on all 65,536 pairs.** So
  TruMD is the same *function* as our design built as a different *circuit*
  (4:2 compressors instead of our full/half adder tree). It is a hardware
  baseline; it is left out of the error-metric correlation (it would count
  four designs twice).

## Reconstruction of Rather et al. Design 1 (software only)

`recon_rather_d1` in `model/designs.py`: our N = 4 truncation plus a constant
+12. It gives EDmax = 37 for 256 pairs, the values Rather et al. 2025 report
for their Design 1. This is a sanity check of our reading of their paper,
**not** a claim that it is their circuit; it has no RTL and is not synthesized.
