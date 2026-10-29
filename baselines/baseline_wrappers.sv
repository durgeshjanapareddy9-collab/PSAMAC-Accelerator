// baseline_wrappers.sv -- give every published baseline multiplier the same
// interface as our approx_mul: ports a, b (8-bit) and p (16-bit), plus a
// parameter N that is ignored. With the same interface, the SAME flow
// (rtl/mul_wrap.sv, flow/synth.tcl, flow/sta.tcl, flow/power.tcl) and the
// SAME exhaustive testbench run on every design, so the comparison is fair.
//
// Each wrapper is only compiled when its macro BASE_<name> is defined
// (model/designs.py passes it), so one design is built at a time. This is
// needed because the four TruMD folders all define modules with the same
// names (FA, HA, AND2, COMP4to2).
//
// Port mapping: our a -> their A, our b -> their B. In every LUT in this
// project, lut[a, b] = design output for (A = a, B = b).
//
// The baseline circuits themselves are NOT modified:
//   EvoApproxLib (MIT licence): baselines/evoapprox/mul8u_*.v (copied unchanged)
//   TruMD (no licence, fetched by baselines/trumd/fetch.sh):
//       baselines/trumd/upstream/Dadda_Exact_8b_Trunc_<k>/OriginalFiles/*.v

// ---- EvoApproxLib 8x8 unsigned multipliers ----
`ifdef BASE_evo_1446
module evo_1446 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    mul8u_1446 u (.A(a), .B(b), .O(p));
endmodule
`endif

`ifdef BASE_evo_GS2
module evo_GS2 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    mul8u_GS2 u (.A(a), .B(b), .O(p));
endmodule
`endif

`ifdef BASE_evo_185Q
module evo_185Q #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    mul8u_185Q u (.A(a), .B(b), .O(p));
endmodule
`endif

`ifdef BASE_evo_JQQ
module evo_JQQ #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    mul8u_JQQ u (.A(a), .B(b), .O(p));
endmodule
`endif

// ---- TruMD: Dadda multiplier (4:2 compressors) with the k lowest columns truncated ----
`ifdef BASE_trumd_2
module trumd_2 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    dadda_8b_trunc_2 u (.A(a), .B(b), .P(p));
endmodule
`endif
`ifdef BASE_trumd_4
module trumd_4 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    dadda_8b_trunc_4b u (.A(a), .B(b), .P(p));
endmodule
`endif
`ifdef BASE_trumd_6
module trumd_6 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    dadda_8b_trunc_6b u (.A(a), .B(b), .P(p));
endmodule
`endif
`ifdef BASE_trumd_8
module trumd_8 #(parameter int N = 0) (input logic [7:0] a, b, output logic [15:0] p);
    dadda_8b_trunc_8 u (.A(a), .B(b), .P(p));
endmodule
`endif
