// tb_approx_mul_exhaustive.sv -- apply all 65,536 input pairs to approx_mul.
//
// The testbench does not judge the answers itself. It writes every result to
// results/rtl_dump_N<N>.txt, one line "a b p" per pair (a outer loop, b inner),
// and model/check_rtl.py compares that file with the Python model.
//
// Run for a given N (normally via `make test`):
//   iverilog -g2012 -P tb_approx_mul_exhaustive.N=4 -o sim.vvp \
//            tb/tb_approx_mul_exhaustive.sv rtl/*.sv   &&  vvp sim.vvp
//
// Other multipliers (the baselines, via `make baselines`): define the macro
// MUL as the module name and set TAG; the output file is then
// results/rtl_dump_<TAG>.txt, e.g.
//   iverilog ... -DMUL=evo_1446 -DBASE_evo_1446 -P tb_approx_mul_exhaustive.TAG=\"evo_1446\"

`timescale 1ns/1ps

`ifndef MUL
`define MUL approx_mul
`endif

module tb_approx_mul_exhaustive;
    parameter int    N   = 0;      // truncation depth under test
    parameter string TAG = "";     // "" = our multiplier, file name uses N

    logic [7:0]  a, b;
    logic [15:0] p;

    // Device under test.
    `MUL #(.N(N)) dut (.a(a), .b(b), .p(p));

    integer fd;                    // output file handle
    integer ia, ib;                // loop counters (integers so they can reach 256)
    string  fname;

    initial begin
        fname = (TAG == "") ? $sformatf("results/rtl_dump_N%0d.txt", N)
                            : $sformatf("results/rtl_dump_%s.txt", TAG);
        fd = $fopen(fname, "w");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", fname);
            $finish;
        end
        for (ia = 0; ia < 256; ia++) begin
            for (ib = 0; ib < 256; ib++) begin
                a = ia[7:0];
                b = ib[7:0];
                #1;                        // let the combinational logic settle
                $fdisplay(fd, "%0d %0d %0d", a, b, p);
            end
        end
        $fclose(fd);
        $display("tb_approx_mul_exhaustive: wrote 65536 results to %s", fname);
        $finish;
    end
endmodule
