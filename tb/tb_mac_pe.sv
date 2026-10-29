// tb_mac_pe.sv -- drive mac_pe from a vector file and dump the accumulator.
//
// Like the multiplier testbench, this one does not judge the answers itself:
//   input : build/pe_vectors_N<N>.txt, one line "clear en a b" per clock cycle
//           (written by `python -m model.systolic gen-pe <N>`)
//   output: results/pe_dump_N<N>.txt, the accumulator after every clock edge
//           (checked by `python -m model.systolic check-pe <N>`)
//
// Inputs change on the falling clock edge, the PE updates on the rising edge,
// and the accumulator is read at the next falling edge.

`timescale 1ns/1ps

module tb_mac_pe;
    parameter int N     = 0;
    parameter int ACC_W = 24;

    logic             clk = 0, clear = 0, en = 0;
    logic [7:0]       a = 0, b = 0;
    logic [7:0]       a_out, b_out;
    logic [ACC_W-1:0] acc;

    mac_pe #(.N(N), .ACC_W(ACC_W)) dut (
        .clk(clk), .clear(clear), .en(en), .a_in(a), .b_in(b),
        .a_out(a_out), .b_out(b_out), .acc(acc));

    always #5 clk = ~clk;                  // 10 ns clock

    integer fin, fout, n_read, c, e, va, vb, cycles = 0;

    initial begin
        fin  = $fopen($sformatf("build/pe_vectors_N%0d.txt", N), "r");
        fout = $fopen($sformatf("results/pe_dump_N%0d.txt", N), "w");
        if (fin == 0 || fout == 0) begin
            $display("ERROR: cannot open the vector or dump file");
            $finish;
        end
        @(negedge clk);
        while (!$feof(fin)) begin
            n_read = $fscanf(fin, "%d %d %d %d\n", c, e, va, vb);
            if (n_read == 4) begin
                clear = c[0]; en = e[0]; a = va[7:0]; b = vb[7:0];
                @(negedge clk);            // exactly one rising edge has happened
                $fdisplay(fout, "%0d", acc);
                cycles++;
            end
        end
        $fclose(fin);
        $fclose(fout);
        $display("tb_mac_pe: N=%0d, %0d cycles", N, cycles);
        $finish;
    end
endmodule
