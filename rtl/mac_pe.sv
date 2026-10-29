// mac_pe.sv -- one processing element (PE): approximate multiply + accumulate.
//
// A MAC ("multiply-accumulate") unit does   acc = acc + a * b   every clock
// cycle. Convolutions and matrix multiplies are long chains of exactly this
// operation, so the MAC is the basic building block of an AI accelerator.
//
// This PE is made for a systolic array (systolic_array.sv), so besides the
// MAC it also passes its inputs on to its neighbours one cycle later:
//
//              b_in (from the PE above)
//                |
//   a_in ---> [ approx_mul #(N) ] --p--> (+) --> acc
//   (from the     |                       ^       |
//    PE on the    |                       +-------+   (accumulator register)
//    left)        v
//             a_out (to the right), b_out (down)   -- registered copies
//
// Control (both synchronous, i.e. they act at the rising clock edge):
//   clear = 1 : acc becomes 0 (start a new sum). clear wins over en.
//   en    = 1 : acc <= acc + p, and a_out/b_out take the new inputs.
//   en    = 0 : everything holds its value (a "stall").
//
// The accumulator is ACC_W bits wide (default 24). Each product is at most
// 255 * 255 = 65,025 < 2^16, so 24 bits hold the sum of at least 2^24 / 65,025
// = 258 products without overflow. If it does overflow, it wraps around
// (keeps the low ACC_W bits), like any fixed-width hardware adder.

module mac_pe #(
    parameter int N     = 0,           // truncation depth of the multiplier, 0..12
    parameter int ACC_W = 24           // accumulator width in bits
) (
    input  logic             clk,
    input  logic             clear,    // synchronous clear of the accumulator
    input  logic             en,       // enable: accumulate and pass data on
    input  logic [7:0]       a_in,     // activation (e.g. image pixel), unsigned
    input  logic [7:0]       b_in,     // weight (e.g. filter coefficient), unsigned
    output logic [7:0]       a_out,    // a_in delayed by one cycle -> PE on the right
    output logic [7:0]       b_out,    // b_in delayed by one cycle -> PE below
    output logic [ACC_W-1:0] acc       // running sum of products
);
    // The (approximate) product of this cycle's inputs.
    logic [15:0] p;
    approx_mul #(.N(N)) u_mul (
        .a (a_in),
        .b (b_in),
        .p (p)
    );

    always_ff @(posedge clk) begin
        if (clear) begin
            acc <= '0;                              // start a new sum
        end else if (en) begin
            acc <= acc + ACC_W'(p);                 // zero-extend p to ACC_W bits and add
        end
        if (en) begin
            a_out <= a_in;                          // pass the activation to the right
            b_out <= b_in;                          // pass the weight down
        end
    end
endmodule
