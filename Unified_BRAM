/*
 * Unified Dual-Port Input Buffer BRAM for TPU-like Design
 * This BRAM will store input activations for the systolic array.
 * Dual-port allows simultaneous read/write operations.
 *
 * Author: BLACKBOXAI
 * Date: 2024
 */

`timescale 1ns/1ps

module Unified_Input_Buffer_BRAM #(
    parameter DATA_WIDTH = 8,         // Width of each data element
    parameter ADDR_WIDTH = 10,        // Address width for 1024 depth
    parameter DEPTH = 1 << ADDR_WIDTH  // Total depth of the buffer
)(
    input wire clk,

    // Port A - Read Port (for reading activations)
    input wire en_a,
    input wire [ADDR_WIDTH-1:0] addr_a,
    output reg [DATA_WIDTH-1:0] dout_a,

    // Port B - Write Port (for writing activations)
    input wire en_b,
    input wire we_b,
    input wire [ADDR_WIDTH-1:0] addr_b,
    input wire [DATA_WIDTH-1:0] din_b
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1]; // Memory array

    // Port A Read
    always @(posedge clk) begin
        if (en_a) begin
            dout_a <= mem[addr_a];
        end
    end

    // Port B Write
    always @(posedge clk) begin
        if (en_b && we_b) begin
            mem[addr_b] <= din_b;
        end
    end

    // Optional initialization for simulation
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = 0;
        end
    end

endmodule
