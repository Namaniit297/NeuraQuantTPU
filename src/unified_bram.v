// Unified_Input_Buffer_BRAM.sv
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
    input logic clk,

    // Port A - Read Port (for reading activations)
    input logic en_a,
    input logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] dout_a,

    // Port B - Write Port (for writing activations)
    input logic en_b,
    input logic we_b,
    input logic [ADDR_WIDTH-1:0] addr_b,
    input logic [DATA_WIDTH-1:0] din_b
);

    // Memory array declaration
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1]; // Memory array

    // Port A Read
    always_ff @(posedge clk) begin
        if (en_a) begin
            dout_a <= mem[addr_a]; // Read data from memory
        end
    end

    // Port B Write
    always_ff @(posedge clk) begin
        if (en_b && we_b) begin
            mem[addr_b] <= din_b; // Write data to memory
        end
    end

    // Optional initialization for simulation
    initial begin
        for (int i = 0; i < DEPTH; i++) begin
            mem[i] = '0; // Initialize memory to zero
        end
    end

endmodule
