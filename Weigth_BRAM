/*
 * Unified Dual-Port Weight BRAM for TPU-like Design
 * This BRAM will store weights for the systolic array.
 * Dual-port allows simultaneous read/write operations.
 *
 * Author: Naman Kalra
 * Date: 2025
 */

`timescale 1ns/1ps

module Unified_Weight_BRAM #(
    parameter DATA_WIDTH = 8,         // Width of each weight
    parameter COLS = 16,              // Number of columns in the systolic array
    parameter ADDR_WIDTH = 8,         // Address width for 256 depth (for 256 weights)
    parameter DEPTH = 1 << ADDR_WIDTH  // Total depth of the weight storage
)(
    input wire clk,

    // Port A - Read Port (for reading weights)
    input wire en_a,
    input wire [ADDR_WIDTH-1:0] addr_a,
    output reg [DATA_WIDTH-1:0] dout_a [0:COLS-1], // Output 16 weights

    // Port B - Write Port (for writing weights)
    input wire en_b,
    input wire we_b,
    input wire [ADDR_WIDTH-1:0] addr_b,
    input wire [DATA_WIDTH*COLS-1:0] din_b // Input 16 weights concatenated
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1][0:COLS-1]; // 2D memory array for weights

    integer i;

    // Port A Read
    always @(posedge clk) begin
        if (en_a) begin
            for (i = 0; i < COLS; i = i + 1) begin
                dout_a[i] <= mem[addr_a][i];
            end
        end
    end

    // Port B Write
    always @(posedge clk) begin
        if (en_b && we_b) begin
            for (i = 0; i < COLS; i = i + 1) begin
                mem[addr_b][i] <= din_b[DATA_WIDTH*i +: DATA_WIDTH];
            end
        end
    end

    // Optional initialization for simulation
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            integer j;
            for (j = 0; j < COLS; j = j + 1) begin
                mem[i][j] = 0;
            end
        end
    end

endmodule
