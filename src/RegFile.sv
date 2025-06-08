// REGISTER_FILE.sv
/*
 * Unified Register File for TPU-like Design
 * This component includes accumulator registers. Registers are accumulated or overwritten.
 * The register file consists of block RAM, which is redundant for a separate accumulation port.
 *
 * Author: BLACKBOXAI
 * Date: 2024
 */

`timescale 1ns/1ps

module REGISTER_FILE #(
    parameter MATRIX_WIDTH = 14,      // Width of each matrix element
    parameter REGISTER_DEPTH = 512    // Number of registers
)(
    input logic clk,                  // Clock signal
    input logic reset,                // Reset signal
    input logic enable,               // Enable signal for writing/reading

    input logic [ACCUMULATOR_ADDRESS_TYPE-1:0] write_address, // Address for writing
    input logic [WORD_ARRAY_TYPE(0:MATRIX_WIDTH-1)] write_port, // Data to write
    input logic write_enable,          // Write enable signal

    input logic accumulate,            // Accumulate signal

    input logic [ACCUMULATOR_ADDRESS_TYPE-1:0] read_address, // Address for reading
    output logic [WORD_ARRAY_TYPE(0:MATRIX_WIDTH-1)] read_port // Data read from the register
);

    // Define the register storage
    logic [4*BYTE_WIDTH*MATRIX_WIDTH-1:0] accumulators [0:REGISTER_DEPTH-1]; // Register array
    logic [4*BYTE_WIDTH*MATRIX_WIDTH-1:0] accumulators_copy [0:REGISTER_DEPTH-1]; // Copy for accumulation

    // Internal signals for DSP operations
    logic [WORD_ARRAY_TYPE(0:MATRIX_WIDTH-1)] dsp_add_port0, dsp_add_port1;
    logic [WORD_ARRAY_TYPE(0:MATRIX_WIDTH-1)] dsp_result_port;
    
    // Pipeline registers for accumulation
    logic [WORD_ARRAY_TYPE(0:MATRIX_WIDTH-1)] accumulate_port_pipe0, accumulate_port_pipe1;

    // Write and read operations
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset all accumulators to zero
            for (int i = 0; i < REGISTER_DEPTH; i++) begin
                accumulators[i] = '0;
                accumulators_copy[i] = '0;
            end
        end else if (enable) begin
            // Write operation
            if (write_enable) begin
                accumulators[write_address] <= write_port; // Write data to the register
                accumulators_copy[write_address] <= write_port; // Copy for accumulation
            end
            
            // Read operation
            read_port <= accumulators[read_address]; // Read data from the register
            
            // Accumulate operation
            if (accumulate) begin
                accumulators[write_address] <= accumulators[write_address] + accumulators_copy[write_address]; // Accumulate data
            end
        end
    end

    // DSP Addition Process
    always_ff @(posedge clk) begin
        for (int i = 0; i < MATRIX_WIDTH; i++) begin
            dsp_result_port[i] <= accumulators[i] + accumulators_copy[i]; // Example DSP operation
        end
    end

endmodule
