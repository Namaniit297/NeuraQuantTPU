// INSTRUCTION_FIFO.sv
module INSTRUCTION_FIFO #(
    parameter FIFO_DEPTH = 32
)(
    input logic clk,                   // Clock signal
    input logic reset,                 // Reset signal
    input logic [WORD_TYPE-1:0] lower_word, // Lower word of the instruction
    input logic [WORD_TYPE-1:0] middle_word, // Middle word of the instruction
    input logic [HALFWORD_TYPE-1:0] upper_word, // Upper halfword of the instruction
    input logic [2:0] write_en,        // Write enable flags for each word
    output logic [INSTRUCTION_TYPE-1:0] output, // Read port of the FIFO
    input logic next_en,               // Read or 'next' enable of the FIFO
    output logic empty,                // Determines if the FIFO is empty
    output logic full                  // Determines if the FIFO is full
);

    // Internal signals
    logic [WORD_TYPE-1:0] lower_output, middle_output;
    logic [HALFWORD_TYPE-1:0] upper_output;

    // Instantiate three FIFOs for each part of the instruction
    FIFO #(.FIFO_WIDTH(4*BYTE_WIDTH), .FIFO_DEPTH(FIFO_DEPTH)) fifo_lower (
        .clk(clk),
        .reset(reset),
        .input_data(lower_word),
        .write_en(write_en[0]),
        .output_data(lower_output),
        .next_en(next_en),
        .empty(empty_lower),
        .full(full_lower)
    );

    FIFO #(.FIFO_WIDTH(4*BYTE_WIDTH), .FIFO_DEPTH(FIFO_DEPTH)) fifo_middle (
        .clk(clk),
        .reset(reset),
        .input_data(middle_word),
        .write_en(write_en[1]),
        .output_data(middle_output),
        .next_en(next_en),
        .empty(empty_middle),
        .full(full_middle)
    );

    FIFO #(.FIFO_WIDTH(2*BYTE_WIDTH), .FIFO_DEPTH(FIFO_DEPTH)) fifo_upper (
        .clk(clk),
        .reset(reset),
        .input_data(upper_word),
        .write_en(write_en[2]),
        .output_data(upper_output),
        .next_en(next_en),
        .empty(empty_upper),
        .full(full_upper)
    );

    // Combine outputs into a single instruction
    assign output = {upper_output, middle_output, lower_output};

    // Determine overall empty and full status
    assign empty = empty_lower | empty_middle | empty_upper;
    assign full = full_lower | full_middle | full_upper;

endmodule
