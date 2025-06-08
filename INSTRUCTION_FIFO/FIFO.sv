// FIFO.sv
module FIFO #(
    parameter FIFO_WIDTH = 8,         // Width of the FIFO
    parameter FIFO_DEPTH = 32          // Depth of the FIFO
)(
    input logic clk,                   // Clock signal
    input logic reset,                 // Reset signal
    input logic [FIFO_WIDTH-1:0] input_data, // Data to write
    input logic write_en,              // Write enable signal
    output logic [FIFO_WIDTH-1:0] output_data, // Data read from FIFO
    input logic next_en,               // Read enable signal
    output logic empty,                // FIFO empty flag
    output logic full                  // FIFO full flag
);

    // Internal signals
    logic [FIFO_WIDTH-1:0] fifo_data [0:FIFO_DEPTH-1]; // FIFO storage
    logic [4:0] write_ptr, read_ptr; // Pointers for write and read
    logic [4:0] size; // Current size of the FIFO

    // Write and read operations
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            write_ptr <= 0;
            read_ptr <= 0;
            size <= 0;
            empty <= 1;
            full <= 0;
        end else begin
            if (write_en && !full) begin
                fifo_data[write_ptr] <= input_data; // Write data
                write_ptr <= (write_ptr + 1) % FIFO_DEPTH; // Circular increment
                size <= size + 1;
            end
            if (next_en && !empty) begin
                output_data <= fifo_data[read_ptr]; // Read data
                read_ptr <= (read_ptr + 1) % FIFO_DEPTH; // Circular increment
                size <= size - 1;
            end
            // Update empty and full flags
            empty <= (size == 0);
            full <= (size == FIFO_DEPTH);
        end
    end

endmodule
