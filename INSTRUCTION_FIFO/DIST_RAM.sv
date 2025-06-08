
// DIST_RAM.sv
module DIST_RAM #(
    parameter DATA_WIDTH = 8,         // Width of a data word
    parameter DATA_DEPTH = 32,        // Depth of the memory
    parameter ADDRESS_WIDTH = 5        // Width of the addresses
)(
    input logic clk,                   // Clock signal
    input logic [ADDRESS_WIDTH-1:0] in_addr, // Input address for writing
    input logic [DATA_WIDTH-1:0] input_data, // Data to write
    input logic write_en,              // Write enable signal
    input logic [ADDRESS_WIDTH-1:0] out_addr, // Output address for reading
    output logic [DATA_WIDTH-1:0] output_data // Data read from memory
);

    // Memory array declaration
    logic [DATA_WIDTH-1:0] ram [0:DATA_DEPTH-1];

    // Write and read operations
    always_ff @(posedge clk) begin
        if (write_en) begin
            ram[in_addr] <= input_data; // Write data to the specified address
        end
    end

    // Read operation (combinational)
    assign output_data = ram[out_addr]; // Read data from the specified address

endmodule
