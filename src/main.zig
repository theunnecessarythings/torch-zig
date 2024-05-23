const Tensor = @import("tensor.zig").Tensor;
const torch = @import("torch.zig");
const std = @import("std");
pub fn main() !void {
    const cuda_available = torch.Cuda.isAvailable();
    std.debug.print("CUDA available: {}\n", .{cuda_available});
    var size = [_]i64{ 2, 3 };
    var a = Tensor.rand(&size, torch.FLOAT_CUDA);
    a.print();
}
