const Tensor = @import("tensor.zig").Tensor;
const torch = @import("torch.zig");
const std = @import("std");
pub fn main() !void {
    var size = [_]i64{ 2, 3 };
    var a = Tensor.rand(&size, torch.FLOAT_CPU);
    a.print();
}
