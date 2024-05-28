const std = @import("std");
const torch = @import("torch.zig");

pub fn reverseRepeatVector(t: []const i64, comptime n: i64) []i64 {
    std.debug.assert(n >= 0);
    var ret = std.ArrayList(i64).init(torch.global_allocator);
    var rit: i64 = n - 1;
    while (rit >= 0) : (rit -= 1) {
        for (0..n) |_| {
            ret.append(t[@intCast(rit)]) catch unreachable;
        }
    }
    return ret.toOwnedSlice() catch unreachable;
}
