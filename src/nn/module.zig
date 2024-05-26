const torch = @import("torch");
const Tensor = @import("torch").Tensor;
const std = @import("std");

pub fn Module(comptime T: type) type {
    return struct {
        pub fn initFields(self: *T) void {
            self.children_ = std.StringArrayHashMap(*T).init(torch.global_allocator);
            self.parameters_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
            self.buffers_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
        }

        pub fn deinitFields(self: *T) void {
            if (self.children_) self.children_.deinit();
            if (self.parameters_) self.parameters_.deinit();
            if (self.buffers_) self.buffers_.deinit();
        }

        pub fn apply(
            self: *const T,
            prefix: []const u8,
            result: anytype,
            function: fn (*const T, []const u8, *anyopaque) void,
        ) !void {
            function(self, prefix, result);
            for (self.children_.keys()) |key| {
                var child = self.children_.get(key).?;
                const new_prefix = std.fmt.allocPrint(torch.global_allocator, "{s}{s}.", .{ prefix, key }) catch unreachable;
                try child.apply(new_prefix, result, function);
            }
        }

        pub fn name(self: *const T) []const u8 {
            if (comptime @hasField(T, "nameImpl")) {
                return self.nameImpl();
            }
            return @typeName(T);
        }

        pub fn clone(self: *const T) T {
            // TODO: Implement clone
            return self;
        }

        pub fn parameters(self: *const T, recurse: bool) std.ArrayList(Tensor) {
            return self.namedParameters(recurse);
        }

        pub fn namedParametersApply(self: *const T, prefix: []const u8, result: *anyopaque) void {
            var result_: *std.StringArrayHashMap(Tensor) = @ptrCast(@alignCast(result));
            for (self.parameters_.keys()) |key| {
                const value = self.parameters_.get(key).?;
                const full_key = std.fmt.allocPrint(torch.global_allocator, "{s}{s}", .{ prefix, key }) catch unreachable;
                result_.put(full_key, value) catch unreachable;
            }
        }

        pub fn namedParameters(self: *const T, recurse: bool) !std.StringArrayHashMap(Tensor) {
            var result = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
            if (!recurse) {
                for (self.parameters_.keys()) |key| {
                    const value = self.parameters_.get(key).?;
                    if (value.defined()) {
                        try result.put(key, value);
                    }
                }
            } else {
                @setEvalBranchQuota(2000);
                try self.apply("", &result, namedParametersApply);
            }
            return result;
        }

        // pub fn buffers(self: *const T,recurse: bool) std.ArrayList(Tensor) {
        //
        // }
        //
        // pub fn namedBuffers(self: *const T, recurse: bool) std.ArrayList(([]const u8, Tensor)) {
        //
        // }
        //
        // pub fn modules(self: *const T, include_self: bool) std.ArrayList(T) {
        //
        // }
        //
        // pub fn namedModules(self: *const T, include_self: bool) std.ArrayList(([]const u8, T)) {
        //
        // }
        //
        // pub fn children(self: *const T) std.ArrayList(T) {
        //
        // }
        //
        // pub fn namedChildren(self: *const T) std.ArrayList(([]const u8, T)) {
        //
        // }
        //
        // pub fn train(self: * T, on: bool) void {
        //
        // }
        //
        // pub fn eval(self: * T) void{
        //
        // }
        //
        // pub fn isTraining(self: *const T) bool {
        //     return false;
        // }
        //
        // pub fn to(self: *T, device: torch.Device, dtype: torch.DType, non_blocking: bool) void {
        //
        // }
        //
        // pub fn zeroGrad(self: *T) bool  {
        //
        // }
        //
        // pub fn save(self: *const T, path: []const u8) void {
        //
        // }
        //
        // pub fn load(self: *T, path: []const u8) void {
        //
        // }
        //
        pub fn registerParameter(self: *T, name_: []const u8, tensor: Tensor, requires_grad_: bool) void {
            // TODO: add checks here, decide on whether to return an error or not
            if (!tensor.defined()) {
                if (requires_grad_) {
                    std.log.warn("An undefined tensor cannot require grad. " ++
                        "Ignoring the `requires_grad` argument.", .{});
                }
            } else {
                _ = tensor.setRequiresGrad(requires_grad_);
            }
            self.parameters_.put(name_, tensor) catch unreachable;
        }
        //
        //
        // pub fn registerBuffer(self: *T, name: []const u8, tensor: Tensor) *Tensor {
        //
        // }
        //
        // pub fn registerModule(self: *T, name: []const u8, module: *T) *T {
        //
        // }
    };
}
