const torch = @import("../torch.zig");
const std = @import("std");
const Tensor = torch.Tensor;
const TensorOptions = torch.TensorOptions;
const module = torch.module;
const Module = module.Module;
const err = torch.utils.err;

pub fn Functional(comptime func: anytype, comptime args: anytype) type {
    return struct {
        base_module: *Module = undefined,

        pub const Self = @This();

        pub fn init() *Self {
            var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
            self.* = Self{};
            self.base_module = Module.init(self);
            self.reset();
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.base_module.deinit();
            torch.global_allocator.destroy(self);
        }

        pub fn reset(self: *Self) void {
            _ = self;
        }

        pub fn forward(self: *Self, input: *const Tensor) Tensor {
            _ = self;
            return @call(.auto, func, .{input} ++ args);
        }
    };
}

pub const Dropout = struct {
    base_module: *Module = undefined,
    p: f32 = 0.5,

    pub const Self = @This();

    pub fn init(p: f32) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .p = p,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        _ = self;
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        return Tensor.dropout(input, self.p, self.base_module.isTraining());
    }
};
