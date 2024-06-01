const torch = @import("../torch.zig");
const Tensor = torch.Tensor;
const std = @import("std");
const Module = @import("module.zig").Module;
const ModuleGen = @import("module.zig").ModuleGen;
const nn_init = @import("init.zig");

pub const LinearOptions = struct {
    in_features: i64,
    out_features: i64,
    bias: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const FlattenOptions = struct {
    start_dim: i64 = 1,
    end_dim: i64 = -1,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const UnflattenOptions = struct {
    dim: i64 = 0,
    dim_name: ?[]const u8 = null,
    sizes: ?[]i64 = null,
    named_shape: ?[]struct { []const u8, i64 } = null,

    pub fn init(dim: i64, sizes: []i64) UnflattenOptions {
        return UnflattenOptions{
            .dim = dim,
            .sizes = sizes,
        };
    }

    pub fn initWithNamedShape(
        dim_name: []const u8,
        named_shape: []struct { []const u8, i64 },
    ) UnflattenOptions {
        return UnflattenOptions{
            .dim_name = dim_name,
            .named_shape = named_shape,
        };
    }
};

pub const BilinearOptions = struct {
    in1_features: i64,
    in2_features: i64,
    out_features: i64,
    bias: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const Identity = struct {
    base_module: *Module = undefined,

    const Self = @This();

    pub fn init() *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn nameImpl(self: *const Self) []const u8 {
        _ = self;
        return "torch::nn::Identity()";
    }

    pub fn reset(self: *Self) void {
        _ = self;
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        _ = self;
        return input.*;
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        _ = self;
        try writer.writeAll("torch::nn::Identity()");
    }
};

pub const Linear = struct {
    base_module: *Module = undefined,

    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: LinearOptions = undefined,

    const Self = @This();

    pub fn init(options: LinearOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        if (self.options.bias) {
            self.bias.?.free();
        }
        self.weight.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        var size = [_]i64{ self.options.out_features, self.options.in_features };
        self.weight = self.base_module.registerParameter(
            "weight",
            Tensor.empty(&size, self.options.tensor_opts),
            true,
        );
        if (self.options.bias) {
            var size_ = [_]i64{self.options.out_features};
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(&size_, self.options.tensor_opts),
                true,
            );
        }
        self.resetParameters();
    }

    pub fn resetParameters(self: *Self) void {
        _ = nn_init.kaimingUniform_(&self.weight, std.math.sqrt(5), .FanIn, .LeakyReLU);
        if (self.bias != null) {
            const fan_in_out = nn_init.caclculateFanInAndFanOut(&self.weight);
            const bound = 1.0 / std.math.sqrt(@as(f64, @floatFromInt(fan_in_out[0])));
            _ = nn_init.uniform_(&self.bias.?, -bound, bound);
        }
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        return Tensor.linear(input, &self.weight, if (self.options.bias) &self.bias.? else null);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "torch::nn::Linear(in_features={d}, out_features={d}, bias={any})",
            .{ self.options.in_features, self.options.out_features, self.options.bias },
        );
    }
};

pub const Flatten = struct {
    base_module: *Module = undefined,

    options: FlattenOptions = FlattenOptions{},

    const Self = @This();

    pub fn init(options: FlattenOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn reset() void {}

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        return input.flatten(self.options.start_dim, self.options.end_dim);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "torch::nn::Flatten(start_dim={d}, end_dim={d})",
            .{ self.options.start_dim, self.options.end_dim },
        );
    }
};

pub const Unflatten = struct {
    base_module: *Module = undefined,

    options: UnflattenOptions = UnflattenOptions{},

    const Self = @This();

    pub fn init(options: UnflattenOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn reset() void {}

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        _ = self;
        _ = input;
        @panic("Unflatten is not implemented yet");
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        if (self.options.named_shape != null) {
            try writer.write("torch::nn::Unflatten(dim_name=\"{s}\", unflattened_size={{", self.options.dim_name);
            for (self.options.named_shape) |shape| {
                try writer.write("{{\"{s}\", {d}}, ", shape[0], shape[1]);
            }
            try writer.write("}})");
        } else {
            try writer.write("torch::nn::Unflatten(dim={d}, unflattened_size={{", self.options.dim);
            for (self.options.sizes) |size| {
                try writer.write("{d}, ", size);
            }
            try writer.write("}})");
        }
    }
};

pub const Bilinear = struct {
    base_module: *Module = undefined,

    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: BilinearOptions = BilinearOptions{},
    const Self = @This();

    pub fn init(options: BilinearOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        if (self.options.bias) {
            self.bias.?.free();
        }
        self.weight.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        var size = [_]i64{ self.options.out_features, self.options.in1_features, self.options.in2_features };
        self.weight = self.base_module.registerParameter(
            "weight",
            Tensor.empty(&size, self.options.tensor_opts),
            true,
        );
        if (self.options.bias) {
            size = [_]i64{self.options.out_features};
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(&size, self.options.tensor_opts),
                true,
            );
        }
        self.resetParameters();
    }

    pub fn resetParameters(self: *Self) void {
        const bound = 1.0 / std.math.sqrt(self.options.in1_features);
        _ = nn_init.uniform_(self.weight, -bound, bound);
        if (self.bias != null) {
            _ = nn_init.uniform_(self.bias.?, -bound, bound);
        }
    }

    pub fn forward(self: *const Self, input1: *const Tensor, input2: *const Tensor) Tensor {
        return Tensor.bilinear(input1, input2, self.weight, self.bias);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "torch::nn::Bilinear(in1_features={d}, in2_features={d}, out_features={d}, bias={any})",
            .{ self.options.in1_features, self.options.in2_features, self.options.out_features, self.options.bias },
        );
    }
};
