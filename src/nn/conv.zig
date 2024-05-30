const std = @import("std");
const torch = @import("../torch.zig");
const Tensor = torch.Tensor;
const module = @import("module.zig");
const nn_init = @import("init.zig");
const Module = module.Module;
const utils = torch.utils;
const ModuleGen = module.ModuleGen;

pub const ConvPaddingMode = enum {
    Zeros,
    Reflect,
    Replicate,
    Circular,
};

pub fn ConvPadding(comptime D: usize) type {
    return union(enum) {
        Same,
        Valid,
        Padding: [D]i64,
    };
}

pub fn ConvOptions(comptime dim: i64) type {
    return struct {
        in_channels: i64,
        out_channels: i64,
        kernel_size: [dim]i64,
        stride: [dim]i64 = [_]i64{1} ** dim,
        padding: ConvPadding(dim) = .{ .Padding = [_]i64{0} ** dim },
        dilation: [dim]i64 = [_]i64{1} ** dim,
        transposed: bool = false,
        output_padding: [dim]i64 = [_]i64{0} ** dim,
        groups: i64 = 1,
        bias: bool = true,
        padding_mode: ConvPaddingMode = .Zeros,
        tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
    };
}

pub fn resetND(self: anytype) void {
    std.debug.assert(self.options.in_channels > 0 and self.options.groups > 0 and self.options.out_channels > 0);
    std.debug.assert(@mod(self.options.in_channels, self.options.groups) == 0);
    std.debug.assert(@mod(self.options.out_channels, self.options.groups) == 0);

    switch (self.options.padding) {
        .Valid => {
            @memset(self._reversed_padding_repeated_twice, 0);
        },
        .Same => {
            for (self.options.stride) |stride| {
                std.debug.assert(stride == 1);
            }

            for (self.options.dilation, 0..) |dilation, i| {
                const kernel_size = self.options.kernel_size[i];
                const total_padding = dilation * (kernel_size - 1);
                const left_pad = @divFloor(total_padding, 2);
                const right_pad = total_padding - left_pad;
                self._reversed_padding_repeated_twice[2 * i] = left_pad;
                self._reversed_padding_repeated_twice[2 * i + 1] = right_pad;
            }
        },
        else => {
            self._reversed_padding_repeated_twice = utils.reverseRepeatVector(&self.options.padding.Padding, 2);
        },
    }

    if (self.options.transposed) {
        const weight_sizes: []i64 = [_]i64{ self.options.in_channels, @divFloor(self.options.out_channels, self.options.groups) } ++ &self.options.kernel_size;
        self.weight = self.base_module.registerParameter("weight", Tensor.empty(weight_sizes, self.options.tensor_opts), true);
    } else {
        const weight_sizes: []i64 = [_]i64{ self.options.out_channels, @divFloor(self.options.in_channels, self.options.groups) } ++ &self.options.kernel_size;
        self.weight = self.base_module.registerParameter("weight", Tensor.empty(weight_sizes, self.options.tensor_opts), true);
    }

    if (self.options.bias) {
        self.bias = self.base_module.registerParameter("bias", Tensor.empty(&[_]i64{self.options.out_channels}, self.options.tensor_opts), true);
    }
    self.resetParameters();
}

fn resetParametersND(self: anytype) void {
    _ = nn_init.kaimingUniform_(&self.weight, std.math.sqrt(5), .FanIn, .LeakyReLU);
    if (self.bias != null) {
        const fan_in_out = nn_init.caclculateFanInAndFanOut(&self.weight);
        const bound = 1.0 / std.math.sqrt(@as(f64, @floatFromInt(fan_in_out[0])));
        _ = nn_init.uniform_(&self.bias.?, -bound, bound);
    }
}

pub const Conv1D = struct {
    base_module: *Module = undefined,
    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: ConvOptions(1) = undefined,
    _reversed_padding_repeated_twice: []i64 = undefined,

    const Self = @This();

    pub fn init(options: ConvOptions(1)) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinitFields();
        if (self.options.bias) {
            self.bias.?.free();
        }
        self.weight.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        resetND(self);
    }

    pub fn resetParameters(self: *Self) void {
        resetParametersND(self);
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        if (self.options.padding_mode != .kZeros) {
            const padded_input = input.pad(self._reversed_padding_repeated_twice, @tagName(self.options.padding_mode), null);
            return Tensor.conv1d(padded_input, self.weight, self.bias, self.options.stride, 0, self.options.dilation, self.options.groups);
        }
        return Tensor.conv1d(input, self.weight, self.bias, self.options.stride, self.options.padding, self.options.dilation, self.options.groups);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Conv1D(in_channels={any}, out_channels={any}, kernel_size={any}, stride={any}, padding={any}, dilation={any}, groups={any}, bias={any}, padding_mode={any})", .{ self.options.in_channels, self.options.out_channels, self.options.kernel_size, self.options.stride, self.options.padding, self.options.dilation, self.options.groups, self.options.bias, self.options.padding_mode });
    }
};

pub const Conv2D = struct {
    base_module: *Module = undefined,

    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: ConvOptions(2) = undefined,

    _reversed_padding_repeated_twice: []i64 = undefined,
    const Self = @This();

    pub fn init(options: ConvOptions(2)) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinitFields();
        if (self.options.bias) {
            self.bias.?.free();
        }
        self.weight.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        resetND(self);
    }

    pub fn resetParameters(self: *Self) void {
        resetParametersND(self);
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        if (self.options.padding_mode != .Zeros) {
            const padding = self._reversed_padding_repeated_twice;
            const padded_input = input.pad(padding, @tagName(self.options.padding_mode), null);
            return Tensor.conv2d(&padded_input, &self.weight, if (self.options.bias) &self.bias.? else null, &self.options.stride, &.{0}, &self.options.dilation, self.options.groups);
        }
        return Tensor.conv2d(input, &self.weight, if (self.options.bias) &self.bias.? else null, &self.options.stride, &self.options.padding.Padding, &self.options.dilation, self.options.groups);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Conv2D(in_channels={any}, out_channels={any}, kernel_size={any}, stride={any}, padding={any}, dilation={any}, groups={any}, bias={any}, padding_mode={any})", .{ self.options.in_channels, self.options.out_channels, self.options.kernel_size, self.options.stride, self.options.padding, self.options.dilation, self.options.groups, self.options.bias, self.options.padding_mode });
    }
};

pub const Conv3D = struct {
    base_module: *Module = undefined,

    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: ConvOptions(3) = undefined,

    _reversed_padding_repeated_twice: []i64 = undefined,
    const Self = @This();

    pub fn init(options: ConvOptions(3)) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinitFields();
        if (self.options.bias) {
            self.bias.?.free();
        }
        self.weight.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        resetND(self);
    }

    pub fn resetParameters(self: *Self) void {
        resetParametersND(self);
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        if (self.options.padding_mode != .kZeros) {
            const padded_input = input.pad(self._reversed_padding_repeated_twice, @tagName(self.options.padding_mode), null);
            return Tensor.conv3d(padded_input, self.weight, self.bias, self.options.stride, 0, self.options.dilation, self.options.groups);
        }
        return Tensor.conv3d(input, self.weight, self.bias, self.options.stride, self.options.padding.Padding, self.options.dilation, self.options.groups);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Conv3D(in_channels={d}, out_channels={d}, kernel_size={any}, stride={any}, padding={any}, dilation={any}, groups={any}, bias={any}, padding_mode={any})", .{ self.options.in_channels, self.options.out_channels, self.options.kernel_size, self.options.stride, self.options.padding, self.options.dilation, self.options.groups, self.options.bias, self.options.padding_mode });
    }
};

// TODO: Implement ConvTranspose1D, ConvTranspose2D, ConvTranspose3D
