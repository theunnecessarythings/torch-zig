const torch = @import("../torch.zig");
const Tensor = torch.Tensor;
const Scalar = torch.Scalar;
const std = @import("std");
const Module = @import("module.zig").Module;
const ModuleGen = @import("module.zig").ModuleGen;
const nn_init = @import("init.zig");

pub const BatchNormOptions = struct {
    num_features: i64,
    eps: f64 = 1e-5,
    momentum: ?f64 = 0.1,
    affine: bool = true,
    track_running_stats: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const InstanceNormOptions = struct {
    num_features: i64,
    eps: f64 = 1e-5,
    momentum: ?f64 = 0.1,
    affine: bool = true,
    track_running_stats: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const LayerNormOptions = struct {
    normalized_shape: []const i64,
    eps: f64 = 1e-5,
    elementwise_affine: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const GroupNormOptions = struct {
    num_groups: i64,
    num_channels: i64,
    eps: f64 = 1e-5,
    affine: bool = true,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

fn resetImpl(self: anytype) void {
    if (self.options.affine) {
        self.weight = self.base_module.registerParameter(
            "weight",
            Tensor.empty(&[_]i64{self.options.num_features}, self.options.tensor_opts),
            true,
        );
        self.bias = self.base_module.registerParameter(
            "bias",
            Tensor.empty(&[_]i64{self.options.num_features}, self.options.tensor_opts),
            true,
        );
    } else {
        self.weight = self.base_module.registerParameter(
            "weight",
            Tensor.empty(&[_]i64{}, self.options.tensor_opts),
            false,
        );
        self.bias = self.base_module.registerParameter(
            "bias",
            Tensor.empty(&[_]i64{}, self.options.tensor_opts),
            false,
        );
    }
    if (self.options.track_running_stats) {
        self.running_mean = self.base_module.registerBuffer(
            "running_mean",
            Tensor.zeros(&[_]i64{self.options.num_features}, self.options.tensor_opts),
        );
        self.running_var = self.base_module.registerBuffer(
            "running_var",
            Tensor.ones(&[_]i64{self.options.num_features}, self.options.tensor_opts),
        );
        self.num_batches_tracked = self.base_module.registerBuffer(
            "num_batches_tracked",
            // Tensor.ones(&.{1}, self.options.tensor_opts.dtype(.Int)),
            Tensor.fromLong(1),
        );
    } else {
        self.running_mean = self.base_module.registerBuffer(
            "running_mean",
            Tensor.empty(&[_]i64{}, self.options.tensor_opts),
        );
        self.running_var = self.base_module.registerBuffer(
            "running_var",
            Tensor.empty(&[_]i64{}, self.options.tensor_opts),
        );
        self.num_batches_tracked = self.base_module.registerBuffer(
            "num_batches_tracked",
            Tensor.empty(&[_]i64{}, self.options.tensor_opts),
        );
    }
    self.resetParameters();
}

fn resetRunningStatsImpl(self: anytype) void {
    if (self.options.track_running_stats) {
        _ = self.running_mean.zero_();
        _ = self.running_var.fill_(torch.Scalar.float(1.0));
        _ = self.num_batches_tracked.zero_();
    }
}

fn resetParametersImpl(self: anytype) void {
    self.resetRunningStats();
    if (self.options.affine) {
        self.weight = nn_init.ones_(&self.weight);
        self.bias = nn_init.zeros_(&self.bias);
    }
}

pub fn BatchNorm(comptime D: usize) type {
    return struct {
        base_module: *Module = undefined,

        bias: Tensor = undefined,
        weight: Tensor = undefined,
        running_mean: Tensor = undefined,
        running_var: Tensor = undefined,
        num_batches_tracked: Tensor = undefined,
        options: BatchNormOptions = undefined,

        const Self = @This();

        pub fn init(options: BatchNormOptions) *Self {
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
            self.weight.free();
            self.bias.free();
            self.running_mean.free();
            self.running_var.free();
            self.num_batches_tracked.free();
            torch.global_allocator.destroy(self);
        }

        pub fn reset(self: *Self) void {
            resetImpl(self);
        }

        pub fn resetParameters(self: *Self) void {
            resetParametersImpl(self);
        }

        pub fn resetRunningStats(self: *Self) void {
            resetRunningStatsImpl(self);
        }

        fn checkInputDim(self: *Self, input: *const Tensor) void {
            _ = self;
            const dim = input.dim();
            if (D == 1) {
                if (dim != 2 and dim != 3) {
                    std.debug.panic("expected 2D or 3D input (got {d}D input)", .{dim});
                }
            }
            if (dim != D + 2) {
                std.debug.panic("expected {d}D input (got {d}D input)", .{ D + 2, dim });
            }
        }

        pub fn forward(self: *Self, input: *const Tensor) Tensor {
            self.checkInputDim(input);
            var exponential_average_factor: f64 = self.options.momentum orelse 0.0;

            if (self.base_module.isTraining() and self.options.track_running_stats) {
                _ = self.num_batches_tracked.addScalar_(Scalar.int(1));
                exponential_average_factor = self.options.momentum orelse 1.0 / self.num_batches_tracked.doubleValue(&.{0});
            }

            return Tensor.batchNorm(input, &self.weight, &self.bias, &self.running_mean, &self.running_var, self.base_module.isTraining() or !self.options.track_running_stats, exponential_average_factor, self.options.eps, true);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("BatchNorm(num_features={d}, eps={d}, momentum={any}, affine={any}, track_running_stats={any})", .{
                self.options.num_features,
                self.options.eps,
                self.options.momentum,
                self.options.affine,
                self.options.track_running_stats,
            });
        }
    };
}

pub fn InstanceNorm(comptime D: i64) type {
    return struct {
        base_module: *Module = undefined,

        bias: Tensor = undefined,
        weight: Tensor = undefined,
        running_mean: Tensor = undefined,
        running_var: Tensor = undefined,
        num_batches_tracked: Scalar = undefined,
        options: InstanceNormOptions = undefined,

        const Self = @This();

        pub fn init(options: InstanceNormOptions) *Self {
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
            self.weight.free();
            self.bias.free();
            self.running_mean.free();
            self.running_var.free();
            self.num_batches_tracked.free();
            torch.global_allocator.destroy(self);
        }

        pub fn reset(self: *Self) void {
            resetImpl(self);
        }

        pub fn resetParameters(self: *Self) void {
            resetParametersImpl(self);
        }

        fn checkInputDim(input: *const Tensor) void {
            const dim = input.dim();
            if (dim != D + 2 and dim != D + 1) {
                std.debug.panic("expected {d}D or {d}D input (got {d}D input)", .{ D + 2, D + 1, dim });
            }
        }

        fn applyInstanceNorm(self: *Self, input: *const Tensor) Tensor {
            return Tensor.instanceNorm(input, self.weight, self.bias, self.running_mean, self.running_var, Module.isTraining(self) or self.options.track_running_stats, self.options.momentum, self.options.eps, true);
        }

        fn handleNoBatchInput(self: *Self, input: *const Tensor) Tensor {
            return self.applyInstanceNorm(input.unsqueeze(0)).squeeze(0);
        }

        pub fn forward(self: *const Self, input: *const Tensor) Tensor {
            checkInputDim(input);
            if (input.dim() == D + 1) {
                return self.handleNoBatchInput(input);
            } else {
                return self.applyInstanceNorm(input);
            }
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
                "InstanceNorm(num_features={d}, eps={d}, momentum={any}, affine={any}, track_running_stats={any})",
                .{
                    self.options.num_features,
                    self.options.eps,
                    self.options.momentum,
                    self.options.affine,
                    self.options.track_running_stats,
                },
            );
        }
    };
}

pub const LayerNorm = struct {
    base_module: *Module = undefined,

    bias: Tensor = undefined,
    weight: Tensor = undefined,
    options: LayerNormOptions = undefined,

    const Self = @This();

    pub fn init(options: LayerNormOptions) *Self {
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
        self.weight.free();
        self.bias.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        if (self.options.elementwise_affine) {
            self.weight = self.base_module.registerParameter(
                "weight",
                Tensor.empty(self.options.normalized_shape, self.options.tensor_opts),
                true,
            );
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(self.options.normalized_shape, self.options.tensor_opts),
                true,
            );
        } else {
            self.weight = self.base_module.registerParameter(
                "weight",
                Tensor.empty(&[_]i64{}, self.options.tensor_opts),
                false,
            );
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(&[_]i64{}, self.options.tensor_opts),
                false,
            );
        }
        self.resetParameters();
    }

    pub fn resetParameters(self: *Self) void {
        if (self.options.elementwise_affine) {
            self.weight = nn_init.ones_(&self.weight);
            self.bias = nn_init.zeros_(&self.bias);
        }
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        return Tensor.layerNorm(input, self.options.normalized_shape, &self.weight, &self.bias, self.options.eps, true);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("LayerNorm(normalized_shape={d}, eps={d}, elementwise_affine={any})", .{
            self.options.normalized_shape,
            self.options.eps,
            self.options.elementwise_affine,
        });
    }
};

pub const GroupNorm = struct {
    base_module: *Module = undefined,

    bias: Tensor = undefined,
    weight: Tensor = undefined,
    options: GroupNormOptions = undefined,

    const Self = @This();

    pub fn init(options: GroupNormOptions) *Self {
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
        self.weight.free();
        self.bias.free();
        torch.global_allocator.destroy(self);
    }

    pub fn reset(self: *Self) void {
        if (self.options.affine) {
            self.weight = self.base_module.registerParameter(
                "weight",
                Tensor.empty(&[_]i64{self.options.num_channels}, self.options.tensor_opts),
                true,
            );
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(&[_]i64{self.options.num_channels}, self.options.tensor_opts),
                true,
            );
        } else {
            self.weight = self.base_module.registerParameter(
                "weight",
                Tensor.empty(&[_]i64{}, self.options.tensor_opts),
                false,
            );
            self.bias = self.base_module.registerParameter(
                "bias",
                Tensor.empty(&[_]i64{}, self.options.tensor_opts),
                false,
            );
        }
        self.resetParameters();
    }

    pub fn resetParameters(self: *Self) void {
        if (self.options.affine) {
            self.weight = nn_init.ones_(self.weight);
            self.bias = nn_init.zeros_(self.bias);
        }
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        return Tensor.groupNorm(input, self.options.num_groups, self.weight, self.bias, self.options.eps, true);
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
            "GroupNorm(num_groups={d}, num_channels={d}, eps={d}, affine={any})",
            .{ self.options.num_groups, self.options.num_channels, self.options.eps, self.options.affine },
        );
    }
};
