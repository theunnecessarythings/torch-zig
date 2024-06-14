const torch = @import("../torch.zig");
const std = @import("std");
const Tensor = torch.Tensor;
const Scalar = torch.Scalar;
const TensorOptions = torch.TensorOptions;
const module = torch.module;
const Module = module.Module;
const conv = torch.conv;
const functional = torch.functional;
const Dropout = functional.Dropout;
const Linear = torch.linear.Linear;
const Functional = functional.Functional;
const Conv2D = torch.conv.Conv2D;
const BatchNorm2D = torch.norm.BatchNorm(2);
const Sequential = module.Sequential;

const InvertedResidual = struct {
    base_module: *Module = undefined,
    layers: *Sequential = undefined,
    apply_residual: bool = true,
    const Self = @This();
    pub fn init(
        c_in: i64,
        c_out: i64,
        ksize: i64,
        stride: i64,
        expansion_factor: i64,
        bn_momentum: f64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .apply_residual = c_in == c_out and stride == 1 };
        self.base_module = Module.init(self);
        const mid_ch = c_in * expansion_factor;
        self.layers = Sequential.init(options)
            .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = mid_ch, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = mid_ch, .momentum = bn_momentum, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Conv2D.init(.{ .in_channels = mid_ch, .out_channels = mid_ch, .kernel_size = .{ ksize, ksize }, .stride = .{ stride, stride }, .groups = mid_ch, .padding = .{ .Padding = .{ @divFloor(ksize, 2), @divFloor(ksize, 2) } }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = mid_ch, .momentum = bn_momentum, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Conv2D.init(.{ .in_channels = mid_ch, .out_channels = c_out, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = c_out, .momentum = bn_momentum, .tensor_opts = options }));
        _ = self.base_module.registerModule("layers", self.layers);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.layers.deinit();
        torch.global_allocator.deallocate(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var ys = self.layers.forward(input);
        if (self.apply_residual) return ys.add(input);
        return ys;
    }
};

fn stack(c_in: i64, c_out: i64, ksize: i64, stride: i64, exp_factor: i64, repeats: i64, bn_momentum: f64, options: TensorOptions) *Sequential {
    var layers = Sequential.init(options);
    layers = layers.add(InvertedResidual.init(c_in, c_out, ksize, stride, exp_factor, bn_momentum, options));
    for (1..@intCast(repeats)) |_| {
        layers = layers.add(InvertedResidual.init(c_out, c_out, ksize, 1, exp_factor, bn_momentum, options));
    }
    return layers;
}

fn _f64(x: i64) f64 {
    return @floatFromInt(x);
}

fn _i64(x: f64) i64 {
    return @intFromFloat(x);
}

fn roundToMultipleOf(val: f64, divisor: i64) i64 {
    const new_val = @max(divisor, @divFloor(_i64(val + _f64(divisor) / 2.0), divisor) * divisor);
    if (_f64(new_val) >= 0.9 * val) return new_val;
    return new_val + divisor;
}

fn getDepths(alpha: f64) [8]i64 {
    var depths = [_]i64{ 32, 16, 24, 40, 80, 96, 192, 320 };
    for (&depths) |*depth| {
        depth.* = roundToMultipleOf(alpha * _f64(depth.*), 8);
    }
    return depths;
}

fn mnasnet(alpha: f64, num_classes: i64, dropout: f32, options: TensorOptions) *Sequential {
    const bn_momentum = 1.0 - 0.9997;
    const depths = getDepths(alpha);
    const layers = Sequential.init(options)
        .add(Conv2D.init(.{ .in_channels = 3, .out_channels = depths[0], .kernel_size = .{ 3, 3 }, .stride = .{ 2, 2 }, .padding = .{ .Padding = .{ 1, 1 } }, .bias = false, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = depths[0], .momentum = bn_momentum, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Conv2D.init(.{ .in_channels = depths[0], .out_channels = depths[0], .kernel_size = .{ 3, 3 }, .stride = .{ 1, 1 }, .padding = .{ .Padding = .{ 1, 1 } }, .groups = depths[0], .bias = false, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = depths[0], .momentum = bn_momentum, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Conv2D.init(.{ .in_channels = depths[0], .out_channels = depths[1], .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = depths[1], .momentum = bn_momentum, .tensor_opts = options }))
        .add(stack(depths[1], depths[2], 3, 2, 3, 3, bn_momentum, options))
        .add(stack(depths[2], depths[3], 5, 2, 3, 3, bn_momentum, options))
        .add(stack(depths[3], depths[4], 5, 2, 6, 3, bn_momentum, options))
        .add(stack(depths[4], depths[5], 3, 1, 6, 2, bn_momentum, options))
        .add(stack(depths[5], depths[6], 5, 2, 6, 4, bn_momentum, options))
        .add(stack(depths[6], depths[7], 3, 1, 6, 1, bn_momentum, options))
        .add(Conv2D.init(.{ .in_channels = depths[7], .out_channels = 1280, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = 1280, .momentum = bn_momentum, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init());
    const classifier = Sequential.init(options)
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = 1280, .out_features = num_classes, .bias = true, .tensor_opts = options }));
    const dim = [2]i64{ 2, 3 };
    const block = Sequential.init(options)
        .addWithName("layers", layers)
        .add(Functional(Tensor.meanDim, .{ @constCast(&dim), false, .Float }).init())
        .addWithName("classifier", classifier);
    return block;
}

pub fn mnasnet0_5(num_classes: i64, options: TensorOptions) *Sequential {
    return mnasnet(0.5, num_classes, 0.2, options);
}

pub fn mnasnet0_75(num_classes: i64, options: TensorOptions) *Sequential {
    return mnasnet(0.75, num_classes, 0.2, options);
}

pub fn mnasnet1_0(num_classes: i64, options: TensorOptions) *Sequential {
    return mnasnet(1.0, num_classes, 0.2, options);
}

pub fn mnasnet1_3(num_classes: i64, options: TensorOptions) *Sequential {
    return mnasnet(1.3, num_classes, 0.2, options);
}
