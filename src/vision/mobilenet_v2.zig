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

fn _f64(x: i64) f64 {
    return @floatFromInt(x);
}

fn _i64(x: f64) i64 {
    return @intFromFloat(x);
}

fn makeDivisible(v: f64, divisor: i64, min_value: ?i64) i64 {
    const min_v = min_value orelse divisor;
    const new_v = @max(min_v, @divFloor(_i64(v) + @divFloor(divisor, 2), divisor) * divisor);
    if (_f64(new_v) < (0.9 * v)) return new_v + divisor;
    return new_v;
}

fn convNormActivation(c_in: i64, c_out: i64, ksize: i64, stride: i64, groups: i64, options: TensorOptions) *Sequential {
    const padding = @divFloor(ksize - 1, 2);
    return Sequential.init(options)
        .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = c_out, .kernel_size = .{ ksize, ksize }, .stride = .{ stride, stride }, .padding = .{ .Padding = .{ padding, padding } }, .groups = groups, .bias = false, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
        .add(Functional(Tensor.relu6, .{}).init());
}

const InvertedResidual = struct {
    base_module: *Module = undefined,
    layers: *Sequential = undefined,
    use_res_connect: bool = false,

    const Self = @This();

    pub fn init(c_in: i64, c_out: i64, stride: i64, expand_ratio: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .use_res_connect = stride == 1 and c_in == c_out };
        self.base_module = Module.init(self);
        const hidden_dim = c_in * expand_ratio;
        var layers = Sequential.init(options);
        if (expand_ratio != 1) {
            layers = layers.add(convNormActivation(c_in, hidden_dim, 1, 1, 1, options));
        }
        self.layers = layers.add(convNormActivation(hidden_dim, hidden_dim, 3, stride, hidden_dim, options))
            .add(Conv2D.init(.{ .in_channels = hidden_dim, .out_channels = c_out, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }));

        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv", self.layers);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        if (self.use_res_connect) return x.add(&self.layers.forward(x));
        return self.layers.forward(x);
    }
};

fn mobilenet(num_classes: i64, width_mult: f64, round_nearest: i64, dropout: f32, options: TensorOptions) *Sequential {
    var c_in = makeDivisible(32 * width_mult, round_nearest, 8);
    const last_channel = makeDivisible(1280 * @max(1.0, width_mult), round_nearest, 8);
    const inverted_residual_setting = [_][4]i64{
        .{ 1, 16, 1, 1 }, .{ 6, 24, 2, 2 }, .{ 6, 32, 3, 2 }, .{ 6, 64, 4, 2 }, .{ 6, 96, 3, 1 }, .{ 6, 160, 3, 2 }, .{ 6, 320, 1, 1 },
    };
    var layers = Sequential.init(options)
        .add(convNormActivation(3, c_in, 3, 2, 1, options));
    for (inverted_residual_setting) |setting| {
        const t, const c, const n, const s = setting;
        const c_out = makeDivisible(_f64(c) * width_mult, round_nearest, null);
        for (0..@intCast(n)) |i| {
            const stride = if (i == 0) s else 1;
            layers = layers.add(InvertedResidual.init(c_in, c_out, stride, t, options));
            c_in = c_out;
        }
    }
    layers = layers.add(convNormActivation(c_in, last_channel, 1, 1, 1, options));
    return Sequential.init(options)
        .addWithName("features", layers)
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .add(Dropout.init(dropout))
        .addWithName("classifier.1", Linear.init(.{ .in_features = last_channel, .out_features = num_classes, .tensor_opts = options }));
}

pub fn mobilenetV2(num_classes: i64, options: TensorOptions) *Sequential {
    return mobilenet(num_classes, 1.0, 8, 0.2, options);
}
