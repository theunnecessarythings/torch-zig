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

fn makeDivisible(v: f64, divisor: i64, min_value: ?i64) i64 {
    const min_v = min_value orelse divisor;
    const new_v = @max(min_v, (v + divisor / 2) / divisor * divisor);
    if (new_v < (0.9 * v)) return new_v + divisor;
    return new_v;
}

const Activation = enum { ReLU, None };

fn convNormActivation(c_in: i64, c_out: i64, ksize: i64, stride: i64, groups: i64, dilation: i64, activation: Activation, options: TensorOptions) *Sequential {
    const padding = (ksize - 1) / 2 * dilation;
    var seq = Sequential.init(options)
        .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = c_out, .kernel_size = ksize, .stride = stride, .groups = groups, .padding = padding, .bias = false, .dilation = dilation, .tensor_opts = options }))
        .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }));
    switch (activation) {
        .ReLU => _ = seq.add(Functional(Tensor.relu, .{}).init()),
        .None => {},
    }
    return seq;
}

fn simpleStemIn(width_in: i64, width_out: i64, activation: Activation, options: TensorOptions) *Sequential {
    return convNormActivation(width_in, width_out, 3, 2, 1, 1, activation, options);
}

const SqueezeExcitation = struct {
    base_module: *Module = undefined,
    scale: *Sequential = undefined,
    const Self = @This();

    pub fn init(c_in: i64, c_squeeze: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.scale = Sequential.init(options)
            .add(Functional(Tensor.adaptiveAvgPool2d, .{.{ 1, 1 }}).init())
            .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = c_squeeze, .kernel_size = 1, .tensor_opts = options }).init())
            .add(Functional(Tensor.relu, .{}).init())
            .add(Conv2D.init(.{ .in_channels = c_squeeze, .out_channels = c_in, .kernel_size = 1, .tensor_opts = options }).init())
            .add(Functional(Tensor.sigmoid, .{}).init());
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("scale", self.scale);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.scale.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        return x.mul(&self.scale.forward(x));
    }
};

fn bottleneckTransform(width_in: i64, width_out: i64, stride: i64, group_width: i64, bottleneck_multiplier: f64, se_ratio: ?f64, options: TensorOptions) *Sequential {
    const w_b = @round(width_out * bottleneck_multiplier);
    const g = w_b / group_width;
    var layers = Sequential.init(options)
        .addWithName("a", convNormActivation(width_in, w_b, 1, 1, 1, 1, .ReLU, options))
        .addWithName("b", convNormActivation(w_b, w_b, 3, stride, g, 1, .ReLU, options));
    if (se_ratio) |ser| {
        const width_se_out = (width_in * ser);
        layers = layers.addWithName("se", SqueezeExcitation.init(w_b, width_se_out, options));
    }
    layers = layers.addWithName("c", convNormActivation(w_b, width_out, 1, 1, 1, 1, .None, options));
    return layers;
}

fn anyStage(width_in: i64, width_out: i64, stride: i64, depth: i64, group_width: i64, bottleneck_multiplier: f64, se_ratio: ?f64, stage_index: i64, options: TensorOptions) *Sequential {
    var layers = Sequential.init(options);
    for (0..depth) |i| {
        const w_in = if (i == 0) width_in else width_out;
        const s = if (i == 0) stride else 1;
        const name = std.fmt.allocPrint(torch.global_allocator, "block{d}-{d}", .{ stage_index, i }) catch unreachable;
        layers = layers.addWithName(name, resBottleneckBlock(w_in, width_out, s, group_width, bottleneck_multiplier, se_ratio, options));
    }
    return layers;
}

const BlockParams = struct {
    depths: []i64,
    widths: []i64,
    group_widths: []i64,
    bottleneck_multipliers: []f64,
    strides: []i64,
    se_ratio: ?f64,

    pub fn fromInitParams(depth: i64, w_0: i64, w_a: f64, w_m: f64, group_width: i64, bottleneck_multiplier: f64, se_ratio: ?f64) BlockParams {
        const quant = 8.0;
        const stride = 2;
        var widths_cont = std.ArrayList(f64).init(torch.global_allocator);
        for (0..depth) |i| {
            widths_cont.append(i * w_a + w_0) catch unreachable;
        }
        var block_capacity = std.ArrayList(f64).init(torch.global_allocator);
        for (widths_cont) |w| {
            block_capacity.append(@round(@log(w / w_0) / @log(w_m))) catch unreachable;
        }
        var block_widths = std.ArrayList(i64).init(torch.global_allocator);
        for (block_capacity) |c| {
            block_widths.append(@round(w_0 * std.math.powi(i64, w_m, c) / quant - 1e-6) * quant) catch unreachable;
        }
        var stages = std.AutoArrayHashMap(i64, void).init(torch.global_allocator);
        for (block_widths) |w| {
            stages.put(w, void{}) catch unreachable;
        }
        const num_stages = stages.count();
        var b0 = block_widths.clone();
        b0.append(0) catch unreachable;
        var b1 = std.ArrayList(i64).init(torch.global_allocator);
        b1.append(0) catch unreachable;
        b1.appendSlice(block_widths.items) catch unreachable;

        var split_helper = std.ArrayList(struct { i64, i64, i64, i64 }).init(torch.global_allocator);
        for (0..block_widths.items.len + 1) |i| {
            split_helper.append(.{ b0.items[i], b1.items[i], b0.items[i], b1.items[i] }) catch unreachable;
        }
    }
};
