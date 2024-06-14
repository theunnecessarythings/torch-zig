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

const Activation = enum { Hardswish, ReLU, None };

fn activationFn(x: *const Tensor, activation: Activation) Tensor {
    switch (activation) {
        .Hardswish => return x.hardswish(),
        .ReLU => return x.relu(),
        .None => return x.shallowClone(),
    }
}

fn convNormActivation(c_in: i64, c_out: i64, ksize: i64, stride: i64, groups: i64, dilation: i64, comptime activation: Activation, options: TensorOptions) *Sequential {
    const padding = @divFloor(ksize - 1, 2) * dilation;

    return Sequential.init(options)
        .add(Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = .{ ksize, ksize },
        .stride = .{ stride, stride },
        .padding = .{ .Padding = .{ padding, padding } },
        .groups = groups,
        .bias = false,
        .dilation = .{ dilation, dilation },
        .tensor_opts = options,
    }))
        .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
        .add(Functional(activationFn, .{activation}).init());
}

const InvertedResidualConfig = struct {
    c_in: i64,
    ksize: i64,
    expanded_channels: i64,
    c_out: i64,
    use_se: bool,
    use_hs: bool,
    stride: i64,
    dilation: i64,
    width_mult: f64,
    options: TensorOptions,

    pub fn adjustChannels(channels: i64, width_mult: f64) i64 {
        return makeDivisible(@as(f64, @floatFromInt(channels)) * width_mult, 8, null);
    }

    pub fn init(c_in: i64, ksize: i64, expanded_channels: i64, c_out: i64, use_se: bool, comptime use_hs: bool, stride: i64, dilation: i64, width_mult: f64, options: TensorOptions) InvertedResidualConfig {
        const new_c_in = adjustChannels(c_in, width_mult);
        const new_expanded_channels = adjustChannels(expanded_channels, width_mult);
        const new_c_out = adjustChannels(c_out, width_mult);
        return .{ .c_in = new_c_in, .ksize = ksize, .expanded_channels = new_expanded_channels, .c_out = new_c_out, .use_se = use_se, .use_hs = use_hs, .stride = stride, .dilation = dilation, .width_mult = width_mult, .options = options };
    }
};

const SqueezeExcitation = struct {
    base_module: *Module = undefined,
    fc1: *Conv2D = undefined,
    fc2: *Conv2D = undefined,

    const Self = @This();

    pub fn init(c_in: i64, c_squeeze: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.fc1 = Conv2D.init(.{ .in_channels = c_in, .out_channels = c_squeeze, .kernel_size = .{ 1, 1 }, .tensor_opts = options });
        self.fc2 = Conv2D.init(.{ .in_channels = c_squeeze, .out_channels = c_in, .kernel_size = .{ 1, 1 }, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        self.fc1.reset();
        self.fc2.reset();
        _ = self.base_module.registerModule("fc1", self.fc1);
        _ = self.base_module.registerModule("fc2", self.fc2);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = x.adaptiveAvgPool2d(&.{ 1, 1 });
        y = self.fc1.forward(&y).relu();
        y = self.fc2.forward(&y).hardsigmoid();
        return x.mul(&y);
    }
};

const InvertedResidual = struct {
    base_module: *Module = undefined,
    layers: *Sequential = undefined,
    use_res_connect: bool,

    const Self = @This();

    pub fn init(comptime cfg: InvertedResidualConfig) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .use_res_connect = cfg.stride == 1 and cfg.c_in == cfg.c_out };
        self.base_module = Module.init(self);
        const activation = switch (cfg.use_hs) {
            true => Activation.Hardswish,
            false => Activation.ReLU,
        };
        self.layers = Sequential.init(cfg.options);
        if (cfg.expanded_channels != cfg.c_in) {
            self.layers = self.layers.add(convNormActivation(cfg.c_in, cfg.expanded_channels, 1, 1, 1, 1, activation, cfg.options));
        }
        const stride = if (cfg.dilation > 1) 1 else cfg.stride;
        self.layers = self.layers.add(convNormActivation(cfg.expanded_channels, cfg.expanded_channels, cfg.ksize, stride, cfg.expanded_channels, cfg.dilation, activation, cfg.options));
        if (cfg.use_se) {
            const c_squeeze = makeDivisible(@divFloor(cfg.expanded_channels, 4), 8, null);
            self.layers = self.layers.add(SqueezeExcitation.init(cfg.expanded_channels, c_squeeze, cfg.options));
        }
        self.layers = self.layers.add(convNormActivation(cfg.expanded_channels, cfg.c_out, 1, 1, 1, 1, Activation.None, cfg.options));

        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("block", self.layers);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.layers.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        const y = self.layers.forward(x);
        if (self.use_res_connect) {
            return x.add(&y);
        }
        return y;
    }
};

fn mobilenetV3(comptime cfgs: []const InvertedResidualConfig, last_channel: i64, num_classes: i64, dropout: f32) *Sequential {
    const firstconv_c_out = cfgs[0].c_in;
    var layers = Sequential.init(cfgs[0].options)
        .add(convNormActivation(3, firstconv_c_out, 3, 2, 1, 1, .Hardswish, cfgs[0].options));
    inline for (cfgs) |cfg| {
        layers = layers.add(InvertedResidual.init(cfg));
    }
    const last_cfg = cfgs[cfgs.len - 1];
    layers = layers.add(convNormActivation(last_cfg.c_out, last_cfg.c_out * 6, 1, 1, 1, 1, .Hardswish, last_cfg.options));
    const classifier = Sequential.init(last_cfg.options)
        .add(Linear.init(.{ .in_features = last_cfg.c_out * 6, .out_features = last_channel, .tensor_opts = last_cfg.options }))
        .add(Functional(Tensor.hardswish, .{}).init())
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = last_channel, .out_features = num_classes, .tensor_opts = last_cfg.options }));

    return Sequential.init(last_cfg.options)
        .addWithName("features", layers)
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .addWithName("classifier", classifier);
}

const MobileNetV3Kind = enum { Small, Large };

fn mobilenetV3Conf(arch: MobileNetV3Kind, width_mult: f64, reduced_tail: bool, dilated: bool, options: TensorOptions) struct { cfgs: []const InvertedResidualConfig, last_channel: i64 } {
    const reduce_divider: i64 = if (reduced_tail) 2 else 1;
    const dilation: i64 = if (dilated) 2 else 1;
    const inverted_residual_setting = switch (arch) {
        .Large => &.{
            InvertedResidualConfig.init(16, 3, 16, 16, false, false, 1, 1, width_mult, options),
            InvertedResidualConfig.init(16, 3, 64, 24, false, false, 2, 1, width_mult, options),
            InvertedResidualConfig.init(24, 3, 72, 24, false, false, 1, 1, width_mult, options),
            InvertedResidualConfig.init(24, 5, 72, 40, true, false, 2, 1, width_mult, options),
            InvertedResidualConfig.init(40, 5, 120, 40, true, false, 1, 1, width_mult, options),
            InvertedResidualConfig.init(40, 5, 120, 40, true, false, 1, 1, width_mult, options),
            InvertedResidualConfig.init(40, 3, 240, 80, false, true, 2, 1, width_mult, options),
            InvertedResidualConfig.init(80, 3, 200, 80, false, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(80, 3, 184, 80, false, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(80, 3, 184, 80, false, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(80, 3, 480, 112, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(112, 3, 672, 112, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(112, 5, 672, @divFloor(160, reduce_divider), true, true, 2, dilation, width_mult, options),
            InvertedResidualConfig.init(@divFloor(160, reduce_divider), 5, @divFloor(960, reduce_divider), @divFloor(160, reduce_divider), true, true, 1, dilation, width_mult, options),
            InvertedResidualConfig.init(@divFloor(160, reduce_divider), 5, @divFloor(960, reduce_divider), @divFloor(160, reduce_divider), true, true, 1, dilation, width_mult, options),
        },
        .Small => &.{
            InvertedResidualConfig.init(16, 3, 16, 16, true, false, 2, 1, width_mult, options),
            InvertedResidualConfig.init(16, 3, 72, 24, false, false, 2, 1, width_mult, options),
            InvertedResidualConfig.init(24, 3, 88, 24, false, false, 1, 1, width_mult, options),
            InvertedResidualConfig.init(24, 5, 96, 40, true, true, 2, 1, width_mult, options),
            InvertedResidualConfig.init(40, 5, 240, 40, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(40, 5, 240, 40, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(40, 5, 120, 48, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(48, 5, 144, 48, true, true, 1, 1, width_mult, options),
            InvertedResidualConfig.init(48, 5, 288, @divFloor(96, reduce_divider), true, true, 2, dilation, width_mult, options),
            InvertedResidualConfig.init(@divFloor(96, reduce_divider), 5, @divFloor(576, reduce_divider), @divFloor(96, reduce_divider), true, true, 1, dilation, width_mult, options),
            InvertedResidualConfig.init(@divFloor(96, reduce_divider), 5, @divFloor(576, reduce_divider), @divFloor(96, reduce_divider), true, true, 1, dilation, width_mult, options),
        },
    };
    const last_channel = switch (arch) {
        .Large => InvertedResidualConfig.adjustChannels(@divFloor(1280, reduce_divider), width_mult),
        .Small => InvertedResidualConfig.adjustChannels(@divFloor(1024, reduce_divider), width_mult),
    };

    return .{ .cfgs = inverted_residual_setting, .last_channel = last_channel };
}

pub fn mobilenetV3Small(num_classes: i64, comptime options: TensorOptions) *Sequential {
    const cfgs = comptime mobilenetV3Conf(MobileNetV3Kind.Small, 1.0, false, false, options);
    return mobilenetV3(cfgs.cfgs, cfgs.last_channel, num_classes, 0.2);
}

pub fn mobilenetV3Large(num_classes: i64, comptime options: TensorOptions) *Sequential {
    const cfgs = comptime mobilenetV3Conf(MobileNetV3Kind.Large, 1.0, false, false, options);
    return mobilenetV3(cfgs.cfgs, cfgs.last_channel, num_classes, 0.2);
}
