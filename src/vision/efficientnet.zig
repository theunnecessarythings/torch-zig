const torch = @import("../torch.zig");
const std = @import("std");
const Tensor = torch.Tensor;
const Scalar = torch.Scalar;
const TensorOptions = torch.TensorOptions;
const module = torch.module;
const Module = module.Module;
const conv = torch.conv;
const functional = torch.functional;
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

const BlockType = enum { MBConv, FusedMBConv };

const MBConvConfig = struct {
    expand_ratio: f64,
    ksize: i64,
    stride: i64,
    c_in: i64,
    c_out: i64,
    num_layers: i64,
    block_type: BlockType,
    options: TensorOptions,

    const Self = @This();
    pub fn adjustChannels(c_in: i64, width_mult: f64, min_value: ?i64) i64 {
        return makeDivisible(_f64(c_in) * width_mult, 8, min_value);
    }

    pub fn adjustDepth(num_layers: i64, depth_mult: f64) i64 {
        return _i64(std.math.ceil(_f64(num_layers) * depth_mult));
    }

    pub fn mbconv(
        expand_ratio: f64,
        ksize: i64,
        stride: i64,
        c_in: i64,
        c_out: i64,
        num_layers: i64,
        width_mult: f64,
        depth_mult: f64,
        options: TensorOptions,
    ) Self {
        return Self{
            .expand_ratio = expand_ratio,
            .ksize = ksize,
            .stride = stride,
            .c_in = adjustChannels(c_in, width_mult, null),
            .c_out = adjustChannels(c_out, width_mult, null),
            .num_layers = adjustDepth(num_layers, depth_mult),
            .block_type = .MBConv,
            .options = options,
        };
    }

    pub fn fusedMBConv(
        expand_ratio: f64,
        ksize: i64,
        stride: i64,
        c_in: i64,
        c_out: i64,
        num_layers: i64,
        options: TensorOptions,
    ) Self {
        return Self{
            .expand_ratio = expand_ratio,
            .ksize = ksize,
            .stride = stride,
            .c_in = c_in,
            .c_out = c_out,
            .num_layers = num_layers,
            .block_type = .FusedMBConv,
            .options = options,
        };
    }
};

const ConvNormActivation = struct {
    base_module: *Module = undefined,
    layers: *Sequential = undefined,
    conv: *Conv2D = undefined,
    norm: *BatchNorm2D = undefined,
    activation: bool,
    options: TensorOptions,

    const Self = @This();

    pub fn init(
        c_in: i64,
        c_out: i64,
        ksize: i64,
        stride: i64,
        groups: i64,
        activation: bool,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .activation = activation, .options = options };
        self.base_module = Module.init(self);
        const padding = @divFloor(ksize - 1, 2);
        self.conv = Conv2D.init(.{
            .in_channels = c_in,
            .out_channels = c_out,
            .kernel_size = .{ ksize, ksize },
            .stride = .{ stride, stride },
            .padding = .{ .Padding = .{ padding, padding } },
            .groups = groups,
            .bias = false,
            .tensor_opts = options,
        });
        self.norm = BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("0", self.conv);
        _ = self.base_module.registerModule("1", self.norm);
        self.conv.reset();
        self.norm.reset();
    }

    pub fn deinit(self: *Self) void {
        self.conv.deinit();
        self.norm.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var ys = self.conv.forward(input);
        ys = self.norm.forward(&ys);
        if (self.activation) return ys.silu();
        return ys;
    }
};

const SqueezeExcitation = struct {
    base_module: *Module = undefined,
    fc1: *Conv2D,
    fc2: *Conv2D,
    options: TensorOptions,

    const Self = @This();

    pub fn init(c_in: i64, c_squeeze: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .fc1 = Conv2D.init(.{
                .in_channels = c_in,
                .out_channels = c_squeeze,
                .kernel_size = .{ 1, 1 },
                .tensor_opts = options,
            }),
            .fc2 = Conv2D.init(.{
                .in_channels = c_squeeze,
                .out_channels = c_in,
                .kernel_size = .{ 1, 1 },
                .tensor_opts = options,
            }),
            .options = options,
        };
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("fc1", self.fc1);
        _ = self.base_module.registerModule("fc2", self.fc2);
        self.fc1.reset();
        self.fc2.reset();
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var scale = input.adaptiveAvgPool2d(&.{ 1, 1 });
        scale = self.fc1.forward(&scale).silu();
        scale = self.fc2.forward(&scale).sigmoid();
        return input.mul(&scale);
    }
};

const StochasticDepthKind = enum {
    Row,
    Batch,
};

pub const StochasticDepth = struct {
    base_module: *Module = undefined,
    prob: f64,
    kind: StochasticDepthKind,
    tensor_opts: TensorOptions,
    const Self = @This();

    pub fn init(prob: f64, kind: StochasticDepthKind, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .prob = prob,
            .kind = kind,
            .tensor_opts = options,
        };
        self.base_module = Module.init(self);
        return self;
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        if (!self.base_module.isTraining() or self.prob == 0.0) {
            return input.shallowClone();
        }
        const survival_rate = 1.0 - self.prob;
        var size = std.ArrayList(i64).init(torch.global_allocator);
        defer size.deinit();
        switch (self.kind) {
            .Row => {
                size.append(input.size()[0]) catch unreachable;
                size.appendNTimes(1, input.dim() - 1) catch unreachable;
            },
            .Batch => {
                size.appendNTimes(1, input.dim()) catch unreachable;
            },
        }
        var noise = Tensor.rand(size.items, self.tensor_opts);
        noise = noise.ge(Scalar.float(survival_rate));
        if (survival_rate > 0.0) {
            return input.mul(&noise).divScalar(Scalar.float(survival_rate));
        }
        return input.mul(&noise);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }
};

const MBConv = struct {
    base_module: *Module = undefined,
    block: *Sequential,
    stochastic_depth: *StochasticDepth = undefined,
    cfg: MBConvConfig,

    const Self = @This();

    pub fn init(cfg: MBConvConfig, stoch_depth_prob: f64) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .cfg = cfg, .block = Sequential.init(cfg.options) };
        self.base_module = Module.init(self);
        const c_expanded = MBConvConfig.adjustChannels(cfg.c_in, cfg.expand_ratio, null);
        if (c_expanded != cfg.c_in) {
            self.block = self.block.add(ConvNormActivation.init(
                cfg.c_in,
                c_expanded,
                1,
                1,
                1,
                true,
                cfg.options,
            ));
        }
        self.block = self.block.add(ConvNormActivation.init(
            c_expanded,
            c_expanded,
            cfg.ksize,
            cfg.stride,
            c_expanded,
            true,
            cfg.options,
        ));
        const c_squeeze = @max(1, @divFloor(cfg.c_in, 4));
        self.block = self.block.add(SqueezeExcitation.init(c_expanded, c_squeeze, cfg.options));
        self.block = self.block.add(ConvNormActivation.init(c_expanded, cfg.c_out, 1, 1, 1, false, cfg.options));
        self.stochastic_depth = StochasticDepth.init(stoch_depth_prob, .Row, cfg.options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("block", self.block);
        _ = self.base_module.registerModule("stochastic_depth", self.stochastic_depth);
    }

    pub fn deinit(self: *Self) void {
        self.block.deinit();
        self.stochastic_depth.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var ys = self.block.forward(input);
        if (self.cfg.stride == 1 and self.cfg.c_in == self.cfg.c_out)
            return self.stochastic_depth.forward(&ys).add(input);
        return ys;
    }
};

const FusedMBConv = struct {
    base_module: *Module = undefined,
    block: *Sequential,
    stochastic_depth: *StochasticDepth = undefined,
    cfg: MBConvConfig,

    const Self = @This();

    pub fn init(cfg: MBConvConfig, stoch_depth_prob: f64) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .cfg = cfg, .block = Sequential.init(cfg.options) };
        self.base_module = Module.init(self);
        const c_expanded = MBConvConfig.adjustChannels(cfg.c_in, cfg.expand_ratio, null);
        if (c_expanded != cfg.c_in) {
            self.block = self.block.add(ConvNormActivation.init(cfg.c_in, c_expanded, cfg.ksize, cfg.stride, 1, true, cfg.options))
                .add(ConvNormActivation.init(c_expanded, cfg.c_out, 1, 1, 1, false, cfg.options));
        } else {
            self.block = self.block.add(ConvNormActivation.init(cfg.c_in, c_expanded, cfg.ksize, cfg.stride, 1, true, cfg.options));
        }
        self.stochastic_depth = StochasticDepth.init(stoch_depth_prob, .Row, cfg.options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("block", self.block);
        _ = self.base_module.registerModule("stochastic_depth", self.stochastic_depth);
    }

    pub fn deinit(self: *Self) void {
        self.block.deinit();
        self.stochastic_depth.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var ys = self.block.forward(input);
        if ((self.cfg.stride == 1) and (self.cfg.c_in == self.cfg.c_out))
            return self.stochastic_depth.forward(&ys).add(input);
        return ys;
    }
};

const EfficientNet = struct {
    base_module: *Module = undefined,
    cfgs: []const MBConvConfig,
    dropout: f64,
    last_channel: ?i64,
    features: *Sequential = undefined,
    classifier: *Linear = undefined,

    const Self = @This();

    pub fn init(cfgs: []const MBConvConfig, dropout: f64, stoch_depth_prob: f64, num_classes: i64, last_channel: ?i64) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .cfgs = cfgs,
            .dropout = dropout,
            .last_channel = last_channel,
        };
        self.base_module = Module.init(self);
        self.features = Sequential.init(cfgs[0].options).add(ConvNormActivation.init(3, cfgs[0].c_in, 3, 2, 1, true, cfgs[0].options));
        var total_stage_blocks: i64 = 0;
        var stage_block_idx: i64 = 0;
        for (self.cfgs) |cfg| {
            total_stage_blocks += cfg.num_layers;
        }
        for (self.cfgs) |cfg| {
            var stage = Sequential.init(cfg.options);
            for (0..@intCast(cfg.num_layers)) |j| {
                var block_cfg = cfg;
                if (j != 0) {
                    block_cfg.c_in = block_cfg.c_out;
                    block_cfg.stride = 1;
                }
                const sd_prob = stoch_depth_prob * @as(f64, @floatFromInt(@divFloor(stage_block_idx, total_stage_blocks)));
                switch (block_cfg.block_type) {
                    .MBConv => {
                        stage = stage.add(MBConv.init(block_cfg, sd_prob));
                    },
                    .FusedMBConv => {
                        stage = stage.add(FusedMBConv.init(block_cfg, sd_prob));
                    },
                }
                stage_block_idx += 1;
            }
            self.features = self.features.add(stage);
        }
        const lastconv_c_in = self.cfgs[self.cfgs.len - 1].c_out;
        const lastconv_c_out = last_channel orelse 4 * lastconv_c_in;
        self.features = self.features.add(ConvNormActivation.init(lastconv_c_in, lastconv_c_out, 1, 1, 1, true, cfgs[0].options));
        self.classifier = Linear.init(.{ .in_features = lastconv_c_out, .out_features = num_classes, .bias = true, .tensor_opts = cfgs[0].options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("features", self.features);
        _ = self.base_module.registerModule("classifier.1", self.classifier);
    }

    pub fn deinit(self: *Self) void {
        self.features.deinit();
        self.classifier.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var ys = self.features.forward(input);
        ys = ys.adaptiveAvgPool2d(&.{ 1, 1 })
            .flatten(1, -1)
            .dropout(self.dropout, self.base_module.isTraining());
        return self.classifier.forward(&ys);
    }
};

const Arch = enum { B0, B1, B2, B3, B4, B5, B6, B7, V2S, V2M, V2L };

fn efficientnetConf(arch: Arch, w_mult: ?f64, d_mult: ?f64, options: TensorOptions) struct { []const MBConvConfig, ?i64 } {
    const width_mult = w_mult orelse 1.0;
    const depth_mult = d_mult orelse 1.0;
    var cfgs = std.ArrayList(MBConvConfig).init(torch.global_allocator);
    const _cfgs = switch (arch) {
        .V2S => &.{
            MBConvConfig.fusedMBConv(1.0, 3, 1, 24, 24, 2, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 24, 48, 4, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 48, 64, 4, options),
            MBConvConfig.mbconv(4.0, 3, 2, 64, 128, 6, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 128, 160, 9, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 2, 160, 256, 15, width_mult, depth_mult, options),
        },
        .V2M => &.{
            MBConvConfig.fusedMBConv(1.0, 3, 1, 24, 24, 3, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 24, 48, 5, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 48, 80, 5, options),
            MBConvConfig.mbconv(4.0, 3, 2, 80, 160, 7, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 160, 176, 14, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 2, 176, 304, 18, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 304, 512, 5, width_mult, depth_mult, options),
        },
        .V2L => &.{
            MBConvConfig.fusedMBConv(1.0, 3, 1, 32, 32, 4, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 32, 64, 7, options),
            MBConvConfig.fusedMBConv(4.0, 3, 2, 64, 96, 7, options),
            MBConvConfig.mbconv(4.0, 3, 2, 96, 192, 10, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 192, 224, 19, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 2, 224, 384, 25, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 384, 640, 7, width_mult, depth_mult, options),
        },
        else => &.{
            MBConvConfig.mbconv(1.0, 3, 1, 32, 16, 1, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 2, 16, 24, 2, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 5, 2, 24, 40, 2, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 2, 40, 80, 3, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 5, 1, 80, 112, 3, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 5, 2, 112, 192, 4, width_mult, depth_mult, options),
            MBConvConfig.mbconv(6.0, 3, 1, 192, 320, 1, width_mult, depth_mult, options),
        },
    };
    cfgs.appendSlice(_cfgs) catch unreachable;
    const last_channel: ?i64 = switch (arch) {
        .V2S, .V2M, .V2L => 1280,
        else => null,
    };
    return .{ cfgs.toOwnedSlice() catch unreachable, last_channel };
}

pub fn efficientnetb0(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B0, 1.0, 1.0, options);
    return EfficientNet.init(cfgs, 0.2, 0.2, num_classes, last_channel);
}

pub fn efficientnetb1(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B1, 1.0, 1.1, options);
    return EfficientNet.init(cfgs, 0.2, 0.2, num_classes, last_channel);
}

pub fn efficientnetb2(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B2, 1.1, 1.2, options);
    return EfficientNet.init(cfgs, 0.3, 0.2, num_classes, last_channel);
}

pub fn efficientnetb3(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B3, 1.2, 1.4, options);
    return EfficientNet.init(cfgs, 0.3, 0.2, num_classes, last_channel);
}

pub fn efficientnetb4(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B4, 1.4, 1.8, options);
    return EfficientNet.init(cfgs, 0.4, 0.2, num_classes, last_channel);
}

pub fn efficientnetb5(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B5, 1.6, 2.2, options);
    return EfficientNet.init(cfgs, 0.4, 0.2, num_classes, last_channel);
}

pub fn efficientnetb6(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B6, 1.8, 2.6, options);
    return EfficientNet.init(cfgs, 0.5, 0.2, num_classes, last_channel);
}

pub fn efficientnetb7(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.B7, 2.0, 3.1, options);
    return EfficientNet.init(cfgs, 0.5, 0.2, num_classes, last_channel);
}

pub fn efficientnetv2s(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.V2S, null, null, options);
    return EfficientNet.init(cfgs, 0.2, 0.2, num_classes, last_channel);
}

pub fn efficientnetv2m(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.V2M, null, null, options);
    return EfficientNet.init(cfgs, 0.3, 0.2, num_classes, last_channel);
}

pub fn efficientnetv2l(num_classes: i64, options: TensorOptions) *EfficientNet {
    const cfgs, const last_channel = efficientnetConf(.V2L, null, null, options);
    return EfficientNet.init(cfgs, 0.4, 0.2, num_classes, last_channel);
}
