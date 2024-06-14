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
const Dropout = functional.Dropout;
const Functional = functional.Functional;
const Conv2D = torch.conv.Conv2D;
const BatchNorm2D = torch.norm.BatchNorm(2);
const Sequential = module.Sequential;
const LayerNorm = torch.norm.LayerNorm;

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

pub const CNBlock = struct {
    base_module: *Module = undefined,
    layer_scale: Tensor = undefined,
    stoch_depth: *StochasticDepth = undefined,
    block: *Sequential = undefined,
    tensor_opts: TensorOptions,
    const Self = @This();

    pub fn init(dim: i64, layer_scale: f64, stochastic_depth_prob: f64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .tensor_opts = options };
        self.base_module = Module.init(self);

        self.layer_scale = self.base_module.registerParameter(
            "layer_scale",
            Tensor.ones(&.{ dim, 1, 1 }, options).mulScalar(Scalar.float(layer_scale)),
            true,
        );
        self.stoch_depth = StochasticDepth.init(stochastic_depth_prob, StochasticDepthKind.Row, options);
        const norm_shape = torch.global_allocator.dupe(i64, &.{dim}) catch unreachable;
        self.block = Sequential.init(options)
            .add(Conv2D.init(
            .{
                .in_channels = dim,
                .out_channels = dim,
                .kernel_size = .{ 7, 7 },
                .padding = .{ .Padding = .{ 3, 3 } },
                .groups = dim,
                .tensor_opts = options,
            },
        ))
            .add(Functional(Tensor.permute, .{&.{ 0, 2, 3, 1 }}).init())
            .add(LayerNorm.init(.{ .normalized_shape = norm_shape, .tensor_opts = options }))
            .add(Linear.init(.{ .in_features = dim, .out_features = 4 * dim, .tensor_opts = options }))
            .add(Functional(Tensor.gelu, .{"none"}).init())
            .add(Linear.init(.{ .in_features = 4 * dim, .out_features = dim, .tensor_opts = options }))
            .add(Functional(Tensor.permute, .{&.{ 0, 3, 1, 2 }}).init());
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("stoch_depth", self.stoch_depth);
        _ = self.base_module.registerModule("block", self.block);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.stoch_depth.deinit();
        self.block.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var x = input.shallowClone();
        x = self.block.forward(&x);
        x = x.mul(&self.layer_scale);
        x = self.stoch_depth.forward(&x);
        return x.add(input);
    }
};

fn convNorm(c_in: i64, c_out: i64, ksize: i64, stride: i64, padding: i64, options: TensorOptions) *Sequential {
    const norm_shape = torch.global_allocator.dupe(i64, &.{c_out}) catch unreachable;
    const seq = Sequential.init(options)
        .add(Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = .{ ksize, ksize },
        .stride = .{ stride, stride },
        .padding = .{ .Padding = .{ padding, padding } },
        .tensor_opts = options,
    }))
        .add(Functional(Tensor.permute, .{&.{ 0, 2, 3, 1 }}).init())
        .addWithName("1", LayerNorm.init(.{ .normalized_shape = norm_shape, .tensor_opts = options }))
        .add(Functional(Tensor.permute, .{&.{ 0, 3, 1, 2 }}).init());

    return seq;
}

pub const ConvNext = struct {
    base_module: *Module = undefined,
    features: *Sequential = undefined,
    classifier: *Sequential = undefined,
    tensor_opts: TensorOptions,

    const Self = @This();

    pub fn init(block_setting: [4][3]i64, stochastic_depth_prob: f64, layer_scale: f64, num_classes: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .tensor_opts = options,
        };
        self.base_module = Module.init(self);
        self.features = Sequential.init(options);

        const firstconv_c_out = block_setting[0][0];
        self.features = self.features.add(convNorm(3, firstconv_c_out, 4, 4, 0, options));
        var total_stage_blocks: i64 = 0;
        for (block_setting) |block| {
            total_stage_blocks += block[2];
        }
        var stage_block_idx: usize = 0;

        for (block_setting) |block| {
            const c_in, const c_out, const num_layers = block;
            var stage = Sequential.init(options);
            for (0..@intCast(num_layers)) |_| {
                const sd_prob = stochastic_depth_prob * @as(f64, @floatFromInt(stage_block_idx)) / @as(f64, @floatFromInt(total_stage_blocks - 1));
                stage = stage.add(CNBlock.init(c_in, layer_scale, sd_prob, options));
                stage_block_idx += 1;
            }
            self.features = self.features.add(stage);
            if (c_out != -1) {
                const norm_shape = torch.global_allocator.dupe(i64, &.{c_in}) catch unreachable;
                self.features = self.features.add(
                    Sequential.init(options)
                        .add(Functional(Tensor.permute, .{&.{ 0, 2, 3, 1 }}).init())
                        .addWithName("0", LayerNorm.init(.{ .normalized_shape = norm_shape, .tensor_opts = options }))
                        .add(Functional(Tensor.permute, .{&.{ 0, 3, 1, 2 }}).init())
                        .addWithName("1", Conv2D.init(.{
                        .in_channels = c_in,
                        .out_channels = c_out,
                        .kernel_size = .{ 2, 2 },
                        .stride = .{ 2, 2 },
                        .tensor_opts = options,
                    })),
                );
            }
        }
        const lastblock = block_setting[block_setting.len - 1];
        const lastconv_c_out = if (lastblock[1] != -1) lastblock[1] else lastblock[0];
        const norm_shape = torch.global_allocator.dupe(i64, &.{lastconv_c_out}) catch unreachable;
        self.classifier = Sequential.init(options)
            .add(Functional(Tensor.permute, .{&.{ 0, 2, 3, 1 }}).init())
            .addWithName("0", LayerNorm.init(.{ .normalized_shape = norm_shape, .tensor_opts = options }))
            .add(Functional(Tensor.permute, .{&.{ 0, 3, 1, 2 }}).init())
            .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
            .addWithName("2", Linear.init(.{ .in_features = lastconv_c_out, .out_features = num_classes, .tensor_opts = options }));

        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("features", self.features);
        _ = self.base_module.registerModule("classifier", self.classifier);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.features.deinit();
        self.classifier.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var x = input.shallowClone();
        x = self.features.forward(&x);
        x = x.adaptiveAvgPool2d(&.{ 1, 1 });
        x = self.classifier.forward(&x);
        return x;
    }
};

pub fn convnextTiny(num_classes: i64, options: TensorOptions) *ConvNext {
    const block_setting = .{
        .{ 96, 192, 3 },
        .{ 192, 384, 3 },
        .{ 384, 768, 9 },
        .{ 768, -1, 3 },
    };
    return ConvNext.init(block_setting, 0.1, 1e-6, num_classes, options);
}

pub fn convnextSmall(num_classes: i64, options: TensorOptions) *ConvNext {
    const block_setting = .{
        .{ 96, 192, 3 },
        .{ 192, 384, 3 },
        .{ 384, 768, 27 },
        .{ 768, -1, 3 },
    };
    return ConvNext.init(block_setting, 0.4, 1e-6, num_classes, options);
}

pub fn convnextBase(num_classes: i64, options: TensorOptions) *ConvNext {
    const block_setting = .{
        .{ 128, 256, 3 },
        .{ 256, 512, 3 },
        .{ 512, 1024, 27 },
        .{ 1024, -1, 3 },
    };
    return ConvNext.init(block_setting, 0.5, 1e-6, num_classes, options);
}

pub fn convnextLarge(num_classes: i64, options: TensorOptions) *ConvNext {
    const block_setting = .{
        .{ 192, 384, 3 },
        .{ 384, 768, 3 },
        .{ 768, 1536, 27 },
        .{ 1536, -1, 3 },
    };
    return ConvNext.init(block_setting, 0.5, 1e-6, num_classes, options);
}
