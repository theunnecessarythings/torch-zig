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

const Fire = struct {
    base_module: *Module = undefined,
    squeeze: *Conv2D = undefined,
    expand1x1: *Conv2D = undefined,
    expand3x3: *Conv2D = undefined,

    const Self = @This();

    pub fn init(c_in: i64, c_squeeze: i64, c_expand1x1: i64, c_expand3x3: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.squeeze = Conv2D.init(.{ .in_channels = c_in, .out_channels = c_squeeze, .kernel_size = .{ 1, 1 }, .tensor_opts = options });
        self.expand1x1 = Conv2D.init(.{ .in_channels = c_squeeze, .out_channels = c_expand1x1, .kernel_size = .{ 1, 1 }, .tensor_opts = options });
        self.expand3x3 = Conv2D.init(.{ .in_channels = c_squeeze, .out_channels = c_expand3x3, .kernel_size = .{ 3, 3 }, .padding = .{ .Padding = .{ 1, 1 } }, .tensor_opts = options });
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        self.squeeze.reset();
        self.expand1x1.reset();
        self.expand3x3.reset();
        _ = self.base_module.registerModule("squeeze", self.squeeze);
        _ = self.base_module.registerModule("expand1x1", self.expand1x1);
        _ = self.base_module.registerModule("expand3x3", self.expand3x3);
    }

    pub fn deinit(self: *Self) void {
        self.squeeze.deinit();
        self.expand1x1.deinit();
        self.expand3x3.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var ys = self.squeeze.forward(x).relu();
        var y = [_]*const Tensor{ &self.expand1x1.forward(&ys).relu(), &self.expand3x3.forward(&ys).relu() };
        return Tensor.cat(&y, 1);
    }
};

const SqueezeNetVersion = enum { SqueezeNet1_0, SqueezeNet1_1 };

fn squeezenet(version: SqueezeNetVersion, num_classes: i64, dropout: f32, options: TensorOptions) *Sequential {
    var features = Sequential.init(options);
    switch (version) {
        .SqueezeNet1_0 => features = features.add(Conv2D.init(.{ .in_channels = 3, .out_channels = 96, .kernel_size = .{ 7, 7 }, .stride = .{ 2, 2 }, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(96, 16, 64, 64, options))
            .add(Fire.init(128, 16, 64, 64, options))
            .add(Fire.init(128, 32, 128, 128, options))
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(256, 32, 128, 128, options))
            .add(Fire.init(256, 48, 192, 192, options))
            .add(Fire.init(384, 48, 192, 192, options))
            .add(Fire.init(384, 64, 256, 256, options))
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(512, 64, 256, 256, options)),
        .SqueezeNet1_1 => features = features.add(Conv2D.init(.{ .in_channels = 3, .out_channels = 64, .kernel_size = .{ 3, 3 }, .stride = .{ 2, 2 }, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(64, 16, 64, 64, options))
            .add(Fire.init(128, 16, 64, 64, options))
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(128, 32, 128, 128, options))
            .add(Fire.init(256, 32, 128, 128, options))
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
            .add(Fire.init(256, 48, 192, 192, options))
            .add(Fire.init(384, 48, 192, 192, options))
            .add(Fire.init(384, 64, 256, 256, options))
            .add(Fire.init(512, 64, 256, 256, options)),
    }
    const final_conv = Conv2D.init(.{ .in_channels = 512, .out_channels = num_classes, .kernel_size = .{ 1, 1 }, .tensor_opts = options });
    return Sequential.init(options)
        .addWithName("features", features)
        .add(Dropout.init(dropout))
        .addWithName("classifier.1", final_conv)
        .add(Functional(Tensor.relu, .{}).init())
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init());
}

pub fn squeezenet1_0(num_classes: i64, options: TensorOptions) *Sequential {
    return squeezenet(.SqueezeNet1_0, num_classes, 0.5, options);
}

pub fn squeezenet1_1(num_classes: i64, options: TensorOptions) *Sequential {
    return squeezenet(.SqueezeNet1_1, num_classes, 0.5, options);
}
