const torch = @import("../torch.zig");
const std = @import("std");
const Tensor = torch.Tensor;
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

fn conv2d(c_in: i64, c_out: i64, ksize: i64, stride: i64, padding: i64, opts: TensorOptions) *Conv2D {
    return Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = .{ ksize, ksize },
        .stride = .{ stride, stride },
        .padding = .{ .Padding = .{ padding, padding } },
        .tensor_opts = opts,
    });
}

pub const Alexnet = struct {
    base_module: *Module = undefined,

    features: *Sequential = undefined,
    classifier: *Sequential = undefined,
    const Self = @This();

    pub fn init(num_classes: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch torch.utils.err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);

        self.features = Sequential.init(options)
            .add(conv2d(3, 64, 11, 4, 2, options))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{3}, &.{2}, &.{0}, &.{1}, false }).init())
            .add(conv2d(64, 192, 5, 1, 2, options))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{3}, &.{2}, &.{0}, &.{1}, false }).init())
            .add(conv2d(192, 384, 3, 1, 1, options))
            .add(Functional(Tensor.relu, .{}).init())
            .add(conv2d(384, 256, 3, 1, 1, options))
            .add(Functional(Tensor.relu, .{}).init())
            .add(conv2d(256, 256, 3, 1, 1, options))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{3}, &.{2}, &.{0}, &.{1}, false }).init());

        self.classifier = Sequential.init(options)
            .add(Dropout.init(0.5))
            .add(Linear.init(.{ .in_features = 256 * 6 * 6, .out_features = 4096, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Dropout.init(0.5))
            .add(Linear.init(.{ .in_features = 4096, .out_features = 4096, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Linear.init(.{ .in_features = 4096, .out_features = num_classes, .tensor_opts = options }));

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
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var x = input.shallowClone();
        x = self.features.forward(&x);
        x = x.adaptiveAvgPool2d(&.{ 6, 6 });
        x = x.flatten(1, -1);
        x = self.classifier.forward(&x);
        return x;
    }
};
