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

fn vgg(num_classes: i64, cfg: []const []const i64, batch_norm: bool, dropout: f32, options: TensorOptions) *Sequential {
    var features = Sequential.init(options);
    var c_in: i64 = 3;
    for (cfg) |block| {
        for (block) |c_out| {
            features = features.add(Conv2D.init(.{
                .in_channels = c_in,
                .out_channels = c_out,
                .kernel_size = .{ 3, 3 },
                .padding = .{ .Padding = .{ 1, 1 } },
                .tensor_opts = options,
            }));
            if (batch_norm) {
                features = features.add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }));
            }
            features = features.add(Functional(Tensor.relu, .{}).init());
            c_in = c_out;
        }
        features = features.add(Functional(Tensor.maxPool2d, .{ &.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init());
    }
    const classifier = Sequential.init(options)
        .add(Linear.init(.{ .in_features = 512 * 7 * 7, .out_features = 4096, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = 4096, .out_features = 4096, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = 4096, .out_features = num_classes, .tensor_opts = options }));

    return Sequential.init(options)
        .addWithName("features", features)
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 7, 7 }}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .addWithName("classifier", classifier);
}

const VGGVersion = enum { VGG11, VGG13, VGG16, VGG19 };

fn getVggCfg(kind: VGGVersion) []const []const i64 {
    switch (kind) {
        .VGG11 => return &.{
            &.{64},
            &.{128},
            &.{ 256, 256 },
            &.{ 512, 512 },
            &.{ 512, 512 },
        },
        .VGG13 => return &.{
            &.{ 64, 64 },
            &.{ 128, 128 },
            &.{ 256, 256 },
            &.{ 512, 512 },
            &.{ 512, 512 },
        },
        .VGG16 => return &.{
            &.{ 64, 64 },
            &.{ 128, 128 },
            &.{ 256, 256, 256 },
            &.{ 512, 512, 512 },
            &.{ 512, 512, 512 },
        },
        .VGG19 => return &.{
            &.{ 64, 64 },
            &.{ 128, 128 },
            &.{ 256, 256, 256, 256 },
            &.{ 512, 512, 512, 512 },
            &.{ 512, 512, 512, 512 },
        },
    }
}

pub fn vgg11(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG11), false, 0.5, options);
}

pub fn vgg13(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG13), false, 0.5, options);
}

pub fn vgg16(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG16), false, 0.5, options);
}

pub fn vgg19(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG19), false, 0.5, options);
}

pub fn vgg11_bn(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG11), true, 0.5, options);
}

pub fn vgg13_bn(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG13), true, 0.5, options);
}

pub fn vgg16_bn(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG16), true, 0.5, options);
}

pub fn vgg19_bn(num_classes: i64, options: TensorOptions) *Sequential {
    return vgg(num_classes, getVggCfg(.VGG19), true, 0.5, options);
}
