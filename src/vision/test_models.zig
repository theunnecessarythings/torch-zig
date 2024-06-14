const std = @import("std");
const torch = @import("torch");
const Alexnet = torch.vision.alexnet.Alexnet;
const Tensor = torch.Tensor;

const Params = []struct { []const u8, []struct { []const u8, []i64 } };
const Model = enum {
    Alexnet,
    ConvnextTiny,
    ConvnextSmall,
    ConvnextLarge,
    ConvnextBase,
    Densenet121,
    Densenet169,
    Densenet201,
    Densenet161,
    EfficientnetB0,
    EfficientnetB1,
    EfficientnetB2,
    EfficientnetB3,
    EfficientnetB4,
    EfficientnetB5,
    EfficientnetB6,
    EfficientnetB7,
    EfficientnetV2S,
    EfficientnetV2M,
    EfficientnetV2L,
    Googlenet,
    InceptionV3,
    Mnasnet05,
    Mnasnet075,
    Mnasnet10,
    Mnasnet13,
    MobilenetV2,
    MobilenetV3Large,
    MobilenetV3Small,
    Resnet18,
    Resnet34,
    Resnet50,
    Resnet101,
    Resnet152,
    Resnext5032x4d,
    Resnext10132x8d,
    WideResnet502,
    WideResnet1012,
    ShufflenetV2X05,
    ShufflenetV2X10,
    ShufflenetV2X15,
    ShufflenetV2X20,
    Squeezenet10,
    Squeezenet11,
    Vgg11,
    Vgg11Bn,
    Vgg13,
    Vgg13Bn,
    Vgg16,
    Vgg16Bn,
    Vgg19,
    Vgg19Bn,
};

fn checkParams(actual: anytype, expected: anytype) !void {
    const model_params = actual.base_module.namedParameters(true);
    const model_buffers = actual.base_module.namedBuffers(true);
    errdefer {
        for (model_params.keys()) |param| {
            const shape = model_params.get(param).?.size();
            std.debug.print("Param: {s}, Shape: {any}\n", .{ param, shape });
        }
        for (model_buffers.keys()) |param| {
            const shape = model_buffers.get(param).?.size();
            std.debug.print("Buffer: {s}, Shape: {any}\n", .{ param, shape });
        }

        for (expected) |param| {
            const name = param[0];
            const shape = param[1];
            std.debug.print("Expected Param: {s}, Shape: {any}\n", .{ name, shape });
        }
    }
    if (expected.len != model_params.keys().len + model_buffers.keys().len) return error.UnexpectedParameterCount;
    for (expected) |param| {
        const name = param[0];
        const shape = param[1];
        if (model_params.contains(name)) {
            const param_shape = model_params.get(name).?.size();
            try std.testing.expectEqualSlices(i64, shape, param_shape);
        } else if (model_buffers.contains(name)) {
            const param_shape = model_buffers.get(name).?.size();
            try std.testing.expectEqualSlices(i64, shape, param_shape);
        } else {
            std.debug.print("Missing Param: {s}\n", .{name});
            return error.MissingParameter;
        }
    }
}

fn loadParamsFile() !std.json.Parsed(Params) {
    const allocator = std.testing.allocator;
    const path = "extra/params.json";
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 2_048_000);
    defer allocator.free(data);
    return std.json.parseFromSlice(Params, allocator, data, .{ .allocate = .alloc_always });
}

fn testModel(model_fn: anytype, model_type: Model) !void {
    var nograd = torch.NoGradGuard.init();
    defer nograd.deinit();
    var guard = torch.MemoryGuard.init("test");
    defer guard.deinit();

    const params = try loadParamsFile();
    defer params.deinit();

    const model_params = params.value[@intFromEnum(model_type)];
    const param = model_params[1];

    const model = model_fn(1000, torch.FLOAT_CUDA);
    const shape = if (model_type == .InceptionV3)
        &.{ 1, 3, 299, 299 }
    else
        &.{ 1, 3, 224, 224 };
    const x = Tensor.randn(shape, torch.FLOAT_CUDA);
    _ = model.forward(&x);

    try checkParams(model, param);
}

test "alexnet" {
    try testModel(Alexnet.init, .Alexnet);
}

test "convnext_tiny" {
    try testModel(torch.vision.convnext.convnextTiny, .ConvnextTiny);
}

test "convnext_small" {
    try testModel(torch.vision.convnext.convnextSmall, .ConvnextSmall);
}

test "convnext_large" {
    try testModel(torch.vision.convnext.convnextLarge, .ConvnextLarge);
}

test "convnext_base" {
    try testModel(torch.vision.convnext.convnextBase, .ConvnextBase);
}

test "densenet121" {
    try testModel(torch.vision.densenet.densenet121, .Densenet121);
}

test "densenet169" {
    try testModel(torch.vision.densenet.densenet169, .Densenet169);
}

test "densenet201" {
    try testModel(torch.vision.densenet.densenet201, .Densenet201);
}

test "densenet161" {
    try testModel(torch.vision.densenet.densenet161, .Densenet161);
}

test "efficientnet_b0" {
    try testModel(torch.vision.efficientnet.efficientnetb0, .EfficientnetB0);
}

test "efficientnet_b1" {
    try testModel(torch.vision.efficientnet.efficientnetb1, .EfficientnetB1);
}

test "efficientnet_b2" {
    try testModel(torch.vision.efficientnet.efficientnetb2, .EfficientnetB2);
}

test "efficientnet_b3" {
    try testModel(torch.vision.efficientnet.efficientnetb3, .EfficientnetB3);
}

test "efficientnet_b4" {
    try testModel(torch.vision.efficientnet.efficientnetb4, .EfficientnetB4);
}

test "efficientnet_b5" {
    try testModel(torch.vision.efficientnet.efficientnetb5, .EfficientnetB5);
}

test "efficientnet_b6" {
    try testModel(torch.vision.efficientnet.efficientnetb6, .EfficientnetB6);
}

test "efficientnet_b7" {
    try testModel(torch.vision.efficientnet.efficientnetb7, .EfficientnetB7);
}

test "efficientnet_v2s" {
    try testModel(torch.vision.efficientnet.efficientnetv2s, .EfficientnetV2S);
}

test "efficientnet_v2m" {
    try testModel(torch.vision.efficientnet.efficientnetv2m, .EfficientnetV2M);
}

test "efficientnet_v2l" {
    try testModel(torch.vision.efficientnet.efficientnetv2l, .EfficientnetV2L);
}

// test "googlenet" {
//     try testModel(torch.vision.googlenet.googlenet, .Googlenet);
// }

test "inception_v3" {
    try testModel(torch.vision.inception.inceptionV3, .InceptionV3);
}

test "mnasnet_05" {
    try testModel(torch.vision.mnasnet.mnasnet0_5, .Mnasnet05);
}

test "mnasnet_075" {
    try testModel(torch.vision.mnasnet.mnasnet0_75, .Mnasnet075);
}

test "mnasnet_10" {
    try testModel(torch.vision.mnasnet.mnasnet1_0, .Mnasnet10);
}

test "mnasnet_13" {
    try testModel(torch.vision.mnasnet.mnasnet1_3, .Mnasnet13);
}

test "mobilenet_v2" {
    try testModel(torch.vision.mobilenetv2.mobilenetV2, .MobilenetV2);
}

test "mobilenet_v3_large" {
    try testModel(torch.vision.mobilenetv3.mobilenetV3Large, .MobilenetV3Large);
}

test "mobilenet_v3_small" {
    try testModel(torch.vision.mobilenetv3.mobilenetV3Small, .MobilenetV3Small);
}

test "resnet18" {
    try testModel(torch.vision.resnet.resnet18, .Resnet18);
}

test "resnet34" {
    try testModel(torch.vision.resnet.resnet34, .Resnet34);
}

test "resnet50" {
    try testModel(torch.vision.resnet.resnet50, .Resnet50);
}

test "resnet101" {
    try testModel(torch.vision.resnet.resnet101, .Resnet101);
}

test "resnet152" {
    try testModel(torch.vision.resnet.resnet152, .Resnet152);
}

test "resnext50_32x4d" {
    try testModel(torch.vision.resnet.resnext50_32x4d, .Resnext5032x4d);
}

test "resnext101_32x8d" {
    try testModel(torch.vision.resnet.resnext101_32x8d, .Resnext10132x8d);
}

test "wide_resnet50_2" {
    try testModel(torch.vision.resnet.wide_resnet50_2, .WideResnet502);
}

test "wide_resnet101_2" {
    try testModel(torch.vision.resnet.wide_resnet101_2, .WideResnet1012);
}

test "shufflenet_v2_x05" {
    try testModel(torch.vision.shufflenetv2.shuffleNetV2_x0_5, .ShufflenetV2X05);
}

test "shufflenet_v2_x10" {
    try testModel(torch.vision.shufflenetv2.shuffleNetV2_x1_0, .ShufflenetV2X10);
}

test "shufflenet_v2_x15" {
    try testModel(torch.vision.shufflenetv2.shuffleNetV2_x1_5, .ShufflenetV2X15);
}

test "shufflenet_v2_x20" {
    try testModel(torch.vision.shufflenetv2.shuffleNetV2_x2_0, .ShufflenetV2X20);
}

test "squeezenet1_0" {
    try testModel(torch.vision.squeezenet.squeezenet1_0, .Squeezenet10);
}

test "squeezenet1_1" {
    try testModel(torch.vision.squeezenet.squeezenet1_1, .Squeezenet11);
}

test "vgg11" {
    try testModel(torch.vision.vgg.vgg11, .Vgg11);
}

test "vgg11_bn" {
    try testModel(torch.vision.vgg.vgg11_bn, .Vgg11Bn);
}

test "vgg13" {
    try testModel(torch.vision.vgg.vgg13, .Vgg13);
}

test "vgg13_bn" {
    try testModel(torch.vision.vgg.vgg13_bn, .Vgg13Bn);
}

test "vgg16" {
    try testModel(torch.vision.vgg.vgg16, .Vgg16);
}

test "vgg16_bn" {
    try testModel(torch.vision.vgg.vgg16_bn, .Vgg16Bn);
}

test "vgg19" {
    try testModel(torch.vision.vgg.vgg19, .Vgg19);
}

test "vgg19_bn" {
    try testModel(torch.vision.vgg.vgg19_bn, .Vgg19Bn);
}
