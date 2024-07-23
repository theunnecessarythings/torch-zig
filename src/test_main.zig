const torch = @import("torch");
const Tensor = torch.Tensor;
const std = @import("std");
const linear = torch.linear;
const Identity = linear.Identity;
const module = torch.module;
const Module = module.Module;
const conv = torch.conv;
const resnet = torch.vision.resnet;
const alexnet = torch.vision.alexnet;
const convnext = torch.vision.convnext;
const densenet = torch.vision.densenet;
const safetensors = torch.safetensors;
// TODO: Memory Management - Need to find a way to free tensors efficiently
// NOTE: Every time a tensor is created I need to have a reference to it so that I can free it,
// so basically my own memory management system, WELL SHIT!! Zig yay
//
// ----------------------------------------
// |                                      |
// TODO: Indexing - Current C++ select function
// TODO: Printing - Current C++ print -> prints entire tensor
// TODO: Arithmetic - Operator overloading is not possible in Zig, so we need to use functions
// TODO: Basic Layers - Linear, Conv2d, to be implemented
// TODO: Generated Docs - Need to generate docs for the library
// TODO: Add fallible versions of all the functions (somebody might need them)

const test_models = @import("vision/test_models.zig");

pub fn main1() !void {
    try test_models.testModel(std.heap.c_allocator, torch.vision.densenet.densenet121, .Densenet121);
    // try test_models.testModel(std.heap.c_allocator, torch.vision.efficientnet.efficientnetv2s, .EfficientnetV2S);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    torch.global_allocator = gpa.allocator();
    const cuda_available = torch.Cuda.isAvailable();
    std.debug.print("CUDA available: {}\n", .{cuda_available});
    var size = [_]i64{ 3, 2 };
    var a = Tensor.rand(&size, torch.FLOAT_CPU);
    a.print();
    a.select(0, 1).print();
    a.add(&a).print();
    const p = a.i(.{-1});
    p.print();

    var id = Identity.init();
    defer id.deinit();
    const b = id.forward(&a).add(&a);
    b.print();
    std.debug.print("Identity Layer: {any}\n", .{id});
    const n = id.base_module.name();
    std.debug.print("Name : {s}\n", .{n});
    _ = id.base_module.registerParameter("test", a, false);
    _ = id.base_module.registerParameter("bias", b, true);

    const map = id.base_module.namedParameters(true);

    for (map.keys()) |key| {
        std.debug.print("Key: {s}\n", .{key});
        map.get(key).?.print();
    }

    var i: usize = 0;
    size = [_]i64{ 1000, 1000 };
    while (i < 5000) : (i += 1) {
        var c = Tensor.rand(&size, torch.FLOAT_CUDA);
        defer c.free();
        var d = c.add(&c);
        d.free();
    }

    defer a.free();

    var fc = linear.Linear.init(linear.LinearOptions{ .in_features = 2, .out_features = 3 });
    fc.base_module.to(a.device(), a.kind(), false);
    defer fc.deinit();

    fc.forward(&a).print();

    var conv2d = conv.Conv2D.init(.{ .in_channels = 3, .out_channels = 3, .kernel_size = [_]i64{ 3, 3 } });
    conv2d.base_module.to(a.device(), a.kind(), false);
    defer conv2d.deinit();
    const input = Tensor.rand(&[_]i64{ 1, 3, 5, 5 }, torch.FLOAT_CPU);
    conv2d.forward(&input).print();

    const x = Tensor.rand(&[_]i64{ 1, 3, 224, 224 }, torch.FLOAT_CUDA);
    var resnet18 = resnet.resnet18(1000, torch.FLOAT_CUDA);
    for (0..100) |_| {
        var nograd = torch.NoGradGuard.init();
        defer nograd.deinit();
        var guard = torch.MemoryGuard.init("resnet18");
        defer guard.deinit();
        _ = resnet18.forward(&x);
    }

    const mods = resnet18.base_module.namedModules("", false);
    for (mods.keys()) |key| {
        std.debug.print("Key: {s} -> ", .{key});
        // const params = mods.get(key).?.namedParameters(true);
        // for (params.keys()) |pkey| {
        //     const sz = params.get(pkey).?.size();
        //     std.debug.print("{s} -> {any}, ", .{ pkey, sz });
        // }
    }

    // var _alexnet = alexnet.Alexnet.init(1000, torch.FLOAT_CUDA);
    // const weights = try torch.utils.downloadFile("https://huggingface.co/theunnecessarythings/vision_models/resolve/main/alexnet.safetensors");
    // try _alexnet.base_module.loadFromSafetensors(weights);

    // for (0..100) |_| {
    //     var nograd = torch.NoGradGuard.init();
    //     defer nograd.deinit();
    //     var guard = torch.MemoryGuard.init("alexnet");
    //     defer guard.deinit();
    //     _ = _alexnet.forward(&x);
    // }
    //
    // var _convnext_t = convnext.convnextTiny(1000, torch.FLOAT_CUDA);
    // for (0..100) |_| {
    //     var nograd = torch.NoGradGuard.init();
    //     defer nograd.deinit();
    //     var guard = torch.MemoryGuard.init("convnext");
    //     defer guard.deinit();
    //     _ = _convnext_t.forward(&x);
    // }
    //
    // var _densenet = densenet.densenet169(1000, torch.FLOAT_CUDA);
    // for (0..100) |_| {
    //     var nograd = torch.NoGradGuard.init();
    //     defer nograd.deinit();
    //     var guard = torch.MemoryGuard.init("densenet");
    //     defer guard.deinit();
    //     _ = _densenet.forward(&x);
    // }

    // const path = "resnet18.safetensors";
    // var safetensor = try safetensors.readSafetensor(torch.global_allocator, path);
    // defer safetensor.deinit();

}

const all_tests = @import("vision/test_models.zig");

test "all_tests" {
    // _ = @import("vision/test_models.zig");
    _ = all_tests;
}
