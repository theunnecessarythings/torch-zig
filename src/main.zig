const torch = @import("torch");
const Tensor = torch.Tensor;
const std = @import("std");
const linear = torch.linear;
const Identity = linear.Identity;
const module = torch.module;
const Module = module.Module;
const conv = torch.conv;
const resnet = torch.vision.resnet;
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

pub fn add() void {
    var size = [_]i64{ 1000, 1000 };
    for (0..1000) |_| {
        var a = Tensor.rand(&size, torch.FLOAT_CUDA);
        defer a.free();
        _ = a.add(&a);
    }
}

pub fn main() !void {
    const cuda_available = torch.Cuda.isAvailable();
    std.debug.print("CUDA available: {}\n", .{cuda_available});
    var size = [_]i64{ 3, 2 };
    var a = Tensor.rand(&size, torch.FLOAT_CUDA);
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

    // var i: usize = 0;
    // size = [_]i64{ 1000, 1000 };
    // while (i < 5000) : (i += 1) {
    //     var c = Tensor.rand(&size, torch.FLOAT_CUDA);
    //     defer c.free();
    //     var d = c.add(&c);
    //     d.free();
    // }
    //
    defer a.free();

    var fc = linear.Linear.init(linear.LinearOptions{ .in_features = 2, .out_features = 3 });
    fc.base_module.to(a.device(), a.kind(), false);
    defer fc.deinit();

    fc.forward(&a).print();

    var conv2d = conv.Conv2D.init(.{ .in_channels = 3, .out_channels = 3, .kernel_size = [_]i64{ 3, 3 } });
    conv2d.base_module.to(a.device(), a.kind(), false);
    defer conv2d.deinit();
    const input = Tensor.rand(&[_]i64{ 1, 3, 5, 5 }, torch.FLOAT_CUDA);
    conv2d.forward(&input).print();

    const x = Tensor.rand(&[_]i64{ 1, 3, 224, 224 }, torch.FLOAT_CUDA);
    var resnet18 = resnet.resnet50(1000);
    resnet18.base_module.to(x.device(), x.kind(), false);

    for (0..1000) |i| {
        var nograd = torch.NoGradGuard.init();
        defer nograd.deinit();
        var guard = torch.MemoryGuard.init("resnet18");
        defer guard.deinit();
        std.debug.print("Iteration: {d}\n", .{i});
        _ = resnet18.forward(&x);
    }

    const params = resnet18.base_module.namedParameters(true);
    for (params.keys()) |key| {
        std.debug.print("Key: {s}\n", .{key});
        const sz = params.get(key).?.size();
        std.debug.print("Size: {any}\n", .{sz});
    }
}

test "leak" {
    add();
}
