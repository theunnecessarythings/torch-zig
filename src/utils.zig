const std = @import("std");
const torch = @import("torch.zig");
const Tensor = torch.Tensor;
var download_progress: ?std.Progress.Node = null;

pub const Model = enum {
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
    ViTB16,
    ViTB32,
    ViTL16,
    ViTL32,
    ViTH14,
    SwinT,
    SwinS,
    SwinB,
    SwinV2T,
    SwinV2S,
    SwinV2B,
};

pub fn getPretrainedWeights(alloc: std.mem.Allocator, model: Model) ![]u8 {
    const url = "https://huggingface.co/theunnecessarythings/vision_models/resolve/main/";
    const data = switch (model) {
        .Alexnet => try downloadFile(alloc, url ++ "alexnet.safetensors"),
        .ConvnextTiny => try downloadFile(alloc, url ++ "convnext_tiny.safetensors"),
        .ConvnextSmall => try downloadFile(alloc, url ++ "convnext_small.safetensors"),
        .ConvnextLarge => try downloadFile(alloc, url ++ "convnext_large.safetensors"),
        .ConvnextBase => try downloadFile(alloc, url ++ "convnext_base.safetensors"),
        .Densenet121 => try downloadFile(alloc, url ++ "densenet121.safetensors"),
        .Densenet169 => try downloadFile(alloc, url ++ "densenet169.safetensors"),
        .Densenet201 => try downloadFile(alloc, url ++ "densenet201.safetensors"),
        .Densenet161 => try downloadFile(alloc, url ++ "densenet161.safetensors"),
        .EfficientnetB0 => try downloadFile(alloc, url ++ "efficientnet_b0.safetensors"),
        .EfficientnetB1 => try downloadFile(alloc, url ++ "efficientnet_b1.safetensors"),
        .EfficientnetB2 => try downloadFile(alloc, url ++ "efficientnet_b2.safetensors"),
        .EfficientnetB3 => try downloadFile(alloc, url ++ "efficientnet_b3.safetensors"),
        .EfficientnetB4 => try downloadFile(alloc, url ++ "efficientnet_b4.safetensors"),
        .EfficientnetB5 => try downloadFile(alloc, url ++ "efficientnet_b5.safetensors"),
        .EfficientnetB6 => try downloadFile(alloc, url ++ "efficientnet_b6.safetensors"),
        .EfficientnetB7 => try downloadFile(alloc, url ++ "efficientnet_b7.safetensors"),
        .EfficientnetV2S => try downloadFile(alloc, url ++ "efficientnet_v2_s.safetensors"),
        .EfficientnetV2M => try downloadFile(alloc, url ++ "efficientnet_v2_m.safetensors"),
        .EfficientnetV2L => try downloadFile(alloc, url ++ "efficientnet_v2_l.safetensors"),
        .Googlenet => try downloadFile(alloc, url ++ "googlenet.safetensors"),
        .InceptionV3 => try downloadFile(alloc, url ++ "inception_v3.safetensors"),
        .Mnasnet05 => try downloadFile(alloc, url ++ "mnasnet0_5.safetensors"),
        .Mnasnet075 => try downloadFile(alloc, url ++ "mnasnet0_75.safetensors"),
        .Mnasnet10 => try downloadFile(alloc, url ++ "mnasnet1_0.safetensors"),
        .Mnasnet13 => try downloadFile(alloc, url ++ "mnasnet1_3.safetensors"),
        .MobilenetV2 => try downloadFile(alloc, url ++ "mobilenet_v2.safetensors"),
        .MobilenetV3Large => try downloadFile(alloc, url ++ "mobilenet_v3_large.safetensors"),
        .MobilenetV3Small => try downloadFile(alloc, url ++ "mobilenet_v3_small.safetensors"),
        .Resnet18 => try downloadFile(alloc, url ++ "resnet18.safetensors"),
        .Resnet34 => try downloadFile(alloc, url ++ "resnet34.safetensors"),
        .Resnet50 => try downloadFile(alloc, url ++ "resnet50.safetensors"),
        .Resnet101 => try downloadFile(alloc, url ++ "resnet101.safetensors"),
        .Resnet152 => try downloadFile(alloc, url ++ "resnet152.safetensors"),
        .Resnext5032x4d => try downloadFile(alloc, url ++ "resnext50_32x4d.safetensors"),
        .Resnext10132x8d => try downloadFile(alloc, url ++ "resnext101_32x8d.safetensors"),
        .WideResnet502 => try downloadFile(alloc, url ++ "wide_resnet50_2.safetensors"),
        .WideResnet1012 => try downloadFile(alloc, url ++ "wide_resnet101_2.safetensors"),
        .ShufflenetV2X05 => try downloadFile(alloc, url ++ "shufflenet_v2_x0_5.safetensors"),
        .ShufflenetV2X10 => try downloadFile(alloc, url ++ "shufflenet_v2_x1_0.safetensors"),
        .ShufflenetV2X15 => try downloadFile(alloc, url ++ "shufflenet_v2_x1_5.safetensors"),
        .ShufflenetV2X20 => try downloadFile(alloc, url ++ "shufflenet_v2_x2_0.safetensors"),
        .Squeezenet10 => try downloadFile(alloc, url ++ "squeezenet1_0.safetensors"),
        .Squeezenet11 => try downloadFile(alloc, url ++ "squeezenet1_1.safetensors"),
        .Vgg11 => try downloadFile(alloc, url ++ "vgg11.safetensors"),
        .Vgg11Bn => try downloadFile(alloc, url ++ "vgg11_bn.safetensors"),
        .Vgg13 => try downloadFile(alloc, url ++ "vgg13.safetensors"),
        .Vgg13Bn => try downloadFile(alloc, url ++ "vgg13_bn.safetensors"),
        .Vgg16 => try downloadFile(alloc, url ++ "vgg16.safetensors"),
        .Vgg16Bn => try downloadFile(alloc, url ++ "vgg16_bn.safetensors"),
        .Vgg19 => try downloadFile(alloc, url ++ "vgg19.safetensors"),
        .Vgg19Bn => try downloadFile(alloc, url ++ "vgg19_bn.safetensors"),
        .ViTB16 => try downloadFile(alloc, url ++ "vit_b_16.safetensors"),
        .ViTB32 => try downloadFile(alloc, url ++ "vit_b_32.safetensors"),
        .ViTL16 => try downloadFile(alloc, url ++ "vit_l_16.safetensors"),
        .ViTL32 => try downloadFile(alloc, url ++ "vit_l_32.safetensors"),
        .SwinT => try downloadFile(alloc, url ++ "swin_t.safetensors"),
        .SwinS => try downloadFile(alloc, url ++ "swin_s.safetensors"),
        .SwinB => try downloadFile(alloc, url ++ "swin_b.safetensors"),
        .SwinV2T => try downloadFile(alloc, url ++ "swin_v2_t.safetensors"),
        .SwinV2S => try downloadFile(alloc, url ++ "swin_v2_s.safetensors"),
        .SwinV2B => try downloadFile(alloc, url ++ "swin_v2_b.safetensors"),
        else => @panic("Pretrained weights not available for this model"),
    };
    return data;
}
pub fn reverseRepeatVector(t: []const i64, comptime n: i64) []i64 {
    std.debug.assert(n >= 0);
    var ret = std.ArrayList(i64).init(torch.global_allocator);
    var rit: i64 = n - 1;
    while (rit >= 0) : (rit -= 1) {
        for (0..n) |_| {
            ret.append(t[@intCast(rit)]) catch err(.AllocFailed);
        }
    }
    return ret.toOwnedSlice() catch err(.AllocFailed);
}

pub fn downloadFile(alloc: std.mem.Allocator, url: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = alloc };
    defer client.deinit();
    const uri: std.Uri = try std.Uri.parse(url);
    var buffer: [4096]u8 = undefined;
    var req = try client.open(.GET, uri, .{ .server_header_buffer = &buffer });
    defer req.deinit();
    try req.send();
    try req.wait();

    var out = std.ArrayList(u8).init(alloc);
    var total: usize = 0;
    var buf: [4096]u8 = undefined;
    var w = out.writer();

    const total_bytes = req.response.content_length.?;

    const filename = std.fs.path.basename(url);
    const path = try std.fs.path.join(alloc, &.{ ".cache", filename });
    defer alloc.free(path);

    const data: []u8 = std.fs.cwd().readFileAlloc(alloc, path, 10_000_000_000) catch "";
    if (data.len == total_bytes)
        return data
    else
        alloc.free(data);

    if (download_progress == null) {
        download_progress = std.Progress.start(.{});
    }
    const progress_node = download_progress.?.start("Downloading", total_bytes);
    defer progress_node.end();
    while (true) {
        const n = try req.readAll(&buf);
        total += n;
        progress_node.setCompletedItems(total);
        if (n == 0) break;
        try w.writeAll(buf[0..n]);
    }

    // Save to file
    std.fs.cwd().access(".cache", .{}) catch |e| {
        switch (e) {
            error.FileNotFound => {
                try std.fs.cwd().makeDir(".cache");
            },
            else => return e,
        }
    };
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(out.items);
    return out.toOwnedSlice();
}

pub const Error = enum { AllocFailed };

pub inline fn err(err_type: Error) noreturn {
    switch (err_type) {
        Error.AllocFailed => std.debug.panic("Allocation failed\n", .{}),
    }
    unreachable;
}
