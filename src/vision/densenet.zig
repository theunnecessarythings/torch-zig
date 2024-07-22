const torch = @import("../torch.zig");
const std = @import("std");
const err = torch.utils.err;
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

const DenseLayer = struct {
    base_module: *Module = undefined,
    bn1: *BatchNorm2D = undefined,
    conv1: *Conv2D = undefined,
    bn2: *BatchNorm2D = undefined,
    conv2: *Conv2D = undefined,
    options: TensorOptions,

    const Self = @This();

    pub fn init(c_in: i64, bn_size: i64, growth: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        const c_inter = bn_size * growth;
        self.bn1 = BatchNorm2D.init(.{ .num_features = c_in, .tensor_opts = options });
        self.conv1 = Conv2D.init(.{
            .in_channels = c_in,
            .out_channels = c_inter,
            .kernel_size = .{ 1, 1 },
            .padding = .{ .Padding = .{ 0, 0 } },
            .stride = .{ 1, 1 },
            .bias = false,
            .tensor_opts = options,
        });
        self.bn2 = BatchNorm2D.init(.{ .num_features = c_inter, .tensor_opts = options });
        self.conv2 = Conv2D.init(.{
            .in_channels = c_inter,
            .out_channels = growth,
            .kernel_size = .{ 3, 3 },
            .padding = .{ .Padding = .{ 1, 1 } },
            .stride = .{ 1, 1 },
            .bias = false,
            .tensor_opts = options,
        });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("norm1", self.bn1);
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("norm2", self.bn2);
        _ = self.base_module.registerModule("conv2", self.conv2);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        // self.bn1.deinit();
        // self.conv1.deinit();
        // self.bn2.deinit();
        // self.conv2.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var out = self.bn1.forward(x).relu();
        out = self.conv1.forward(&out);
        out = self.bn2.forward(&out).relu();
        out = self.conv2.forward(&out);
        return out;
    }
};

const DenseBlock = struct {
    base_module: *Module = undefined,
    layers: std.ArrayList(*DenseLayer) = undefined,
    options: TensorOptions = undefined,

    const Self = @This();

    pub fn init(c_in: i64, bn_size: i64, growth: i64, nlayers: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .options = options,
            .layers = std.ArrayList(*DenseLayer).init(torch.global_allocator),
        };
        self.layers.resize(@intCast(nlayers)) catch err(.AllocFailed);
        self.base_module = Module.init(self);
        for (0..@intCast(nlayers)) |i| {
            self.layers.items[i] = DenseLayer.init(c_in + @as(i64, @intCast(i)) * growth, bn_size, growth, options);
        }
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        for (0..self.layers.items.len) |i| {
            const name = std.fmt.allocPrint(torch.global_allocator, "denselayer{d}", .{i + 1}) catch err(.AllocFailed);
            _ = self.base_module.registerModule(name, self.layers.items[i]);
        }
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        for (0..self.layers.items.len) |i| {
            self.layers.items[i].deinit();
        }
        self.layers.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var features = std.ArrayList(*Tensor).init(torch.global_allocator);
        defer features.deinit();
        features.resize(self.layers.items.len + 1) catch err(.AllocFailed);
        features.items[0] = @constCast(x);
        for (0..self.layers.items.len) |i| {
            var ys = self.layers.items[i].forward(&Tensor.cat(features.items[0..(i + 1)], 1));
            features.items[i + 1] = &ys;
        }
        return Tensor.cat(features.items, 1);
    }
};

fn transition(c_in: i64, c_out: i64, options: TensorOptions) *Sequential {
    const seq = Sequential.init(options)
        .addWithName("norm", BatchNorm2D.init(.{ .num_features = c_in, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .addWithName("conv", Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = .{ 1, 1 },
        .padding = .{ .Padding = .{ 0, 0 } },
        .stride = .{ 1, 1 },
        .bias = false,
        .tensor_opts = options,
    }))
        .add(Functional(Tensor.avgPool2d, .{ &.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, false, true, null }).init());
    return seq;
}

const DenseNet = struct {
    base_module: *Module = undefined,
    features: *Sequential = undefined,
    classifier: *Linear = undefined,

    const Self = @This();
    pub fn init(c_in: i64, bn_size: i64, growth: i64, block_config: []const i64, c_out: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.features = Sequential.init(options)
            .addWithName("conv0", Conv2D.init(.{
            .in_channels = 3,
            .out_channels = c_in,
            .kernel_size = .{ 7, 7 },
            .padding = .{ .Padding = .{ 3, 3 } },
            .stride = .{ 2, 2 },
            .bias = false,
            .tensor_opts = options,
        }))
            .addWithName("norm0", BatchNorm2D.init(.{ .num_features = c_in, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 1, 1 }, &.{ 1, 1 }, false }).init());
        var nfeat = c_in;
        for (block_config, 0..) |nlayers, i| {
            const name = std.fmt.allocPrint(torch.global_allocator, "denseblock{d}", .{i + 1}) catch err(.AllocFailed);
            self.features = self.features.addWithName(name, DenseBlock.init(nfeat, bn_size, growth, nlayers, options));
            nfeat += nlayers * growth;
            if (i + 1 != block_config.len) {
                const layer_name = std.fmt.allocPrint(torch.global_allocator, "transition{d}", .{i + 1}) catch err(.AllocFailed);
                self.features = self.features.addWithName(layer_name, transition(nfeat, @divFloor(nfeat, 2), options));
                nfeat = @divFloor(nfeat, 2);
            }
        }
        self.features = self.features.addWithName("norm5", BatchNorm2D.init(.{ .num_features = nfeat, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
            .add(Functional(Tensor.flatten, .{ 1, -1 }).init());
        self.classifier = Linear.init(.{ .in_features = nfeat, .out_features = c_out, .bias = true, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("features", self.features);
        _ = self.base_module.registerModule("classifier", self.classifier);
    }

    pub fn deinit(self: *Self) void {
        self.features.deinit();
        // self.classifier.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = self.features.forward(x);
        return self.classifier.forward(&y);
    }
};

pub fn densenet121(comptime nclasses: i64, options: TensorOptions) *DenseNet {
    return DenseNet.init(64, 4, 32, &.{ 6, 12, 24, 16 }, nclasses, options);
}

pub fn densenet169(comptime nclasses: i64, options: TensorOptions) *DenseNet {
    return DenseNet.init(64, 4, 32, &.{ 6, 12, 32, 32 }, nclasses, options);
}

pub fn densenet201(comptime nclasses: i64, options: TensorOptions) *DenseNet {
    return DenseNet.init(64, 4, 32, &.{ 6, 12, 48, 32 }, nclasses, options);
}

pub fn densenet161(comptime nclasses: i64, options: TensorOptions) *DenseNet {
    return DenseNet.init(96, 4, 48, &.{ 6, 12, 36, 24 }, nclasses, options);
}
