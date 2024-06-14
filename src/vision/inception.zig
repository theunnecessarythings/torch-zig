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

fn basicConv2d(c_in: i64, c_out: i64, ksize: i64, stride: i64, padding: i64, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .addWithName("conv", Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = .{ ksize, ksize },
        .stride = .{ stride, stride },
        .padding = .{ .Padding = .{ padding, padding } },
        .bias = false,
        .tensor_opts = options,
    }))
        .addWithName("bn", BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init());
}

fn basicConv2d2(c_in: i64, c_out: i64, ksize: [2]i64, pad: [2]i64, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .addWithName("conv", Conv2D.init(.{
        .in_channels = c_in,
        .out_channels = c_out,
        .kernel_size = ksize,
        .padding = .{ .Padding = pad },
        .bias = false,
        .tensor_opts = options,
    }))
        .addWithName("bn", BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init());
}

const InceptionA = struct {
    base_module: *Module = undefined,
    branch1x1: *Sequential = undefined,
    branch5x5_1: *Sequential = undefined,
    branch5x5_2: *Sequential = undefined,
    branch3x3dbl_1: *Sequential = undefined,
    branch3x3dbl_2: *Sequential = undefined,
    branch3x3dbl_3: *Sequential = undefined,
    branch_pool: *Sequential = undefined,
    const Self = @This();

    pub fn init(c_in: i64, pool_feats: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch1x1 = basicConv2d(c_in, 64, 1, 1, 0, options);
        self.branch5x5_1 = basicConv2d(c_in, 48, 1, 1, 0, options);
        self.branch5x5_2 = basicConv2d(48, 64, 5, 1, 2, options);
        self.branch3x3dbl_1 = basicConv2d(c_in, 64, 1, 1, 0, options);
        self.branch3x3dbl_2 = basicConv2d(64, 96, 3, 1, 1, options);
        self.branch3x3dbl_3 = basicConv2d(96, 96, 3, 1, 1, options);
        self.branch_pool = basicConv2d(c_in, pool_feats, 1, 1, 0, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch1x1", self.branch1x1);
        _ = self.base_module.registerModule("branch5x5_1", self.branch5x5_1);
        _ = self.base_module.registerModule("branch5x5_2", self.branch5x5_2);
        _ = self.base_module.registerModule("branch3x3dbl_1", self.branch3x3dbl_1);
        _ = self.base_module.registerModule("branch3x3dbl_2", self.branch3x3dbl_2);
        _ = self.base_module.registerModule("branch3x3dbl_3", self.branch3x3dbl_3);
        _ = self.base_module.registerModule("branch_pool", self.branch_pool);
    }

    pub fn deinit(self: *Self) void {
        self.branch1x1.deinit();
        self.branch5x5_1.deinit();
        self.branch5x5_2.deinit();
        self.branch3x3dbl_1.deinit();
        self.branch3x3dbl_2.deinit();
        self.branch3x3dbl_3.deinit();
        self.branch_pool.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const b1 = self.branch1x1.forward(input);
        const b2 = self.branch5x5_2.forward(&self.branch5x5_1.forward(input));
        const b3 = self.branch3x3dbl_3.forward(&self.branch3x3dbl_2.forward(&self.branch3x3dbl_1.forward(input)));
        var b4 = input.avgPool2d(&.{ 3, 3 }, &.{ 1, 1 }, &.{ 1, 1 }, false, true, null);
        b4 = self.branch_pool.forward(&b4);
        var ys = [_]*const Tensor{ &b1, &b2, &b3, &b4 };
        return Tensor.cat(&ys, 1);
    }
};

const InceptionB = struct {
    base_module: *Module = undefined,
    branch3x3: *Sequential = undefined,
    branch3x3dbl_1: *Sequential = undefined,
    branch3x3dbl_2: *Sequential = undefined,
    branch3x3dbl_3: *Sequential = undefined,
    const Self = @This();

    pub fn init(c_in: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch3x3 = basicConv2d(c_in, 384, 3, 2, 0, options);
        self.branch3x3dbl_1 = basicConv2d(c_in, 64, 1, 1, 0, options);
        self.branch3x3dbl_2 = basicConv2d(64, 96, 3, 1, 1, options);
        self.branch3x3dbl_3 = basicConv2d(96, 96, 3, 2, 0, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch3x3", self.branch3x3);
        _ = self.base_module.registerModule("branch3x3dbl_1", self.branch3x3dbl_1);
        _ = self.base_module.registerModule("branch3x3dbl_2", self.branch3x3dbl_2);
        _ = self.base_module.registerModule("branch3x3dbl_3", self.branch3x3dbl_3);
    }

    pub fn deinit(self: *Self) void {
        self.branch3x3.deinit();
        self.branch3x3dbl_1.deinit();
        self.branch3x3dbl_2.deinit();
        self.branch3x3dbl_3.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const b1 = self.branch3x3.forward(input);
        const b2 = self.branch3x3dbl_3.forward(&self.branch3x3dbl_2.forward(&self.branch3x3dbl_1.forward(input)));
        const b3 = input.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false);
        var ys = [_]*const Tensor{ &b1, &b2, &b3 };
        return Tensor.cat(&ys, 1);
    }
};

const InceptionC = struct {
    base_module: *Module = undefined,
    branch1x1: *Sequential = undefined,
    branch7x7_1: *Sequential = undefined,
    branch7x7_2: *Sequential = undefined,
    branch7x7_3: *Sequential = undefined,
    branch7x7dbl_1: *Sequential = undefined,
    branch7x7dbl_2: *Sequential = undefined,
    branch7x7dbl_3: *Sequential = undefined,
    branch7x7dbl_4: *Sequential = undefined,
    branch7x7dbl_5: *Sequential = undefined,
    branch_pool: *Sequential = undefined,

    const Self = @This();

    pub fn init(c_in: i64, channels_7x7: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch1x1 = basicConv2d(c_in, 192, 1, 1, 0, options);
        self.branch7x7_1 = basicConv2d(c_in, channels_7x7, 1, 1, 0, options);
        self.branch7x7_2 = basicConv2d2(channels_7x7, channels_7x7, .{ 1, 7 }, .{ 0, 3 }, options);
        self.branch7x7_3 = basicConv2d2(channels_7x7, 192, .{ 7, 1 }, .{ 3, 0 }, options);
        self.branch7x7dbl_1 = basicConv2d(c_in, channels_7x7, 1, 1, 0, options);
        self.branch7x7dbl_2 = basicConv2d2(channels_7x7, channels_7x7, .{ 7, 1 }, .{ 3, 0 }, options);
        self.branch7x7dbl_3 = basicConv2d2(channels_7x7, channels_7x7, .{ 1, 7 }, .{ 0, 3 }, options);
        self.branch7x7dbl_4 = basicConv2d2(channels_7x7, channels_7x7, .{ 7, 1 }, .{ 3, 0 }, options);
        self.branch7x7dbl_5 = basicConv2d2(channels_7x7, 192, .{ 1, 7 }, .{ 0, 3 }, options);
        self.branch_pool = basicConv2d(c_in, 192, 1, 1, 0, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch1x1", self.branch1x1);
        _ = self.base_module.registerModule("branch7x7_1", self.branch7x7_1);
        _ = self.base_module.registerModule("branch7x7_2", self.branch7x7_2);
        _ = self.base_module.registerModule("branch7x7_3", self.branch7x7_3);
        _ = self.base_module.registerModule("branch7x7dbl_1", self.branch7x7dbl_1);
        _ = self.base_module.registerModule("branch7x7dbl_2", self.branch7x7dbl_2);
        _ = self.base_module.registerModule("branch7x7dbl_3", self.branch7x7dbl_3);
        _ = self.base_module.registerModule("branch7x7dbl_4", self.branch7x7dbl_4);
        _ = self.base_module.registerModule("branch7x7dbl_5", self.branch7x7dbl_5);
        _ = self.base_module.registerModule("branch_pool", self.branch_pool);
    }

    pub fn deinit(self: *Self) void {
        self.branch1x1.deinit();
        self.branch7x7_1.deinit();
        self.branch7x7_2.deinit();
        self.branch7x7_3.deinit();
        self.branch7x7dbl_1.deinit();
        self.branch7x7dbl_2.deinit();
        self.branch7x7dbl_3.deinit();
        self.branch7x7dbl_4.deinit();
        self.branch7x7dbl_5.deinit();
        self.branch_pool.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const b1 = self.branch1x1.forward(input);
        const b2 = self.branch7x7_3.forward(&self.branch7x7_2.forward(&self.branch7x7_1.forward(input)));
        const b3 = self.branch7x7dbl_5.forward(&self.branch7x7dbl_4.forward(&self.branch7x7dbl_3.forward(&self.branch7x7dbl_2.forward(&self.branch7x7dbl_1.forward(input)))));
        var b4 = input.avgPool2d(&.{ 3, 3 }, &.{ 1, 1 }, &.{ 1, 1 }, false, true, null);
        b4 = self.branch_pool.forward(&b4);
        var ys = [_]*const Tensor{ &b1, &b2, &b3, &b4 };
        return Tensor.cat(&ys, 1);
    }
};

const InceptionD = struct {
    base_module: *Module = undefined,
    branch3x3_1: *Sequential = undefined,
    branch3x3_2: *Sequential = undefined,
    branch7x7x3_1: *Sequential = undefined,
    branch7x7x3_2: *Sequential = undefined,
    branch7x7x3_3: *Sequential = undefined,
    branch7x7x3_4: *Sequential = undefined,

    const Self = @This();

    pub fn init(c_in: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch3x3_1 = basicConv2d(c_in, 192, 1, 1, 0, options);
        self.branch3x3_2 = basicConv2d(192, 320, 3, 2, 0, options);
        self.branch7x7x3_1 = basicConv2d(c_in, 192, 1, 1, 0, options);
        self.branch7x7x3_2 = basicConv2d2(192, 192, .{ 1, 7 }, .{ 0, 3 }, options);
        self.branch7x7x3_3 = basicConv2d2(192, 192, .{ 7, 1 }, .{ 3, 0 }, options);
        self.branch7x7x3_4 = basicConv2d(192, 192, 3, 2, 0, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch3x3_1", self.branch3x3_1);
        _ = self.base_module.registerModule("branch3x3_2", self.branch3x3_2);
        _ = self.base_module.registerModule("branch7x7x3_1", self.branch7x7x3_1);
        _ = self.base_module.registerModule("branch7x7x3_2", self.branch7x7x3_2);
        _ = self.base_module.registerModule("branch7x7x3_3", self.branch7x7x3_3);
        _ = self.base_module.registerModule("branch7x7x3_4", self.branch7x7x3_4);
    }

    pub fn deinit(self: *Self) void {
        self.branch3x3_1.deinit();
        self.branch3x3_2.deinit();
        self.branch7x7x3_1.deinit();
        self.branch7x7x3_2.deinit();
        self.branch7x7x3_3.deinit();
        self.branch7x7x3_4.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const b1 = self.branch3x3_2.forward(&self.branch3x3_1.forward(input));
        const b2 = self.branch7x7x3_4.forward(&self.branch7x7x3_3.forward(&self.branch7x7x3_2.forward(&self.branch7x7x3_1.forward(input))));
        const b3 = input.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false);
        var ys = [_]*const Tensor{ &b1, &b2, &b3 };
        return Tensor.cat(&ys, 1);
    }
};

const InceptionE = struct {
    base_module: *Module = undefined,
    branch1x1: *Sequential = undefined,
    branch3x3_1: *Sequential = undefined,
    branch3x3_2a: *Sequential = undefined,
    branch3x3_2b: *Sequential = undefined,
    branch3x3dbl_1: *Sequential = undefined,
    branch3x3dbl_2: *Sequential = undefined,
    branch3x3dbl_3a: *Sequential = undefined,
    branch3x3dbl_3b: *Sequential = undefined,
    branch_pool: *Sequential = undefined,

    const Self = @This();

    pub fn init(c_in: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch1x1 = basicConv2d(c_in, 320, 1, 1, 0, options);
        self.branch3x3_1 = basicConv2d(c_in, 384, 1, 1, 0, options);
        self.branch3x3_2a = basicConv2d2(384, 384, .{ 1, 3 }, .{ 0, 1 }, options);
        self.branch3x3_2b = basicConv2d2(384, 384, .{ 3, 1 }, .{ 1, 0 }, options);
        self.branch3x3dbl_1 = basicConv2d(c_in, 448, 1, 1, 0, options);
        self.branch3x3dbl_2 = basicConv2d(448, 384, 3, 1, 1, options);
        self.branch3x3dbl_3a = basicConv2d2(384, 384, .{ 1, 3 }, .{ 0, 1 }, options);
        self.branch3x3dbl_3b = basicConv2d2(384, 384, .{ 3, 1 }, .{ 1, 0 }, options);
        self.branch_pool = basicConv2d(c_in, 192, 1, 1, 0, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch1x1", self.branch1x1);
        _ = self.base_module.registerModule("branch3x3_1", self.branch3x3_1);
        _ = self.base_module.registerModule("branch3x3_2a", self.branch3x3_2a);
        _ = self.base_module.registerModule("branch3x3_2b", self.branch3x3_2b);
        _ = self.base_module.registerModule("branch3x3dbl_1", self.branch3x3dbl_1);
        _ = self.base_module.registerModule("branch3x3dbl_2", self.branch3x3dbl_2);
        _ = self.base_module.registerModule("branch3x3dbl_3a", self.branch3x3dbl_3a);
        _ = self.base_module.registerModule("branch3x3dbl_3b", self.branch3x3dbl_3b);
        _ = self.base_module.registerModule("branch_pool", self.branch_pool);
    }

    pub fn deinit(self: *Self) void {
        self.branch1x1.deinit();
        self.branch3x3_1.deinit();
        self.branch3x3_2a.deinit();
        self.branch3x3_2b.deinit();
        self.branch3x3dbl_1.deinit();
        self.branch3x3dbl_2.deinit();
        self.branch3x3dbl_3a.deinit();
        self.branch3x3dbl_3b.deinit();
        self.branch_pool.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const b1 = self.branch1x1.forward(input);
        var b2 = self.branch3x3_1.forward(input);
        var b2_temp = [_]*const Tensor{ &self.branch3x3_2a.forward(&b2), &self.branch3x3_2b.forward(&b2) };
        b2 = Tensor.cat(&b2_temp, 1);
        var b3 = self.branch3x3dbl_2.forward(&self.branch3x3dbl_1.forward(input));
        var b3_temp = [_]*const Tensor{ &self.branch3x3dbl_3a.forward(&b3), &self.branch3x3dbl_3b.forward(&b3) };
        b3 = Tensor.cat(&b3_temp, 1);
        var b4 = input.avgPool2d(&.{ 3, 3 }, &.{ 1, 1 }, &.{ 1, 1 }, false, true, null);
        b4 = self.branch_pool.forward(&b4);
        var ys = [_]*const Tensor{ &b1, &b2, &b3, &b4 };
        return Tensor.cat(&ys, 1);
    }
};

const InceptionAux = struct {
    base_module: *Module = undefined,
    conv0: *Sequential = undefined,
    conv1: *Sequential = undefined,
    fc: *Linear = undefined,
    aux: bool = true,

    const Self = @This();

    pub fn init(c_in: i64, num_classes: i64, aux: bool, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .aux = aux };
        self.base_module = Module.init(self);
        self.conv0 = basicConv2d(c_in, 128, 1, 1, 0, options);
        self.conv1 = basicConv2d(128, 768, 5, 1, 0, options);
        self.fc = Linear.init(.{ .in_features = 768, .out_features = num_classes, .bias = true, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv0", self.conv0);
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("fc", self.fc);
    }

    pub fn deinit(self: *Self) void {
        self.conv0.deinit();
        self.conv1.deinit();
        self.fc.deinit();
        _ = torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        if (!self.aux) {
            return input.*;
        }
        var x = input.avgPool2d(&.{ 5, 5 }, &.{ 3, 3 }, &.{ 0, 0 }, false, true, null);
        x = self.conv0.forward(&x);
        x = self.conv1.forward(&x);
        x = x.flatten(1, -1);
        x = self.fc.forward(&x);
        return x;
    }
};

fn transformInput(x: *const Tensor, transform_input: bool) Tensor {
    if (transform_input) {
        const x_ch0 = x.select(1, 0).unsqueeze(1).mulScalar(Scalar.float(0.229))
            .addScalar(Scalar.float(-0.485)).divScalar(Scalar.float(0.5));
        const x_ch1 = x.select(1, 1).unsqueeze(1).mulScalar(Scalar.float(0.224))
            .addScalar(Scalar.float(-0.456)).divScalar(Scalar.float(0.5));
        const x_ch2 = x.select(1, 2).unsqueeze(1).mulScalar(Scalar.float(0.225))
            .addScalar(Scalar.float(-0.406)).divScalar(Scalar.float(0.5));
        var y = [_]*const Tensor{ &x_ch0, &x_ch1, &x_ch2 };
        return Tensor.cat(&y, 1);
    }
    return x.*;
}

fn inception3(num_classes: i64, comptime transform_input: bool, dropout: f32, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .add(Functional(transformInput, .{transform_input}).init())
        .addWithName("Conv2d_1a_3x3", basicConv2d(3, 32, 3, 2, 0, options))
        .addWithName("Conv2d_2a_3x3", basicConv2d(32, 32, 3, 1, 0, options))
        .addWithName("Conv2d_2b_3x3", basicConv2d(32, 64, 3, 1, 1, options))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
        .addWithName("Conv2d_3b_1x1", basicConv2d(64, 80, 1, 1, 0, options))
        .addWithName("Conv2d_4a_3x3", basicConv2d(80, 192, 3, 1, 0, options))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init())
        .addWithName("Mixed_5b", InceptionA.init(192, 32, options))
        .addWithName("Mixed_5c", InceptionA.init(256, 64, options))
        .addWithName("Mixed_5d", InceptionA.init(288, 64, options))
        .addWithName("Mixed_6a", InceptionB.init(288, options))
        .addWithName("Mixed_6b", InceptionC.init(768, 128, options))
        .addWithName("Mixed_6c", InceptionC.init(768, 160, options))
        .addWithName("Mixed_6d", InceptionC.init(768, 160, options))
        .addWithName("Mixed_6e", InceptionC.init(768, 192, options))
        .addWithName("AuxLogits", InceptionAux.init(768, num_classes, false, options))
        .addWithName("Mixed_7a", InceptionD.init(768, options))
        .addWithName("Mixed_7b", InceptionE.init(1280, options))
        .addWithName("Mixed_7c", InceptionE.init(2048, options))
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
        .add(Dropout.init(dropout))
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .addWithName("fc", Linear.init(.{ .in_features = 2048, .out_features = num_classes, .bias = true, .tensor_opts = options }));
}

pub fn inceptionV3(num_classes: i64, options: TensorOptions) *Sequential {
    return inception3(num_classes, true, 0.5, options);
}
