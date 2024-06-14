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

const Inception = struct {
    base_module: *Module = undefined,
    branch1: *Sequential = undefined,
    branch2: *Sequential = undefined,
    branch3: *Sequential = undefined,
    branch4: *Sequential = undefined,

    const Self = @This();

    pub fn init(
        c_in: i64,
        ch1x1: i64,
        ch3x3red: i64,
        ch3x3: i64,
        ch5x5red: i64,
        ch5x5: i64,
        pool_proj: i64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        self.branch1 = basicConv2d(c_in, ch1x1, 1, 1, 0, options);
        self.branch2 = Sequential.init(options)
            .add(basicConv2d(c_in, ch3x3red, 1, 1, 0, options))
            .add(basicConv2d(ch3x3red, ch3x3, 3, 1, 1, options));
        self.branch3 = Sequential.init(options)
            .add(basicConv2d(c_in, ch5x5red, 1, 1, 0, options))
            .add(basicConv2d(ch5x5red, ch5x5, 3, 1, 1, options));
        self.branch4 = Sequential.init(options)
            .add(Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 1, 1 }, &.{ 1, 1 }, &.{ 1, 1 }, true }).init())
            .add(basicConv2d(c_in, pool_proj, 1, 1, 0, options));
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch1", self.branch1);
        _ = self.base_module.registerModule("branch2", self.branch2);
        _ = self.base_module.registerModule("branch3", self.branch3);
        _ = self.base_module.registerModule("branch4", self.branch4);
    }

    pub fn deinit(self: *Self) void {
        self.branch1.deinit();
        self.branch2.deinit();
        self.branch3.deinit();
        self.branch4.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const branch1 = self.branch1.forward(input);
        const branch2 = self.branch2.forward(input);
        const branch3 = self.branch3.forward(input);
        const branch4 = self.branch4.forward(input);
        var y = [_]*const Tensor{ &branch1, &branch2, &branch3, &branch4 };
        return Tensor.cat(&y, 1);
    }
};

fn inceptionAux(c_in: i64, num_classes: i64, dropout: f32, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 4, 4 }}).init())
        .addWithName("conv", basicConv2d(c_in, 128, 1, 1, 0, options))
        .addWithName("fc1", Linear.init(.{ .in_features = 2048, .out_features = 1024, .bias = true, .tensor_opts = options }))
        .add(Functional(Tensor.relu, .{}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .add(Dropout.init(dropout))
        .addWithName("fc2", Linear.init(.{ .in_features = 1024, .out_features = num_classes, .bias = true, .tensor_opts = options }));
}

const GoogleNet = struct {
    base_module: *Module = undefined,
    conv1: *Sequential = undefined,
    conv2: *Sequential = undefined,
    conv3: *Sequential = undefined,
    inception3a: *Inception = undefined,
    inception3b: *Inception = undefined,
    inception4a: *Inception = undefined,
    inception4b: *Inception = undefined,
    inception4c: *Inception = undefined,
    inception4d: *Inception = undefined,
    inception4e: *Inception = undefined,
    inception5a: *Inception = undefined,
    inception5b: *Inception = undefined,
    aux1: ?*Sequential = null,
    aux2: ?*Sequential = null,
    fc: *Linear = undefined,
    transform_input: bool = false,
    dropout: f64 = 0.5,

    const Self = @This();

    pub fn init(num_classes: i64, aux_logits: bool, transform_input: bool, dropout: f32, dropout_aux: f32, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{ .transform_input = transform_input, .dropout = dropout };
        self.base_module = Module.init(self);
        self.conv1 = basicConv2d(3, 64, 7, 2, 3, options);
        self.conv2 = basicConv2d(64, 64, 1, 1, 0, options);
        self.conv3 = basicConv2d(64, 192, 3, 1, 1, options);
        self.inception3a = Inception.init(192, 64, 96, 128, 16, 32, 32, options);
        self.inception3b = Inception.init(256, 128, 128, 192, 32, 96, 64, options);
        self.inception4a = Inception.init(480, 192, 96, 208, 16, 48, 64, options);
        self.inception4b = Inception.init(512, 160, 112, 224, 24, 64, 64, options);
        self.inception4c = Inception.init(512, 128, 128, 256, 24, 64, 64, options);
        self.inception4d = Inception.init(512, 112, 144, 288, 32, 64, 64, options);
        self.inception4e = Inception.init(528, 256, 160, 320, 32, 128, 128, options);
        self.inception5a = Inception.init(832, 256, 160, 320, 32, 128, 128, options);
        self.inception5b = Inception.init(832, 384, 192, 384, 48, 128, 128, options);
        if (aux_logits) {
            self.aux1 = inceptionAux(512, num_classes, dropout_aux, options);
            self.aux2 = inceptionAux(528, num_classes, dropout_aux, options);
        }
        self.fc = Linear.init(.{ .in_features = 1024, .out_features = num_classes, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        self.conv1.reset();
        self.conv2.reset();
        self.conv3.reset();
        self.inception3a.reset();
        self.inception3b.reset();
        self.inception4a.reset();
        self.inception4b.reset();
        self.inception4c.reset();
        self.inception4d.reset();
        self.inception4e.reset();
        self.inception5a.reset();
        self.inception5b.reset();
        if (self.aux1 != null) {
            self.aux1.?.reset();
            _ = self.base_module.registerModule("aux1", self.aux1.?);
        }
        if (self.aux2 != null) {
            self.aux2.?.reset();
            _ = self.base_module.registerModule("aux2", self.aux2.?);
        }
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("conv2", self.conv2);
        _ = self.base_module.registerModule("conv3", self.conv3);
        _ = self.base_module.registerModule("inception3a", self.inception3a);
        _ = self.base_module.registerModule("inception3b", self.inception3b);
        _ = self.base_module.registerModule("inception4a", self.inception4a);
        _ = self.base_module.registerModule("inception4b", self.inception4b);
        _ = self.base_module.registerModule("inception4c", self.inception4c);
        _ = self.base_module.registerModule("inception4d", self.inception4d);
        _ = self.base_module.registerModule("inception4e", self.inception4e);
        _ = self.base_module.registerModule("inception5a", self.inception5a);
        _ = self.base_module.registerModule("inception5b", self.inception5b);
        _ = self.base_module.registerModule("fc", self.fc);
    }

    pub fn deinit(self: *Self) void {
        self.conv1.deinit();
        self.conv2.deinit();
        self.conv3.deinit();
        self.inception3a.deinit();
        self.inception3b.deinit();
        self.inception4a.deinit();
        self.inception4b.deinit();
        self.inception4c.deinit();
        self.inception4d.deinit();
        self.inception4e.deinit();
        self.inception5a.deinit();
        self.inception5b.deinit();
        if (self.aux1 != null) {
            self.aux1.?.deinit();
        }
        if (self.aux2 != null) {
            self.aux2.?.deinit();
        }
        self.fc.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        var xs = input.shallowClone();
        if (self.transform_input) {
            const x_ch0 = xs.select(1, 0).unsqueeze(1).mulScalar(Scalar.float(0.229))
                .addScalar(Scalar.float(-0.485)).divScalar(Scalar.float(0.5));
            const x_ch1 = xs.select(1, 1).unsqueeze(1).mulScalar(Scalar.float(0.224))
                .addScalar(Scalar.float(-0.456)).divScalar(Scalar.float(0.5));
            const x_ch2 = xs.select(1, 2).unsqueeze(1).mulScalar(Scalar.float(0.225))
                .addScalar(Scalar.float(-0.406)).divScalar(Scalar.float(0.5));
            var y = [_]*const Tensor{ &x_ch0, &x_ch1, &x_ch2 };
            xs = Tensor.cat(&y, 1);
        }

        xs = self.conv1.forward(&xs);
        xs = xs.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, true);
        xs = self.conv2.forward(&xs);
        xs = self.conv3.forward(&xs);
        xs = xs.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, true);
        xs = self.inception3a.forward(&xs);
        xs = self.inception3b.forward(&xs);
        xs = xs.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, true);
        xs = self.inception4a.forward(&xs);

        var aux1_out: ?Tensor = null;
        if (self.aux1 != null) {
            aux1_out = self.aux1.?.forward(&xs);
        }
        xs = self.inception4b.forward(&xs);
        xs = self.inception4c.forward(&xs);
        xs = self.inception4d.forward(&xs);

        var aux2_out: ?Tensor = null;
        if (self.aux2 != null) {
            aux2_out = self.aux2.?.forward(&xs);
        }
        xs = self.inception4e.forward(&xs);
        xs = xs.maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, true);
        xs = self.inception5a.forward(&xs);
        xs = self.inception5b.forward(&xs);
        xs = xs.adaptiveAvgPool2d(&.{ 1, 1 });
        xs = xs.flatten(1, -1);
        xs = xs.dropout(self.dropout, true);
        xs = self.fc.forward(&xs);

        return xs;
    }
};

pub fn googlenet(num_classes: i64, options: TensorOptions) *GoogleNet {
    return GoogleNet.init(num_classes, true, true, 0.2, 0.7, options);
}
