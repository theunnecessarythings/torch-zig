const torch = @import("../torch.zig");
const std = @import("std");
const Tensor = torch.Tensor;
const TensorOptions = torch.TensorOptions;
const module = torch.module;
const Module = module.Module;
const conv = torch.conv;
const Conv2D = torch.conv.Conv2D;
const BatchNorm2D = torch.norm.BatchNorm(2);
const Sequential = module.Sequential;

pub const BlockType = enum {
    BasicBlock,
    Bottleneck,
};

pub const ResnetOptions = struct {
    block_type: BlockType,
    layers: [4]usize,
    num_classes: i64,
    groups: i64,
    width_per_group: i64,
    tensor_options: TensorOptions,
};

fn conv3x3(in_planes: i64, out_planes: i64, stride: i64, groups: i64, dilation: i64) *Conv2D {
    return Conv2D.init(.{
        .in_channels = in_planes,
        .out_channels = out_planes,
        .kernel_size = .{ 3, 3 },
        .stride = .{ stride, stride },
        .padding = .{ .Padding = .{ dilation, dilation } },
        .bias = false,
        .groups = groups,
    });
}

fn conv1x1(in_planes: i64, out_planes: i64, stride: i64) *Conv2D {
    return Conv2D.init(.{
        .in_channels = in_planes,
        .out_channels = out_planes,
        .kernel_size = .{ 1, 1 },
        .stride = .{ stride, stride },
        .bias = false,
        .padding = .{ .Padding = .{ 0, 0 } },
    });
}

const BasicBlock = struct {
    base_module: *Module = undefined,

    conv1: *Conv2D = undefined,
    bn1: *BatchNorm2D = undefined,
    conv2: *Conv2D = undefined,
    bn2: *BatchNorm2D = undefined,
    downsample: *Sequential = undefined,

    const Self = @This();

    pub fn init(
        inplanes: i64,
        planes: i64,
        stride: i64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);

        self.conv1 = conv3x3(inplanes, planes, stride, 1, 1);
        self.bn1 = BatchNorm2D.init(.{ .num_features = planes });
        self.conv2 = conv3x3(planes, planes, 1, 1, 1);
        self.bn2 = BatchNorm2D.init(.{ .num_features = planes });
        if (stride != 1 or (inplanes != planes)) {
            self.downsample = Sequential.init(options);
            const conv_ = conv1x1(inplanes, planes, stride);
            self.downsample = self.downsample.add(conv_.base_module);
            const bn_ = BatchNorm2D.init(.{ .num_features = planes });
            self.downsample = self.downsample.add(bn_.base_module);
        } else {
            self.downsample = Sequential.init(options);
        }
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv1", self.conv1.base_module);
        _ = self.base_module.registerModule("bn1", self.bn1.base_module);
        _ = self.base_module.registerModule("conv2", self.conv2.base_module);
        _ = self.base_module.registerModule("bn2", self.bn2.base_module);
        _ = self.base_module.registerModule("downsample", self.downsample.base_module);
        self.conv1.reset();
        self.bn1.reset();
        self.conv2.reset();
        self.bn2.reset();
        self.downsample.reset();
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.conv1.deinit();
        self.bn1.deinit();
        self.conv2.deinit();
        self.bn2.deinit();
        self.downsample.deinit();
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var residual = x.shallowClone();
        var out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = out.relu();
        out = self.conv2.forward(&out);
        out = self.bn2.forward(&out);
        out = out.add(&self.downsample.forward(&residual));
        out = out.relu();
        return out;
    }
};

pub const Bottleneck = struct {
    base_module: *Module = undefined,

    conv1: *Conv2D = undefined,
    bn1: *BatchNorm2D = undefined,
    conv2: *Conv2D = undefined,
    bn2: *BatchNorm2D = undefined,
    conv3: *Conv2D = undefined,
    bn3: *BatchNorm2D = undefined,
    downsample: *Sequential = undefined,

    const Self = @This();

    pub fn init(
        inplanes: i64,
        planes: i64,
        stride: i64,
        groups: i64,
        base_width: i64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{};
        self.base_module = Module.init(self);
        const width = @divExact(planes * base_width, 64);
        self.conv1 = conv1x1(inplanes, width, 1);
        self.bn1 = BatchNorm2D.init(.{ .num_features = width });
        self.conv2 = conv3x3(width, width, stride, groups, 1);
        self.bn2 = BatchNorm2D.init(.{ .num_features = width });
        self.conv3 = conv1x1(width, planes * 4, 1);
        self.bn3 = BatchNorm2D.init(.{ .num_features = planes * 4 });
        if (stride != 1 or (inplanes != planes * 4)) {
            self.downsample = Sequential.init(options);
            const conv_ = conv1x1(inplanes, planes * 4, stride);
            self.downsample = self.downsample.add(conv_.base_module);
            const bn_ = BatchNorm2D.init(.{ .num_features = planes * 4 });
            self.downsample = self.downsample.add(bn_.base_module);
        } else {
            self.downsample = Sequential.init(options);
        }
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv1", self.conv1.base_module);
        _ = self.base_module.registerModule("bn1", self.bn1.base_module);
        _ = self.base_module.registerModule("conv2", self.conv2.base_module);
        _ = self.base_module.registerModule("bn2", self.bn2.base_module);
        _ = self.base_module.registerModule("conv3", self.conv3.base_module);
        _ = self.base_module.registerModule("bn3", self.bn3.base_module);
        _ = self.base_module.registerModule("downsample", self.downsample.base_module);
        self.conv1.reset();
        self.bn1.reset();
        self.conv2.reset();
        self.bn2.reset();
        self.conv3.reset();
        self.bn3.reset();
        self.downsample.reset();
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.conv1.deinit();
        self.bn1.deinit();
        self.conv2.deinit();
        self.bn2.deinit();
        self.conv3.deinit();
        self.bn3.deinit();
        self.downsample.deinit();
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var residual = x.shallowClone();
        var out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = out.relu();
        out = self.conv2.forward(&out);
        out = self.bn2.forward(&out);
        out = out.relu();
        out = self.conv3.forward(&out);
        out = self.bn3.forward(&out);
        out = out.add(&self.downsample.forward(&residual));
        out = out.relu();
        return out;
    }
};

pub const Resnet = struct {
    base_module: *Module = undefined,
    options: ResnetOptions = undefined,
    conv1: *Conv2D = undefined,
    bn1: *BatchNorm2D = undefined,
    layer1: *Sequential = undefined,
    layer2: *Sequential = undefined,
    layer3: *Sequential = undefined,
    layer4: *Sequential = undefined,
    fc: *torch.linear.Linear = undefined,

    const Self = @This();

    pub fn init(options: ResnetOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.base_module = Module.init(self);
        self.conv1 = Conv2D.init(.{
            .in_channels = 3,
            .out_channels = 64,
            .kernel_size = .{ 7, 7 },
            .stride = .{ 2, 2 },
            .padding = .{ .Padding = .{ 3, 3 } },
            .bias = false,
        });
        const expansion: i64 = switch (options.block_type) {
            .BasicBlock => 1,
            .Bottleneck => 4,
        };
        self.bn1 = BatchNorm2D.init(.{ .num_features = 64 });
        self.layer1 = self.makeLayer(
            options.block_type,
            64,
            options.layers[0],
            1,
            options.groups,
            options.width_per_group,
            64,
        );
        self.layer2 = self.makeLayer(
            options.block_type,
            128,
            options.layers[1],
            2,
            options.groups,
            options.width_per_group,
            64 * expansion,
        );
        self.layer3 = self.makeLayer(
            options.block_type,
            256,
            options.layers[2],
            2,
            options.groups,
            options.width_per_group,
            128 * expansion,
        );
        self.layer4 = self.makeLayer(
            options.block_type,
            512,
            options.layers[3],
            2,
            options.groups,
            options.width_per_group,
            256 * expansion,
        );
        self.fc = torch.linear.Linear.init(.{
            .in_features = 512 * expansion,
            .out_features = options.num_classes,
        });
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.conv1.deinit();
        self.bn1.deinit();
        self.layer1.deinit();
        self.layer2.deinit();
        self.layer3.deinit();
        self.layer4.deinit();
        self.fc.deinit();
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv1", self.conv1.base_module);
        _ = self.base_module.registerModule("bn1", self.bn1.base_module);
        _ = self.base_module.registerModule("layer1", self.layer1.base_module);
        _ = self.base_module.registerModule("layer2", self.layer2.base_module);
        _ = self.base_module.registerModule("layer3", self.layer3.base_module);
        _ = self.base_module.registerModule("layer4", self.layer4.base_module);
        _ = self.base_module.registerModule("fc", self.fc.base_module);
        self.conv1.reset();
        self.bn1.reset();
        self.layer1.reset();
        self.layer2.reset();
        self.layer3.reset();
        self.layer4.reset();
        self.fc.reset();
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = out.relu();
        out = out.maxPool2d(&.{3}, &.{2}, &.{1}, &.{1}, false);
        out = self.layer1.forward(&out);
        out = self.layer2.forward(&out);
        out = self.layer3.forward(&out);
        out = self.layer4.forward(&out);
        out = out.adaptiveAvgPool2d(&.{ 1, 1 });
        out = out.flatten(0, -1);
        out = self.fc.forward(&out);
        return out;
    }

    fn makeLayer(
        self: *Self,
        block: BlockType,
        planes: i64,
        blocks: usize,
        stride: i64,
        groups: i64,
        base_width: i64,
        inplanes: i64,
    ) *Sequential {
        var layers = Sequential.init(self.options.tensor_options);
        switch (block) {
            .BasicBlock => {
                const blk = BasicBlock.init(
                    inplanes,
                    planes,
                    1,
                    self.options.tensor_options,
                );
                layers = layers.add(blk.base_module);
                for (1..blocks) |_| {
                    const blk_i = BasicBlock.init(
                        planes,
                        planes,
                        1,
                        self.options.tensor_options,
                    );
                    layers = layers.add(blk_i.base_module);
                }
            },
            .Bottleneck => {
                const blk = Bottleneck.init(
                    inplanes,
                    planes,
                    stride,
                    groups,
                    base_width,
                    self.options.tensor_options,
                );
                layers = layers.add(blk.base_module);
                const in_channels = planes * 4;
                for (1..blocks) |_| {
                    const blk_i = Bottleneck.init(
                        in_channels,
                        planes,
                        1,
                        groups,
                        base_width,
                        self.options.tensor_options,
                    );
                    layers = layers.add(blk_i.base_module);
                }
            },
        }
        return layers;
    }
};

pub fn resnet18(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .BasicBlock,
        .layers = .{ 2, 2, 2, 2 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnet34(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .BasicBlock,
        .layers = .{ 3, 4, 6, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnet50(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 6, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnet101(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 23, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnet152(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 8, 36, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnext50_32x4d(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 6, 3 },
        .num_classes = num_classes,
        .groups = 32,
        .width_per_group = 4,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnext101_32x8d(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 23, 3 },
        .num_classes = num_classes,
        .groups = 32,
        .width_per_group = 8,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn resnext101_64x4d(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 23, 3 },
        .num_classes = num_classes,
        .groups = 64,
        .width_per_group = 4,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn wide_resnet50_2(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 6, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64 * 2,
        .tensor_options = torch.FLOAT_CPU,
    });
}

pub fn wide_resnet101_2(num_classes: i64) *Resnet {
    return Resnet.init(.{
        .block_type = .Bottleneck,
        .layers = .{ 3, 4, 23, 3 },
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 64 * 2,
        .tensor_options = torch.FLOAT_CPU,
    });
}