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
const Dropout = functional.Dropout;
const Linear = torch.linear.Linear;
const Functional = functional.Functional;
const Conv2D = torch.conv.Conv2D;
const BatchNorm2D = torch.norm.BatchNorm(2);
const Sequential = module.Sequential;

fn channelShuffle(x: *const Tensor, groups: i64) Tensor {
    const bs, const c, const h, const w = x.sizeDims(4);
    const channels_per_group = @divFloor(c, groups);
    return x.view(&.{ bs, groups, channels_per_group, h, w })
        .transpose(1, 2).contiguous()
        .view(&.{ bs, c, h, w });
}

fn depthwiseConv(c_in: i64, c_out: i64, ksize: i64, stride: i64, padding: i64, bias: bool, options: TensorOptions) *Conv2D {
    return Conv2D.init(.{ .in_channels = c_in, .out_channels = c_out, .kernel_size = .{ ksize, ksize }, .stride = .{ stride, stride }, .padding = .{ .Padding = .{ padding, padding } }, .groups = c_in, .bias = bias, .tensor_opts = options });
}

const InvertedResidual = struct {
    base_module: *Module = undefined,
    branch1: *Sequential = undefined,
    branch2: *Sequential = undefined,
    stride: i64,

    const Self = @This();

    pub fn init(c_in: i64, c_out: i64, stride: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{ .stride = stride };
        self.base_module = Module.init(self);
        self.branch1 = Sequential.init(options);
        const branch_features = @divFloor(c_out, 2);
        if (stride > 1) {
            self.branch1 = self.branch1.add(depthwiseConv(c_in, c_in, 3, stride, 1, false, options))
                .add(BatchNorm2D.init(.{ .num_features = c_in, .tensor_opts = options }))
                .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = branch_features, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
                .add(BatchNorm2D.init(.{ .num_features = branch_features, .tensor_opts = options }))
                .add(Functional(Tensor.relu, .{}).init());
        }
        self.branch2 = Sequential.init(options)
            .add(Conv2D.init(.{ .in_channels = if (stride > 1) c_in else branch_features, .out_channels = branch_features, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = branch_features, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .add(depthwiseConv(branch_features, branch_features, 3, stride, 1, false, options))
            .add(BatchNorm2D.init(.{ .num_features = branch_features, .tensor_opts = options }))
            .add(Conv2D.init(.{ .in_channels = branch_features, .out_channels = branch_features, .kernel_size = .{ 1, 1 }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = branch_features, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init());
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("branch1", self.branch1);
        _ = self.base_module.registerModule("branch2", self.branch2);
    }

    pub fn deinit(self: *Self) void {
        self.branch1.deinit();
        self.branch2.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var ys: Tensor = undefined;
        if (self.stride == 1) {
            const xs = x.chunk(2, 1);
            var y = [_]*const Tensor{ &xs[0], &self.branch2.forward(&xs[1]) };
            ys = Tensor.cat(&y, 1);
        } else {
            var y = [_]*const Tensor{ &self.branch1.forward(x), &self.branch2.forward(x) };
            ys = Tensor.cat(&y, 1);
        }
        return channelShuffle(&ys, 2);
    }
};

const ShuffleNetV2 = struct {
    base_module: *Module = undefined,
    conv1: *Sequential = undefined,
    stages: [3]*Sequential = undefined,
    conv5: *Sequential = undefined,
    fc: *Linear = undefined,

    const Self = @This();

    fn init(stages_repeats: []const i64, stages_c_out: []const i64, num_classes: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        var c_in: i64 = 3;
        var c_out = stages_c_out[0];
        self.conv1 = Sequential.init(options)
            .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = c_out, .kernel_size = .{ 3, 3 }, .stride = .{ 2, 2 }, .padding = .{ .Padding = .{ 1, 1 } }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init());
        c_in = c_out;
        for (2..5) |i| {
            const repeats = stages_repeats[i - 2];
            c_out = stages_c_out[i - 1];
            var stage = Sequential.init(options)
                .add(InvertedResidual.init(c_in, c_out, 2, options));
            for (0..@intCast(repeats - 1)) |_| {
                stage = stage.add(InvertedResidual.init(c_out, c_out, 1, options));
            }
            c_in = c_out;
            self.stages[i - 2] = stage;
        }
        c_out = stages_c_out[stages_c_out.len - 1];
        self.conv5 = Sequential.init(options)
            .add(Conv2D.init(.{ .in_channels = c_in, .out_channels = c_out, .kernel_size = .{ 1, 1 }, .stride = .{ 1, 1 }, .padding = .{ .Padding = .{ 0, 0 } }, .bias = false, .tensor_opts = options }))
            .add(BatchNorm2D.init(.{ .num_features = c_out, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init());
        self.fc = Linear.init(.{ .in_features = c_out, .out_features = num_classes, .bias = true, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("stage2", self.stages[0]);
        _ = self.base_module.registerModule("stage3", self.stages[1]);
        _ = self.base_module.registerModule("stage4", self.stages[2]);
        _ = self.base_module.registerModule("conv5", self.conv5);
        _ = self.base_module.registerModule("fc", self.fc);
    }

    pub fn deinit(self: *Self) void {
        // self.conv1.deinit();
        self.stages[0].deinit();
        self.stages[1].deinit();
        self.stages[2].deinit();
        // self.conv5.deinit();
        // self.fc.deinit();
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = self.conv1.forward(x)
            .maxPool2d(&.{ 3, 3 }, &.{ 2, 2 }, &.{ 1, 1 }, &.{ 1, 1 }, false);
        y = self.stages[0].forward(&y);
        y = self.stages[1].forward(&y);
        y = self.stages[2].forward(&y);
        y = self.conv5.forward(&y).adaptiveAvgPool2d(&.{ 1, 1 }).flatten(1, -1);
        y = self.fc.forward(&y);
        return y;
    }
};
pub fn shuffleNetV2_x0_5(num_classes: i64, options: TensorOptions) *ShuffleNetV2 {
    return ShuffleNetV2.init(&.{ 4, 8, 4 }, &.{ 24, 48, 96, 192, 1024 }, num_classes, options);
}

pub fn shuffleNetV2_x1_0(num_classes: i64, options: TensorOptions) *ShuffleNetV2 {
    return ShuffleNetV2.init(&.{ 4, 8, 4 }, &.{ 24, 116, 232, 464, 1024 }, num_classes, options);
}

pub fn shuffleNetV2_x1_5(num_classes: i64, options: TensorOptions) *ShuffleNetV2 {
    return ShuffleNetV2.init(&.{ 4, 8, 4 }, &.{ 24, 176, 352, 704, 1024 }, num_classes, options);
}

pub fn shuffleNetV2_x2_0(num_classes: i64, options: TensorOptions) *ShuffleNetV2 {
    return ShuffleNetV2.init(&.{ 4, 8, 4 }, &.{ 24, 244, 488, 976, 2048 }, num_classes, options);
}
