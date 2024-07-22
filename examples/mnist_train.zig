const std = @import("std");
const torch = @import("torch");
const Tensor = torch.Tensor;
const Module = torch.module.Module;
const Conv2D = torch.conv.Conv2D;
const Dropout = torch.functional.Dropout;
const Linear = torch.linear.Linear;

const Dataset = struct {
    train_images: Tensor,
    train_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    labels: u32,
};

fn readU32(reader: *std.fs.File.Reader) !u32 {
    var b: [4]u8 = undefined;
    _ = try reader.readAll(&b);
    return (@as(u32, @intCast(b[0])) << 24) | (@as(u32, @intCast(b[1])) << 16) | (@as(u32, @intCast(b[2])) << 8) | b[3];
}

fn checkMagicNumber(reader: *std.fs.File.Reader, expected: u32) !void {
    const magic_number = try readU32(reader);
    if (magic_number != expected) {
        return error.InvalidMagicNumber;
    }
}

fn readLabels(filename: []const u8) !Tensor {
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();
    var reader = file.reader();
    try checkMagicNumber(&reader, 2049);
    const samples = try readU32(&reader);
    const data = try std.heap.c_allocator.alloc(u8, samples);
    defer std.heap.c_allocator.free(data);
    _ = try reader.readAll(data);
    return Tensor.fromSlice(u8, data).totype(torch.Kind.Int64);
}

fn readImages(filename: []const u8) !Tensor {
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();
    var reader = file.reader();
    try checkMagicNumber(&reader, 2051);
    const samples = try readU32(&reader);
    const rows = try readU32(&reader);
    const cols = try readU32(&reader);
    const data_len = samples * rows * cols;
    const data = try std.heap.c_allocator.alloc(u8, data_len);
    defer std.heap.c_allocator.free(data);
    _ = try reader.readAll(data);
    return Tensor.fromSlice(u8, data)
        .view(&[_]i64{ samples, rows * cols })
        .totype(torch.Kind.Float)
        .divScalar(torch.Scalar.float(255.0));
}

pub fn loadDir() !Dataset {
    const train_images = try readImages("examples/mnist/train-images.idx3-ubyte");
    const train_labels = try readLabels("examples/mnist/train-labels.idx1-ubyte");
    const test_images = try readImages("examples/mnist/t10k-images.idx3-ubyte");
    const test_labels = try readLabels("examples/mnist/t10k-labels.idx1-ubyte");
    return Dataset{ .train_images = train_images.to(.{ .Cuda = 0 }), .train_labels = train_labels.to(.{ .Cuda = 0 }), .test_images = test_images.to(.{ .Cuda = 0 }), .test_labels = test_labels.to(.{ .Cuda = 0 }), .labels = 10 };
}
const DATA_ROOT = "./data";
const TRAIN_BATCH_SIZE: i64 = 256;
const TEST_BATCH_SIZE: i64 = 1000;

const Net = struct {
    base_module: *Module = undefined,
    conv1: *Conv2D = undefined,
    conv2: *Conv2D = undefined,
    dropout: *Dropout = undefined,
    fc1: *Linear = undefined,
    fc2: *Linear = undefined,

    const Self = @This();

    pub fn init(options: torch.TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch torch.utils.err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.conv1 = Conv2D.init(.{ .in_channels = 1, .out_channels = 10, .kernel_size = .{ 5, 5 }, .tensor_opts = options });
        self.conv2 = Conv2D.init(.{ .in_channels = 10, .out_channels = 20, .kernel_size = .{ 5, 3 }, .tensor_opts = options });
        self.dropout = Dropout.init(0.5);
        self.fc1 = Linear.init(.{ .in_features = 400, .out_features = 50, .tensor_opts = options });
        self.fc2 = Linear.init(.{ .in_features = 50, .out_features = 10, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        self.conv1.reset();
        self.conv2.reset();
        self.dropout.reset();
        self.fc1.reset();
        self.fc2.reset();
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("conv2", self.conv2);
        _ = self.base_module.registerModule("dropout", self.dropout);
        _ = self.base_module.registerModule("fc1", self.fc1);
        _ = self.base_module.registerModule("fc2", self.fc2);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.conv1.deinit();
        self.conv2.deinit();
        self.dropout.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const x = input.reshape(&.{ -1, 1, 28, 28 });
        var y = self.conv1.forward(&x).maxPool2d(&.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false).relu();
        y = self.conv2.forward(&y);
        y = self.dropout.forward(&y);
        y = y.maxPool2d(&.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false).relu();
        y = y.view(&.{ -1, 400 });
        y = self.fc1.forward(&y).relu();
        y = self.dropout.forward(&y);
        y = self.fc2.forward(&y);
        return y;
    }
};

pub fn main() !void {
    torch.Utils.manualSeed(1);
    const device_type = torch.Device.cudaIfAvailable();
    if (device_type.isCuda()) {
        std.debug.print("CUDA available! Training on GPU.\n", .{});
    } else {
        std.debug.print("Training on CPU.\n", .{});
    }
    const options = if (device_type.isCuda()) torch.FLOAT_CUDA else torch.FLOAT_CPU;

    const model = Net.init(options);
    const ds = try loadDir();
    std.debug.print("Loaded data.\n", .{});
    std.debug.print("Train images size: {any}\n", .{ds.train_images.size()});
    std.debug.print("Train labels size: {any}\n", .{ds.train_labels.size()});
    std.debug.print("Test images size: {any}\n", .{ds.test_images.size()});
    std.debug.print("Test labels size: {any}\n", .{ds.test_labels.size()});

    var opt = torch.COptimizer.adam(1e-3, 0.99, 0.999, 1e-5, 1e-5, false);
    opt.addParameters(model.base_module.parameters(true), 0);

    for (1..50_000) |step| {
        var mem_guard = torch.MemoryGuard.init("mnist");
        defer mem_guard.deinit();

        const batch_indices = Tensor.randintLow(0, 50_000, &.{TRAIN_BATCH_SIZE}, torch.INT64_CUDA);
        const batch_images = ds.train_images.indexSelect(0, &batch_indices);
        const batch_labels = ds.train_labels.indexSelect(0, &batch_indices);
        opt.zeroGrad();
        var loss = model.forward(&batch_images).crossEntropyForLogits(&batch_labels);
        loss.backward();
        opt.step();

        // Evaluate the model
        var guard = torch.NoGradGuard.init();
        defer guard.deinit();
        const test_accuracy = model.forward(&ds.test_images).accuracyForLogits(&ds.test_labels);
        std.debug.print("Iteration: {d}, Loss: {d}, Test accuracy: {d}\r", .{ step, loss.toFloat(), test_accuracy.toFloat() });
    }
}
