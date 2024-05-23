const std = @import("std");
const c = @cImport({
    @cInclude("stddef.h");
    @cInclude("stdbool.h");
    @cInclude("stdlib.h");
    @cInclude("torch_api.h");
    @cInclude("torch_api_generated.h");
});
// TODO: we don't have the error generating/fallible functions right now, probably need
// // to add them for usecases where fallback is needed instead of panic
const Tensor = @import("tensor.zig").Tensor;

// TODO: Do we name the stuff here proper, or follow tch-rs and its conventions?

pub const FLOAT_CPU = TensorOptions{ .kind = Kind.Float, .device = Device.Cpu };
pub const FLOAT_CUDA = TensorOptions{ .kind = Kind.Float, .device = Device.Cuda(0) };
pub const DOUBLE_CPU = TensorOptions{ .kind = Kind.Double, .device = Device.Cpu };
pub const DOUBLE_CUDA = TensorOptions{ .kind = Kind.Double, .device = Device.Cuda(0) };
pub const INT64_CPU = TensorOptions{ .kind = Kind.Int64, .device = Device.Cpu };
pub const INT64_CUDA = TensorOptions{ .kind = Kind.Int64, .device = Device.Cuda(0) };

pub const global_allocator: std.mem.Allocator = std.heap.raw_c_allocator;

pub fn setGlobalAllocator(allocator: std.mem.Allocator) void {
    global_allocator = allocator;
}

pub const Device = union(enum) {
    Cuda: usize,
    Cpu,
    Mps,
    Vulkan,

    pub fn cInt(self: Device) c_int {
        return switch (self) {
            .Cpu => -1,
            .Cuda => |device_index| @intCast(device_index),
            .Mps => -2,
            .Vulkan => -3,
        };
    }

    pub fn fromCInt(v: c_int) !Device {
        return switch (v) {
            -1 => Device.Cpu,
            -2 => Device.Mps,
            -3 => Device.Vulkan,
            _ => if (v >= 0) {
                Device.Cuda(@intCast(v));
            } else {
                std.debug.err("Invalid device index: {}\n", .{v});
                return error.InvalidDeviceIndex;
            },
        };
    }

    pub fn cudaIfAvailable() Device {
        if (Cuda.isAvailable()) {
            return Device.Cuda(0);
        } else {
            return Device.Cpu;
        }
    }

    pub fn isCuda(self: Device) bool {
        return switch (self) {
            .Cuda => true,
            _ => false,
        };
    }
};

pub const Cuda = enum {
    pub fn deviceCount() i64 {
        return c.atc_cuda_device_count();
    }

    pub fn isAvailable() bool {
        return c.atc_cuda_is_available();
    }

    pub fn cudnn_is_available() bool {
        return c.atc_cudnn_is_available();
    }

    pub fn manualSeed(seed: u64) void {
        c.atc_manual_seed(seed);
    }

    pub fn manualSeedAll(seed: u64) void {
        c.atc_manual_seed_all(seed);
    }

    pub fn synchronize(device_index: i64) void {
        c.atc_synchronize(device_index);
    }

    pub fn userEnabledCudnn() bool {
        return c.atc_user_enabled_cudnn();
    }

    pub fn setUserEnabledCudnn(b: bool) void {
        c.atc_set_user_enabled_cudnn(b);
    }

    pub fn cudnnSetBenchmark(b: bool) void {
        c.atc_set_benchmark_cudnn(b);
    }
};

pub const Kind = enum {
    UInt8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,

    pub fn cInt(self: Kind) c_int {
        return switch (self) {
            .UInt8 => 0,
            .Int8 => 1,
            .Int16 => 2,
            .Int => 3,
            .Int64 => 4,
            .Half => 5,
            .Float => 6,
            .Double => 7,
            .ComplexHalf => 8,
            .ComplexFloat => 9,
            .ComplexDouble => 10,
            .Bool => 11,
            .QInt8 => 12,
            .QUInt8 => 13,
            .QInt32 => 14,
            .BFloat16 => 15,
        };
    }

    pub fn fromCInt(v: c_int) !Kind {
        return switch (v) {
            0 => Kind.UInt8,
            1 => Kind.Int8,
            2 => Kind.Int16,
            3 => Kind.Int,
            4 => Kind.Int64,
            5 => Kind.Half,
            6 => Kind.Float,
            7 => Kind.Double,
            8 => Kind.ComplexHalf,
            9 => Kind.ComplexFloat,
            10 => Kind.ComplexDouble,
            11 => Kind.Bool,
            12 => Kind.QInt8,
            13 => Kind.QUInt8,
            14 => Kind.QInt32,
            15 => Kind.BFloat16,
            _ => {
                std.debug.err("Invalid kind: {}\n", .{v});
                return error.InvalidKind;
            },
        };
    }

    pub fn eltSizeInBytes(self: Kind) usize {
        return switch (self) {
            .UInt8 => 1,
            .Int8 => 1,
            .Int16 => 2,
            .Int => 4,
            .Int64 => 8,
            .Half => 2,
            .Float => 4,
            .Double => 8,
            .ComplexHalf => 4,
            .ComplexFloat => 8,
            .ComplexDouble => 16,
            .Bool => 1,
            .QInt8 => 1,
            .QUInt8 => 1,
            .QInt32 => 4,
            .BFloat16 => 2,
        };
    }
};

pub const TensorOptions = struct {
    kind: Kind,
    device: Device,
};

pub const Layout = enum {
    Strided,
    Sparse,
    SparseCsr,
    Mkldnn,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    NumOptions,

    pub fn toI8(self: Layout) i8 {
        return switch (self) {
            .Strided => 0,
            .Sparse => 1,
            .SparseCsr => 2,
            .Mkldnn => 3,
            .SparseCsc => 4,
            .SparseBsr => 5,
            .SparseBsc => 6,
            .NumOptions => 7,
        };
    }
};

pub const Scalar = struct {
    c_scalar: c.scalar,

    pub fn int(v: i64) Scalar {
        return Scalar{ .c_scalar = c.ats_int(v) };
    }

    pub fn float(v: f64) Scalar {
        return Scalar{ .c_scalar = c.ats_float(v) };
    }

    pub fn toInt(self: Scalar) i64 {
        const ret = c.ats_to_int(self.c_scalar);
        readAndCleanError();
        return ret;
    }

    pub fn toFloat(self: Scalar) f64 {
        const ret = c.ats_to_float(self.c_scalar);
        readAndCleanError();
        return ret;
    }

    pub fn toString(self: Scalar) [*c]u8 {
        const ret = c.ats_to_string(self.c_scalar);
        readAndCleanError();
        return ret;
    }

    pub fn free(self: Scalar) void {
        c.ats_free(self.c_scalar);
        readAndCleanError();
    }
};

pub const TorchError = error{
    Convert,
    FileFormat,
    TensorNameNotFound,
    Io,
    Kind,
    MissingImage,
    Null,
    ParseInt,
    Shape,
    UnknownKind,
    Torch,
    NdArray,
    SafeTensorError,
};

// TODO: Do i need to this all the time? How much overhead does it have? Maybe on flag?
pub fn readAndCleanError() void {
    const err = c.get_and_reset_last_err() orelse return;
    defer c.free(err);
    std.debug.panic("Torch Error: {s}\n", .{err});
}

// Namespace for the torch::utils functions
pub const Utils = struct {
    pub fn manualSeed(seed: u64) void {
        c.atc_manual_seed(seed);
        readAndCleanError();
    }

    pub fn getNumInteropThreads() i32 {
        const ret = c.at_get_num_interop_threads();
        readAndCleanError();
        return ret;
    }

    pub fn getNumThreads() i32 {
        const ret = c.at_get_num_threads();
        readAndCleanError();
        return ret;
    }

    pub fn setNumInteropThreads(num_threads: i32) void {
        c.at_set_num_interop_threads(num_threads);
        readAndCleanError();
    }

    pub fn setNumThreads(num_threads: i32) void {
        c.at_set_num_threads(num_threads);
        readAndCleanError();
    }

    pub fn hasOpenmp() bool {
        const ret = c.at_context_has_openmp();
        readAndCleanError();
        return ret;
    }

    pub fn hasMkl() bool {
        const ret = c.at_context_has_mkl();
        readAndCleanError();
        return ret;
    }

    pub fn hasMkldnn() bool {
        const ret = c.at_context_has_mkldnn();
        readAndCleanError();
        return ret;
    }

    pub fn hasLapack() bool {
        const ret = c.at_context_has_lapack();
        readAndCleanError();
        return ret;
    }

    pub fn hasMagma() bool {
        const ret = c.at_context_has_magma();
        readAndCleanError();
        return ret;
    }

    pub fn hasCuda() bool {
        const ret = c.at_context_has_cuda();
        readAndCleanError();
        return ret;
    }

    pub fn hasCudnn() bool {
        const ret = c.at_context_has_cudnn();
        readAndCleanError();
        return ret;
    }

    pub fn hasCudart() bool {
        const ret = c.at_context_has_cudart();
        readAndCleanError();
        return ret;
    }

    pub fn hasCuSolver() bool {
        const ret = c.at_context_has_cusolver();
        readAndCleanError();
        return ret;
    }

    pub fn hasIpu() bool {
        const ret = c.at_context_has_ipu();
        readAndCleanError();
        return ret;
    }

    pub fn hasXla() bool {
        const ret = c.at_context_has_xla();
        readAndCleanError();
        return ret;
    }

    pub fn hasLazy() bool {
        const ret = c.at_context_has_lazy();
        readAndCleanError();
        return ret;
    }

    pub fn hasMps() bool {
        const ret = c.at_context_has_mps();
        readAndCleanError();
        return ret;
    }

    pub fn hasOrt() bool {
        const ret = c.at_context_has_ort();
        readAndCleanError();
        return ret;
    }

    pub fn hasVulkan() bool {
        const ret = c.atg_is_vulkan_available();
        readAndCleanError();
        return ret;
    }

    pub fn versionCudnn() i64 {
        const ret = c.at_context_version_cudnn();
        readAndCleanError();
        return ret;
    }

    pub fn versionCudart() i64 {
        const ret = c.at_context_version_cudart();
        readAndCleanError();
        return ret;
    }
};

pub const QEngine = enum {
    NoQEngine,
    FBGEMM,
    Qnnpack,

    pub fn toCInt(self: QEngine) c_int {
        return switch (self) {
            .NoQEngine => 0,
            .FBGEMM => 1,
            .Qnnpack => 2,
        };
    }
    pub fn set(self: QEngine) void {
        c.at_set_qengine(self.toCInt());
        readAndCleanError();
    }
};

// Namespace for the torch::image functions
pub const Image = struct {
    pub fn loadHWC(path: []const u8) !Tensor {
        const path_ = try global_allocator.dupeZ(u8, path);
        defer global_allocator.free(path_);
        const ret = c.at_load_image(path_);
        readAndCleanError();
        return Tensor{ .c_tensor = ret };
    }

    pub fn loadHWCFromMemory(data: []const u8) !Tensor {
        const data_ = try global_allocator.dupe(u8, data);
        defer global_allocator.free(data_);
        const ret = c.at_load_image_from_memory(data_, data.len);
        readAndCleanError();
        return Tensor{ .c_tensor = ret };
    }

    pub fn saveHWC(tensor: Tensor, path: []const u8) !void {
        const path_ = try global_allocator.dupeZ(u8, path);
        defer global_allocator.free(path_);
        c.at_save_image(tensor.c_tensor, path_);
        readAndCleanError();
    }

    pub fn resizeHWC(tensor: Tensor, width: i64, height: i64) Tensor {
        const ret = c.at_resize_image(tensor.c_tensor, width, height);
        readAndCleanError();
        return Tensor{ .c_tensor = ret };
    }
};

pub const COptimizer = struct {
    c_optimizer: c.optimizer,

    pub fn adam(lr: f64, beta1: f64, beta2: f64, wd: f64, eps: f64, amsgrad: bool) COptimizer {
        const ret = c.ato_adam(lr, beta1, beta2, wd, eps, amsgrad);
        readAndCleanError();
        return COptimizer{ .c_optimizer = ret };
    }

    pub fn adamw(lr: f64, beta1: f64, beta2: f64, wd: f64, eps: f64, amsgrad: bool) COptimizer {
        const ret = c.ato_adamw(lr, beta1, beta2, wd, eps, amsgrad);
        readAndCleanError();
        return COptimizer{ .c_optimizer = ret };
    }

    pub fn rmsprop(lr: f64, alpha: f64, eps: f64, wd: f64, momentum: f64, centered: bool) COptimizer {
        const ret = c.ato_rmsprop(lr, alpha, eps, wd, momentum, centered);
        readAndCleanError();
        return COptimizer{ .c_optimizer = ret };
    }

    pub fn sgd(lr: f64, momentum: f64, dampening: f64, wd: f64, nesterov: bool) COptimizer {
        const ret = c.ato_sgd(lr, momentum, dampening, wd, nesterov);
        readAndCleanError();
        return COptimizer{ .c_optimizer = ret };
    }

    pub fn addParameters(self: *COptimizer, parameters: *Tensor, group: usize) void {
        c.ato_add_parameters(self.c_optimizer, parameters.c_tensor, group);
        readAndCleanError();
    }

    pub fn setLearningRate(self: *COptimizer, lr: f64) void {
        c.ato_set_learning_rate(self.c_optimizer, lr);
        readAndCleanError();
    }

    pub fn setLearningRateGroup(self: *COptimizer, lr: f64, group: usize) void {
        c.ato_set_learning_rate_group(self.c_optimizer, group, lr);
        readAndCleanError();
    }

    pub fn setMomentum(self: *COptimizer, momentum: f64) void {
        c.ato_set_momentum(self.c_optimizer, momentum);
        readAndCleanError();
    }

    pub fn setMomentumGroup(self: *COptimizer, momentum: f64, group: usize) void {
        c.ato_set_momentum_group(self.c_optimizer, group, momentum);
        readAndCleanError();
    }

    pub fn setWeightDecay(self: *COptimizer, wd: f64) void {
        c.ato_set_weight_decay(self.c_optimizer, wd);
        readAndCleanError();
    }

    pub fn setWeightDecayGroup(self: *COptimizer, wd: f64, group: usize) void {
        c.ato_set_weight_decay_group(self.c_optimizer, group, wd);
        readAndCleanError();
    }

    pub fn zeroGrad(self: *COptimizer) void {
        c.ato_zero_grad(self.c_optimizer);
        readAndCleanError();
    }

    pub fn step(self: *COptimizer) void {
        c.ato_step(self.c_optimizer);
        readAndCleanError();
    }

    pub fn free(self: COptimizer) void {
        c.ato_free(self.c_optimizer);
        readAndCleanError();
    }
};
