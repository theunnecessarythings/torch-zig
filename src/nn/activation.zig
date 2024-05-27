const torch = @import("torch");
const Tensor = @import("torch").Tensor;
const std = @import("std");
const Module = @import("module.zig").Module;
const ModuleGen = @import("module.zig").ModuleGen;

pub const ELUOptions = struct {
    alpha: f64 = 1.0,
    inplace: bool = false,
};

pub const SELUOptions = struct {
    inplace: bool = false,
};

pub const GLUOptions = struct {
    dim: i64 = -1,
};

pub const GELOUOptions = struct {
    approximate: []const u8 = "none",
};

pub const HardShrinkOptions = struct {
    lambda: f64 = 0.5,
};

pub const HardtanhOptions = struct {
    min_val: f64 = -1.0,
    max_val: f64 = 1.0,
    inplace: bool = false,
};

pub const LeakyReLUOptions = struct {
    negative_slope: f64 = 1e-2,
    inplace: bool = false,
};

pub const SoftmaxOptions = struct {
    dim: i64 = -1,
};

pub const SoftminOptions = struct {
    dim: i64 = -1,
};

pub const LogSoftmaxOptions = struct {
    dim: i64 = -1,
};

pub const PReLUOptions = struct {
    num_parameters: i64 = 1,
    init: f64 = 0.25,
};

pub const ReLUOptions = struct {
    inplace: bool = false,
};

pub const ReLU6Options = struct {
    inplace: bool = false,
};

pub const RReLUOptions = struct {
    lower: f64 = 1.0 / 8.0,
    upper: f64 = 1.0 / 3.0,
    inplace: bool = false,
};

pub const CELUOptions = struct {
    alpha: f64 = 1.0,
    inplace: bool = false,
};

pub const SoftplusOptions = struct {
    beta: f64 = 1.0,
    threshold: f64 = 20.0,
};

pub const SoftshrinkOptions = struct {
    lambda: f64 = 0.5,
};

pub const ThresholdOptions = struct {
    threshold: f64 = 1e-6,
    value: f64 = 0.0,
    inplace: bool = false,
};

pub const GumbelSoftmaxOptions = struct {
    tau: f64 = 1.0,
    hard: bool = false,
    dim: i64 = -1,
};

pub const MultiheadAttentionOptions = struct {
    embed_dim: i64,
    num_heads: i64,
    dropout: f64 = 0.0,
    bias: bool = true,
    add_bias_kv: bool = false,
    add_zero_attn: bool = false,
    kdim: i64 = 0,
    vdim: i64 = 0,
};

pub const ELU = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,

    bias: ?Tensor = null,
    weight: Tensor = undefined,
    options: ELUOptions = ELUOptions{},

    const Self = @This();
    const M = ModuleGen(Self);
    pub usingnamespace M;
};
