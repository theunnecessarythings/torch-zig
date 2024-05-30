const torch = @import("torch");
const Tensor = torch.Tensor;
const std = @import("std");
const NoGradGuard = torch.NoGradGuard;
const Module = @import("module.zig").Module;
const ModuleGen = @import("module.zig").ModuleGen;
const nn_init = @import("init.zig");

pub const EmbeddingOptions = struct {
    num_embeddings: i64,
    embedding_dim: i64,
    padding_idx: ?i64 = null,
    max_norm: ?f64 = null,
    norm_type: f64 = 2.0,
    scale_grad_by_freq: bool = false,
    sparse: bool = false,
    _weight: ?Tensor = null,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const EmbeddingFromPretrainedOptions = struct {
    freeze: bool = true,
    padding_idx: ?i64 = null,
    max_norm: ?f64 = null,
    norm_type: f64 = 2.0,
    scale_grad_by_freq: bool = false,
    sparse: bool = false,
};

pub const Embedding = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,

    weight: Tensor = undefined,
    options: EmbeddingOptions = undefined,

    const Self = @This();

    pub fn init(options: EmbeddingOptions) Self {
        var self = Self{
            .options = options,
        };
        Module.initFields(self);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        Module.deinitFields(self);
        self.weight.free();
    }

    pub fn fromPretrained(embeddings: *const Tensor, options: EmbeddingFromPretrainedOptions) Self {
        std.debug.assert(embeddings.dim() == 2);
        const size = embeddings.size();
        const opts = EmbeddingOptions{
            .num_embeddings = size[0],
            .embedding_dim = size[1],
            .padding_idx = options.padding_idx,
            .max_norm = options.max_norm,
            .norm_type = options.norm_type,
            .scale_grad_by_freq = options.scale_grad_by_freq,
            .sparse = options.sparse,
            ._weight = embeddings,
        };
        var embedding = Embedding.init(opts);
        embedding.weight.setRequiresGrad(!options.freeze);
        return embedding;
    }

    pub fn reset(self: *Self) void {
        if (self.options.padding_idx) |padding_idx| {
            if (padding_idx > 0) {
                std.debug.assert(padding_idx < self.options.num_embeddings);
            } else if (padding_idx < 0) {
                std.debug.assert(padding_idx >= -self.options.num_embeddings);
                self.options.padding_idx = self.options.num_embeddings + padding_idx;
            }
        }

        if (!self.options._weight) {
            self.weight = Module.registerParameter(self, "weight", Tensor.empty(.{ self.options.num_embeddings, self.options.embedding_dim }, self.options.tensor_opts), true);
            self.resetParameters();
        } else {
            const size = self.options._weight.?.size();
            std.debug.assert(size[0] == self.options.num_embeddings);
            std.debug.assert(size[1] == self.options.embedding_dim);
            self.weight = Module.registerParameter(self, "weight", self.options._weight.?, true);
        }
    }

    pub fn resetParameters(self: *Self) void {
        nn_init.normal_(self.weight, 0.0, 1.0);
        if (self.options.padding_idx) |padding_idx| {
            var guard = NoGradGuard.init();
            defer guard.deinit();
            const idx_tensor = Tensor.fromSlice(i64, &.{padding_idx});
            self.weight.indexFill_(0, idx_tensor, 0.0);
        }
    }

    pub fn forward(self: *const Self, input: *const Tensor) Tensor {
        return input.embedding(self.weight, self.options.padding_idx, self.options.scale_grad_by_freq, self.options.sparse);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.write(
            "Embedding(num_embeddings: {}, embedding_dim: {}, padding_idx: {}, max_norm: {}, norm_type: {}, scale_grad_by_freq: {}, sparse: {})",
            .{
                self.options.num_embeddings,
                self.options.embedding_dim,
                self.options.padding_idx,
                self.options.max_norm,
                self.options.norm_type,
                self.options.scale_grad_by_freq,
                self.options.sparse,
            },
        );
    }
};
