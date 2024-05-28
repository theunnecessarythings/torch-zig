const torch = @import("torch");
const Tensor = torch.Tensor;
const Scalar = torch.Scalar;
const std = @import("std");
const Module = @import("module.zig").Module;
const ModuleGen = @import("module.zig").ModuleGen;
const nn_init = @import("init.zig");
const linear = @import("linear.zig");

pub const Activation = enum {
    ReLU,
    GELU,
    Function,
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
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const TransformerOptions = struct {
    d_model: i64 = 512,
    nhead: i64 = 8,
    num_encoder_layers: i64 = 6,
    num_decoder_layers: i64 = 6,
    dim_feedforward: i64 = 2048,
    dropout: f64 = 0.1,
    activation: Activation = .ReLU,
    custom_encoder: ?*Module,
    custom_decoder: ?*Module,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const TransformerEncoderLayerOptions = struct {
    d_model: i64,
    nhead: i64,
    dim_feedforward: i64 = 2048,
    dropout: f64 = 0.1,
    activation: Activation = .ReLU,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const TransformerDecoderLayerOptions = struct {
    d_model: i64,
    nhead: i64,
    dim_feedforward: i64 = 2048,
    dropout: f64 = 0.1,
    activation: Activation = .ReLU,
    tensor_opts: torch.TensorOptions = torch.FLOAT_CPU,
};

pub const MultiheadAttention = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,

    _qkv_same_embed_dim: bool = false,
    in_proj_weight: Tensor = undefined,
    in_proj_bias: Tensor = undefined,
    bias_k: Tensor = undefined,
    bias_v: Tensor = undefined,
    out_proj: ?linear.Linear = null,
    q_proj_weight: Tensor = undefined,
    k_proj_weight: Tensor = undefined,
    v_proj_weight: Tensor = undefined,
    head_dim: i64,

    options: MultiheadAttentionOptions = undefined,

    const Self = @This();
    const M = ModuleGen(Self);
    pub usingnamespace M;

    pub fn init(options: MultiheadAttentionOptions) Self {
        var self = Self{
            .options = options,
        };
        self.initFields();
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.deinitFields();
        self.in_proj_weight.free();
        self.in_proj_bias.free();
        self.bias_k.free();
        self.bias_v.free();
        if (self.out_proj != null) {
            self.out_proj.?.deinit();
        }
        self.q_proj_weight.free();
        self.k_proj_weight.free();
        self.v_proj_weight.free();
    }

    pub fn reset(self: *Self) void {
        self._qkv_same_embed_dim = self.options.kdim == self.options.embed_dim and self.options.vdim == self.options.embed_dim;
        self.head_dim = self.options.embed_dim / self.options.num_heads;
        std.debug.assert(self.head_dim * self.options.num_heads == self.options.embed_dim);
        if (!self._qkv_same_embed_dim) {
            self.q_proj_weight = self.registerParameter("q_proj_weight", Tensor.empty(&.{ self.options.embed_dim, self.options.embed_dim }), true);
            self.k_proj_weight = self.registerParameter("k_proj_weight", Tensor.empty(&.{ self.options.embed_dim, self.options.kdim }), true);
            self.v_proj_weight = self.registerParameter("v_proj_weight", Tensor.empty(&.{ self.options.embed_dim, self.options.vdim }), true);
        } else {
            self.in_proj_weight = self.registerParameter("in_proj_weight", Tensor.empty(&.{ 3 * self.options.embed_dim, self.options.embed_dim }), true);
        }
        if (self.options.bias) {
            self.in_proj_bias = self.registerParameter("in_proj_bias", Tensor.empty(&.{3 * self.options.embed_dim}), true);
        }
        self.out_proj = linear.Linear.init(.{
            .in_features = self.options.embed_dim,
            .out_features = self.options.embed_dim,
        });
        _ = self.registerModule("out_proj", Module.init(self.out_proj));
        if (self.options.add_bias_kv) {
            self.bias_k = self.registerParameter("bias_k", Tensor.empty(&.{ 1, 1, self.options.embed_dim }), true);
            self.bias_v = self.registerParameter("bias_v", Tensor.empty(&.{ 1, 1, self.options.embed_dim }), true);
        }
        self.resetParameters();
    }

    pub fn resetParameters(self: *Self) void {
        if (!self._qkv_same_embed_dim) {
            nn_init.xavierUniform_(self.q_proj_weight);
            nn_init.xavierUniform_(self.k_proj_weight);
            nn_init.xavierUniform_(self.v_proj_weight);
        } else {
            nn_init.xavierUniform_(self.in_proj_weight);
        }
        if (self.options.bias) {
            nn_init.constant_(self.in_proj_bias, 0.0);
        }
        if (self.options.add_bias_kv) {
            nn_init.xavierNormal_(self.bias_k);
            nn_init.xavierNormal_(self.bias_v);
        }
    }

    pub fn forward(self: *Self, query: *const Tensor, key: *const Tensor, value: *const Tensor, key_padding_mask: *const Tensor, need_weights: bool, attn_mask: *const Tensor, average_attn_weights: bool) Tensor {
        if (!self._qkv_same_embed_dim) {
            return Tensor.internalNativeMultiHeadAttention(
                query,
                key,
                value,
                self.options.embed_dim,
                self.options.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.?.weight,
                self.out_proj.?.bias,
                key_padding_mask,
                need_weights,
                average_attn_weights,
            );
        }
        // TODO: Implement the rest of the function
    }
};

pub const TransformerEncoderLayer = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,

    self_attn: MultiheadAttention = undefined,
    linear1: linear.Linear = undefined,
    dropout: torch.Dropout = undefined,
    linear2: linear.Linear = undefined,
    norm1: torch.LayerNorm = undefined,
    norm2: torch.LayerNorm = undefined,
    dropout1: torch.Dropout = undefined,
    dropout2: torch.Dropout = undefined,
    options: TransformerEncoderLayerOptions = undefined,

    const Self = @This();
    const M = ModuleGen(Self);
    pub usingnamespace M;

    pub fn init(options: TransformerEncoderLayerOptions) Self {
        var self = Self{
            .options = options,
        };
        self.initFields();
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.deinitFields();
        self.self_attn.deinit();
        self.linear1.deinit();
        self.dropout.deinit();
        self.linear2.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
        self.dropout1.deinit();
        self.dropout2.deinit();
    }

    pub fn reset(self: *Self) void {
        self.self_attn = MultiheadAttention.init(.{
            .d_model = self.options.d_model,
            .nhead = self.options.nhead,
            .dropout = self.options.dropout,
        });
        _ = self.registerModule("self_attn", Module.init(self.self_attn));
        self.linear1 = linear.Linear.init(.{
            .in_features = self.options.d_model,
            .out_features = self.options.dim_feedforward,
        });
        _ = self.registerModule("linear1", Module.init(self.linear1));
        self.dropout = torch.Dropout.init(.{
            .p = self.options.dropout,
        });
        _ = self.registerModule("dropout", Module.init(self.dropout));
        self.linear2 = linear.Linear.init(.{
            .in_features = self.options.dim_feedforward,
            .out_features = self.options.d_model,
        });
        _ = self.registerModule("linear2", Module.init(self.linear2));
        self.norm1 = torch.LayerNorm.init(.{
            .normalized_shape = self.options.d_model,
        });
        _ = self.registerModule("norm1", Module.init(self.norm1));
        self.norm2 = torch.LayerNorm.init(.{
            .normalized_shape = self.options.d_model,
        });
        _ = self.registerModule("norm2", Module.init(self.norm2));
        self.dropout1 = torch.Dropout.init(.{
            .p = self.options.dropout,
        });
        _ = self.registerModule("dropout1", Module.init(self.dropout1));
        self.dropout2 = torch.Dropout.init(.{
            .p = self.options.dropout,
        });
        _ = self.registerModule("dropout2", Module.init(self.dropout2));
    }

    pub fn resetParameters(self: *Self) void {
        self.self_attn.resetParameters();
        self.linear1.resetParameters();
        self.linear2.resetParameters();
        self.norm1.resetParameters();
        self.norm2.resetParameters();
    }

    pub fn forward(self: *Self, src: *const Tensor, src_mask: *const Tensor, src_key_padding_mask: *const Tensor) Tensor {
        var src2 = self.self_attn.forward(src, src, src, src_key_padding_mask, true, src_mask);
        const ret = self.norm1.forward(&src.add(self.dropout1.forward(&src2)));

        switch (self.options.activation) {
            .GELU => {
                src2 = self.linear2.forward(&self.dropout.forward(&Tensor.gelu(&self.linear1.forward(&ret), "none")));
            },
            .ReLU => {
                src2 = self.linear2.forward(&self.dropout.forward(&Tensor.relu(&self.linear1.forward(&ret))));
            },
            else => {
                @panic("Function activation not implemented");
            },
        }

        return self.norm2.forward(&ret.add(self.dropout2.forward(&src2)));
    }
};

pub const TransformerDecoderLayer = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,

    self_attn: MultiheadAttention = undefined,
    dropout1: torch.Dropout = undefined,
    norm1: torch.LayerNorm = undefined,
    multihead_attn: MultiheadAttention = undefined,
    dropout2: torch.Dropout = undefined,
    norm2: torch.LayerNorm = undefined,
    linear1: linear.Linear = undefined,
    dropout: torch.Dropout = undefined,
    linear2: linear.Linear = undefined,
    norm3: torch.LayerNorm = undefined,
    dropout3: torch.Dropout = undefined,
    options: TransformerDecoderLayerOptions = undefined,

    const Self = @This();
    const M = ModuleGen(Self);
    pub usingnamespace M;

    pub fn init(options: TransformerDecoderLayerOptions) Self {
        var self = Self{
            .options = options,
        };
        self.initFields();
        self.reset();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.deinitFields();
        self.weight.free();
        self.bias.free();
    }
};
