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
const LayerNorm = torch.norm.LayerNorm;
const Sequential = module.Sequential;

fn mlpBlock(in_dim: i64, mlp_dim: i64, dropout: f32, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .add(Linear.init(.{ .in_features = in_dim, .out_features = mlp_dim, .tensor_opts = options }))
        .add(Functional(Tensor.gelu, .{"none"}).init())
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = mlp_dim, .out_features = in_dim, .tensor_opts = options }))
        .add(Dropout.init(dropout));
}

const Attention = struct {
    base_module: *Module = undefined,
    in_proj_bias: Tensor = undefined,
    in_proj_weight: Tensor = undefined,
    out_proj: *Linear = undefined,
    scale: f64,
    num_heads: i64,

    const Self = @This();

    pub fn init(dim: i64, num_heads: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = .{ .num_heads = num_heads, .scale = 1.0 / @sqrt(@as(f64, @floatFromInt(@divFloor(dim, num_heads)))) };
        self.base_module = Module.init(self);
        self.in_proj_weight = self.base_module.registerParameter("in_proj_weight", Tensor.randn(&.{ dim * 3, dim }, options), true);
        self.in_proj_bias = self.base_module.registerParameter("in_proj_bias", Tensor.zeros(&.{dim * 3}, options), true);
        self.out_proj = Linear.init(.{ .in_features = dim, .out_features = dim, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("out_proj", self.out_proj);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        const b, const n, const c = x.sizeDims(3);
        const qkv = (x.matmul(&self.in_proj_weight.transpose(0, 1)).add(&self.in_proj_bias))
            .reshape(&.{ b, n, 3, self.num_heads, @divFloor(c, self.num_heads) })
            .permute(&.{ 2, 0, 3, 1, 4 });
        const q = qkv.get(0).mulScalar(Scalar.float(self.scale));
        const k = qkv.get(1);
        const v = qkv.get(2);
        const attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, torch.Kind.Float);
        const out = attn.matmul(&v)
            .transpose(1, 2)
            .reshape(&.{ b, n, c });
        return self.out_proj.forward(&out);
    }
};

const EncoderBlock = struct {
    base_module: *Module = undefined,
    attention: *Attention = undefined,
    mlp: *Sequential = undefined,
    ln1: *LayerNorm = undefined,
    ln2: *LayerNorm = undefined,
    dropout: f64,

    const Self = @This();

    pub fn init(num_heads: i64, hidden_dim: i64, mlp_dim: i64, dropout: f32, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{ .dropout = dropout };
        self.base_module = Module.init(self);
        self.ln1 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{hidden_dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.attention = Attention.init(hidden_dim, num_heads, options);
        self.ln2 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{hidden_dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.mlp = mlpBlock(hidden_dim, mlp_dim, dropout, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("self_attention", self.attention);
        _ = self.base_module.registerModule("mlp", self.mlp);
        _ = self.base_module.registerModule("ln_1", self.ln1);
        _ = self.base_module.registerModule("ln_2", self.ln2);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = self.ln1.forward(x);
        y = self.attention.forward(&y).dropout(self.dropout, self.base_module.isTraining())
            .add(x);
        var y_ = self.ln2.forward(&y);
        y_ = self.mlp.forward(&y_);
        return y.add(&y_);
    }
};

const Encoder = struct {
    base_module: *Module = undefined,
    pos_embedding: Tensor = undefined,
    layers: *Sequential = undefined,
    ln: *LayerNorm = undefined,
    dropout: f64,

    const Self = @This();

    pub fn init(seq_length: i64, num_layers: usize, num_heads: i64, hidden_dim: i64, mlp_dim: i64, dropout: f32, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{ .dropout = dropout };
        self.base_module = Module.init(self);
        self.pos_embedding = self.base_module.registerParameter("pos_embedding", Tensor.randn(&.{ 1, seq_length, hidden_dim }, options), true);
        self.layers = Sequential.init(options);

        for (0..num_layers) |i| {
            const name = std.fmt.allocPrint(torch.global_allocator, "encoder_layer_{d}", .{i}) catch err(.AllocFailed);
            self.layers = self.layers.addWithName(name, EncoderBlock.init(num_heads, hidden_dim, mlp_dim, dropout, options));
        }
        self.ln = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{hidden_dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("ln", self.ln);
        _ = self.base_module.registerModule("layers", self.layers);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = x.add(&self.pos_embedding);
        y = y.dropout(self.dropout, self.base_module.isTraining());
        y = self.layers.forward(&y);
        return self.ln.forward(&y);
    }
};

const ViT = struct {
    base_module: *Module = undefined,
    conv_proj: *Conv2D = undefined,
    class_token: Tensor = undefined,
    encoder: *Encoder = undefined,
    head: *Linear = undefined,
    patch_size: i64,
    hidden_dim: i64,

    const Self = @This();

    pub fn init(image_size: i64, patch_size: i64, num_layers: usize, num_heads: i64, hidden_dim: i64, mlp_dim: i64, dropout: f32, num_classes: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{ .patch_size = patch_size, .hidden_dim = hidden_dim };
        self.base_module = Module.init(self);
        self.conv_proj = Conv2D.init(.{ .in_channels = 3, .out_channels = hidden_dim, .kernel_size = .{ patch_size, patch_size }, .stride = .{ patch_size, patch_size }, .tensor_opts = options });
        const seq_length = @divFloor(image_size, patch_size) * @divFloor(image_size, patch_size) + 1;
        self.class_token = self.base_module.registerParameter("class_token", Tensor.zeros(&.{ 1, 1, hidden_dim }, options), true);
        self.encoder = Encoder.init(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, options);
        self.head = Linear.init(.{ .in_features = hidden_dim, .out_features = num_classes, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("conv_proj", self.conv_proj);
        _ = self.base_module.registerModule("encoder", self.encoder);
        _ = self.base_module.registerModule("heads.head", self.head);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        const b, _, const h, const w = x.sizeDims(4);
        const n_h, const n_w = .{ @divFloor(h, self.patch_size), @divFloor(w, self.patch_size) };
        var y = self.conv_proj.forward(x)
            .reshape(&.{ b, self.hidden_dim, n_h * n_w })
            .permute(&.{ 0, 2, 1 });
        const class_token = self.class_token.expand(&.{ b, -1, -1 }, true);
        var temp = [_]*const Tensor{ &class_token, &y };
        y = Tensor.cat(&temp, 1);
        y = self.encoder.forward(&y).select(1, 0);
        return self.head.forward(&y);
    }
};

pub fn vitB16(num_classes: i64, options: TensorOptions) *ViT {
    return ViT.init(224, 16, 12, 12, 768, 3072, 0.0, num_classes, options);
}

pub fn vitB32(num_classes: i64, options: TensorOptions) *ViT {
    return ViT.init(224, 32, 12, 12, 768, 3072, 0.0, num_classes, options);
}

pub fn vitL16(num_classes: i64, options: TensorOptions) *ViT {
    return ViT.init(224, 16, 24, 16, 1024, 4096, 0.0, num_classes, options);
}

pub fn vitL32(num_classes: i64, options: TensorOptions) *ViT {
    return ViT.init(224, 32, 24, 16, 1024, 4096, 0.0, num_classes, options);
}

pub fn vitH14(num_classes: i64, options: TensorOptions) *ViT {
    return ViT.init(224, 14, 32, 16, 1280, 5120, 0.0, num_classes, options);
}
