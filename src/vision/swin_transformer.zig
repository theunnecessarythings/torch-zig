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
const LayerNorm = torch.norm.LayerNorm;
const Sequential = module.Sequential;
const NoGradGuard = torch.NoGradGuard;
const err = torch.utils.err;

const StochasticDepthKind = enum { Row, Batch };

pub const StochasticDepth = struct {
    base_module: *Module = undefined,
    prob: f64,
    kind: StochasticDepthKind,
    tensor_opts: TensorOptions,
    const Self = @This();

    pub fn init(prob: f64, kind: StochasticDepthKind, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .prob = prob,
            .kind = kind,
            .tensor_opts = options,
        };
        self.base_module = Module.init(self);
        return self;
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        if (!self.base_module.isTraining() or self.prob == 0.0) {
            return input.shallowClone();
        }
        const survival_rate = 1.0 - self.prob;
        var size = std.ArrayList(i64).init(torch.global_allocator);
        defer size.deinit();
        switch (self.kind) {
            .Row => {
                size.append(input.size()[0]) catch err(.AllocFailed);
                size.appendNTimes(1, input.dim() - 1) catch err(.AllocFailed);
            },
            .Batch => {
                size.appendNTimes(1, input.dim()) catch err(.AllocFailed);
            },
        }
        var noise = Tensor.rand(size.items, self.tensor_opts);
        noise = noise.ge(Scalar.float(survival_rate));
        if (survival_rate > 0.0) {
            return input.mul(&noise).divScalar(Scalar.float(survival_rate));
        }
        return input.mul(&noise);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }
};

fn mlpBlock(in_dim: i64, mlp_dim: i64, dropout: f32, options: TensorOptions) *Sequential {
    return Sequential.init(options)
        .add(Linear.init(.{ .in_features = in_dim, .out_features = mlp_dim, .tensor_opts = options }))
        .add(Functional(Tensor.gelu, .{"none"}).init())
        .add(Dropout.init(dropout))
        .add(Linear.init(.{ .in_features = mlp_dim, .out_features = in_dim, .tensor_opts = options }))
        .add(Dropout.init(dropout));
}

fn patchMergingPad(x: *const Tensor) Tensor {
    const size = x.dim();
    const h, const w = .{ x.size()[size - 3], x.size()[size - 2] };
    const xs = x.pad(&.{ 0, 0, 0, @mod(w, 2), 0, @mod(h, 2) }, "constant", 0.0);

    const h_indices_0 = Tensor.arangeStartStep(Scalar.int(0), Scalar.int(xs.size()[1]), Scalar.int(2), TensorOptions{ .kind = .Int, .device = xs.device() });
    const w_indices_0 = Tensor.arangeStartStep(Scalar.int(0), Scalar.int(xs.size()[2]), Scalar.int(2), TensorOptions{ .kind = .Int, .device = xs.device() });
    const h_indices_1 = Tensor.arangeStartStep(Scalar.int(1), Scalar.int(xs.size()[1]), Scalar.int(2), TensorOptions{ .kind = .Int, .device = xs.device() });
    const w_indices_1 = Tensor.arangeStartStep(Scalar.int(1), Scalar.int(xs.size()[2]), Scalar.int(2), TensorOptions{ .kind = .Int, .device = xs.device() });

    const x0 = xs.indexSelect(1, &h_indices_0)
        .indexSelect(2, &w_indices_0);
    const x1 = xs.indexSelect(1, &h_indices_1)
        .indexSelect(2, &w_indices_0);
    const x2 = xs.indexSelect(1, &h_indices_0)
        .indexSelect(2, &w_indices_1);
    const x3 = xs.indexSelect(1, &h_indices_1)
        .indexSelect(2, &w_indices_1);
    var t = [_]*const Tensor{ &x0, &x1, &x2, &x3 };
    return Tensor.cat(&t, -1);
}

const PatchMerging = struct {
    base_module: *Module = undefined,
    reduction: *Linear = undefined,
    norm: *LayerNorm = undefined,

    const Self = @This();

    pub fn init(dim: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.reduction = Linear.init(.{ .in_features = 4 * dim, .out_features = 2 * dim, .tensor_opts = options, .bias = false });
        self.norm = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{4 * dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("reduction", self.reduction);
        _ = self.base_module.registerModule("norm", self.norm);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = patchMergingPad(x);
        y = self.norm.forward(&y);
        return self.reduction.forward(&y);
    }
};

const PatchMergingV2 = struct {
    base_module: *Module = undefined,
    reduction: *Linear = undefined,
    norm: *LayerNorm = undefined,

    const Self = @This();
    pub fn init(dim: i64, options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.reduction = Linear.init(.{ .in_features = 4 * dim, .out_features = 2 * dim, .tensor_opts = options, .bias = false });
        self.norm = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{2 * dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("reduction", self.reduction);
        _ = self.base_module.registerModule("norm", self.norm);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var y = patchMergingPad(x);
        y = self.reduction.forward(&y);
        return self.norm.forward(&y);
    }
};

fn normalize(x: *const Tensor, p: f64, dim: i64, eps: f64) Tensor {
    const norm = x.normScalaroptDim(Scalar.float(p), &.{dim}, true)
        .clampMin(Scalar.float(eps))
        .expandAs(x);
    return x.div(&norm);
}

fn shiftedWindowAttention(
    x: *const Tensor,
    qkv_weight: *const Tensor,
    proj_weight: *const Tensor,
    relative_position_bias: *const Tensor,
    window_size: [2]i64,
    num_heads: i64,
    shift_size: []i64,
    attention_dropout: f64,
    dropout: f64,
    qkv_bias: ?*const Tensor,
    proj_bias: ?*const Tensor,
    logit_scale: ?*const Tensor,
    training: bool,
) Tensor {
    const b, const h, const w, const c = x.sizeDims(4);
    const pad_r = @mod((window_size[1] - @mod(w, window_size[1])), window_size[1]);
    const pad_b = @mod((window_size[0] - @mod(h, window_size[0])), window_size[0]);
    var xs = x.pad(&.{ 0, 0, 0, pad_r, 0, pad_b }, "constant", 0.0);
    _, const pad_h, const pad_w, _ = xs.sizeDims(4);

    var shift_sz = shift_size;
    if (window_size[0] >= pad_h) {
        shift_sz[0] = 0;
    }
    if (window_size[1] >= pad_w) {
        shift_sz[1] = 0;
    }

    var shift_size_sum: i64 = 0;
    for (shift_size) |shift| {
        shift_size_sum += shift;
    }
    if (shift_size_sum > 0) {
        xs = xs.roll(&.{ -shift_sz[0], -shift_sz[1] }, &.{ 1, 2 });
    }
    const num_windows = @divFloor(pad_h, window_size[0]) * @divFloor(pad_w, window_size[1]);
    xs = xs.view(&.{
        b,
        @divFloor(pad_h, window_size[0]),
        window_size[0],
        @divFloor(pad_w, window_size[1]),
        window_size[1],
        c,
    });
    xs = xs.permute(&.{ 0, 1, 3, 2, 4, 5 })
        .reshape(&.{ b * num_windows, window_size[0] * window_size[1], c });
    var qkv: Tensor = undefined;
    if (logit_scale != null and qkv_bias != null) {
        var guard = NoGradGuard.init();
        defer guard.deinit();
        const length: i64 = @intCast(@divFloor(qkv_bias.?.numel(), 3));
        var t = qkv_bias.?.narrow(0, length, length);
        _ = t.zero_();
        qkv = xs.linear(qkv_weight, qkv_bias);
    } else {
        qkv = xs.linear(qkv_weight, qkv_bias);
    }
    qkv = qkv.reshape(&.{ xs.size()[0], xs.size()[1], 3, num_heads, @divFloor(c, num_heads) })
        .permute(&.{ 2, 0, 3, 1, 4 });
    const _qkv = qkv.chunk(3, 0);
    var q, var k, var v = .{ _qkv[0], _qkv[1], _qkv[2] };

    var attn: Tensor = undefined;
    if (logit_scale) |scale| {
        attn = normalize(&q, 2.0, -1, 1e-12).matmul(&normalize(&k, 2.0, -1, 1e-12).transpose(-2, -1));
        attn = attn.mul(&scale.clampMax(Scalar.float(@log(100.0)))).exp();
    } else {
        q = q.mulScalar(Scalar.float(std.math.pow(f64, @floatFromInt(@divFloor(c, num_heads)), -0.5)));
        attn = q.matmul(&k.transpose(-2, -1));
    }
    attn = attn.add(relative_position_bias);

    if (shift_size_sum > 0) {
        var attn_mask = xs.newZeros(&.{ pad_h, pad_w }, xs.options());
        const h_slices = [_][2]i64{
            .{ 0, pad_w - window_size[1] },
            .{ pad_h - window_size[1], pad_w - shift_sz[1] },
            .{ pad_h - shift_sz[0], pad_h },
        };
        const w_slices = [_][2]i64{
            .{ 0, pad_w - window_size[1] },
            .{ pad_h - window_size[1], pad_w - shift_sz[1] },
            .{ pad_h - shift_sz[1], pad_w },
        };
        var count: i64 = 0;
        for (h_slices) |h_| {
            for (w_slices) |w_| {
                var t = attn_mask.narrow(0, h_[0], h_[1] - h_[0])
                    .narrow(1, w_[0], w_[1] - w_[0]);
                _ = t.fill_(Scalar.int(count));
                count += 1;
            }
        }
        attn_mask = attn_mask
            .view(&.{ @divFloor(pad_h, window_size[0]), window_size[0], @divFloor(pad_w, window_size[1]), window_size[1] })
            .permute(&.{ 0, 2, 1, 3 })
            .reshape(&.{ num_windows, window_size[0] * window_size[1] });
        attn_mask = attn_mask.unsqueeze(1).sub(&attn_mask.unsqueeze(2));
        attn_mask = attn_mask
            .maskedFill(&attn_mask.ne(Scalar.float(0.0)), Scalar.float(-100.0))
            .maskedFill(&attn_mask.eq(Scalar.float(0.0)), Scalar.float(0.0));
        attn = attn.view(&.{
            @divFloor(xs.size()[0], num_windows),
            num_heads,
            xs.size()[1],
            xs.size()[2],
        });
        attn = attn.add(&attn_mask.unsqueeze(1).unsqueeze(0));
        attn = attn.view(&.{ -1, num_heads, xs.size()[1], xs.size()[1] });
    }

    attn = attn.softmax(-1, .Float).dropout(attention_dropout, training);
    xs = attn.matmul(&v).transpose(1, 2)
        .reshape(&.{ xs.size()[0], xs.size()[1], c })
        .linear(proj_weight, proj_bias)
        .dropout(dropout, training)
        .view(&.{ b, @divFloor(pad_h, window_size[0]), @divFloor(pad_w, window_size[1]), window_size[0], window_size[1], c })
        .permute(&.{ 0, 1, 3, 2, 4, 5 })
        .reshape(&.{ b, pad_h, pad_w, c });

    if (shift_size_sum > 0) {
        xs = xs.roll(&.{ shift_sz[0], shift_sz[1] }, &.{ 1, 2 });
    }
    return xs.narrow(1, 0, h).narrow(2, 0, w).contiguous();
}

fn defineRelativePositionIndex(window_size: [2]i64) Tensor {
    const coords_h = Tensor.arange(Scalar.int(window_size[0]), TensorOptions{ .kind = .Int64, .device = .Cpu });
    const coords_w = Tensor.arange(Scalar.int(window_size[1]), TensorOptions{ .kind = .Int64, .device = .Cpu });
    var t = [_]*const Tensor{ &coords_h, &coords_w };
    const mesh_grid = Tensor.meshgridIndexing(&t, "ij");
    var mesh_grid_list = std.ArrayList(*const Tensor).init(torch.global_allocator);
    mesh_grid_list.resize(mesh_grid.len) catch err(.AllocFailed);
    defer mesh_grid_list.deinit();
    for (mesh_grid, 0..) |m, i| {
        mesh_grid_list.items[i] = &m;
    }
    const coords = Tensor.stack(mesh_grid_list.items, 0).flatten(1, -1);

    var relative_coords = coords.unsqueeze(-1).sub(&coords.unsqueeze(1));
    relative_coords = relative_coords.permute(&.{ 1, 2, 0 }).contiguous();
    var temp = relative_coords.select(-1, 0);
    temp = temp.addScalar_(Scalar.int(window_size[0] - 1));
    temp = relative_coords.select(-1, 1);
    temp = temp.addScalar_(Scalar.int(window_size[1] - 1));
    temp = relative_coords.select(-1, 0);
    temp = temp.mulScalar_(Scalar.int(2 * window_size[1] - 2));
    var s = [_]i64{-1};
    return relative_coords.sumDimIntlist(&s, false, .Int).flatten(0, -1);
}

fn getRelativePositionBias(relative_position_bias_table: *const Tensor, relative_position_index: *const Tensor, window_size: [2]i64) Tensor {
    const n = window_size[0] * window_size[1];
    var t = [_]?*const Tensor{relative_position_index};
    return relative_position_bias_table.index(&t)
        .view(&.{ n, n, -1 }).permute(&.{ 2, 0, 1 }).contiguous().unsqueeze(0);
}

const ShiftedWindowAttentionBlock = struct {
    base_module: *Module = undefined,
    qkv: *Linear = undefined,
    proj: *Linear = undefined,
    relative_position_bias_table: Tensor = undefined,
    relative_position_index: Tensor = undefined,
    shift_size: []i64,
    dropout: f64,
    attention_dropout: f64,
    window_size: [2]i64,
    num_heads: i64,
    const Self = @This();

    pub fn init(
        dim: i64,
        window_size: [2]i64,
        shift_size: []i64,
        num_heads: i64,
        qkv_bias: bool,
        proj_bias: bool,
        attention_dropout: f64,
        dropout: f64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .shift_size = shift_size,
            .dropout = dropout,
            .attention_dropout = attention_dropout,
            .window_size = window_size,
            .num_heads = num_heads,
        };
        self.base_module = Module.init(self);
        self.qkv = Linear.init(.{ .in_features = dim, .out_features = 3 * dim, .bias = qkv_bias, .tensor_opts = options });
        self.proj = Linear.init(.{ .in_features = dim, .out_features = dim, .bias = proj_bias, .tensor_opts = options });
        self.relative_position_bias_table = self.base_module.registerParameter("relative_position_bias_table", Tensor.randn(&.{ (2 * window_size[0] - 1 * (2 * window_size[1] - 1)), num_heads }, options), true);
        const rpi = defineRelativePositionIndex(window_size);
        self.relative_position_index = self.base_module.registerBuffer("relative_position_index", rpi);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("qkv", self.qkv);
        _ = self.base_module.registerModule("proj", self.proj);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        const relative_position_bias = getRelativePositionBias(&self.relative_position_bias_table, &self.relative_position_index, self.window_size);
        return shiftedWindowAttention(
            x,
            &self.qkv.weight,
            &self.proj.weight,
            &relative_position_bias,
            self.window_size,
            self.num_heads,
            self.shift_size,
            self.attention_dropout,
            self.dropout,
            &self.qkv.bias.?,
            &self.proj.bias.?,
            null,
            self.base_module.isTraining(),
        );
    }
};

fn defineRelativePositionBiasTableV2(window_size: [2]i64) Tensor {
    const relative_coords_h = Tensor.arangeStart(Scalar.int(-(window_size[0] - 1)), Scalar.int(window_size[0]), torch.FLOAT_CPU);
    const relative_coords_w = Tensor.arangeStart(Scalar.int(-(window_size[1] - 1)), Scalar.int(window_size[1]), torch.FLOAT_CPU);
    var t = [_]*const Tensor{ &relative_coords_h, &relative_coords_w };
    const mesh_grid = Tensor.meshgrid(&t);
    var mesh_grid_list = std.ArrayList(*const Tensor).init(torch.global_allocator);
    mesh_grid_list.resize(mesh_grid.len) catch err(.AllocFailed);
    defer mesh_grid_list.deinit();
    for (mesh_grid, 0..) |m, i| {
        mesh_grid_list.items[i] = &m;
    }
    var relative_coords_table = Tensor.stack(mesh_grid_list.items, 0)
        .permute(&.{ 1, 2, 0 }).contiguous().unsqueeze(0);
    var temp = relative_coords_table.select(-1, 0);
    temp = temp.divScalar_(Scalar.int(window_size[0] - 1));
    temp = relative_coords_table.select(-1, 1);
    temp = temp.divScalar_(Scalar.int(window_size[1] - 1));
    _ = relative_coords_table.mulScalar_(Scalar.int(8));
    return relative_coords_table.sign().mul(&relative_coords_table.abs().addScalar(Scalar.float(1.0)).log2().divScalar(Scalar.float(3.0)));
}

const ShiftedWindowAttentionBlockV2 = struct {
    base_module: *Module = undefined,
    qkv: *Linear = undefined,
    proj: *Linear = undefined,
    relative_coords_table: Tensor = undefined,
    relative_position_index: Tensor = undefined,
    logit_scale: Tensor = undefined,
    cbp_mlp: *Sequential = undefined,
    shift_size: []i64,
    dropout: f64,
    attention_dropout: f64,
    num_heads: i64,
    window_size: [2]i64,

    const Self = @This();

    pub fn init(
        dim: i64,
        window_size: [2]i64,
        shift_size: []i64,
        num_heads: i64,
        qkv_bias: bool,
        proj_bias: bool,
        attention_dropout: f64,
        dropout: f64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{
            .shift_size = shift_size,
            .dropout = dropout,
            .attention_dropout = attention_dropout,
            .num_heads = num_heads,
            .window_size = window_size,
        };
        self.base_module = Module.init(self);
        self.qkv = Linear.init(.{ .in_features = dim, .out_features = 3 * dim, .bias = qkv_bias, .tensor_opts = options });
        self.proj = Linear.init(.{ .in_features = dim, .out_features = dim, .bias = proj_bias, .tensor_opts = options });
        const rct = defineRelativePositionBiasTableV2(window_size);
        self.relative_coords_table = self.base_module.registerBuffer("relative_coords_table", rct);
        const rpi = defineRelativePositionIndex(window_size);
        self.relative_position_index = self.base_module.registerBuffer("relative_position_index", rpi);

        self.logit_scale = self.base_module.registerParameter("logit_scale", Tensor.ones(&.{ num_heads, 1, 1 }, options).mulScalar(Scalar.float(@log(10.0))), true);

        self.cbp_mlp = Sequential.init(options)
            .addWithName("cpb_mlp.0", Linear.init(.{ .in_features = 2, .out_features = 512, .tensor_opts = options }))
            .add(Functional(Tensor.relu, .{}).init())
            .addWithName("cpb_mlp.2", Linear.init(.{ .in_features = 512, .out_features = num_heads, .tensor_opts = options, .bias = false }));

        if (qkv_bias) {
            var guard = NoGradGuard.init();
            defer guard.deinit();
            const length: i64 = @intCast(@divFloor(self.qkv.bias.?.numel(), 3));
            var t = self.qkv.bias.?.narrow(0, length, length);
            _ = t.zero_();
        }
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("qkv", self.qkv);
        _ = self.base_module.registerModule("proj", self.proj);
        _ = self.base_module.registerModule("cpb_mlp", self.cbp_mlp);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        const relative_position_bias = getRelativePositionBias(&self.cbp_mlp.forward(&self.relative_coords_table).view(&.{ -1, self.num_heads }), &self.relative_position_index, self.window_size).sigmoid().mulScalar(Scalar.int(16));

        return shiftedWindowAttention(
            x,
            &self.qkv.weight,
            &self.proj.weight,
            &relative_position_bias,
            self.window_size,
            self.num_heads,
            self.shift_size,
            self.attention_dropout,
            self.dropout,
            &self.qkv.bias.?,
            &self.proj.bias.?,
            &self.logit_scale,
            self.base_module.isTraining(),
        );
    }
};

const SwinTransformerBlock = struct {
    base_module: *Module = undefined,
    norm1: *LayerNorm = undefined,
    attn: *ShiftedWindowAttentionBlock = undefined,
    stoch_depth: *StochasticDepth = undefined,
    norm2: *LayerNorm = undefined,
    mlp: *Sequential = undefined,

    const Self = @This();

    pub fn init(
        dim: i64,
        num_heads: i64,
        window_size: [2]i64,
        shift_size: []i64,
        mlp_ratio: f64,
        dropout: f32,
        attention_dropout: f32,
        stochastic_depth_prob: f64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.norm1 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.attn = ShiftedWindowAttentionBlock.init(dim, window_size, shift_size, num_heads, true, true, attention_dropout, dropout, options);
        self.stoch_depth = StochasticDepth.init(stochastic_depth_prob, StochasticDepthKind.Row, options);
        self.norm2 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.mlp = mlpBlock(dim, @intFromFloat(@as(f64, @floatFromInt(dim)) * mlp_ratio), dropout, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("norm1", self.norm1);
        _ = self.base_module.registerModule("attn", self.attn);
        _ = self.base_module.registerModule("stoch_depth", self.stoch_depth);
        _ = self.base_module.registerModule("norm2", self.norm2);
        _ = self.base_module.registerModule("mlp", self.mlp);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var residual = x.shallowClone();
        var y = self.norm1.forward(x);
        y = self.attn.forward(&y);
        y = self.stoch_depth.forward(&y);
        y = y.add(&residual);
        residual = y.shallowClone();
        y = self.norm2.forward(&y);
        y = self.mlp.forward(&y);
        y = self.stoch_depth.forward(&y);
        y = y.add(&residual);
        return y;
    }
};

const SwinTransformerBlockV2 = struct {
    base_module: *Module = undefined,
    norm1: *LayerNorm = undefined,
    attn: *ShiftedWindowAttentionBlockV2 = undefined,
    stoch_depth: *StochasticDepth = undefined,
    norm2: *LayerNorm = undefined,
    mlp: *Sequential = undefined,

    const Self = @This();

    pub fn init(
        dim: i64,
        num_heads: i64,
        window_size: [2]i64,
        shift_size: []i64,
        mlp_ratio: f64,
        dropout: f32,
        attention_dropout: f32,
        stochastic_depth_prob: f64,
        options: TensorOptions,
    ) *Self {
        var self = torch.global_allocator.create(Self) catch err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self);
        self.norm1 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.attn = ShiftedWindowAttentionBlockV2.init(dim, window_size, shift_size, num_heads, true, true, attention_dropout, dropout, options);
        self.stoch_depth = StochasticDepth.init(stochastic_depth_prob, StochasticDepthKind.Row, options);
        self.norm2 = LayerNorm.init(.{
            .normalized_shape = torch.global_allocator.dupe(i64, &.{dim}) catch err(.AllocFailed),
            .tensor_opts = options,
        });
        self.mlp = mlpBlock(dim, @intFromFloat(@as(f64, @floatFromInt(dim)) * mlp_ratio), dropout, options);
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self.base_module.registerModule("norm1", self.norm1);
        _ = self.base_module.registerModule("attn", self.attn);
        _ = self.base_module.registerModule("stoch_depth", self.stoch_depth);
        _ = self.base_module.registerModule("norm2", self.norm2);
        _ = self.base_module.registerModule("mlp", self.mlp);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, x: *const Tensor) Tensor {
        var residual = x.shallowClone();
        var y = self.attn.forward(x);
        y = self.norm1.forward(&y);
        y = self.stoch_depth.forward(&y);
        y = y.add(&residual);
        residual = y.shallowClone();
        y = self.mlp.forward(&y);
        y = self.norm2.forward(&y);
        y = self.stoch_depth.forward(&y);
        y = y.add(&residual);
        return y;
    }
};

fn swinTransformer(
    patch_size: i64,
    embed_dim: i64,
    depths: [4]i64,
    num_heads: [4]i64,
    window_size: [2]i64,
    mlp_ratio: f64,
    dropout: f32,
    attention_dropout: f32,
    stochastic_depth_prob: f64,
    num_classes: i64,
    v2: bool,
    options: TensorOptions,
) *Sequential {
    var layers = Sequential.init(options)
        .add(Sequential.init(options)
        .addWithName("features.0.0", Conv2D.init(.{
        .in_channels = 3,
        .out_channels = embed_dim,
        .kernel_size = .{ patch_size, patch_size },
        .stride = .{ patch_size, patch_size },
        .tensor_opts = options,
    }))
        .add(Functional(Tensor.permute, .{&.{ 0, 2, 3, 1 }}).init())
        .addWithName("features.0.2", LayerNorm.init(.{
        .normalized_shape = torch.global_allocator.dupe(i64, &.{embed_dim}) catch err(.AllocFailed),
        .tensor_opts = options,
    })));
    var total_stage_blocks: i64 = 0;
    for (depths) |depth| {
        total_stage_blocks += depth;
    }
    var stage_block_idx: i64 = 0;
    for (depths, 0..) |depth, i_stage| {
        var stage = Sequential.init(options);
        const dim = embed_dim * std.math.pow(i64, 2, @intCast(i_stage));
        for (0..@intCast(depth)) |i_layer| {
            const sd_prob = stochastic_depth_prob * @as(f64, @floatFromInt(stage_block_idx)) / @as(f64, @floatFromInt(total_stage_blocks - 1));
            var ss = torch.global_allocator.dupe(i64, &window_size) catch err(.AllocFailed);
            for (0..ss.len) |i| {
                if (i_layer % 2 == 0) ss[i] = @divFloor(ss[i], 2);
            }
            if (v2) {
                const name = std.fmt.allocPrint(torch.global_allocator, "{d}.{d}", .{ layers.modules.items.len, i_layer }) catch err(.AllocFailed);
                stage = stage.addWithName(name, SwinTransformerBlockV2.init(dim, num_heads[i_stage], window_size, ss, mlp_ratio, dropout, attention_dropout, sd_prob, options));
            } else {
                const name = std.fmt.allocPrint(torch.global_allocator, "{d}.{d}", .{ layers.modules.items.len, i_layer }) catch err(.AllocFailed);
                stage = stage.addWithName(name, SwinTransformerBlock.init(dim, num_heads[i_stage], window_size, ss, mlp_ratio, dropout, attention_dropout, sd_prob, options));
            }
            stage_block_idx += 1;
        }
        layers = layers.add(stage);
        if (i_stage < depths.len - 1) {
            if (v2) {
                const name = std.fmt.allocPrint(torch.global_allocator, "{d}", .{layers.modules.items.len}) catch err(.AllocFailed);
                layers = layers.addWithName(name, PatchMergingV2.init(dim, options));
            } else {
                const name = std.fmt.allocPrint(torch.global_allocator, "{d}", .{layers.modules.items.len}) catch err(.AllocFailed);
                layers = layers.addWithName(name, PatchMerging.init(dim, options));
            }
        }
    }

    const num_features = embed_dim * std.math.pow(i64, 2, depths.len - 1);
    return Sequential.init(options)
        .addWithName("features", layers)
        .addWithName("norm", LayerNorm.init(.{
        .normalized_shape = torch.global_allocator.dupe(i64, &.{num_features}) catch err(.AllocFailed),
        .tensor_opts = options,
    }))
        .add(Functional(Tensor.permute, .{&.{ 0, 3, 1, 2 }}).init())
        .add(Functional(Tensor.adaptiveAvgPool2d, .{&.{ 1, 1 }}).init())
        .add(Functional(Tensor.flatten, .{ 1, -1 }).init())
        .addWithName("head", Linear.init(.{ .in_features = num_features, .out_features = num_classes, .tensor_opts = options }));
}

pub fn swinT(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        96,
        .{ 2, 2, 6, 2 },
        .{ 3, 6, 12, 24 },
        .{ 7, 7 },
        4.0,
        0.0,
        0.0,
        0.2,
        num_classes,
        false,
        options,
    );
}

pub fn swinS(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        96,
        .{ 2, 2, 18, 2 },
        .{ 3, 6, 12, 24 },
        .{ 7, 7 },
        4.0,
        0.0,
        0.0,
        0.3,
        num_classes,
        false,
        options,
    );
}

pub fn swinB(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        128,
        .{ 2, 2, 18, 2 },
        .{ 4, 8, 16, 32 },
        .{ 7, 7 },
        4.0,
        0.0,
        0.0,
        0.5,
        num_classes,
        false,
        options,
    );
}

pub fn swinV2T(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        96,
        .{ 2, 2, 6, 2 },
        .{ 3, 6, 12, 24 },
        .{ 8, 8 },
        4.0,
        0.0,
        0.0,
        0.2,
        num_classes,
        true,
        options,
    );
}

pub fn swinV2S(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        96,
        .{ 2, 2, 18, 2 },
        .{ 3, 6, 12, 24 },
        .{ 8, 8 },
        4.0,
        0.0,
        0.0,
        0.3,
        num_classes,
        true,
        options,
    );
}

pub fn swinV2B(num_classes: i64, options: TensorOptions) *Sequential {
    return swinTransformer(
        4,
        128,
        .{ 2, 2, 18, 2 },
        .{ 4, 8, 16, 32 },
        .{ 8, 8 },
        4.0,
        0.0,
        0.0,
        0.5,
        num_classes,
        true,
        options,
    );
}
