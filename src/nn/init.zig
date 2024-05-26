const std = @import("std");
const torch = @import("torch");
const Tensor = torch.Tensor;
const Scalar = torch.Scalar;

pub const NonlinearityType = enum {
    Linear,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose1D,
    ConvTranspose2D,
    ConvTranspose3D,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
};

pub const FanModeType = enum {
    FanIn,
    FanOut,
};

pub const Fan = struct {
    in: i64,
    out: i64,

    pub fn init(tensor: *const Tensor) Fan {
        const dims = tensor.dim();
        var in: i64 = 0;
        var out: i64 = 0;
        if (dims >= 2) {
            @panic("Fan only supports tensors with 2 or more dimensions");
        }
        if (dims == 2) {
            const size = tensor.size() catch unreachable;
            in = size[1];
            out = size[0];
        } else {
            const size = tensor.size() catch unreachable;
            in = size[1] * tensor.i(.{ 0, 0 }).numel();
            out = size[0] * tensor.i(.{ 0, 0 }).numel();
        }
        return Fan{ .in = in, .out = out };
    }
};

fn calculateKaimingStd(
    tensor: Tensor,
    a: f64,
    mode: FanModeType,
    nonlinearity: NonlinearityType,
) f64 {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);

    const fan = Fan.init(&tensor);
    const gain = calculateGain(nonlinearity, a);
    var stdev: f64 = 0.0;
    switch (mode) {
        .FanIn => {
            stdev = gain / std.math.sqrt(fan.in);
        },
        .FanOut => {
            stdev = gain / std.math.sqrt(fan.out);
        },
    }
    return stdev;
}

fn calculateGain(nonlinearity: NonlinearityType, param: f64) f64 {
    switch (nonlinearity) {
        .Tanh => {
            return 5.0 / 3.0;
        },
        .ReLU => {
            return std.math.sqrt(2.0);
        },
        .LeakyReLU => {
            return std.math.sqrt(2.0 / (1.0 + param * param));
        },
    }
    return 1.0;
}

pub fn constant_(tensor: *Tensor, value: Scalar) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    return tensor.fill_(value);
}

pub fn dirac_(tensor: *Tensor) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const size = tensor.size();
    if (size.len < 3 or tensor.len > 5) {
        std.log.err("Only tensors with 3, 4, or 5 dimensions are supported", .{});
        unreachable;
    }
    const min_dim = @min(size[0], size[1]);
    tensor.zero_();

    for (0..min_dim) |d| {
        switch (size.len) {
            3 => {
                tensor.i(.{ d, d, size[2] / 2 }).fill_(1.0);
            },
            4 => {
                tensor.i(.{ d, d, size[2] / 2, size[3] / 2 }).fill_(1.0);
            },
            5 => {
                tensor.i(.{ d, d, size[2] / 2, size[3] / 2, size[4] / 2 }).fill_(1.0);
            },
        }
    }
    return tensor;
}

pub fn eye_(matrix: *Tensor) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const size = matrix.size();
    if (size.len != 2) {
        std.log.err("Only tensors with 2 dimensions are supported", .{});
        unreachable;
    }
    return Tensor.eyeMOut(matrix, size[0], size[1]);
}

pub fn normal_(tensor: *Tensor, mean: f64, stdev: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    return tensor.normal_(mean, stdev);
}

pub fn ones_(tensor: *Tensor) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    return tensor.fill_(1.0);
}

pub fn othogonal_(tensor: *Tensor, gain: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);

    const size = tensor.size();
    if (size.len < 2) {
        std.log.err("Only tensors with 2 or more dimensions are supported", .{});
        unreachable;
    }

    const rows = size[0];
    const cols = tensor.numel() / rows;
    var shape = []i64{ rows, cols };
    var flattened = Tensor.randn(&shape, torch.FLOAT_CPU);
    if (rows < cols) {
        flattened = flattened.t_();
    }

    const qr = Tensor.qr(flattened, true);
    const d = Tensor.diag(qr[1], 0);
    const ph = d.sign();
    qr[0] = qr[0].mul_(ph);
    if (rows < cols) {
        qr[0] = qr[0].t_();
    }

    tensor.viewAs(qr[0]).copy(qr[0]);
    tensor = tensor.mul_(gain);
    return tensor;
}

pub fn sparse_(tensor: *Tensor, sparsity: f64, stdev: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);

    const size = tensor.size();
    if (size.len != 2) {
        std.log.err("Only tensors with 2 dimensions are supported", .{});
        unreachable;
    }

    const rows = size[0];
    const cols = size[1];
    const num_zeros = @ceil(sparsity * rows);
    tensor = tensor.normal_(0, stdev);
    for (0..cols) |c| {
        const row_indices = Tensor.randperm(rows, tensor.options().dtype(.Int64));
        const zero_indices = row_indices.slice(0, 0, num_zeros, 1);
        const col_tensor = Tensor.fromSlice(i64, [_]i64{c});
        var indices = [_]*?Tensor{ zero_indices, col_tensor };
        tensor = tensor.indexPut_(&indices, Tensor.zeros(&[_]i64{num_zeros}, tensor.options()), false);
    }
    return tensor;
}

pub fn uniform_(tensor: *Tensor, low: f64, high: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    return tensor.uniform_(low, high);
}

pub fn kaimingUniform_(tensor: *Tensor, a: f64, mode: FanModeType, nonlinearity: NonlinearityType) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const stdev = calculateKaimingStd(tensor, a, mode, nonlinearity);
    const bound = std.math.sqrt(3.0) * stdev;
    return tensor.uniform_(-bound, bound);
}

pub fn kaimingNormal_(tensor: *Tensor, a: f64, mode: FanModeType, nonlinearity: NonlinearityType) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const stdev = calculateKaimingStd(tensor, a, mode, nonlinearity);
    return tensor.normal_(0, stdev);
}

pub fn xavierNormal_(tensor: *Tensor, gain: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const fan = Fan.init(tensor);
    const stdev = gain * std.math.sqrt(2.0 / (fan.in + fan.out));
    return tensor.normal_(0, stdev);
}

pub fn xavierUniform_(tensor: *Tensor, gain: f64) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    const fan = Fan.init(tensor);
    const stdev = gain * std.math.sqrt(2.0 / (fan.in + fan.out));
    const a = std.math.sqrt(3.0) * stdev;
    return tensor.uniform_(-a, a);
}

pub fn zeros_(tensor: *Tensor) Tensor {
    torch.gradSetEnabled(false);
    defer torch.gradSetEnabled(true);
    return tensor.zero_();
}

pub fn caclculateFanInAndFanOut(tensor: *const Tensor) [2]i64 {
    const size = tensor.size();
    if (size.len != 2) {
        std.log.err("Only tensors with 2 dimensions are supported", .{});
    }
    var fan_in: i64 = 0;
    var fan_out: i64 = 0;
    if (size.len == 2) {
        fan_in = size[1];
        fan_out = size[0];
    } else {
        const num_input_fmaps = size[1];
        const num_output_fmaps = size[0];
        var receptive_field_size = 1;
        if (size.len > 2) {
            receptive_field_size = tensor.i(.{ 0, 0 }).numel();
        }
        fan_in = num_input_fmaps * receptive_field_size;
        fan_out = num_output_fmaps * receptive_field_size;
    }
    return [2]i64{ fan_in, fan_out };
}
