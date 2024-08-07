// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! ";
const __c = @cImport({
    @cInclude("stddef.h");
    @cInclude("stdbool.h");
    @cInclude("torch_api.h");
    @cInclude("torch_api_generated.h");
});
const std = @import("std");
const torch = @import("torch.zig");
const err = torch.utils.err;
const TchError = torch.TchError;
const TensorOptions = torch.TensorOptions;
const Device = torch.Device;
const Kind = torch.Kind;
const Scalar = torch.Scalar;
const Layout = torch.Layout;
const C_tensor = __c.tensor;

pub const Reduction = enum {
    None,
    Mean,
    Sum,

    pub fn toInt(self: Reduction) i64 {
        return switch (self) {
            .None => 0,
            .Mean => 1,
            .Sum => 2,
        };
    }
};

fn ptrListOpt(l: []?*const Tensor) []C_tensor {
    var ret = std.ArrayList(C_tensor).init(torch.global_allocator);
    for (l) |x| {
        if (x == null) {
            ret.append(null) catch err(.AllocFailed);
            continue;
        }
        ret.append(x.?.c_tensor) catch err(.AllocFailed);
    }
    return ret.toOwnedSlice() catch err(.AllocFailed);
}
fn ptrList(l: []*const Tensor) []C_tensor {
    var ret = std.ArrayList(C_tensor).init(torch.global_allocator);
    for (l) |x| {
        ret.append(x.c_tensor) catch err(.AllocFailed);
    }
    return ret.toOwnedSlice() catch err(.AllocFailed);
}

pub const TensorIndexer = union(enum) {
    Select: i64,
    Narrow: struct {
        start: ?i64,
        end: ?i64,
    },
    IndexSelect: *const Tensor,
    InsertNewAxis: void,
};

pub const NewAxis = struct {};

pub const Tensor = struct {
    c_tensor: C_tensor,
    pub fn new() Tensor {
        const ret = __c.at_new_tensor();
        torch.readAndCleanError();
        return Tensor{ .c_tensor = ret };
    }
    // TODO: implement this for formatted printing, for now we can just use the default print
    // pub fn format(self: Tensor, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype,) !void {
    // }

    pub fn i(self: *const Tensor, index_spec: anytype) Tensor {
        // TODO: add case for just a single int index
        var specs = std.ArrayList(TensorIndexer).init(torch.global_allocator);
        defer specs.deinit();
        inline for (index_spec) |spec| {
            const spec_ = switch (@TypeOf(spec)) {
                NewAxis => TensorIndexer.InsertNewAxis,
                i64, comptime_int => TensorIndexer{ .Select = spec },
                struct { start: ?i64, end: ?i64 } => TensorIndexer{ .Narrow = spec },
                []i64 => blk_i64: {
                    const index_tensor = Tensor.fromSlice(i64, spec);
                    break :blk_i64 TensorIndexer{ .IndexSelect = index_tensor };
                },
                Tensor => TensorIndexer{ .IndexSelect = spec.shallowClone() },
                else => {
                    @panic("unsupported index spec type");
                },
            };
            specs.append(spec_) catch err(.AllocFailed);
        }
        return self.indexer(specs.items);
    }

    fn indexer(self: *const Tensor, index_spec: []TensorIndexer) Tensor {
        var n_newaxis: usize = 0;
        for (index_spec) |spec| {
            if (spec == TensorIndexer.InsertNewAxis) {
                n_newaxis += 1;
            }
        }
        const dim_ = self.dim();
        if (index_spec.len > dim_ + n_newaxis) {
            @panic("too many indices for tensor");
        }

        for (index_spec) |spec| {
            switch (spec) {
                .IndexSelect => |tensor| {
                    if (dim_ != 1) {
                        @panic("expected 1-d tensor");
                    }

                    switch (tensor.kind()) {
                        .Int64, .Int16, .Int8, .Int => {},
                        else => {
                            @panic("expected int tensor for indices");
                        },
                    }
                },
                else => {},
            }
        }

        var curr_tensor = self.shallowClone();
        var curr_idx: usize = 0;

        for (index_spec) |spec| {
            switch (spec) {
                TensorIndexer.InsertNewAxis => {
                    curr_tensor = curr_tensor.unsqueeze(@intCast(curr_idx));
                    curr_idx += 1;
                },
                TensorIndexer.Select => |idx| {
                    curr_tensor = curr_tensor.select(@intCast(curr_idx), @intCast(idx));
                },
                TensorIndexer.Narrow => |narrow_| {
                    const size_ = self.size();
                    const start = narrow_.start orelse 0;
                    const end = narrow_.end orelse size_[curr_idx];
                    if (start < 0 or end < start or end > size_[curr_idx]) {
                        @panic("invalid start/end for narrow");
                    }
                    const length = end - start;
                    curr_tensor = curr_tensor.narrow(@intCast(curr_idx), start, length);
                    curr_idx += 1;
                },
                TensorIndexer.IndexSelect => |index_tensor| {
                    curr_tensor = curr_tensor.indexSelect(@intCast(curr_idx), index_tensor);
                    curr_idx += 1;
                },
            }
        }

        return curr_tensor;
    }

    pub fn options(self: *const Tensor) TensorOptions {
        return TensorOptions{ .kind = self.kind(), .device = self.device() };
    }

    pub fn fromPtr(c_tensor: C_tensor) Tensor {
        return Tensor{ .c_tensor = c_tensor };
    }

    pub fn fromLong(v: i64) Tensor {
        var c_tensor = [1]C_tensor{__c.at_new_long(v)};
        torch.memory_pool.put(&c_tensor);
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor[0] };
    }

    pub fn fromFloat(v: f64) Tensor {
        var c_tensor = [1]C_tensor{__c.at_new_double(v)};
        torch.memory_pool.put(&c_tensor);
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor[0] };
    }

    pub fn cloneFromPtr(c_tensor: C_tensor) Tensor {
        const tensor = __c.at_shallow_clone(c_tensor);
        torch.memory_pool.put(&.{tensor});
        torch.readAndCleanError();
        return Tensor{ .c_tensor = tensor };
    }

    pub fn asPtr(self: *const Tensor) C_tensor {
        return self.c_tensor;
    }

    pub fn dim(self: *const Tensor) usize {
        const ret = __c.at_dim(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }

    pub fn size(self: *const Tensor) []i64 {
        const dim_ = self.dim();
        var buffer: [10]i64 = undefined;
        __c.at_shape(self.c_tensor, buffer[0..dim_].ptr);
        torch.readAndCleanError();
        return torch.global_allocator.dupe(i64, buffer[0..dim_]) catch err(.AllocFailed);
    }

    pub fn sizeDims(self: *const Tensor, comptime dims: usize) [dims]i64 {
        const size_ = self.size();
        if (size_.len != dims) {
            @panic("expected one dim");
        }
        return size_[0..dims].*;
    }

    pub fn stride(self: *const Tensor) ![]i64 {
        const dim_ = self.dim();
        var sz = std.ArrayList(i64).init(torch.global_allocator);
        try sz.resize(dim_);
        __c.at_stride(self.c_tensor, sz.items);
        torch.readAndCleanError();
        return sz.toOwnedSlice();
    }

    pub fn strideDims(self: *const Tensor, comptime dims: usize) ![dims]i64 {
        const stride_ = self.stride();
        if (stride_.len != dims) {
            @panic("expected one dim");
        }
        return stride_[0..dims];
    }

    pub fn kind(self: *const Tensor) Kind {
        const kind_ = __c.at_scalar_type(self.c_tensor);
        torch.readAndCleanError();
        return Kind.fromCInt(kind_);
    }

    pub fn device(self: *const Tensor) Device {
        const device_ = __c.at_device(self.c_tensor);
        torch.readAndCleanError();
        return Device.fromCInt(device_);
    }

    pub fn print(self: *const Tensor) void {
        __c.at_print(self.c_tensor);
        torch.readAndCleanError();
    }

    pub fn doubleValue(self: *const Tensor, idx: []const i64) f64 {
        const ret = __c.at_double_value_at_indexes(
            self.c_tensor,
            @constCast(@ptrCast(idx)),
            @intCast(idx.len),
        );
        torch.readAndCleanError();
        return ret;
    }

    pub fn int64Value(self: *const Tensor, idx: []const i64) i64 {
        const ret = __c.at_int64_value_at_indexes(
            self.c_tensor,
            @constCast(@ptrCast(idx)),
            @intCast(idx.len),
        );
        torch.readAndCleanError();
        return ret;
    }

    pub fn requiresGrad(self: *const Tensor) bool {
        const ret = __c.at_requires_grad(self.c_tensor);
        torch.readAndCleanError();
        return if (ret != 0) true else false;
    }

    // pub fn dataPtr(self: *Tensor) *c_void {
    //     const ret = __c.at_data_ptr(self.c_tensor);
    //     torch.readAndCleanError();
    //     return ret;
    // }

    pub fn defined(self: *const Tensor) bool {
        const ret = __c.at_defined(self.c_tensor);
        torch.readAndCleanError();
        return if (ret != 0) true else false;
    }

    pub fn isMkldnn(self: *const Tensor) bool {
        const ret = __c.at_is_mkldnn(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }

    pub fn isSparse(self: *const Tensor) bool {
        const ret = __c.at_is_sparse(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }

    pub fn isContiguous(self: *const Tensor) bool {
        const ret = __c.at_is_contiguous(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }

    pub fn zeroGrad(self: *Tensor) void {
        var grad_ = self.grad();
        if (grad_.defined()) {
            _ = grad_.detach().zero();
        }
    }

    pub fn backward(self: *Tensor) void {
        __c.at_backward(self.c_tensor, 0, 0);
        torch.readAndCleanError();
    }

    pub fn runBackward(tensors: []Tensor, inputs: []Tensor, keep_graph: bool, create_graph: bool) !std.ArrayList(Tensor) {
        var outputs = std.ArrayList(C_tensor).init(torch.global_allocator);
        defer outputs.deinit();
        try outputs.resize(inputs.len);
        var inputs_ = ptrList(inputs);
        var tensors_ = ptrList(tensors);

        __c.at_run_backward(&tensors_, @intCast(tensors_.len), &inputs_, @intCast(inputs_.len), outputs.items, keep_graph, create_graph);
        torch.readAndCleanError();
        var res = std.ArrayList(Tensor).init(torch.global_allocator);
        for (outputs.items) |output| {
            res.append(Tensor{ .c_tensor = output });
        }
        return res;
    }

    pub fn copyDataU8(self: *Tensor, dst: []u8, numel_: usize) !void {
        const elt_size_in_bytes = self.kind().eltSizeInBytes();
        if (dst.len < numel_ * elt_size_in_bytes) {
            std.log.err("expected buffer of size {}, got {}", .{ numel_ * elt_size_in_bytes, dst.len });
            return error.BufferTooSmall;
        }
        __c.at_copy_data(self.c_tensor, dst.ptr, numel_, elt_size_in_bytes);
        torch.readAndCleanError();
    }

    pub fn internalAmpNonFiniteCheckAndUnscale(self: *Tensor, found_inf: *Tensor, inv_scale: *const Tensor) void {
        __c.at__amp_non_finite_check_and_unscale(self.c_tensor, found_inf.c_tensor, inv_scale.c_tensor);
        torch.readAndCleanError();
    }

    pub fn copyData(self: *const Tensor, dst: []*Tensor, numel_: usize) !void {
        // TODO: Fix this function
        if (self.kind() != dst.kind()) {
            std.log.err("expected same kind, got {any} and {any}", .{ self.kind(), dst.kind() });
            return error.UnexpectedKind;
        }

        if (dst.len < numel_) {
            std.log.err("expected buffer of size {}, got {}", .{ numel_, dst.len });
            return error.BufferTooSmall;
        }

        __c.at_copy_data(self.c_tensor, dst.c_tensor, numel_);
        torch.readAndCleanError();
    }

    pub fn numel(self: *const Tensor) usize {
        const size_ = self.size();
        var ret: usize = 1;
        for (size_) |s| {
            ret *= @intCast(s);
        }
        return ret;
    }

    pub fn fromSlice(comptime T: type, data_: []T) Tensor {
        var size_ = [_]i64{@intCast(data_.len)};
        const kind_ = torch.elementKind(T);
        const c_tensor = __c.at_tensor_of_data(data_.ptr, &size_, 1, kind_.eltSizeInBytes(), kind_.cInt());
        var c_tensors = [_]C_tensor{c_tensor};
        torch.memory_pool.put(&c_tensors);
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor };
    }

    pub fn free(self: *Tensor) void {
        __c.at_free(self.c_tensor);
        torch.readAndCleanError();
    }

    pub fn crossEntropyForLogits(self: *const Tensor, targets: *const Tensor) Tensor {
        return self.logSoftmax(-1, .Float).nllLoss(targets, null, .Mean, -100);
    }

    pub fn accuracyForLogits(self: *const Tensor, targets: *const Tensor) Tensor {
        return self.argmax(-1, false).eqTensor(targets).totype(.Float).mean(.Float);
    }

    // TODO: finish rest of the functions

    pub fn shallowClone(self: *const Tensor) Tensor {
        const c_tensor = __c.at_shallow_clone(self.c_tensor);
        var __t = [_]__c.tensor{c_tensor};
        torch.memory_pool.put(&__t);
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor };
    }

    pub fn get(self: *const Tensor, idx: i64) Tensor {
        const c_tensor = __c.at_get(self.c_tensor, @intCast(idx));
        var __t = [_]__c.tensor{c_tensor};
        torch.memory_pool.put(&__t);
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor };
    }

    pub fn copy(self: *const Tensor, src: *const Tensor) void {
        __c.at_copy_(self.c_tensor, src.c_tensor);
        torch.readAndCleanError();
    }

    pub fn load(path: []const u8) Tensor {
        const c_path = torch.global_allocator.dupeZ(path);
        defer torch.global_allocator.free(c_path);
        const c_tensor = __c.at_load(c_path);
        torch.memory_pool.put(&.{c_tensor});
        torch.readAndCleanError();
        return Tensor{ .c_tensor = c_tensor };
    }

    pub fn save(self: *const Tensor, path: []const u8) void {
        const c_path = torch.global_allocator.dupeZ(path);
        defer torch.global_allocator.free(c_path);
        __c.at_save(self.c_tensor, c_path);
        torch.readAndCleanError();
    }

    pub fn toString(self: *const Tensor, lw: i64) []const u8 {
        const s = __c.at_to_string(self.c_tensor, @intCast(lw));
        torch.readAndCleanError();
        return std.mem.span(s);
    }

    pub fn toInt(self: *const Tensor) i64 {
        const ret = __c.at_tensor_item_int64(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }

    pub fn toFloat(self: *const Tensor) f32 {
        const ret = __c.at_tensor_item_float(self.c_tensor);
        torch.readAndCleanError();
        return ret;
    }
};
