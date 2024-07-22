const std = @import("std");
const torch = @import("torch.zig");
const Tensor = torch.Tensor;
//
// +-------------------------+-------------------------+----------------------+
// |         8 bytes         |         N bytes         |   Rest of the file   |
// | containing size of the  |  JSON UTF-8 string      |    (Tensor Data)     |
// |         header          |  representing header    |                      |
// +-------------------------+-------------------------+----------------------+
// |                         |                         |                      |
// |                         | {                       |                      |
// |                         |   "TENSOR_NAME_1": {    |                      |
// |                         |     "dtype": DATA_TYPE, |                      |
// |                         |     "shape": [1, 16,    |                      |
// |                         |              256],      |                      |
// |                         |     "offsets": [BEGIN,  |                      |
// |                         |                  END]   |                      |
// |                         |   },                    |                      |
// |                         |   "TENSOR_NAME_2": {...}|                      |
// |                         |   ...                   |                      |
// |                         |   "__metadata__": {...} |                      |
// |                         | }                       |                      |
// +-------------------------------------------------------------|------------+
//                                                               |
//                                                   +-----------+
//                                                   |
//                                       Offsets -> [BEGIN, END]

pub const DType = enum {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    BF16,
    pub fn size(self: DType) usize {
        switch (self) {
            .Bool => return @sizeOf(bool),
            .U8 => return @sizeOf(u8),
            .U16 => return @sizeOf(u16),
            .U32 => return @sizeOf(u32),
            .U64 => return @sizeOf(u64),
            .I8 => return @sizeOf(i8),
            .I16 => return @sizeOf(i16),
            .I32 => return @sizeOf(i32),
            .I64 => return @sizeOf(i64),
            .F16 => return @sizeOf(f16),
            .F32 => return @sizeOf(f32),
            .F64 => return @sizeOf(f64),
            .BF16 => return 2,
        }
    }
};

fn strToDType(s: []const u8) !DType {
    if (std.mem.eql(u8, "BOOL", s)) return .Bool;
    if (std.mem.eql(u8, "U8", s)) return .U8;
    if (std.mem.eql(u8, "U16", s)) return .U16;
    if (std.mem.eql(u8, "U32", s)) return .U32;
    if (std.mem.eql(u8, "U64", s)) return .U64;
    if (std.mem.eql(u8, "I8", s)) return .I8;
    if (std.mem.eql(u8, "I16", s)) return .I16;
    if (std.mem.eql(u8, "I32", s)) return .I32;
    if (std.mem.eql(u8, "I64", s)) return .I64;
    if (std.mem.eql(u8, "F16", s)) return .F16;
    if (std.mem.eql(u8, "F32", s)) return .F32;
    if (std.mem.eql(u8, "F64", s)) return .F64;
    if (std.mem.eql(u8, "BF16", s)) return .BF16;
    return error.InvalidDType;
}

fn getShape(alloc: std.mem.Allocator, s: []std.json.Value) ![]i64 {
    var shape = std.ArrayList(i64).init(alloc);
    try shape.resize(s.len);
    for (s, 0..) |val, i| {
        shape.items[i] = val.integer;
    }
    return shape.toOwnedSlice();
}

fn getByteSize(dtype: DType, shape: []i64) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= @intCast(dim);
    }
    return size * dtype.size();
}

pub fn readSafetensorFrom(alloc: std.mem.Allocator, data: []u8) !std.StringArrayHashMap(Tensor) {
    const N: u64 = std.mem.readInt(u64, data[0..8], .little);
    const header = data[8 .. 8 + N];
    const tree = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer tree.deinit();
    var weights = std.StringArrayHashMap(Tensor).init(alloc);
    const tensors = tree.value.object;
    var start_offset: usize = N + 8;
    for (tensors.keys()) |key| {
        const val = tensors.get(key).?.object;
        const dtype = try strToDType(val.get("dtype").?.string);
        const shape = try getShape(alloc, val.get("shape").?.array.items);
        defer alloc.free(shape);
        const size = getByteSize(dtype, shape);
        const end = start_offset + size;
        const data_offsets = val.get("data_offsets");
        const offsets: [2]i64 = .{
            data_offsets.?.array.items[0].integer,
            data_offsets.?.array.items[1].integer,
        };
        _ = offsets; // Same as the calculated offsets
        var tensor: Tensor = undefined;
        switch (dtype) {
            .Bool => {
                const tensor_data: []bool = @alignCast(std.mem.bytesAsSlice(bool, data[start_offset..end]));
                tensor = Tensor.fromSlice(bool, tensor_data);
            },
            .U8 => {
                const tensor_data: []u8 = @alignCast(std.mem.bytesAsSlice(u8, data[start_offset..end]));
                tensor = Tensor.fromSlice(u8, tensor_data);
            },
            .U16 => {
                const tensor_data: []u16 = @alignCast(std.mem.bytesAsSlice(u16, data[start_offset..end]));
                tensor = Tensor.fromSlice(u16, tensor_data);
            },
            .U32 => {
                const tensor_data: []u32 = @alignCast(std.mem.bytesAsSlice(u32, data[start_offset..end]));
                tensor = Tensor.fromSlice(u32, tensor_data);
            },
            .U64 => {
                const tensor_data: []u64 = @alignCast(std.mem.bytesAsSlice(u64, data[start_offset..end]));
                tensor = Tensor.fromSlice(u64, tensor_data);
            },
            .I8 => {
                const tensor_data: []i8 = @alignCast(std.mem.bytesAsSlice(i8, data[start_offset..end]));
                tensor = Tensor.fromSlice(i8, tensor_data);
            },
            .I16 => {
                const tensor_data: []i16 = @alignCast(std.mem.bytesAsSlice(i16, data[start_offset..end]));
                tensor = Tensor.fromSlice(i16, tensor_data);
            },
            .I32 => {
                const tensor_data: []i32 = @alignCast(std.mem.bytesAsSlice(i32, data[start_offset..end]));
                tensor = Tensor.fromSlice(i32, tensor_data);
            },
            .I64 => {
                const tensor_data: []i64 = @alignCast(std.mem.bytesAsSlice(i64, data[start_offset..end]));
                tensor = Tensor.fromSlice(i64, tensor_data);
            },
            .F16 => {
                const tensor_data: []f16 = @alignCast(std.mem.bytesAsSlice(f16, data[start_offset..end]));
                tensor = Tensor.fromSlice(f16, tensor_data);
            },
            .F32 => {
                const tensor_data: []f32 = @alignCast(std.mem.bytesAsSlice(f32, data[start_offset..end]));
                tensor = Tensor.fromSlice(f32, tensor_data);
            },
            .F64 => {
                const tensor_data: []f64 = @alignCast(std.mem.bytesAsSlice(f64, data[start_offset..end]));
                tensor = Tensor.fromSlice(f64, tensor_data);
            },
            // .BF16 => {
            //     const tensor_data = std.mem.bytesAsSlice(bf16, data[start_offset..end]);
            //     tensor = Tensor.fromSlice(bf16, tensor_data);
            // },
            else => return error.InvalidDType,
        }
        tensor = tensor.view(shape);
        try weights.put(try alloc.dupe(u8, key), tensor);
        start_offset = end;
    }
    return weights;
}

pub fn readSafetensor(alloc: std.mem.Allocator, path: []const u8) !std.StringArrayHashMap(Tensor) {
    const data = try std.fs.cwd().readFileAlloc(alloc, path, 1_000_000_000);
    defer alloc.free(data);
    return readSafetensorFrom(alloc, data);
}

test "load" {
    const path = "resnet18.safetensors";
    var safetensor = try readSafetensor(std.testing.allocator, path);
    defer safetensor.deinit();
}
