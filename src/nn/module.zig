const torch = @import("../torch.zig");
const Tensor = torch.Tensor;
const TensorOptions = torch.TensorOptions;
const std = @import("std");

pub const Module = struct {
    children_: std.StringArrayHashMap(*Module) = undefined,
    parameters_: std.StringArrayHashMap(Tensor) = undefined,
    buffers_: std.StringArrayHashMap(Tensor) = undefined,
    is_training: bool,
    ptr: *anyopaque,
    forwardFn: *const fn (*anyopaque, *const Tensor) Tensor,

    pub fn init(obj: anytype) *Module {
        const Ptr = @TypeOf(obj);
        const impl = struct {
            fn forward(ptr: *anyopaque, input: *const Tensor) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ptr));
                return self.forward(input);
            }
        };
        var self = torch.global_allocator.create(Module) catch unreachable;
        self.* = Module{
            .ptr = @ptrCast(obj),
            .forwardFn = &impl.forward,
            .is_training = true,
        };
        self.initFields();
        return self;
    }

    pub fn forward(self: *Module, input: *const Tensor) Tensor {
        return self.forwardFn(self.ptr, input);
    }

    fn initFields(self: *Module) void {
        self.children_ = std.StringArrayHashMap(*Module).init(torch.global_allocator);
        self.parameters_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
        self.buffers_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
    }

    pub fn deinit(self: *Module) void {
        self.deinitFields();
        torch.global_allocator.destroy(self);
    }

    fn deinitFields(self: *Module) void {
        self.children_.deinit();
        self.parameters_.deinit();
        self.buffers_.deinit();
    }

    pub fn apply(
        self: *Module,
        prefix: []const u8,
        result: anytype,
        function: *const fn (type, *anyopaque, []const u8, *anyopaque) void,
    ) void {
        function(@TypeOf(self), self, prefix, result);
        for (self.children_.keys()) |key| {
            var child = self.children_.get(key).?;
            const new_prefix = std.fmt.allocPrint(torch.global_allocator, "{s}{s}.", .{ prefix, key }) catch unreachable;
            child.apply(new_prefix, result, function);
        }
    }

    pub fn name(self: *Module) []const u8 {
        // if (comptime @hasField(@TypeOf(self), "nameImpl")) {
        //     return self.nameImpl();
        // }
        return @typeName(@TypeOf(self));
    }

    pub fn parameters(self: *Module, recurse: bool) []Tensor {
        return self.namedParameters(recurse).values();
    }

    pub fn namedParametersApply(comptime T: type, ptr: *anyopaque, prefix: []const u8, result: *anyopaque) void {
        var result_: *std.StringArrayHashMap(Tensor) = @ptrCast(@alignCast(result));
        var self: T = @ptrCast(@alignCast(ptr));
        for (self.parameters_.keys()) |key| {
            const value = self.parameters_.get(key).?;
            const full_key = std.fmt.allocPrint(torch.global_allocator, "{s}{s}", .{ prefix, key }) catch unreachable;
            result_.put(full_key, value) catch unreachable;
        }
    }

    pub fn namedBuffersApply(comptime T: type, ptr: *anyopaque, prefix: []const u8, result: *anyopaque) void {
        var result_: *std.StringArrayHashMap(Tensor) = @ptrCast(@alignCast(result));
        var self: T = @ptrCast(@alignCast(ptr));
        for (self.buffers_.keys()) |key| {
            const value = self.buffers_.get(key).?;
            const full_key = std.fmt.allocPrint(torch.global_allocator, "{s}{s}", .{ prefix, key }) catch unreachable;
            result_.put(full_key, value) catch unreachable;
        }
    }

    pub fn namedParameters(self: *Module, recurse: bool) std.StringArrayHashMap(Tensor) {
        var result = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
        if (!recurse) {
            for (self.parameters_.keys()) |key| {
                const value = self.parameters_.get(key).?;
                if (value.defined()) {
                    result.put(key, value) catch unreachable;
                }
            }
        } else {
            @setEvalBranchQuota(2000);
            self.apply("", &result, namedParametersApply);
        }
        return result;
    }

    pub fn buffers(self: *Module, recurse: bool) std.ArrayList(Tensor) {
        return self.namedBuffers(recurse).values();
    }

    pub fn namedBuffers(self: *Module, recurse: bool) std.StringArrayHashMap(Tensor) {
        var result = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
        if (!recurse) {
            for (self.buffers_.keys()) |key| {
                const value = self.buffers_.get(key).?;
                if (value.defined()) {
                    result.put(key, value) catch unreachable;
                }
            }
        } else {
            @setEvalBranchQuota(2000);
            self.apply("", &result, namedBuffersApply);
        }
        return result;
    }
    //
    // pub fn modules(self: *const T, include_self: bool) std.ArrayList(T) {
    //
    // }
    //
    // pub fn namedModules(self: *const T, include_self: bool) std.ArrayList(([]const u8, T)) {
    //
    // }
    //
    pub fn children(self: *const Module) std.ArrayList(*Module) {
        return self.children_.values();
    }

    pub fn namedChildren(self: *const Module) std.StringArrayHashMap(*Module) {
        return self.children_;
    }

    pub fn train(self: *Module, on: bool) void {
        for (self.children_.keys()) |key| {
            const child = self.children_.get(key).?;
            _ = child.train(on);
        }
        self.is_training = on;
    }

    pub fn eval(self: *Module) void {
        self.train(false);
    }

    pub fn isTraining(self: *const Module) bool {
        return self.is_training;
    }

    pub fn to(self: *Module, device: torch.Device, kind: torch.Kind, non_blocking: bool) void {
        for (self.children_.keys()) |key| {
            const child = self.children_.get(key).?;
            child.to(device, kind, non_blocking);
        }

        for (self.parameters_.keys()) |key| {
            var tensor = self.parameters_.get(key).?;
            tensor.setData(&tensor.toDevice(device, kind, non_blocking, false));
        }

        for (self.buffers_.keys()) |key| {
            var tensor = self.buffers_.get(key).?;
            tensor.setData(&tensor.toDevice(device, kind, non_blocking, false));
        }
    }

    // pub fn zeroGrad(self: *Module, set_to_none: bool) void  {
    //     for(self.children_.keys()) |key| {
    //         const child = self.children_.get(key).?;
    //         child.zeroGrad(set_to_none);
    //     }
    //
    //     for(self.namedParameters(false).keys()) |key| {
    //         var param = self.parameters_.get(key).?;
    //         if (param.requires_grad()) {
    //             param.grad().zero_();
    //             if (set_to_none) {
    //                 param.grad().setData(undefined);
    //             }
    //         }
    //     }
    // }
    //
    // pub fn save(self: *const T, path: []const u8) void {
    //
    // }
    //
    // pub fn load(self: *T, path: []const u8) void {
    //
    // }
    //
    pub fn registerParameter(self: *Module, name_: []const u8, tensor: Tensor, requires_grad_: bool) Tensor {
        // TODO: add checks here, decide on whether to return an error or not
        if (!tensor.defined()) {
            if (requires_grad_) {
                std.log.warn("An undefined tensor cannot require grad. " ++
                    "Ignoring the `requires_grad` argument.", .{});
            }
        } else {
            _ = tensor.setRequiresGrad(requires_grad_);
        }
        self.parameters_.put(name_, tensor) catch unreachable;
        return tensor;
    }

    pub fn registerBuffer(self: *Module, name_: []const u8, tensor: Tensor) Tensor {
        _ = self.buffers_.put(name_, tensor) catch unreachable;
        return tensor;
    }

    pub fn registerModule(self: *Module, name_: []const u8, module: anytype) *Module {
        if (!@hasField(@TypeOf(module.*), "base_module")) {
            @compileError("The module must have a `base_module` field.");
        }
        _ = self.children_.put(name_, module.base_module) catch unreachable;
        return module.base_module;
    }
};

pub const Sequential = struct {
    base_module: *Module = undefined,

    modules: std.ArrayList(*Module) = undefined,
    options: TensorOptions = undefined,

    const Self = @This();

    pub fn init(options: TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch unreachable;
        self.* = Self{
            .options = options,
        };
        self.modules = std.ArrayList(*Module).init(torch.global_allocator);
        self.base_module = Module.init(self);
        self.reset();
        return self;
    }

    pub fn add(self: *Self, module: anytype) *Self {
        if (!@hasField(@TypeOf(module.*), "base_module")) {
            @compileError("The module must have a `base_module` field.");
        }
        const name = std.fmt.allocPrint(torch.global_allocator, "{d}", .{self.modules.items.len}) catch unreachable;
        self.modules.append(module.base_module) catch unreachable;
        _ = self.base_module.registerModule(name, module);
        return self;
    }

    pub fn addWithName(self: *Self, name: []const u8, module: anytype) *Self {
        if (!@hasField(@TypeOf(module.*), "base_module")) {
            @compileError("The module must have a `base_module` field.");
        }
        self.modules.append(module.base_module) catch unreachable;
        _ = self.base_module.registerModule(name, module);
        return self;
    }

    pub fn reset(self: *Self) void {
        _ = self;
    }

    pub fn forward(self: *Self, inputs: anytype) Tensor {
        var output = inputs;
        var out: Tensor = undefined;
        for (self.modules.items) |module| {
            out = module.forward(output);
            output = &out;
        }
        return output.*;
    }

    pub fn deinit(self: *Self) void {
        for (self.modules.items) |module| {
            module.deinit();
        }
        torch.global_allocator.destroy(self);
    }
};
