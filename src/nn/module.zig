const torch = @import("torch");
const Tensor = @import("torch").Tensor;
const std = @import("std");

pub const Module = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        initFields: *const fn (*anyopaque) void,
        deinitFields: *const fn (*anyopaque) void,
        apply: *const fn (*const anyopaque, []const u8, *anyopaque, *const fn (*const anyopaque, []const u8, *anyopaque) void) void,
        name: *const fn (*const anyopaque) []const u8,
        parameters: *const fn (*const anyopaque, bool) []Tensor,
        namedParameters: *const fn (*const anyopaque, bool) std.StringArrayHashMap(Tensor),
        registerParameter: *const fn (*anyopaque, []const u8, Tensor, bool) Tensor,
        to: *const fn (*anyopaque, torch.Device, torch.Kind, bool) void,
        registerBuffer: *const fn (*anyopaque, []const u8, Tensor) Tensor,
        registerModule: *const fn (*anyopaque, []const u8, *Module) *Module,
    };

    pub fn initFields(self: *Module) void {
        self.vtable.initFields(self.ptr);
    }

    pub fn deinitFields(self: *Module) void {
        self.vtable.deinitFields(self.ptr);
    }

    pub fn apply(self: *const Module, prefix: []const u8, result: *anyopaque, function: *const fn (*const anyopaque, []const u8, *anyopaque) void) void {
        self.vtable.apply(self.ptr, prefix, result, function);
    }

    pub fn name(self: *const Module) []const u8 {
        return self.vtable.name(self.ptr);
    }

    pub fn parameters(self: *const Module, recurse: bool) []Tensor {
        return self.vtable.parameters(self.ptr, recurse);
    }

    pub fn to(self: *Module, device: torch.Device, kind: torch.Kind, non_blocking: bool) void {
        self.vtable.to(self.ptr, device, kind, non_blocking);
    }

    pub fn namedParameters(self: *const Module, recurse: bool) std.StringArrayHashMap(Tensor) {
        return self.vtable.namedParameters(self.ptr, recurse);
    }

    pub fn registerParameter(self: *Module, name_: []const u8, tensor: Tensor, requires_grad: bool) Tensor {
        return self.vtable.registerParameter(self.ptr, name_, tensor, requires_grad);
    }

    pub fn registerBuffer(self: *Module, name_: []const u8, tensor: Tensor) Tensor {
        return self.vtable.registerBuffer(self.ptr, name_, tensor);
    }

    pub fn registerModule(self: *Module, name_: []const u8, module: *Module) *Module {
        return self.vtable.registerModule(self.ptr, name_, module);
    }

    pub fn init(obj: anytype) Module {
        const Ptr = @TypeOf(obj);
        const ptr_info = @typeInfo(Ptr);
        const const_ptr_info = std.builtin.Type{ .Pointer = .{
            .is_const = true,
            .size = ptr_info.Pointer.size,
            .child = ptr_info.Pointer.child,
            .sentinel = ptr_info.Pointer.sentinel,
            .alignment = ptr_info.Pointer.alignment,
            .is_volatile = ptr_info.Pointer.is_volatile,
            .is_allowzero = ptr_info.Pointer.is_allowzero,
            .address_space = ptr_info.Pointer.address_space,
        } };
        const ConstPtr = @Type(const_ptr_info);
        std.debug.assert(ptr_info == .Pointer);
        std.debug.assert(ptr_info.Pointer.size == .One);
        std.debug.assert(@typeInfo(ptr_info.Pointer.child) == .Struct);
        const impl = struct {
            fn initFields(ptr: *anyopaque) void {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                self.initFields();
            }

            fn deinitFields(ptr: *anyopaque) void {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                self.deinitFields();
            }

            fn apply(ptr: *const anyopaque, prefix: []const u8, result: *anyopaque, function: *const fn (*const anyopaque, []const u8, *anyopaque) void) void {
                const self: ConstPtr = @alignCast(@ptrCast(ptr));
                self.apply(prefix, result, function);
            }

            fn name(ptr: *const anyopaque) []const u8 {
                const self: ConstPtr = @alignCast(@ptrCast(ptr));
                return self.name();
            }

            fn parameters(ptr: *const anyopaque, recurse: bool) []Tensor {
                const self: ConstPtr = @alignCast(@ptrCast(ptr));
                return self.parameters(recurse);
            }

            fn namedParameters(ptr: *const anyopaque, recurse: bool) std.StringArrayHashMap(Tensor) {
                const self: ConstPtr = @alignCast(@ptrCast(ptr));
                return self.namedParameters(recurse);
            }

            fn registerParameter(ptr: *anyopaque, name_: []const u8, tensor: Tensor, requires_grad: bool) Tensor {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                return self.registerParameter(name_, tensor, requires_grad);
            }

            fn to(ptr: *anyopaque, device: torch.Device, kind: torch.Kind, non_blocking: bool) void {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                self.to(device, kind, non_blocking);
            }

            fn registerBuffer(ptr: *anyopaque, name_: []const u8, tensor: Tensor) Tensor {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                return self.registerBuffer(name_, tensor);
            }

            fn registerModule(ptr: *anyopaque, name_: []const u8, module: *Module) *Module {
                const self: Ptr = @alignCast(@ptrCast(ptr));
                return self.registerModule(name_, module);
            }
        };
        return Module{
            .ptr = @ptrCast(obj),
            .vtable = &.{
                .initFields = impl.initFields,
                .deinitFields = impl.deinitFields,
                .apply = impl.apply,
                .name = impl.name,
                .parameters = impl.parameters,
                .namedParameters = impl.namedParameters,
                .registerParameter = impl.registerParameter,
                .to = impl.to,
                .registerBuffer = impl.registerBuffer,
                .registerModule = impl.registerModule,
            },
        };
    }
};

pub fn ModuleGen(comptime T: type) type {
    return struct {
        pub fn initFields(self: *T) void {
            self.children_ = std.StringArrayHashMap(*Module).init(torch.global_allocator);
            self.parameters_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
            self.buffers_ = std.StringArrayHashMap(Tensor).init(torch.global_allocator);
        }

        pub fn deinitFields(self: *T) void {
            self.children_.deinit();
            self.parameters_.deinit();
            self.buffers_.deinit();
        }

        pub fn apply(
            self: *const T,
            prefix: []const u8,
            result: anytype,
            function: *const fn (*const anyopaque, []const u8, *anyopaque) void,
        ) void {
            function(self, prefix, result);
            for (self.children_.keys()) |key| {
                var child = self.children_.get(key).?;
                const new_prefix = std.fmt.allocPrint(torch.global_allocator, "{s}{s}.", .{ prefix, key }) catch unreachable;
                child.apply(new_prefix, result, function);
            }
        }

        pub fn name(self: *const T) []const u8 {
            if (comptime @hasField(T, "nameImpl")) {
                return self.nameImpl();
            }
            return @typeName(T);
        }

        pub fn parameters(self: *const T, recurse: bool) []Tensor {
            return self.namedParameters(recurse).values();
        }

        pub fn namedParametersApply(ptr: *const anyopaque, prefix: []const u8, result: *anyopaque) void {
            const self: *const T = @ptrCast(@alignCast(ptr));
            var result_: *std.StringArrayHashMap(Tensor) = @ptrCast(@alignCast(result));
            for (self.parameters_.keys()) |key| {
                const value = self.parameters_.get(key).?;
                const full_key = std.fmt.allocPrint(torch.global_allocator, "{s}{s}", .{ prefix, key }) catch unreachable;
                result_.put(full_key, value) catch unreachable;
            }
        }

        pub fn namedParameters(self: *const T, recurse: bool) std.StringArrayHashMap(Tensor) {
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

        // pub fn buffers(self: *const T,recurse: bool) std.ArrayList(Tensor) {
        //
        // }
        //
        // pub fn namedBuffers(self: *const T, recurse: bool) std.ArrayList(([]const u8, Tensor)) {
        //
        // }
        //
        // pub fn modules(self: *const T, include_self: bool) std.ArrayList(T) {
        //
        // }
        //
        // pub fn namedModules(self: *const T, include_self: bool) std.ArrayList(([]const u8, T)) {
        //
        // }
        //
        // pub fn children(self: *const T) std.ArrayList(T) {
        //
        // }
        //
        // pub fn namedChildren(self: *const T) std.ArrayList(([]const u8, T)) {
        //
        // }
        //
        // pub fn train(self: * T, on: bool) void {
        //
        // }
        //
        // pub fn eval(self: * T) void{
        //
        // }
        //
        // pub fn isTraining(self: *const T) bool {
        //     return false;
        // }
        //
        pub fn to(self: *T, device: torch.Device, kind: torch.Kind, non_blocking: bool) void {
            for (self.children_.keys()) |key| {
                var child = self.children_.get(key).?;
                _ = child.to(device, kind, non_blocking);
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
        //
        // pub fn zeroGrad(self: *T) bool  {
        //
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
        pub fn registerParameter(self: *T, name_: []const u8, tensor: Tensor, requires_grad_: bool) Tensor {
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

        pub fn registerBuffer(self: *T, name_: []const u8, tensor: Tensor) Tensor {
            _ = self.buffers_.put(name_, tensor) catch unreachable;
            return tensor;
        }

        pub fn registerModule(self: *T, name_: []const u8, module: *Module) *Module {
            _ = self.children_.put(name_, module) catch unreachable;
            return module;
        }
    };
}
