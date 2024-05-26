const std = @import("std");

fn findLibtorchLibrary(b: *std.Build, path: []const u8, lib: []const u8) ?[]const u8 {
    var dir = std.fs.openDirAbsolute(path, .{ .iterate = true }) catch return null;
    var iterator = dir.iterate();
    while (iterator.next() catch return null) |entry| {
        const lib_name = std.fmt.allocPrint(b.allocator, "lib{s}.so", .{lib}) catch return null;
        defer b.allocator.free(lib_name);
        if (std.mem.eql(u8, entry.name, lib_name)) {
            return std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ path, entry.name }) catch return null;
        }
    }
    iterator = dir.iterate();
    while (iterator.next() catch return null) |entry| {
        const lib_name = std.fmt.allocPrint(b.allocator, "lib{s}", .{lib}) catch return null;
        defer b.allocator.free(lib_name);
        if (std.mem.startsWith(u8, entry.name, lib_name)) {
            return std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ path, entry.name }) catch return null;
        }
    }
    return null;
}

pub fn build(b: *std.Build) void {
    const LIBTORCH_LIB = "/home/sreeraj/libtorch/lib";
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const torch_module = b.addModule("torch", .{
        // .name = "torch",
        .root_source_file = b.path("src/torch.zig"),
        .target = target,
        .optimize = optimize,
    });
    torch_module.addObjectFile(b.path("libtch/libtch.a"));
    torch_module.addIncludePath(b.path("libtch/"));
    torch_module.addLibraryPath(.{ .path = LIBTORCH_LIB });
    // exe.linkLibC();
    // exe.linkLibCpp();
    torch_module.linkSystemLibrary("pthread", .{});
    torch_module.linkSystemLibrary("m", .{});
    torch_module.linkSystemLibrary("dl", .{});
    torch_module.linkSystemLibrary("rt", .{});

    torch_module.addObjectFile(.{
        .path = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    });
    torch_module.addObjectFile(.{
        .path = "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
    });
    torch_module.addIncludePath(.{
        .path = "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    });
    torch_module.addIncludePath(.{
        .path = "/usr/include/c++/11",
    });
    torch_module.addIncludePath(.{
        .path = "/usr/include/x86_64-linux-gnu/c++/11",
    });

    const torch_libs = [_][]const u8{
        "c10",
        "torch",
        "torch_cpu",
        "torch_global_deps",
        // "torch_python",
        "gomp",
        "c10_cuda",
        "torch_cuda",
        "nvToolsExt",
        "cublas",
        "cudart",
        "cudnn.so",
        "cublasLt",
        "caffe2",
    };

    for (torch_libs) |lib| {
        const lib_path = findLibtorchLibrary(b, LIBTORCH_LIB, lib) orelse {
            std.log.err("Could not find libtorch library: {s}", .{lib});
            return;
        };
        torch_module.addObjectFile(.{
            .path = lib_path,
        });
    }

    const exe = b.addExecutable(.{
        .name = "torch_test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();
    exe.addIncludePath(b.path("libtch/"));
    exe.root_module.addImport("torch", torch_module);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_unit_tests.linkLibC();
    exe_unit_tests.addIncludePath(b.path("libtch/"));
    exe_unit_tests.root_module.addImport("torch", torch_module);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
