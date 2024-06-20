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
        .root_source_file = b.path("src/torch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    torch_module.addObjectFile(b.path("libtch/libtch.a"));
    torch_module.addIncludePath(b.path("libtch/"));
    torch_module.addLibraryPath(.{ .cwd_relative = LIBTORCH_LIB });
    torch_module.linkSystemLibrary("pthread", .{});
    torch_module.linkSystemLibrary("m", .{});
    torch_module.linkSystemLibrary("dl", .{});
    torch_module.linkSystemLibrary("rt", .{});

    torch_module.addObjectFile(.{
        .cwd_relative = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    });
    torch_module.addObjectFile(.{
        .cwd_relative = "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
    });
    torch_module.addIncludePath(.{
        .cwd_relative = "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    });
    torch_module.addIncludePath(.{
        .cwd_relative = "/usr/include/c++/11",
    });
    torch_module.addIncludePath(.{
        .cwd_relative = "/usr/include/x86_64-linux-gnu/c++/11",
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
            .cwd_relative = lib_path,
        });
    }

    const exe = b.addExecutable(.{
        .name = "torch_test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        // .use_llvm = false,
        // .use_lld = false,
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

    const example_option = b.option([]const u8, "example", "Run example") orelse return;
    const options = b.addOptions();
    options.addOption([]const u8, "example", example_option);

    const example = b.addExecutable(.{
        .name = "mnist",
        .root_source_file = .{ .cwd_relative = std.fmt.allocPrint(b.allocator, "examples/{s}.zig", .{example_option}) catch unreachable },
        .target = target,
        .optimize = optimize,
    });
    example.linkLibC();
    example.addIncludePath(b.path("libtch/"));
    example.root_module.addImport("torch", torch_module);
    b.installArtifact(example);
    const example_run = b.addRunArtifact(example);
    const example_step = b.step("example", "Run example");
    example_step.dependOn(&example_run.step);
}
