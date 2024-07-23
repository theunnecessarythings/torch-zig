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
    const LIBTORCH = b.option([]const u8, "LIBTORCH", "Path to libtorch") orelse "/home/sreeraj/libtorch";
    const CUDA_HOME = b.option([]const u8, "CUDA_HOME", "Path to CUDA") orelse "/usr/local/cuda";
    const CXX_COMPILER = b.option([]const u8, "CXX_COMPILER", "C++ compiler") orelse "g++";
    const LIBTORCH_LIB = b.fmt("{s}/lib", .{LIBTORCH});

    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const CXX_FLAGS = [_][]const u8{
        "-std=c++17",
        // "-stdlib=libstdc++",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-DUSE_C10D_GLOO",
        "-DUSE_DISTRIBUTED",
        "-DUSE_RPC",
        "-DUSE_TENSORPIPE",
        "-c",
    };

    const cmd_1 = b.addSystemCommand(&.{CXX_COMPILER});
    cmd_1.addArgs(&CXX_FLAGS ++ [_][]const u8{
        "torch_api.cpp",
        "-o",
        "torch_api.o",
        b.fmt("-I{s}/include/", .{LIBTORCH}),
        b.fmt("-I{s}/include/torch/csrc/api/include/", .{LIBTORCH}),
        b.fmt("-I{s}/include/", .{CUDA_HOME}),
    });
    cmd_1.setCwd(b.path("libtch/"));

    const cmd_2 = b.addSystemCommand(&.{CXX_COMPILER});
    cmd_2.addArgs(&CXX_FLAGS ++ [_][]const u8{
        "torch_api_generated.cpp",
        "-o",
        "torch_api_generated.o",
        b.fmt("-I{s}/include/", .{LIBTORCH}),
        b.fmt("-I{s}/include/torch/csrc/api/include/", .{LIBTORCH}),
        b.fmt("-I{s}/include/", .{CUDA_HOME}),
    });
    cmd_2.setCwd(b.path("libtch/"));
    cmd_1.step.dependOn(&cmd_2.step);
    const static_lib_cmd = b.addSystemCommand(&.{
        "ar",
        "crs",
        "libtch/libtch.a",
        "libtch/torch_api.o",
        "libtch/torch_api_generated.o",
    });
    static_lib_cmd.step.dependOn(&cmd_2.step);

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
    torch_module.addObjectFile(.{
        .cwd_relative = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
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
        .root_source_file = b.path("src/test_main.zig"),
        .target = target,
        .optimize = optimize,
        // .strip = true,
        // .use_llvm = false,
        // .use_lld = false,
    });

    exe.linkLibC();
    exe.linkLibCpp();
    exe.addIncludePath(b.path("libtch/"));
    exe.root_module.addImport("torch", torch_module);

    b.installArtifact(exe);

    const build_lib_step = b.option(bool, "lib", "Build libtch.a") orelse false;
    if (build_lib_step) {
        exe.step.dependOn(&static_lib_cmd.step);
    } else {
        std.fs.cwd().access("libtch/libtch.a", .{}) catch {
            std.log.info("libtch not built, building it!!!", .{});
            std.log.info("To force libtch rebuild if changes are made to `torch_api.cpp` or `torch_api_generated.cpp`, run `zig build -Dlib=true`\n", .{});
            exe.step.dependOn(&static_lib_cmd.step);
        };
    }
    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/test_main.zig"),
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
