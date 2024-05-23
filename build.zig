const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "torch-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.addObjectFile(b.path("libtch/libtch.a"));
    exe.addIncludePath(b.path("libtch/"));
    exe.addLibraryPath(.{ .path = "/home/sreeraj/libtorch/lib" });
    // exe.linkLibC();
    // exe.linkLibCpp();
    exe.linkSystemLibrary("pthread");
    exe.linkSystemLibrary("m");
    exe.linkSystemLibrary("dl");

    exe.addObjectFile(.{
        .path = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    });
    exe.addObjectFile(.{
        .path = "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
    });
    exe.addIncludePath(.{
        .path = "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    });
    exe.addIncludePath(.{
        .path = "/usr/include/c++/11",
    });
    exe.addIncludePath(.{
        .path = "/usr/include/x86_64-linux-gnu/c++/11",
    });

    exe.linkSystemLibrary("c10");
    exe.linkSystemLibrary("torch");
    exe.linkSystemLibrary("torch_cpu");
    exe.linkSystemLibrary("torch_global_deps");
    exe.linkSystemLibrary("torch_python");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
