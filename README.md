# torch-zig
Zig bindings for the C++ api of PyTorch.

> [!CAUTION]
> The code is crap, JUST DON'T USE IT. One would also call it an `alpha` version.
> I AM NOT RESPONSIBLE FOR ANY DAMAGE CAUSED BY THIS CODE. THAT INCLUDES EMOTIONAL ONES TOO.

## Usage (As Is)
Now that you have been warned, here is how you can use it:

To build the library, you need to have Zig installed. Then you can run the following command:
```sh
zig build lib # Builds the C++ bindings to a static library, Only needs to be run once
zig build run # Runs the test main program 
zig build -Dexample=mnist_train run # Runs the mnist training example
```
- You can write your own `main` and modify the `build.zig` as you like.
- Or you can add your own examples in the `examples` directory and run them using the `-Dexample` flag.

## Usage (As a Library)
Look, I really don't understand how the zig dependency system works, but here is how you can use this library in your project:

```
# Fetch the library
zig fetch --save https://github.com/theunnecessarythings/torch-zig/tarball/master
```

Then in your `build.zig` file, you can add the following code:

```zig
const comp: []const u8 = "clang++";
const CUDA_HOME: []const u8 = "/usr/local/cuda";
const LIBTORCH: []const u8 = "/path/to/libtorch";
const torch_dep = b.dependency("torch-zig", .{
    .target = target,
    .optimize = optimize,
    // .lib = true, // Force rebuild the library
    // .CXX_COMPILER = comp, // Defaults to g++
    // .CUDA_HOME = CUDA_HOME, // Defaults to /usr/local/cuda
    .LIBTORCH = LIBTORCH,
});
exe.step.dependOn(&torch_dep.artifact("torch-zig").step);
const torch = torch_dep.module("torch");
exe.root_module.addImport("torch", torch);

```

And then you can use the library in your code like this:

```zig
const torch = @import("torch");
const Tensor = torch.Tensor;

pub fn main() !void {
  const a = Tensor.randn(&.{2, 3}, torch.FLOAT_CPU);
  a.print();
}
```
This will print a random tensor of shape 2x3.
```sh
 0.3737 -2.6723 -0.0300
 1.1182 -0.2807 -0.3038
 0.2195 -1.3325 -1.2250
[ CPUFloatType{3,3} ]
```

