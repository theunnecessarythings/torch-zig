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
  const a = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
  a.print();
}
```
This will print a random tensor of shape 3x3.
```sh
 0.3737 -2.6723 -0.0300
 1.1182 -0.2807 -0.3038
 0.2195 -1.3325 -1.2250
[ CPUFloatType{3,3} ]
```

## Managing Memory
You chose zig, now you have to manage memory yourself. Now that's a good thing, right? Right? RIGHT?

Freeing a tensor is easy enough:
```zig
const a = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
a.free();
```

Except its not!!!
```zig
const a = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
const b = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
const c = a.add(&b).mul(&b).mulScalar(Scalar.float(2.0));
a.free();
b.free();
c.free();
// What about all the interemediate tensors created by the operations?
```

You can use `MemoryGuard` to manage memory for you. Internally it uses a TensorPool. Here is how you can use it:
```zig
{
  const guard = torch.MemoryGuard.init("temp_guard"); // Name identifies the TensorPool, so that you can have multiple pools
  defer guard.deinit(); // Frees all tensors created within the scope of the guard
  const a = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
  const b = Tensor.randn(&.{3, 3}, torch.FLOAT_CPU);
  const c = a.add(&b).mul(&b).mulScalar(Scalar.float(2.0));
  // All tensors will be freed when `guard` goes out of scope
}
```

torch-zig by default stores all tensors in the `default` pool. When you create a new MemoryGuard the default pool is temporarily switched to the new one and switched back to the previous one on its deinit.

For example, you can manually create and free pools using the TensorPool API
```zig
const size = torch.memory_pool.getPoolSize("default"); // Returns memory usage of the default pool
torch.memory_pool.addPool("temp_pool"); // Adds a new pool
torch.memory_pool.freePool("temp_pool"); // Frees the pool along with all tensors in it
torch.memory_pool.freeAll(); // Frees all pools
```

## NoGradGuard
You can use `NoGradGuard` to disable gradient computation. Here is how you can use it:
```zig
{
  const guard = torch.NoGradGuard.init();
  defer guard.deinit();
  
  var resnet = torch.vision.resnet18(1000, torch.FLOAT_CUDA); // Creates a resnet model on the GPU
  _ = resnet.forward(&input); // Forward pass
  // Gradient computation is disabled for all operations within the scope of the guard
}
```

## Available Neural Network Layers
- [x] Identity -> `torch.linear.Identity`
- [x] Linear -> `torch.linear.Linear`
- [x] Flatten -> `torch.linear.Flatten`
- [x] Unflatten -> `torch.linear.Unflatten`
- [x] Bilinear -> `torch.linear.Bilinear`
- [x] Conv1D -> `torch.conv.Conv1D`
- [x] Conv2D -> `torch.conv.Conv2D`
- [x] Conv3D -> `torch.conv.Conv3D`
- [ ] ConvTranspose1D -> `torch.conv.ConvTranspose1D`
- [ ] ConvTranspose2D -> `torch.conv.ConvTranspose2D`
- [ ] ConvTranspose3D -> `torch.conv.ConvTranspose3D`
- [x] Embedding -> `torch.embedding.Embedding`
- [x] Sequential -> `torch.module.Sequential`
- [x] BatchNormND -> `torch.norm.BatchNorm(D)`
- [x] InstanceNormND -> `torch.norm.InstanceNorm(D)`
- [x] LayerNorm -> `torch.norm.LayerNorm`
- [x] GroupNorm -> `torch.norm.GroupNorm`
- [x] Dropout -> `torch.functional.Dropout`

### Special `Functional` Layer for Functional API
Example ReLu and MaxPool2D layers using `Functional`
```zig
const relu_layer = Functional(Tensor.relu, .{}).init();
const maxpool_2d = Functional(Tensor.maxPool2d, .{ &.{ 3, 3 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false }).init();
```

## Available Vision Models
Just for shits and giggles, I have also implemented some vision models from torchvision. Here is how you can use them:
```zig
const torch = @import("torch");
const vision = torch.vision;

var _alexnet = alexnet.Alexnet.init(1000, torch.FLOAT_CUDA);
// Utility function to download weights from the huggingface model hub
const weights = try torch.utils.downloadFile("https://huggingface.co/theunnecessarythings/vision_models/resolve/main/alexnet.safetensors");
// Load safetensor weights
try _alexnet.base_module.loadFromSafetensors(weights);
```

For loading from `safetensors` I have implemented a basic safetensors reader in zig. It's working so far, so I am happy with it. If it breaks, your fault. Deal with it. (Or open an issue)

### Implemented Vision Models 
- [x] Alexnet 
- [x] ConvNext
- [x] DenseNet
- [x] EfficientNet - V2S, V2M, V2L -> Testing Fails
- [x] Inception
- [x] MnasNet
- [x] MobileNetV2
- [x] MobileNetV3
- [ ] RegNet
- [x] ResNet
- [x] ShuffleNetV2
- [x] SqueezeNet
- [x] SwinTransformer -> Testing Fails
- [x] VGG
- [x] VisionTransformer


## Example MNIST Training Code
```zig
// Module imports and Dataset loading omitted. See the full code in examples/mnist_train.zig
const Net = struct {
    base_module: *Module = undefined, // required -> for registering modules, parameters, etc.
    conv1: *Conv2D = undefined,
    conv2: *Conv2D = undefined,
    dropout: *Dropout = undefined,
    fc1: *Linear = undefined,
    fc2: *Linear = undefined,

    const Self = @This();

    pub fn init(options: torch.TensorOptions) *Self {
        var self = torch.global_allocator.create(Self) catch torch.utils.err(.AllocFailed);
        self.* = Self{};
        self.base_module = Module.init(self); // You can initialize base_module like this
        self.conv1 = Conv2D.init(.{ .in_channels = 1, .out_channels = 10, .kernel_size = .{ 5, 5 }, .tensor_opts = options });
        self.conv2 = Conv2D.init(.{ .in_channels = 10, .out_channels = 20, .kernel_size = .{ 5, 3 }, .tensor_opts = options });
        self.dropout = Dropout.init(0.5);
        self.fc1 = Linear.init(.{ .in_features = 400, .out_features = 50, .tensor_opts = options });
        self.fc2 = Linear.init(.{ .in_features = 50, .out_features = 10, .tensor_opts = options });
        self.reset();
        return self;
    }

    pub fn reset(self: *Self) void {
        // Reset usually initializes the parameters of the model
        self.conv1.reset();
        self.conv2.reset();
        self.dropout.reset();
        self.fc1.reset();
        self.fc2.reset();
        // Registering modules with the base_module is necessary for the optimizer to work
        _ = self.base_module.registerModule("conv1", self.conv1);
        _ = self.base_module.registerModule("conv2", self.conv2);
        _ = self.base_module.registerModule("dropout", self.dropout);
        _ = self.base_module.registerModule("fc1", self.fc1);
        _ = self.base_module.registerModule("fc2", self.fc2);
    }

    pub fn deinit(self: *Self) void {
        self.base_module.deinit();
        self.conv1.deinit();
        self.conv2.deinit();
        self.dropout.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
        torch.global_allocator.destroy(self);
    }

    pub fn forward(self: *Self, input: *const Tensor) Tensor {
        const x = input.reshape(&.{ -1, 1, 28, 28 });
        var y = self.conv1.forward(&x).maxPool2d(&.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false).relu();
        y = self.conv2.forward(&y);
        y = self.dropout.forward(&y);
        y = y.maxPool2d(&.{ 2, 2 }, &.{ 2, 2 }, &.{ 0, 0 }, &.{ 1, 1 }, false).relu();
        y = y.view(&.{ -1, 400 });
        y = self.fc1.forward(&y).relu();
        y = self.dropout.forward(&y);
        y = self.fc2.forward(&y);
        return y;
    }
};

pub fn main() !void {
    torch.Utils.manualSeed(1);
    const device_type = torch.Device.cudaIfAvailable();
    if (device_type.isCuda()) {
        std.debug.print("CUDA available! Training on GPU.\n", .{});
    } else {
        std.debug.print("Training on CPU.\n", .{});
    }
    const options = if (device_type.isCuda()) torch.FLOAT_CUDA else torch.FLOAT_CPU;

    const model = Net.init(options);
    const ds = try loadDir();
    std.debug.print("Loaded data.\n", .{});
    std.debug.print("Train images size: {any}\n", .{ds.train_images.size()});
    std.debug.print("Train labels size: {any}\n", .{ds.train_labels.size()});
    std.debug.print("Test images size: {any}\n", .{ds.test_images.size()});
    std.debug.print("Test labels size: {any}\n", .{ds.test_labels.size()});

    var opt = torch.COptimizer.adam(1e-3, 0.99, 0.999, 1e-5, 1e-5, false);
    opt.addParameters(model.base_module.parameters(true), 0);

    for (1..50_000) |step| {
        var mem_guard = torch.MemoryGuard.init("mnist");
        defer mem_guard.deinit();

        const batch_indices = Tensor.randintLow(0, 50_000, &.{TRAIN_BATCH_SIZE}, torch.INT64_CUDA);
        const batch_images = ds.train_images.indexSelect(0, &batch_indices);
        const batch_labels = ds.train_labels.indexSelect(0, &batch_indices);
        opt.zeroGrad();
        var loss = model.forward(&batch_images).crossEntropyForLogits(&batch_labels);
        loss.backward();
        opt.step();

        // Evaluate the model
        var guard = torch.NoGradGuard.init();
        defer guard.deinit();
        const test_accuracy = model.forward(&ds.test_images).accuracyForLogits(&ds.test_labels);
        std.debug.print("Iteration: {d}, Loss: {d}, Test accuracy: {d}\r", .{ step, loss.toFloat(), test_accuracy.toFloat() });
    }
}
```
