# Neo

Neo is a tiny v0 graphics-kernel language prototype for explicit, SIMD-oriented
GPU work on NVIDIA hardware. The current backend lowers `.neo` kernels to CUDA C,
compiles them with NVRTC through Rust, launches them with the CUDA driver API,
and writes a compute-generated image.

```powershell
cargo run -p neo-cli -- doctor
cargo run -p neo-cli -- compile examples/gradient.neo --out target/gradient.cu
cargo run -p neo-cli -- run examples/gradient.neo --out target/gradient.png
cargo run -p neo-live-window -- examples/live-window/live.neo
cargo run -p neo-live-window -- examples/live-window/live.neo --width 960 --height 540 --seconds 3
cargo run -p neo-live-window --release -- examples/live-window/live.neo --width 960 --height 540 --seconds 3 --mode kernel-throughput --sample-every 256
cargo run -p neo-live-window --release -- examples/live-window/live.neo --width 3440 --height 1440 --seconds 3 --mode kernel-throughput --present-target-fps 60
```

The compiler is intentionally small in v0: kernel signatures are parsed as Neo,
while kernel bodies use a modern C-like subset with targeted rewrites for `let`,
Neo scalar/vector names, and builtins such as `thread_id()` and `block_id()`.

## v0 Language Shape

```neo
kernel fn image(global u8* pixels, u32 width, u32 height) {
    let x: u32 = block_id().x * block_dim().x + thread_id().x;
}
```

Supported v0 concepts:
- `kernel fn` entrypoints with explicit parameters.
- Address-space markers: `global`, `shared`, `local`.
- Scalar/vector names: `bool`, `i32`, `u8`, `u32`, `f32`, `vec2f`, `vec3f`, `vec4f`.
- SIMD launch builtins: `thread_id()`, `block_id()`, `block_dim()`, `grid_dim()`.
- Explicit host launch through `neo-runtime`; no hidden global graphics state.

## Runtime Notes

`neo doctor` reports CUDA driver and NVRTC discovery. On Windows, Neo checks
`PATH`, `CUDA_PATH`, installed CUDA Toolkit folders, and a couple of NVIDIA app
SDK locations before compiling. A full CUDA Toolkit install remains the cleanest
setup, but v0 can use a compatible `nvrtc64_120_0.dll` already present on this
machine.

## Live Window

`neo-live-window` opens a Win32 window, launches the Neo `image` kernel into a
CUDA-owned device buffer every frame, downloads into a reused host buffer, and
presents through a no-interop Win32 presenter. This path intentionally avoids
CUDA/OpenGL, CUDA/D3D, Vulkan external memory, registered graphics buffers, and
mapped graphics resources. The live kernel ABI is:

```neo
kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame)
```

The app watches the `.neo` file and hot reloads it. If compilation fails, it
keeps rendering the last good kernel and prints the error to stderr. FPS is
logged to the console once per second. The live presenter expects BGRA8 pixels
so Win32 can consume the buffer without a CPU color swizzle.

Use `--mode kernel-throughput` to measure completed Neo kernels separately from
sampled readbacks and visible presents. This is the intended mode for the 10k
kernel FPS target. Use `--present-target-fps N` when you want a specific visible
preview cadence while leaving kernel throughput uncapped.
