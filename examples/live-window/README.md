# Neo Live Window

Run:

```powershell
cargo run -p neo-live-window -- examples/live-window/live.neo
```

For bounded stress runs:

```powershell
cargo run -p neo-live-window -- examples/live-window/live.neo --width 960 --height 540 --seconds 3
```

For completed-kernel throughput runs:

```powershell
cargo run -p neo-live-window --release -- examples/live-window/live.neo --width 960 --height 540 --seconds 3 --mode kernel-throughput --sample-every 256
```

To request a visible preview rate directly while keeping kernel throughput
uncapped:

```powershell
cargo run -p neo-live-window --release -- examples/live-window/live.neo --width 3440 --height 1440 --seconds 3 --mode kernel-throughput --present-target-fps 60
```

The app opens a Win32 window, launches the Neo `image` kernel into a CUDA-owned
device buffer, downloads pixels to a reused host buffer, and presents them with a
no-interop Win32 presenter. There is no CUDA/OpenGL, CUDA/D3D, Vulkan external
memory, registered graphics buffer, or mapped graphics resource path in this
example.

Expected kernel ABI:

```neo
kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame)
```

For this presenter, `pixels` is BGRA8 byte order so the Win32 DIB path can use
the kernel output directly without a CPU swizzle.

Edit `live.neo` while the app is running to hot reload. If a compile fails, the
window keeps rendering the last good kernel and the error is printed to stderr.
FPS is logged to the console once per second.

`--mode kernel-throughput` reports completed `kernel_fps` separately from
sampled readback FPS and visible present FPS. The window updates from sampled
completed frames, not every counted kernel. Use either `--sample-every N` for
kernel-count sampling or `--present-target-fps N` for a wall-clock preview rate.
