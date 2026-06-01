# Neo

Neo is a Rust-based, NVIDIA-first graphics kernel language and runtime. It began
as a tiny `.neo` to CUDA C prototype, and is now growing into an explicit,
SIMD-oriented graphics stack: kernels, GPU-owned buffers, D3D12/CUDA interop
presentation, mesh and instance resources, AoSoA layouts, and live stress demos.

The design goal is still the same: no hidden global graphics state. Host code owns
resources and launch dimensions; Neo kernels receive explicit pointers and
metadata. The friendly app layer can make common workflows concise, but the
low-level handles stay available.

## Current Highlights

- `.neo` kernels lower to CUDA C and compile at runtime with NVRTC.
- `neo-runtime` provides CUDA context/module/kernel launch helpers, device
  buffers, pinned buffers, CUDA fences/streams/graphs, image output, mesh buffers,
  instance buffers, and D3D12/CUDA external-memory interop.
- `neo-live-window` runs live kernels in a Win32 window with hot reload, FPS
  logging, D3D12 interop presentation, fallback upload presenters, kernel caps,
  presentation caps, idle/visibility throttling, and stress modes.
- `MeshBuffer`, `InstanceBuffer`, `StructuredBuffer`, `DataLayout::AoS`,
  `DataLayout::SoA`, and `DataLayout::AoSoA { group_size }` make layout explicit.
- The current flagship stress path renders an 8,388,608 instance
  `256x256x128` 3D quad field through D3D12/CUDA interop, with AoSoA32/AoSoA64
  comparison and GPU-side debug heatmaps.
- `neo-app` is the friendlier Rust layer: builder-style configuration on top of
  the runtime, not a replacement for it.

## Quick Commands

Basic compiler/runtime checks:

```powershell
cargo run -p neo-cli -- doctor
cargo run -p neo-cli -- compile examples/gradient.neo --out target/gradient.cu
cargo run -p neo-cli -- run examples/gradient.neo --out target/gradient.png
```

Live plasma/image kernel:

```powershell
cargo run -p neo-live-window --release -- examples/live-window/live.neo --presenter d3d12-interop
```

Kernel throughput with visible interop presentation:

```powershell
cargo run -p neo-live-window --release -- examples/live-window/live.neo --width 3440 --height 1440 --mode kernel-throughput --presenter d3d12-interop --kernel-target-fps 1000 --present-target-fps 1000 --no-hot-reload
```

Flagship 8M 3D quad stress launcher:

```powershell
cargo run -p neo-quad-stress-3d --release
```

Tweakable 8M stress command with debug views:

```powershell
cargo run -p neo-live-window --release -- examples/stress-quads/three_d_instances.neo --mode instance-stress --presenter d3d12-interop --kernel-target-fps 1000 --present-target-fps 1000 --max-inflight 8 --present-ring 8 --instance-grid 256x256x128 --instance-stress-variant tiled --instance-layout aosoa32 --instance-debug-view iterations --render-policy auto --no-hot-reload --interop-fallback fail
```

Debug views:

- `--instance-debug-view off`: normal render.
- `--instance-debug-view tile-range`: tile candidate layer-window heatmap.
- `--instance-debug-view iterations`: per-pixel traversal-count heatmap.
- `--instance-debug-view hit-miss`: hit/miss/early-background diagnostic colors.

Layout comparison:

- `--instance-layout aosoa32`: default NVIDIA warp-sized AoSoA grouping.
- `--instance-layout aosoa64`: two-warp grouping for cache/neighbor experiments.

## Language Shape

Neo source files contain explicit kernel entrypoints:

```neo
kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {
    let x: u32 = block_id().x * block_dim().x + thread_id().x;
    let y: u32 = block_id().y * block_dim().y + thread_id().y;
}
```

Supported concepts include:

- `kernel fn` entrypoints with explicit parameters.
- Address-space markers: `global`, `shared`, `local`.
- Scalar/vector names: `bool`, `i32`, `u8`, `u32`, `f32`, `vec2f`, `vec3f`,
  `vec4f`, `u8x4_unorm`.
- SIMD launch builtins: `thread_id()`, `block_id()`, `block_dim()`, `grid_dim()`.
- Block synchronization via `block_barrier()`.
- Declarative layout syntax such as `layout aosoa(32) Instance { ... }`.

The parser/lowering layer is intentionally compact. Kernel bodies are a modern
C-like subset with targeted Neo rewrites rather than a full CUDA clone. Runtime
preludes provide mesh and instance helpers when compiling through `neo-runtime`.

## Runtime And Presentation

The fast live path is Windows + NVIDIA + D3D12/CUDA interop. Neo owns linear BGRA
frame buffers in shared GPU memory; CUDA writes them, D3D12 copies/presents them,
and there is no CPU readback or upload in the interop path.

Fallback no-interop presenters remain useful for comparison and debugging:

- `--presenter d3d12`: D3D12 flip-model upload path.
- `--presenter d3d11`: D3D11 flip-model upload path.
- `--presenter gdi`: simple CPU fallback.

`--mode kernel-throughput` separates completed CUDA kernel FPS from sampled
readbacks and visible presentation. `--kernel-target-fps` caps GPU work;
`--present-target-fps` caps visible updates. `--render-policy auto` can pause or
throttle when the window is minimized, occluded, or the structured stress scene
is off-screen.

## Resource Model

Neo's modern buffer direction is explicit and runtime-owned:

- `MeshBuffer` is the replacement for loose vertex/index buffers.
- `InstanceBuffer` stores repeated instance data on the GPU.
- `StructuredBuffer` and `DataLayout` make AoS/SoA/AoSoA choices visible.
- AoSoA defaults to 32 lanes for NVIDIA-friendly warp access, with explicit
  alternatives such as 64 lanes when benchmarking says it helps.

The stress renderer uses one base quad mesh plus millions of GPU-resident
instances. The specialized tiled kernels read known AoSoA streams directly for
the hot path, while generic helper-based kernels remain available for
compatibility and experiments.

## Workspace Map

- `crates/neo-lang`: lexer/parser, AST, diagnostics, and CUDA C lowering.
- `crates/neo-runtime`: CUDA/NVRTC runtime, buffers, mesh/instance resources,
  layout packing, D3D12 interop, and launch helpers.
- `crates/neo-cli`: doctor/compile/run commands.
- `crates/neo-app`: friendly Rust builder layer over the runtime/live stack.
- `examples/live-window`: main live window, throughput, mesh, and instance stress
  frontend.
- `examples/stress-quads`: 2D and 3D stress kernels.
- `examples/stress-quads-3d`: fixed flagship 8M quad stress launcher.
- `examples/friendly-demo`: concise app-layer demo.

## Setup Notes

Neo currently targets Windows + NVIDIA first. A working NVIDIA driver is required,
and NVRTC must be discoverable. `neo doctor` checks `PATH`, `CUDA_PATH`, installed
CUDA Toolkit folders, and common NVIDIA SDK locations.

Recommended verification before pushing changes:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## Aspirations

Neo is heading toward a graphics language/runtime where high-level convenience
does not erase low-level control:

- friendly app construction without hiding ownership,
- first-class layout and data-oriented GPU resources,
- compute-first rendering experiments that can later meet raster/graphics API
  interop cleanly,
- specialization where performance matters,
- diagnostics and heatmaps that make GPU work visible instead of mystical.

The v0 contract is still preserved in [docs/v0.md](docs/v0.md), but the root
README now tracks the living project: a fast, explicit, weirdly fun graphics
language taking shape one measured stress test at a time.
