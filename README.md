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
  `256x256x128` 3D quad field through Neo's CUDA tiled renderer and D3D12/CUDA
  interop presentation, with AoSoA32/AoSoA64 comparison and GPU-side debug
  heatmaps.
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

Hardware-raster stress comparisons:

```powershell
cargo run -p neo-quad-stress-3d --release -- --draw-backend draw-execution
cargo run -p neo-quad-stress-3d --release -- --draw-backend hardware-raster
cargo run -p neo-quad-stress-3d --release -- --draw-backend hardware-raster-draw-all
```

`draw-execution` is Neo's friendly name for the optimized hardware-backed draw
path: Neo compute builds the indirect draw, uses `--cull-order stable-dense`,
and opts into `--visibility projected-size` with `--min-projected-pixels 0.85`.
That means tiny projected instances are skipped instead of submitting invisible
detail to fixed-function raster. `hardware-raster` is the concrete backend
comparison spelling; `raster` and `raster-culled` remain aliases for this
optimized path.
`hardware-raster-draw-all` is the explicit brute-force baseline for measuring
how expensive it is to ask D3D12 to draw all 8M instances; `raster-draw-all` and
`raster-baseline` remain compatibility aliases. This baseline passes
`--draw-depth off` to avoid paying for depth clears/tests/writes while measuring
raw all-instance submission. Use `--draw-depth on` when you want stable depth
ordering instead, or `--draw-depth auto` to let draw-all run without depth while
compute-culled draw execution keeps depth enabled. The default launcher still
uses the faster CUDA tiled path: `cargo run -p neo-quad-stress-3d --release`, or
explicitly `--draw-backend primary-neo` / `--draw-backend cuda-tiled`.
The older `--renderer` flag remains a compatibility alias.
When the launcher expands to `neo-live-window`, the hardware draw comparison now
uses `--mode draw-stress`; the older `--mode raster-stress` spelling remains
accepted.

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

## Modern Draw Vocabulary

Neo's draw direction is intentionally not "OpenGL, but renamed." The friendly
layer describes graphics work as explicit owned pieces:

- `GeometryStream`: mesh vertices and indices.
- `InstanceStream`: repeated GPU instance data and layout.
- `MaterialKernel`: a Neo shader/kernel entrypoint contract, such as hardware
  vertex/fragment stages or a CUDA tiled instance kernel.
- `Target`: the window or named render target dimensions.
- `DrawPolicy` / `DrawPolicyConfig`: how work is submitted, such as `DrawAll`
  or `ComputeCulled`, plus policy details such as depth mode, cull ordering, and
  visibility thresholds.
- `DrawPass`: the runtime pass destination, currently a `Target` wrapper; older
  `RasterPass`, `RasterTarget`, and `RenderTarget` spellings remain aliases.

Material kernels, policies, configs, draw specs, graph draws, run plans, and
runtime draws all report their resolved backend explicitly with `backend()`.
Draw-shaped objects also report their unified policy with `policy_config()`.
Draw policies, backends, cull orders, and visibility modes also expose stable
lowercase labels through `label()` and `Display`, such as `cuda-tiled`,
`hardware-raster`, and `compute-culled`.
Use `CullOrder`, `VisibilityMode`, and `DEFAULT_MIN_PROJECTED_MILLIPIXELS` for
new app/runtime code; `RasterCullOrder`, `RasterVisibilityMode`, and
`DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS` remain compatibility spellings.
`DrawBackend::primary_neo()` currently returns `cuda-tiled`; this is the
intentional default for dense structured instance workloads. App code can set
`draw_backend_preference(DrawBackendPreference::DrawExecution)` when it wants
Neo's explicit draw-execution path, `DrawBackendPreference::HardwareRaster`
when it wants the concrete fixed-function comparison backend, or
`DrawBackendPreference::FirstConfigured` when it wants old-fashioned graph order
to win. `RendererPreference` and `renderer_preference(...)` remain compatibility
spellings.
`DrawPolicyConfig` mirrors those helpers with `policy_label()`,
`backend_label()`, `depth_label()`, `uses_depth()`, `cull_order_label()`,
`visibility_label()`, and `min_projected_pixels()`.
`MaterialAbi` / runtime `MaterialKernelKind` expose stable labels too, such as
`direct-instance-color` or `cuda-tiled`, so tools can distinguish material
contracts from execution backends.
Material contracts expose `is_draw_execution()` for hardware-backed vertex /
fragment execution and `is_cuda_tiled()` for Neo's CUDA tiled material kernels.
Runtime `MaterialKernelKind::DrawExecution` is the primary material-kind name
for vertex/fragment execution; the older `MaterialKernelKind::HardwareRaster`
still works as a compatibility kind and maps to the same hardware-raster backend.
At the app layer, `MaterialExecutionKind` answers the same coarse question:
`DrawExecution` for vertex/fragment materials, or `CudaTiled` for Neo's primary
CUDA tiled material path. Detailed `MaterialAbi` values such as
`ComputeCulledInstanceColor` remain available after that first classification.
`neo-app::DrawContract` snapshots the user-facing draw shape as names for the
`GeometryStream`, optional `InstanceStream`, `MaterialKernel`, `Target`,
`DrawPolicyConfig`, and backend. Configured specs keep unresolved details as
optional, while resolved draw graphs and run plans also report instance
count/layout, material ABI labels, and target dimensions.
`DrawSpec`, `DrawGraph`, and `DrawGraphDraw` expose accessor methods for the
same questions, so tooling does not have to couple itself to backend-specific
field names.
Contract helpers such as `depth_label()`, `uses_depth()`,
`instance_layout_label()`, and `target_dimensions()` make those resolved fields
straightforward for tooling and logs.
`neo_runtime::DrawContract` snapshots materialized draw handles as geometry
counts, optional instance count/layout, material label/kind, target size,
`DrawPolicyConfig`, and backend; it is available from runtime `DrawRecipe`
values, `DrawExecution`, `CudaDraw`, and app `RuntimeDraw` escape hatches.
Material binding contracts use `MaterialBindingKind::DrawParams` for the shared
draw parameter block; `MaterialBindingKind::RasterParams` remains a compatibility
alias for older hardware-raster callers and resolves to the same binding.
The runtime fixed-function comparison surface uses `DrawDevice` and
`DrawPipeline` as the neutral resource names.
`RasterDraw` / `RasterDrawBuilder` remain compatibility spellings for the same
hardware-backed draw execution path.
Live throughput logs use the same neutral terms, for example `draw_policy`,
`cull_order`, `draw_visibility`, and `min_projected_px`.
The live-window hardware-raster executor also exposes
`HardwareRasterDrawContract`, so even backend-specific execution plans can be
inspected through draw name, `GeometryStream`, `InstanceStream`, layout,
`MaterialKernel`, `Target`, `DrawPolicyConfig`, and backend.
It mirrors the same label and dimension helpers used by app/runtime contracts.
New integration code can use the neutral live-window plan types
`DrawExecutionPlan`, `GeometryStreamPlan`, `InstanceStreamPlan`,
`MaterialKernelPlan`, `MaterialKernelPlanKind`, `TargetPlan`,
`DrawPolicyPlan`, `DrawCullOrder`, and `DrawVisibilityMode`; the older
`HardwareRaster*` names remain compatibility aliases for the D3D12 executor.
`DrawExecutionPlan` also exposes `draw_name()`, `geometry_stream()`,
`instance_stream()`, `material()`, `target()`, `draw_policy()`, `policy()`,
`cull_order()`, `visibility()`, `min_projected_pixels()`, and
`policy_config()` so executor callers can inspect the same draw shape without
reaching into fields.

The app layer can express that directly:

```rust
NeoApp::new()
    .geometry_stream("quad", GeometryBuilder::quad().colored())
    .instance_stream_aosoa32("instances", InstanceGrid::new(32, 32, 8))
    .instance_material_kernel("quad-material", "quad_vs", "quad_fs")
    .compute_cull_with_order(
        "raster_cull",
        "examples/stress-quads/hardware_raster.neo",
        CullOrder::StableDense,
    )
    .draw_compute_culled_projected(
        "main",
        "quad",
        "instances",
        "quad-material",
        TargetSpec::window(),
        850,
    );
```

`material_kernel(...)`, `instance_material_kernel(...)`, and
`direct_instance_material_kernel(...)` are the concise defaults. When code wants
to name the execution model explicitly, the matching
`draw_execution_material_kernel(...)`,
`draw_execution_instance_material_kernel(...)`, and
`direct_draw_execution_instance_material_kernel(...)` builders produce the same
MaterialKernel contracts.

For the faster CUDA tiled path, the same stream/material/target vocabulary uses a
different draw policy:

```rust
NeoApp::new()
    .geometry_stream("quad", GeometryBuilder::quad().colored())
    .instance_stream_aosoa32("instances", InstanceGrid::new(256, 256, 128))
    .cuda_tiled_material_kernel(
        "quad-material",
        "instance_raster",
        "examples/stress-quads/three_d_instances.neo",
    )
    .draw_cuda_tiled(
        "main",
        "quad",
        "instances",
        "quad-material",
        TargetSpec::window(),
    );
```

Older `draw_indirect*` helpers remain as compatibility spelling, but new code
should prefer the stream/material/target/policy vocabulary above. Hardware
raster is a backend; CUDA tiled is currently the primary renderer for the dense
tiny-quad stress workload.

For lower-level control, `neo-app` can materialize the friendly draw graph into
runtime-owned resources and a policy-selected draw recipe. The app-level
`draw_run_plan()` exposes the five owned pieces before choosing a low-level
backend:

```rust
let mut app = NeoApp::new()
    .geometry_stream("quad", GeometryBuilder::quad().colored())
    .instance_stream_aosoa32("instances", InstanceGrid::new(128, 128, 64))
    .cuda_tiled_material_kernel(
        "quad-material",
        "instance_raster",
        "examples/stress-quads/three_d_instances.neo",
    )
    .draw_cuda_tiled(
        "main",
        "quad",
        "instances",
        "quad-material",
        TargetSpec::window(),
    );

let plan = app.draw_run_plan()?.expect("configured draw");
let geometry = plan.geometry();
let instances = plan.instances();
let material = plan.material();
let target = plan.target();
let policy = plan.policy_config();
let backend = plan.backend();
let contract = plan.contract();

let resources = app.build_runtime_draw_resources()?;
let graph_draw = resources.graph.draw("main")?;
let draw = resources.draw("main")?;
```

Resolved app plans implement `neo_app::DrawPlanRecipe`, so tooling can inspect
the same owned pieces before any GPU resources are built.
Use `app.draw_execution_run_plan()` when you want the hardware-backed execution
plan directly; `app.hardware_raster_run_plan()` and `app.raster_run_plan()`
remain compatibility spellings.
`neo-live-window` accepts that plan through
`run_from_args_with_draw_execution_plan(...)`; the shorter
`run_from_args_with_draw_plan(...)` and older raster spelling remain compatibility
entrypoints.
The runtime draw returned by `resources.draw("main")` is either a
`neo_runtime::CudaDraw` or hardware-backed `neo_runtime::DrawExecution`, chosen
from `DrawPolicy`, while still exposing the underlying `GeometryStream`, `InstanceStream`,
`MaterialKernel`, `Target`, and `DrawPolicyConfig` handles for advanced control.
Use `resources.draw_execution("main")` when you want the hardware-backed
execution handle directly; `resources.raster_draw("main")` remains the old
compatibility spelling.
Both concrete runtime draw types, plus the app-level `RuntimeDraw` wrapper,
implement `neo_runtime::DrawRecipe`, the common runtime contract for those owned
draw pieces.
`RuntimeDraw` also exposes `as_draw_execution()` and `as_cuda_draw()` so tools
can take a concrete handle without matching on older backend-shaped enum
variants.
New app code should prefer `DrawRunPlan::DrawExecution` and
`RuntimeDraw::DrawExecution` for the hardware-backed execution path;
`DrawRunPlan::HardwareRaster` and `RuntimeDraw::Raster` remain compatibility
variants for older callers.
The app-level plan reports the resolved backend explicitly with
`DrawRunPlan::backend()`, and materialized runtime draws report the same choice
through `RuntimeDraw::backend()` / `neo_runtime::DrawBackend`. That lets tools
prefer CUDA tiled rendering for dense microgeometry while still keeping hardware
raster as an intentional comparison or conventional-mesh backend.
`DrawRunPlan` exposes `as_cuda_plan()` and `as_draw_execution_plan()` for
backend-specific details without making callers match on old backend-shaped
variant names.
When a draw is lowered into the live-window hardware-raster executor, its
backend-specific plan still exposes the same neutral questions: `backend()`,
`geometry_stream()`, `instance_stream()`, `material()`, `target()`,
`draw_policy()`, and `policy_config()`.

The same configured pieces are inspectable through `geometry_stream_specs()`,
`instance_stream_specs()`, `material_specs()`, `target_specs()`, and
`draw_specs()` before any window is launched. `MeshBuilder`, `MeshSpec`, and
`MeshSource` remain compatibility aliases, but new app code should prefer
`GeometryBuilder`, treat `GeometryStreamConfig` as the configuration type for
geometry input, and use `GeometryStreamSource` for geometry sources.

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
- `examples/friendly-demo`: concise app-layer CUDA tiled demo.

## Setup Notes

Neo currently targets Windows + NVIDIA first. A working NVIDIA driver is required,
and NVRTC must be discoverable. `neo doctor` checks `PATH`, `CUDA_PATH`, installed
CUDA Toolkit folders, and common NVIDIA SDK locations.

CUDA 13 is the default developer build:

```powershell
cargo run -p neo-quad-stress-3d --release
```

For older NVIDIA systems that are pinned to CUDA 12.6, build Neo with the CUDA
12.6 feature path:

```powershell
cargo run --no-default-features --features cuda-12060 -p neo-quad-stress-3d --release -- --draw-backend cuda-tiled
```

On a first run, confirm the CUDA 12.6 NVRTC DLL is discoverable:

```powershell
cargo run --no-default-features --features cuda-12060 -p neo-cli -- doctor
```

The CUDA tiled path is the recommended compatibility test before trying
hardware raster experiments.

Recommended verification before pushing changes:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo test --workspace --no-default-features --features cuda-12060
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
