use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use neo_lang::{AddressSpace, TypeName};
use neo_runtime::{
    Context as NeoContext, CudaFence, CudaGraph, DataLayout, DeviceBuffer, DrawBackend,
    DrawDepthMode as RuntimeDrawDepthMode, DrawExecution, DrawIndexedIndirectCommand, DrawPolicy,
    DrawPolicyConfig, GeometryStream, IndexFormat, IndirectDrawBuffer, InstanceAttribute,
    InstanceBuffer, InstanceBufferDesc, InstanceFormat, InstanceLayout, InstanceSemantic,
    InstanceStream, Kernel, LaunchDims, MaterialBindingKind, MaterialFragmentRequirement,
    MaterialKernel, MaterialKernelAbi, MaterialVertexRequirement, MeshBuffer, MeshBufferDesc,
    NeoD3d12InteropDevice, PrimitiveTopology, RasterCullOrder as RuntimeRasterCullOrder,
    RasterVisibilityMode as RuntimeRasterVisibilityMode, ReadablePinnedHostBuffer, SharedFrameRing,
    SharedGpuBuffer, SharedInstanceStream, Stream as CudaStream, Target, VertexAttribute,
    VertexFormat, VertexLayout, VertexSemantic, VisibleInstanceStream,
};
use notify::{Event as NotifyEvent, RecursiveMode, Watcher as _};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes},
};

const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 540;
const BLOCK: (u32, u32) = (16, 16);
const INSTANCE_CULL_TILE: u32 = 8;
const TILE_CULL_RECORD_BYTES: usize = 16;
const INSTANCE_MACROCELL_SIZE: u32 = 8;
const VISIBILITY_HEADER_U32S: usize = 8;
const VISIBILITY_RECORD_U32S: usize = 6;
const VISIBILITY_MAGIC: u32 = 0x4e45_4f4d;
const EMPTY_IDLE_FPS: f32 = 15.0;
const UNFOCUSED_IDLE_FPS: f32 = 15.0;
const CAMERA_MOVE_UNITS_PER_SEC: f32 = 4.0;
const CAMERA_MAX_STEP_SECONDS: f32 = 1.0 / 30.0;
const DEFAULT_MIN_PROJECTED_MILLIPIXELS: u32 = 850;
const DEFAULT_INSTANCE_GRID: InstanceGrid = InstanceGrid {
    x: 256,
    y: 256,
    z: 128,
};

pub fn main_entry() -> Result<()> {
    run_from_args(std::env::args().skip(1))
}

pub fn run_from_args(args: impl IntoIterator<Item = String>) -> Result<()> {
    run(LiveOptions::parse(args)?)
}

pub fn run_from_args_with_draw_plan(
    args: impl IntoIterator<Item = String>,
    draw_plan: DrawPlan,
) -> Result<()> {
    run_from_args_with_draw_execution_plan(args, draw_plan)
}

pub fn run_from_args_with_draw_execution_plan(
    args: impl IntoIterator<Item = String>,
    draw_plan: DrawExecutionPlan,
) -> Result<()> {
    let mut options = LiveOptions::parse(args)?;
    options.raster_plan = draw_plan;
    run(options)
}

pub fn run_from_args_with_raster_plan(
    args: impl IntoIterator<Item = String>,
    raster_plan: HardwareRasterPlan,
) -> Result<()> {
    run_from_args_with_draw_plan(args, raster_plan)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawExecutionPlan {
    pub draw_name: String,
    pub geometry_stream: GeometryStreamPlan,
    pub instance_stream: InstanceStreamPlan,
    pub target: TargetPlan,
    pub material: MaterialKernelPlan,
    pub draw_policy: DrawPolicyPlan,
    pub depth: DrawDepthMode,
    pub cull_order: DrawCullOrder,
    pub visibility: DrawVisibilityMode,
    pub min_projected_millipixels: u32,
}

pub type DrawPlan = DrawExecutionPlan;
pub type HardwareRasterPlan = DrawExecutionPlan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawExecutionContract {
    pub draw_name: String,
    pub geometry_stream: String,
    pub instance_stream: String,
    pub instance_layout: StressInstanceLayout,
    pub material: String,
    pub target: String,
    pub target_width: u32,
    pub target_height: u32,
    pub policy_config: DrawPolicyConfig,
    pub backend: DrawBackend,
}

pub type DrawContract = DrawExecutionContract;
pub type HardwareRasterDrawContract = DrawExecutionContract;

impl DrawExecutionContract {
    pub fn policy(&self) -> DrawPolicy {
        self.policy_config.policy
    }

    pub fn policy_label(&self) -> &'static str {
        self.policy_config.policy_label()
    }

    pub fn backend_label(&self) -> &'static str {
        self.policy_config.backend_label()
    }

    pub fn depth_label(&self) -> &'static str {
        self.policy_config.depth_label()
    }

    pub fn uses_depth(&self) -> bool {
        self.policy_config.uses_depth()
    }

    pub fn cull_order_label(&self) -> &'static str {
        self.policy_config.cull_order_label()
    }

    pub fn visibility_label(&self) -> &'static str {
        self.policy_config.visibility_label()
    }

    pub fn min_projected_pixels(&self) -> f32 {
        self.policy_config.min_projected_pixels()
    }

    pub fn instance_layout_label(&self) -> &'static str {
        self.instance_layout.label()
    }

    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }
}

impl DrawExecutionPlan {
    pub fn stock() -> Self {
        Self {
            draw_name: "main".to_string(),
            geometry_stream: GeometryStreamPlan::stock_quad(),
            instance_stream: InstanceStreamPlan::stock_instances(
                DEFAULT_INSTANCE_GRID,
                StressInstanceLayout::AoSoA32,
            ),
            target: TargetPlan::window(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            material: MaterialKernelPlan::direct_instance_color(
                "hardware-raster",
                "quad_vs_direct",
                "quad_fs",
            ),
            draw_policy: DrawPolicyPlan::DrawAll,
            depth: DrawDepthMode::Auto,
            cull_order: DrawCullOrder::StableDense,
            visibility: DrawVisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    fn material_kernel(&self) -> MaterialKernel {
        self.material.material_kernel()
    }

    pub fn draw_name(&self) -> &str {
        &self.draw_name
    }

    pub fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    pub fn geometry_stream(&self) -> &GeometryStreamPlan {
        &self.geometry_stream
    }

    pub fn instance_stream(&self) -> &InstanceStreamPlan {
        &self.instance_stream
    }

    pub fn target(&self) -> &TargetPlan {
        &self.target
    }

    pub fn material(&self) -> &MaterialKernelPlan {
        &self.material
    }

    pub fn draw_policy(&self) -> DrawPolicyPlan {
        self.draw_policy
    }

    pub fn depth(&self) -> DrawDepthMode {
        self.depth
    }

    pub fn uses_depth(&self) -> bool {
        self.depth.uses_depth(self.draw_policy)
    }

    pub fn policy(&self) -> DrawPolicy {
        self.draw_policy.into()
    }

    pub fn cull_order(&self) -> DrawCullOrder {
        self.cull_order
    }

    pub fn visibility(&self) -> DrawVisibilityMode {
        self.visibility
    }

    pub fn min_projected_pixels(&self) -> f32 {
        self.min_projected_millipixels as f32 / 1000.0
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        match self.draw_policy {
            DrawPolicyPlan::DrawAll => DrawPolicyConfig::draw_all(),
            DrawPolicyPlan::ComputeCulled => DrawPolicyConfig::compute_culled_with_visibility(
                self.cull_order.runtime_order(),
                self.visibility.runtime_visibility(),
            )
            .with_min_projected_millipixels(self.min_projected_millipixels),
        }
        .with_depth(self.depth.runtime_mode())
    }

    pub fn contract(&self) -> DrawExecutionContract {
        DrawExecutionContract {
            draw_name: self.draw_name.clone(),
            geometry_stream: self.geometry_stream.name.clone(),
            instance_stream: self.instance_stream.name.clone(),
            instance_layout: self.instance_stream.layout,
            material: self.material.name.clone(),
            target: self.target.name.clone(),
            target_width: self.target.width,
            target_height: self.target.height,
            policy_config: self.policy_config(),
            backend: self.backend(),
        }
    }

    fn sync_stock_material_to_draw_policy(&mut self) {
        if !self.material.is_stock_hardware_raster() {
            return;
        }
        self.material = match self.draw_policy {
            DrawPolicyPlan::DrawAll => MaterialKernelPlan::direct_instance_color(
                "hardware-raster",
                "quad_vs_direct",
                "quad_fs",
            ),
            DrawPolicyPlan::ComputeCulled => MaterialKernelPlan::compute_culled_instance_color(
                "hardware-raster",
                "quad_vs",
                "quad_fs",
            ),
        };
    }

    fn validate_executor_contract(&self) -> Result<()> {
        if self.geometry_stream.indices_u16.is_empty() {
            bail!(
                "hardware raster draw `{}` references GeometryStream `{}` with no indices",
                self.draw_name,
                self.geometry_stream.name
            );
        }
        if self.geometry_stream.vertex_bytes.is_empty() {
            bail!(
                "hardware raster draw `{}` references GeometryStream `{}` with no vertex data",
                self.draw_name,
                self.geometry_stream.name
            );
        }
        if self.geometry_stream.vertex_stride < 16 {
            bail!(
                "hardware raster draw `{}` references GeometryStream `{}` with vertex stride smaller than position+color payload",
                self.draw_name,
                self.geometry_stream.name
            );
        }
        if self.geometry_stream.indices_u16.len() > u32::MAX as usize {
            bail!(
                "hardware raster draw `{}` references GeometryStream `{}` with too many indices",
                self.draw_name,
                self.geometry_stream.name
            );
        }
        self.instance_stream.grid.validate()?;
        if self.target.width == 0 || self.target.height == 0 {
            bail!(
                "hardware raster draw `{}` references Target `{}` with zero size",
                self.draw_name,
                self.target.name
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawPolicyPlan {
    DrawAll,
    ComputeCulled,
}

pub type HardwareRasterDrawPolicy = DrawPolicyPlan;

impl DrawPolicyPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::DrawAll => "draw-all",
            Self::ComputeCulled => "compute-culled",
        }
    }
}

impl From<DrawPolicyPlan> for DrawPolicy {
    fn from(value: DrawPolicyPlan) -> Self {
        match value {
            DrawPolicyPlan::DrawAll => Self::DrawAll,
            DrawPolicyPlan::ComputeCulled => Self::ComputeCulled,
        }
    }
}

impl std::str::FromStr for DrawPolicyPlan {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "draw-all" | "all" => Ok(Self::DrawAll),
            "compute-culled" | "culled" => Ok(Self::ComputeCulled),
            _ => bail!("unknown raster draw policy `{value}`; expected draw-all or compute-culled"),
        }
    }
}

impl std::fmt::Display for DrawPolicyPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawDepthMode {
    Auto,
    On,
    Off,
}

impl DrawDepthMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    pub fn uses_depth(self, policy: DrawPolicyPlan) -> bool {
        match self {
            Self::Auto => policy != DrawPolicyPlan::DrawAll,
            Self::On => true,
            Self::Off => false,
        }
    }

    fn runtime_mode(self) -> RuntimeDrawDepthMode {
        match self {
            Self::Auto => RuntimeDrawDepthMode::Auto,
            Self::On => RuntimeDrawDepthMode::On,
            Self::Off => RuntimeDrawDepthMode::Off,
        }
    }
}

impl std::str::FromStr for DrawDepthMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" | "true" | "depth" => Ok(Self::On),
            "off" | "false" | "none" | "no-depth" => Ok(Self::Off),
            _ => bail!("unknown draw depth mode `{value}`; expected auto, on, or off"),
        }
    }
}

impl std::fmt::Display for DrawDepthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawCullOrder {
    AtomicCompact,
    StableDense,
}

pub type HardwareRasterCullOrder = DrawCullOrder;

impl DrawCullOrder {
    pub fn label(self) -> &'static str {
        match self {
            Self::AtomicCompact => "atomic-compact",
            Self::StableDense => "stable-dense",
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::AtomicCompact => 0,
            Self::StableDense => 1,
        }
    }

    fn runtime_order(self) -> RuntimeRasterCullOrder {
        match self {
            Self::AtomicCompact => RuntimeRasterCullOrder::AtomicCompact,
            Self::StableDense => RuntimeRasterCullOrder::StableDense,
        }
    }
}

impl std::str::FromStr for DrawCullOrder {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "atomic" | "atomic-compact" => Ok(Self::AtomicCompact),
            "stable" | "stable-dense" => Ok(Self::StableDense),
            _ => bail!(
                "unknown raster cull order `{value}`; expected atomic-compact or stable-dense"
            ),
        }
    }
}

impl std::fmt::Display for DrawCullOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawVisibilityMode {
    Frustum,
    ProjectedSize,
}

pub type HardwareRasterVisibilityMode = DrawVisibilityMode;

impl DrawVisibilityMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Frustum => "frustum",
            Self::ProjectedSize => "projected-size",
        }
    }

    fn code(&self) -> u32 {
        match self {
            Self::Frustum => 0,
            Self::ProjectedSize => 1,
        }
    }

    fn runtime_visibility(self) -> RuntimeRasterVisibilityMode {
        match self {
            Self::Frustum => RuntimeRasterVisibilityMode::Frustum,
            Self::ProjectedSize => RuntimeRasterVisibilityMode::ProjectedSize,
        }
    }
}

impl std::str::FromStr for DrawVisibilityMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "frustum" | "frustum-only" => Ok(Self::Frustum),
            "projected-size" | "projected" | "pixel-size" => Ok(Self::ProjectedSize),
            _ => bail!(
                "unknown raster visibility mode `{value}`; expected frustum or projected-size"
            ),
        }
    }
}

impl std::fmt::Display for DrawVisibilityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryStreamPlan {
    pub name: String,
    pub vertex_bytes: Vec<u8>,
    pub vertex_stride: u32,
    pub color_offset: u32,
    pub indices_u16: Vec<u16>,
}

pub type HardwareRasterGeometryStreamPlan = GeometryStreamPlan;

impl GeometryStreamPlan {
    pub fn stock_quad() -> Self {
        let vertices = [
            DemoVertex {
                position: [-1.0, -1.0, 0.0],
                color_bgra: 0xffff_ffff,
            },
            DemoVertex {
                position: [1.0, -1.0, 0.0],
                color_bgra: 0xffff_ffff,
            },
            DemoVertex {
                position: [-1.0, 1.0, 0.0],
                color_bgra: 0xffff_ffff,
            },
            DemoVertex {
                position: [1.0, 1.0, 0.0],
                color_bgra: 0xffff_ffff,
            },
        ];
        Self {
            name: "quad".to_string(),
            vertex_bytes: vertices_as_bytes(&vertices),
            vertex_stride: std::mem::size_of::<DemoVertex>() as u32,
            color_offset: 12,
            indices_u16: vec![0, 1, 2, 2, 1, 3],
        }
    }

    pub fn indexed_u16(
        name: impl Into<String>,
        vertex_bytes: impl Into<Vec<u8>>,
        vertex_stride: u32,
        color_offset: u32,
        indices: impl Into<Vec<u16>>,
    ) -> Self {
        Self {
            name: name.into(),
            vertex_bytes: vertex_bytes.into(),
            vertex_stride,
            color_offset,
            indices_u16: indices.into(),
        }
    }

    pub fn index_count(&self) -> u32 {
        self.indices_u16.len() as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceStreamPlan {
    pub name: String,
    pub grid: InstanceGrid,
    pub layout: StressInstanceLayout,
}

pub type HardwareRasterInstanceStreamPlan = InstanceStreamPlan;

impl InstanceStreamPlan {
    pub fn stock_instances(grid: InstanceGrid, layout: StressInstanceLayout) -> Self {
        Self {
            name: "instances".to_string(),
            grid,
            layout,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetPlan {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

pub type HardwareRasterTargetPlan = TargetPlan;

impl TargetPlan {
    pub fn window(width: u32, height: u32) -> Self {
        Self {
            name: "window".to_string(),
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialKernelPlan {
    pub name: String,
    pub vertex_entrypoint: String,
    pub fragment_entrypoint: String,
    pub kind: MaterialKernelPlanKind,
}

pub type HardwareRasterMaterialPlan = MaterialKernelPlan;

impl MaterialKernelPlan {
    pub fn direct_instance_color(
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            vertex_entrypoint: vertex_entrypoint.into(),
            fragment_entrypoint: fragment_entrypoint.into(),
            kind: MaterialKernelPlanKind::DirectInstanceColor,
        }
    }

    pub fn compute_culled_instance_color(
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            vertex_entrypoint: vertex_entrypoint.into(),
            fragment_entrypoint: fragment_entrypoint.into(),
            kind: MaterialKernelPlanKind::ComputeCulledInstanceColor,
        }
    }

    fn material_kernel(&self) -> MaterialKernel {
        let material = MaterialKernel::from_stages(
            self.name.clone(),
            self.vertex_entrypoint.clone(),
            self.fragment_entrypoint.clone(),
        );
        match self.kind {
            MaterialKernelPlanKind::DirectInstanceColor => {
                material.with_abi(MaterialKernelAbi::direct_instance_color(
                    self.vertex_entrypoint.clone(),
                    self.fragment_entrypoint.clone(),
                ))
            }
            MaterialKernelPlanKind::ComputeCulledInstanceColor => {
                material.with_abi(MaterialKernelAbi::compute_culled_instance_color(
                    self.vertex_entrypoint.clone(),
                    self.fragment_entrypoint.clone(),
                ))
            }
        }
    }

    fn is_stock_hardware_raster(&self) -> bool {
        self.name == "hardware-raster"
            && self.fragment_entrypoint == "quad_fs"
            && matches!(
                self.vertex_entrypoint.as_str(),
                "quad_vs" | "quad_vs_direct"
            )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialKernelPlanKind {
    DirectInstanceColor,
    ComputeCulledInstanceColor,
}

pub type HardwareRasterMaterialKind = MaterialKernelPlanKind;

impl MaterialKernelPlanKind {
    fn requires_visible_stream(self) -> bool {
        matches!(self, Self::ComputeCulledInstanceColor)
    }
}

#[allow(deprecated)]
fn run(mut options: LiveOptions) -> Result<()> {
    if options.mode == RunMode::DrawStress {
        options.raster_plan.validate_executor_contract()?;
        options.instance_grid = options.raster_plan.instance_stream.grid;
        options.instance_layout = options.raster_plan.instance_stream.layout;
        options.width = options.raster_plan.target.width;
        options.height = options.raster_plan.target.height;
    }
    let source_path = options
        .source_path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", options.source_path.display()))?;
    let instance_source_path = if options.mode == RunMode::InstanceStress {
        options
            .instance_stress_variant
            .source_path(&source_path, options.instance_layout)
    } else if options.mode == RunMode::DrawStress {
        raster_stress_source_path(&source_path)
    } else {
        source_path.clone()
    };
    let event_loop = EventLoop::new()?;
    let window = create_window(&event_loop, &options.title, options.width, options.height)?;
    let neo = NeoContext::new_default_device()?;
    let mut interop_device = None;
    let presenter_kind = if options.presenter == PresenterKind::D3d12Interop {
        match NeoD3d12InteropDevice::new(&neo) {
            Ok(device) => {
                interop_device = Some(device);
                PresenterKind::D3d12Interop
            }
            Err(err) if options.interop_fallback == InteropFallback::NoInterop => {
                eprintln!("D3D12/CUDA interop unavailable; falling back to d3d12: {err:#}");
                PresenterKind::D3d12
            }
            Err(err) => return Err(err).context("failed to initialize D3D12/CUDA interop"),
        }
    } else {
        options.presenter
    };
    if matches!(
        options.mode,
        RunMode::MeshDemo | RunMode::InstanceStress | RunMode::DrawStress
    ) && presenter_kind != PresenterKind::D3d12Interop
    {
        bail!(
            "{} requires D3D12/CUDA interop; use --presenter d3d12-interop --interop-fallback fail",
            options.mode
        );
    }
    let presenter = WindowPresenter::new(
        &window,
        presenter_kind,
        options.present_ring,
        options.d3d_upload,
        interop_device.as_ref(),
    )?;
    let mut presenter = if matches!(options.mode, RunMode::KernelThroughput)
        && presenter_kind.uses_present_thread()
    {
        PresentSink::threaded(presenter)
    } else {
        PresentSink::Direct(presenter)
    };

    if matches!(options.mode, RunMode::KernelThroughput) {
        unsafe {
            neo.disable_automatic_event_tracking();
        }
    }
    let mut live_reload = if matches!(
        options.mode,
        RunMode::MeshDemo | RunMode::InstanceStress | RunMode::DrawStress
    ) {
        None
    } else {
        Some(ReloadState::new(LiveKernel::compile(&neo, &source_path)?))
    };
    let mut mesh_reload = if options.mode == RunMode::MeshDemo {
        Some(ReloadState::new(MeshKernel::compile(&neo, &source_path)?))
    } else {
        None
    };
    let mut instance_reload = if options.mode == RunMode::InstanceStress {
        Some(ReloadState::new(InstanceKernel::compile(
            &neo,
            &instance_source_path,
            options.instance_stress_variant,
        )?))
    } else {
        None
    };
    let mut raster_reload = if options.mode == RunMode::DrawStress {
        Some(ReloadState::new(RasterStressKernel::compile(
            &neo,
            &instance_source_path,
            &options.raster_plan,
        )?))
    } else {
        None
    };
    let mesh_buffer = if options.mode == RunMode::MeshDemo {
        Some(create_demo_mesh(&neo)?)
    } else {
        None
    };
    let mut instance_assets =
        if matches!(options.mode, RunMode::InstanceStress | RunMode::DrawStress) {
            Some(create_instance_stress_assets(
                &neo,
                interop_device
                    .as_ref()
                    .filter(|_| options.mode == RunMode::DrawStress),
                options.instance_grid,
                options.present_ring,
                options.instance_layout,
            )?)
        } else {
            None
        };
    let mut raster_resources: Option<RasterStressResources> = None;
    let watched_source_path = if options.mode == RunMode::InstanceStress {
        &instance_source_path
    } else {
        &source_path
    };
    let (_watcher, reload_rx) = if options.hot_reload {
        let (watcher, rx) = watch_source(watched_source_path)?;
        (Some(watcher), Some(rx))
    } else {
        (None, None)
    };
    let mut frame_resources: Option<FrameResources> = None;
    let mut throughput_resources: Option<ThroughputResources> = None;
    let mut interop_throughput_resources: Option<InteropThroughputResources> = None;
    let mut throughput_generation = live_reload
        .as_ref()
        .map(|reload| reload.generation)
        .or_else(|| mesh_reload.as_ref().map(|reload| reload.generation))
        .or_else(|| instance_reload.as_ref().map(|reload| reload.generation))
        .or_else(|| raster_reload.as_ref().map(|reload| reload.generation))
        .unwrap_or(0);
    let mut fps = FpsCounter::new();
    let mut throughput = ThroughputCounter::new();
    let start = Instant::now();
    let mut frame = 0u32;
    let mut completed_kernels = 0u64;
    let mut next_sample_at = options.sample_every as u64;
    let mut next_present_at = options.present_interval().map(|interval| start + interval);
    let mut kernel_limiter =
        KernelRateLimiter::new(options.kernel_cap(), options.max_inflight, start);
    let mut unfocused_kernel_limiter = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
    let mut interop_present_limiter = PresentRateLimiter::new(options.present_target_fps, start);
    let mut camera = CameraController::new();
    let mut window_visibility = WindowVisibilityState::default();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                if let Some(reload_rx) = reload_rx.as_ref() {
                    let reload_result = match options.mode {
                        RunMode::MeshDemo => handle_mesh_reload_events(
                            &neo,
                            &source_path,
                            reload_rx,
                            mesh_reload
                                .as_mut()
                                .expect("mesh reload exists in mesh-demo mode"),
                        ),
                        RunMode::InstanceStress => handle_instance_reload_events(
                            &neo,
                            &instance_source_path,
                            options.instance_stress_variant,
                            reload_rx,
                            instance_reload
                                .as_mut()
                                .expect("instance reload exists in instance-stress mode"),
                        ),
                        RunMode::DrawStress => handle_raster_reload_events(
                            &neo,
                            &instance_source_path,
                            &options.raster_plan,
                            reload_rx,
                            raster_reload
                                .as_mut()
                                .expect("draw reload exists in draw-stress mode"),
                        ),
                        _ => handle_reload_events(
                            &neo,
                            &source_path,
                            reload_rx,
                            live_reload
                                .as_mut()
                                .expect("live reload exists outside mesh-demo mode"),
                        ),
                    };
                    if let Err(err) = reload_result {
                        eprintln!("hot reload watcher error: {err:#}");
                    }
                }

                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    window_visibility.minimized = true;
                    elwt.set_control_flow(ControlFlow::Wait);
                    return;
                }
                window_visibility.minimized = false;

                if matches!(options.mode, RunMode::InstanceStress) {
                    let reload = instance_reload
                        .as_mut()
                        .expect("instance reload exists in instance-stress mode");
                    if presenter.kind() != PresenterKind::D3d12Interop {
                        eprintln!("instance-stress requires --presenter d3d12-interop");
                        elwt.exit();
                        return;
                    }
                    if throughput_generation != reload.generation {
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let mut camera_params =
                        camera.params(size, start.elapsed().as_secs_f32(), options.instance_grid);
                    camera_params.config = [
                        options.instance_debug_view.code(),
                        options.instance_layout.group_size(),
                        0,
                        0,
                    ];
                    let visibility = instance_render_visibility(
                        options.render_policy,
                        window_visibility,
                        &camera_params,
                    );
                    let max_inflight = grant_instance_inflight(
                        now,
                        visibility,
                        options.render_policy,
                        window_visibility.focused,
                        options.max_inflight,
                        &mut kernel_limiter,
                        &mut unfocused_kernel_limiter,
                    );
                    let has_live_interop_work = interop_throughput_resources
                        .as_ref()
                        .is_some_and(InteropThroughputResources::has_live_work);
                    let skip_idle_gpu_work = should_skip_idle_instance_gpu_work(
                        visibility,
                        options.render_policy,
                        max_inflight,
                        has_live_interop_work,
                    );
                    let result = if skip_idle_gpu_work {
                        Ok(ThroughputBatchStats::default())
                    } else {
                        match (interop_device.as_ref(), instance_assets.as_mut()) {
                            (Some(interop), Some(assets)) => {
                                run_instance_stress_batch(InstanceStressBatch {
                                    neo: &neo,
                                    interop,
                                    resources: &mut interop_throughput_resources,
                                    kernel: &reload.active,
                                    assets,
                                    camera: camera_params,
                                    presenter: &mut presenter,
                                    size,
                                    start,
                                    next_frame: &mut frame,
                                    completed_kernels: &mut completed_kernels,
                                    present_limiter: &mut interop_present_limiter,
                                    max_inflight,
                                    present_ring: options.present_ring,
                                })
                            }
                            _ => Err(anyhow!(
                                "missing D3D12 interop device or instance stress assets"
                            )),
                        }
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: Some(options.instance_stress_variant),
                                instance_layout: instance_assets
                                    .as_ref()
                                    .map(|assets| assets.instances.layout_label()),
                                instance_debug_view: Some(options.instance_debug_view),
                                renderer: None,
                                draw_policy: None,
                                draw_depth: None,
                                uses_depth: None,
                                cull_order: None,
                                draw_visibility: None,
                                min_projected_millipixels: None,
                                visible_instances: None,
                                indirect_draws: None,
                                render_policy: options.render_policy,
                                visibility,
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                            if max_inflight == 0
                                && let Some(next_kernel_at) =
                                    next_instance_tick_at(InstanceTickPacing {
                                        now,
                                        visibility,
                                        kernel_limiter: &kernel_limiter,
                                        unfocused_limiter: &unfocused_kernel_limiter,
                                        render_policy: options.render_policy,
                                        focused: window_visibility.focused,
                                        wait_for_kernel_token: should_wait_for_kernel_token(
                                            presenter.kind(),
                                            options.present_target_fps,
                                        ),
                                    })
                            {
                                elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                            } else if max_inflight == 0
                                && options.render_policy != RenderPolicy::ForceRender
                                && matches!(
                                    visibility,
                                    RenderVisibility::Occluded | RenderVisibility::Minimized
                                )
                            {
                                elwt.set_control_flow(ControlFlow::Wait);
                            }
                        }
                        Err(err) => {
                            eprintln!("instance stress error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                if matches!(options.mode, RunMode::DrawStress) {
                    let reload = raster_reload
                        .as_mut()
                        .expect("draw reload exists in draw-stress mode");
                    if presenter.kind() != PresenterKind::D3d12Interop {
                        eprintln!("draw-stress requires --presenter d3d12-interop");
                        elwt.exit();
                        return;
                    }
                    if throughput_generation != reload.generation {
                        raster_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let camera_params =
                        camera.params(size, start.elapsed().as_secs_f32(), options.instance_grid);
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    let result = match interop_device.as_ref() {
                        Some(interop) => run_raster_stress_batch(RasterStressBatch {
                            neo: &neo,
                            interop,
                            resources: &mut raster_resources,
                            kernel: &reload.active,
                            assets: instance_assets
                                .as_mut()
                                .expect("draw-stress creates instance assets"),
                            presenter: &mut presenter,
                            size,
                            start,
                            next_frame: &mut frame,
                            completed_kernels: &mut completed_kernels,
                            present_limiter: &mut interop_present_limiter,
                            max_inflight,
                            present_ring: options.present_ring,
                            instance_grid: options.instance_grid,
                            camera: camera_params,
                            raster_plan: &options.raster_plan,
                        }),
                        None => Err(anyhow!("missing D3D12 interop device")),
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: None,
                                instance_layout: Some(
                                    options.raster_plan.instance_stream.layout.to_string(),
                                ),
                                instance_debug_view: None,
                                renderer: Some(options.raster_plan.backend()),
                                draw_policy: Some(options.raster_plan.draw_policy),
                                draw_depth: Some(options.raster_plan.depth),
                                uses_depth: Some(options.raster_plan.uses_depth()),
                                cull_order: Some(options.raster_plan.cull_order),
                                draw_visibility: Some(options.raster_plan.visibility),
                                min_projected_millipixels: Some(
                                    options.raster_plan.min_projected_millipixels,
                                ),
                                visible_instances: hardware_raster_visible_instances_for_log(
                                    options.raster_plan.draw_policy,
                                    options.instance_grid,
                                ),
                                indirect_draws: Some(1),
                                render_policy: options.render_policy,
                                visibility: RenderVisibility::Visible,
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                raster_resources = None;
                                elwt.exit();
                            }
                            if max_inflight == 0
                                && let Some(next_kernel_at) = kernel_limiter.next_token_at(now)
                            {
                                elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                            }
                        }
                        Err(err) => {
                            eprintln!("raster stress error: {err:#}");
                            raster_resources = None;
                            elwt.exit();
                        }
                    }
                    return;
                }

                if matches!(options.mode, RunMode::MeshDemo) {
                    let reload = mesh_reload
                        .as_mut()
                        .expect("mesh reload exists in mesh-demo mode");
                    if presenter.kind() != PresenterKind::D3d12Interop {
                        eprintln!("mesh-demo requires --presenter d3d12-interop");
                        elwt.exit();
                        return;
                    }
                    if throughput_generation != reload.generation {
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    let result = match (interop_device.as_ref(), mesh_buffer.as_ref()) {
                        (Some(interop), Some(mesh)) => run_mesh_demo_batch(MeshDemoBatch {
                            neo: &neo,
                            interop,
                            resources: &mut interop_throughput_resources,
                            kernel: &reload.active.kernel,
                            mesh,
                            presenter: &mut presenter,
                            size,
                            start,
                            next_frame: &mut frame,
                            completed_kernels: &mut completed_kernels,
                            present_limiter: &mut interop_present_limiter,
                            max_inflight,
                            present_ring: options.present_ring,
                        }),
                        _ => Err(anyhow!("missing D3D12 interop device or mesh buffer")),
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: None,
                                instance_layout: None,
                                instance_debug_view: None,
                                renderer: None,
                                draw_policy: None,
                                draw_depth: None,
                                uses_depth: None,
                                cull_order: None,
                                draw_visibility: None,
                                min_projected_millipixels: None,
                                visible_instances: None,
                                indirect_draws: None,
                                render_policy: options.render_policy,
                                visibility: window_visibility.render_visibility(),
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                        }
                        Err(err) => {
                            eprintln!("mesh demo error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                if matches!(options.mode, RunMode::KernelThroughput) {
                    let reload = live_reload
                        .as_mut()
                        .expect("live reload exists in kernel-throughput mode");
                    if throughput_generation != reload.generation {
                        throughput_resources = None;
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    let result = if presenter.kind() == PresenterKind::D3d12Interop {
                        match interop_device.as_ref() {
                            Some(interop) => run_interop_throughput_batch(InteropThroughputBatch {
                                neo: &neo,
                                interop,
                                resources: &mut interop_throughput_resources,
                                kernel: &reload.active.kernel,
                                presenter: &mut presenter,
                                size,
                                start,
                                next_frame: &mut frame,
                                completed_kernels: &mut completed_kernels,
                                present_limiter: &mut interop_present_limiter,
                                max_inflight,
                                present_ring: options.present_ring,
                            }),
                            None => Err(anyhow!("missing D3D12 interop device")),
                        }
                    } else {
                        run_kernel_throughput_batch(ThroughputBatch {
                            neo: &neo,
                            resources: &mut throughput_resources,
                            graph_kernel: &reload.active.graph_kernel,
                            presenter: &mut presenter,
                            size,
                            start,
                            next_frame: &mut frame,
                            completed_kernels: &mut completed_kernels,
                            next_sample_at: &mut next_sample_at,
                            next_present_at: &mut next_present_at,
                            sample_every: options.sample_every,
                            present_interval: options.present_interval(),
                            max_inflight,
                            present_ring: options.present_ring,
                        })
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: None,
                                instance_layout: None,
                                instance_debug_view: None,
                                renderer: None,
                                draw_policy: None,
                                draw_depth: None,
                                uses_depth: None,
                                cull_order: None,
                                draw_visibility: None,
                                min_projected_millipixels: None,
                                visible_instances: None,
                                indirect_draws: None,
                                render_policy: options.render_policy,
                                visibility: window_visibility.render_visibility(),
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                            if max_inflight == 0
                                && should_wait_for_kernel_token(
                                    presenter.kind(),
                                    options.present_target_fps,
                                )
                                && let Some(next_kernel_at) = kernel_limiter.next_token_at(now)
                            {
                                elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                            }
                        }
                        Err(err) => {
                            eprintln!("throughput error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                let reload = live_reload
                    .as_ref()
                    .expect("live reload exists in live mode");
                let frame_start = Instant::now();
                let mut timings = match render_frame(RenderFrame {
                    neo: &neo,
                    resources: &mut frame_resources,
                    kernel: &reload.active.kernel,
                    size,
                    time: start.elapsed().as_secs_f32(),
                    frame,
                }) {
                    Ok(timings) => timings,
                    Err(err) => {
                        eprintln!("render error: {err:#}");
                        elwt.exit();
                        return;
                    }
                };

                let resources = frame_resources
                    .as_ref()
                    .expect("frame resources are created before presentation");
                let before_present = Instant::now();
                let present_timings =
                    match presenter.present_sync(size, resources.host_bgra.as_slice()) {
                        Ok(timings) => timings,
                        Err(err) => {
                            eprintln!("present error: {err:#}");
                            elwt.exit();
                            return;
                        }
                    };
                timings.render = before_present - frame_start;
                timings.present = present_timings.total;
                timings.total = frame_start.elapsed();

                frame = frame.wrapping_add(1);
                fps.tick(
                    size,
                    frame,
                    reload.last_error.as_deref(),
                    presenter.kind(),
                    timings,
                );
                if options.should_stop(frame, start.elapsed()) {
                    elwt.exit();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera.handle_raw_mouse_delta(delta.0 as f32, delta.1 as f32);
            }
            Event::WindowEvent { event, .. } => {
                match &event {
                    WindowEvent::Occluded(occluded) => {
                        window_visibility.occluded = *occluded;
                    }
                    WindowEvent::Focused(focused) => {
                        window_visibility.focused = *focused;
                    }
                    WindowEvent::Resized(size) => {
                        window_visibility.minimized = size.width == 0 || size.height == 0;
                    }
                    _ => {}
                }
                match camera.handle_window_event(&event) {
                    CameraWindowAction::None => {}
                    CameraWindowAction::CaptureMouse => capture_mouse(&window),
                    CameraWindowAction::ReleaseMouse => release_mouse(&window),
                    CameraWindowAction::RecenterMouse => recenter_mouse(&window),
                }
            }
            _ => {}
        }
    })?;
    Ok(())
}

fn wait_for_interop_idle(
    resources: &mut Option<InteropThroughputResources>,
    interop: Option<&NeoD3d12InteropDevice>,
) {
    if let (Some(resources), Some(interop)) = (resources.as_mut(), interop)
        && let Err(err) = resources.ring.wait_idle(interop.queue())
    {
        eprintln!("failed to wait for D3D12/CUDA interop idle: {err:#}");
    }
}

fn interop_trace(message: &str) {
    if std::env::var_os("NEO_INTEROP_TRACE").is_some() {
        eprintln!("[interop] {message}");
    }
}

fn capture_mouse(window: &Window) {
    if window.set_cursor_grab(CursorGrabMode::Locked).is_err()
        && let Err(err) = window.set_cursor_grab(CursorGrabMode::Confined)
    {
        eprintln!("mouse look capture unavailable: {err}");
    }
    window.set_cursor_visible(false);
    recenter_mouse(window);
}

fn release_mouse(window: &Window) {
    if let Err(err) = window.set_cursor_grab(CursorGrabMode::None) {
        eprintln!("failed to release mouse look capture: {err}");
    }
    window.set_cursor_visible(true);
}

fn recenter_mouse(window: &Window) {
    let size = window.inner_size();
    if size.width == 0 || size.height == 0 {
        return;
    }
    if let Err(err) = window.set_cursor_position(PhysicalPosition::new(
        f64::from(size.width) * 0.5,
        f64::from(size.height) * 0.5,
    )) {
        if std::env::var_os("NEO_MOUSE_TRACE").is_some() {
            eprintln!("failed to recenter mouse look cursor: {err}");
        }
    }
}

fn should_wait_for_kernel_token(presenter: PresenterKind, present_target_fps: Option<f32>) -> bool {
    const POLL_PRECISE_INTEROP_FPS: f32 = 120.0;
    !(presenter == PresenterKind::D3d12Interop
        && present_target_fps.is_some_and(|fps| fps >= POLL_PRECISE_INTEROP_FPS))
}

fn grant_instance_inflight(
    now: Instant,
    visibility: RenderVisibility,
    render_policy: RenderPolicy,
    focused: bool,
    max_inflight: u32,
    kernel_limiter: &mut KernelRateLimiter,
    unfocused_limiter: &mut KernelRateLimiter,
) -> u32 {
    let visibility_limit = visibility.limit_max_inflight(max_inflight);
    if visibility_limit == 0 {
        return 0;
    }
    if render_policy == RenderPolicy::Auto && !focused {
        let kernel_available = kernel_limiter.available(now, visibility_limit);
        let unfocused_available = unfocused_limiter.available(now, 1);
        let granted = kernel_available.min(unfocused_available);
        kernel_limiter.consume(granted);
        unfocused_limiter.consume(granted);
        granted
    } else {
        kernel_limiter.grant(now, visibility_limit)
    }
}

fn should_skip_idle_instance_gpu_work(
    visibility: RenderVisibility,
    render_policy: RenderPolicy,
    max_inflight: u32,
    has_live_interop_work: bool,
) -> bool {
    render_policy != RenderPolicy::ForceRender
        && max_inflight == 0
        && !has_live_interop_work
        && matches!(
            visibility,
            RenderVisibility::Empty | RenderVisibility::Occluded | RenderVisibility::Minimized
        )
}

struct InstanceTickPacing<'a> {
    now: Instant,
    visibility: RenderVisibility,
    kernel_limiter: &'a KernelRateLimiter,
    unfocused_limiter: &'a KernelRateLimiter,
    render_policy: RenderPolicy,
    focused: bool,
    wait_for_kernel_token: bool,
}

fn next_instance_tick_at(pacing: InstanceTickPacing<'_>) -> Option<Instant> {
    if pacing.render_policy == RenderPolicy::ForceRender {
        return if pacing.wait_for_kernel_token {
            pacing.kernel_limiter.next_token_at(pacing.now)
        } else {
            None
        };
    }
    match pacing.visibility {
        RenderVisibility::Visible => {
            if pacing.render_policy == RenderPolicy::Auto && !pacing.focused {
                return latest_token_time(
                    pacing.kernel_limiter.next_token_at(pacing.now),
                    pacing.unfocused_limiter.next_token_at(pacing.now),
                );
            }
            if pacing.wait_for_kernel_token {
                pacing.kernel_limiter.next_token_at(pacing.now)
            } else {
                None
            }
        }
        RenderVisibility::Empty => Some(pacing.now + Duration::from_secs_f32(1.0 / EMPTY_IDLE_FPS)),
        RenderVisibility::Occluded | RenderVisibility::Minimized => None,
    }
}

fn latest_token_time(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[derive(Debug, Clone)]
struct LiveOptions {
    source_path: PathBuf,
    title: String,
    width: u32,
    height: u32,
    max_frames: Option<u32>,
    max_seconds: Option<f32>,
    presenter: PresenterKind,
    mode: RunMode,
    sample_every: u32,
    present_target_fps: Option<f32>,
    kernel_target_fps: Option<f32>,
    max_inflight: u32,
    present_ring: usize,
    instance_grid: InstanceGrid,
    instance_stress_variant: InstanceStressVariant,
    instance_debug_view: InstanceDebugView,
    instance_layout: StressInstanceLayout,
    render_policy: RenderPolicy,
    d3d_upload: D3dUploadMode,
    interop_fallback: InteropFallback,
    hot_reload: bool,
    raster_plan: HardwareRasterPlan,
}

impl LiveOptions {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            source_path: PathBuf::from("examples/live-window/live.neo"),
            title: "Neo Live Window".to_string(),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            max_frames: None,
            max_seconds: None,
            presenter: PresenterKind::D3d11,
            mode: RunMode::Live,
            sample_every: 256,
            present_target_fps: None,
            kernel_target_fps: None,
            max_inflight: 2,
            present_ring: 6,
            instance_grid: DEFAULT_INSTANCE_GRID,
            instance_stress_variant: InstanceStressVariant::Tiled,
            instance_debug_view: InstanceDebugView::Off,
            instance_layout: StressInstanceLayout::AoSoA32,
            render_policy: RenderPolicy::Auto,
            d3d_upload: D3dUploadMode::MappedCopy,
            interop_fallback: InteropFallback::NoInterop,
            hot_reload: true,
            raster_plan: HardwareRasterPlan::stock(),
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--width" => options.width = parse_next(&mut args, "--width")?,
                "--height" => options.height = parse_next(&mut args, "--height")?,
                "--title" => options.title = parse_next(&mut args, "--title")?,
                "--frames" => options.max_frames = Some(parse_next(&mut args, "--frames")?),
                "--seconds" => options.max_seconds = Some(parse_next(&mut args, "--seconds")?),
                "--presenter" => options.presenter = parse_next(&mut args, "--presenter")?,
                "--mode" => options.mode = parse_next(&mut args, "--mode")?,
                "--sample-every" => options.sample_every = parse_next(&mut args, "--sample-every")?,
                "--present-target-fps" => {
                    options.present_target_fps =
                        Some(parse_next(&mut args, "--present-target-fps")?)
                }
                "--kernel-target-fps" => {
                    options.kernel_target_fps = Some(parse_next(&mut args, "--kernel-target-fps")?)
                }
                "--max-inflight" => options.max_inflight = parse_next(&mut args, "--max-inflight")?,
                "--present-ring" => options.present_ring = parse_next(&mut args, "--present-ring")?,
                "--instance-grid" => {
                    options.instance_grid = parse_next(&mut args, "--instance-grid")?
                }
                "--instance-stress-variant" => {
                    options.instance_stress_variant =
                        parse_next(&mut args, "--instance-stress-variant")?
                }
                "--instance-debug-view" => {
                    options.instance_debug_view = parse_next(&mut args, "--instance-debug-view")?
                }
                "--instance-layout" => {
                    options.instance_layout = parse_next(&mut args, "--instance-layout")?
                }
                "--draw-policy" | "--raster-draw-policy" => {
                    options.raster_plan.draw_policy = parse_next(&mut args, arg.as_str())?;
                    options.raster_plan.sync_stock_material_to_draw_policy();
                }
                "--draw-depth" | "--raster-depth" => {
                    options.raster_plan.depth = parse_next(&mut args, arg.as_str())?
                }
                "--cull-order" | "--raster-cull-order" => {
                    options.raster_plan.cull_order = parse_next(&mut args, arg.as_str())?
                }
                "--visibility" | "--raster-visibility" => {
                    options.raster_plan.visibility = parse_next(&mut args, arg.as_str())?
                }
                "--min-projected-pixels" | "--raster-min-projected-pixels" => {
                    options.raster_plan.min_projected_millipixels = parse_min_projected_pixels(
                        arg.as_str(),
                        parse_next::<String>(&mut args, arg.as_str())?,
                    )?
                }
                "--render-policy" => {
                    options.render_policy = parse_next(&mut args, "--render-policy")?
                }
                "--d3d-upload" => options.d3d_upload = parse_next(&mut args, "--d3d-upload")?,
                "--interop-fallback" => {
                    options.interop_fallback = parse_next(&mut args, "--interop-fallback")?
                }
                "--hot-reload" => options.hot_reload = true,
                "--no-hot-reload" => options.hot_reload = false,
                "--help" | "-h" => bail!(
                    "usage: neo-live-window [path.neo] [--title TEXT] [--width N] [--height N] [--frames N] [--seconds N] [--presenter d3d12-interop|d3d12|d3d11|gdi] [--mode live|kernel-throughput|mesh-demo|instance-stress|draw-stress|raster-stress] [--sample-every N] [--present-target-fps N] [--kernel-target-fps N] [--max-inflight N] [--present-ring N] [--instance-grid XxYxZ] [--instance-stress-variant baseline|fast|culled|tiled|macrocell] [--instance-debug-view off|tile-range|iterations|hit-miss] [--instance-layout aosoa32|aosoa64] [--draw-policy draw-all|compute-culled] [--draw-depth auto|on|off] [--cull-order atomic-compact|stable-dense] [--visibility frustum|projected-size] [--min-projected-pixels N] [--render-policy auto|force-render|pause-when-empty] [--d3d-upload mapped-copy|update-subresource] [--interop-fallback no-interop|fail] [--hot-reload|--no-hot-reload]"
                ),
                value if value.starts_with('-') => bail!("unknown option `{value}`"),
                value => options.source_path = PathBuf::from(value),
            }
        }
        options.validate()?;
        Ok(options)
    }

    fn should_stop(&self, frame: u32, elapsed: Duration) -> bool {
        self.max_frames.is_some_and(|max| frame >= max)
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    fn should_stop_completed(&self, completed: u64, elapsed: Duration) -> bool {
        self.max_frames
            .is_some_and(|max| completed >= u64::from(max))
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    fn validate(&self) -> Result<()> {
        if self.width == 0 {
            bail!("--width must be greater than zero");
        }
        if self.height == 0 {
            bail!("--height must be greater than zero");
        }
        if self.sample_every == 0 {
            bail!("--sample-every must be greater than zero");
        }
        if self.max_inflight == 0 {
            bail!("--max-inflight must be greater than zero");
        }
        if self.present_ring == 0 {
            bail!("--present-ring must be greater than zero");
        }
        self.instance_grid.validate()?;
        if self
            .present_target_fps
            .is_some_and(|fps| !fps.is_finite() || fps <= 0.0)
        {
            bail!("--present-target-fps must be greater than zero");
        }
        if self
            .kernel_target_fps
            .is_some_and(|fps| !fps.is_finite() || fps < 0.0)
        {
            bail!("--kernel-target-fps must be finite and non-negative");
        }
        Ok(())
    }

    fn present_interval(&self) -> Option<Duration> {
        self.present_target_fps
            .map(|fps| Duration::from_secs_f32(1.0 / fps))
    }

    fn kernel_cap(&self) -> Option<f32> {
        self.kernel_target_fps.filter(|fps| *fps > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Live,
    KernelThroughput,
    MeshDemo,
    InstanceStress,
    DrawStress,
}

impl std::str::FromStr for RunMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "live" => Ok(Self::Live),
            "kernel-throughput" => Ok(Self::KernelThroughput),
            "mesh-demo" => Ok(Self::MeshDemo),
            "instance-stress" => Ok(Self::InstanceStress),
            "draw-stress" | "raster-stress" => Ok(Self::DrawStress),
            _ => bail!(
                "unknown mode `{value}`; expected live, kernel-throughput, mesh-demo, instance-stress, draw-stress, or raster-stress"
            ),
        }
    }
}

impl std::fmt::Display for RunMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live => f.write_str("live"),
            Self::KernelThroughput => f.write_str("kernel-throughput"),
            Self::MeshDemo => f.write_str("mesh-demo"),
            Self::InstanceStress => f.write_str("instance-stress"),
            Self::DrawStress => f.write_str("draw-stress"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderPolicy {
    Auto,
    ForceRender,
    PauseWhenEmpty,
}

impl std::str::FromStr for RenderPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "force-render" => Ok(Self::ForceRender),
            "pause-when-empty" => Ok(Self::PauseWhenEmpty),
            _ => bail!(
                "unknown render policy `{value}`; expected auto, force-render, or pause-when-empty"
            ),
        }
    }
}

impl std::fmt::Display for RenderPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::ForceRender => f.write_str("force-render"),
            Self::PauseWhenEmpty => f.write_str("pause-when-empty"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderVisibility {
    Visible,
    Empty,
    Occluded,
    Minimized,
}

impl RenderVisibility {
    fn limit_max_inflight(self, max_inflight: u32) -> u32 {
        match self {
            Self::Visible => max_inflight,
            Self::Empty | Self::Occluded | Self::Minimized => 0,
        }
    }
}

impl std::fmt::Display for RenderVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Visible => f.write_str("visible"),
            Self::Empty => f.write_str("empty"),
            Self::Occluded => f.write_str("occluded"),
            Self::Minimized => f.write_str("minimized"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WindowVisibilityState {
    minimized: bool,
    occluded: bool,
    focused: bool,
}

impl Default for WindowVisibilityState {
    fn default() -> Self {
        Self {
            minimized: false,
            occluded: false,
            focused: true,
        }
    }
}

impl WindowVisibilityState {
    fn render_visibility(self) -> RenderVisibility {
        if self.minimized {
            RenderVisibility::Minimized
        } else if self.occluded {
            RenderVisibility::Occluded
        } else {
            RenderVisibility::Visible
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceGrid {
    x: u32,
    y: u32,
    z: u32,
}

impl InstanceGrid {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn x(self) -> u32 {
        self.x
    }

    pub fn y(self) -> u32 {
        self.y
    }

    pub fn z(self) -> u32 {
        self.z
    }

    pub fn validate(self) -> Result<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            bail!("--instance-grid dimensions must be greater than zero");
        }
        self.count()
            .ok_or_else(|| anyhow!("--instance-grid instance count overflow"))?;
        Ok(())
    }

    pub fn count(self) -> Option<u32> {
        self.x.checked_mul(self.y)?.checked_mul(self.z)
    }
}

impl std::str::FromStr for InstanceGrid {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let parts = value.split('x').collect::<Vec<_>>();
        if parts.len() != 3 {
            bail!("invalid --instance-grid `{value}`; expected XxYxZ");
        }
        let grid = Self {
            x: parts[0]
                .parse()
                .with_context(|| format!("invalid X dimension in --instance-grid `{value}`"))?,
            y: parts[1]
                .parse()
                .with_context(|| format!("invalid Y dimension in --instance-grid `{value}`"))?,
            z: parts[2]
                .parse()
                .with_context(|| format!("invalid Z dimension in --instance-grid `{value}`"))?,
        };
        grid.validate()?;
        Ok(grid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceStressVariant {
    Baseline,
    Fast,
    Culled,
    Tiled,
    Macrocell,
}

impl InstanceStressVariant {
    fn source_path(self, requested: &Path, layout: StressInstanceLayout) -> PathBuf {
        if requested
            .file_name()
            .is_some_and(|name| name == "three_d_instances.neo")
        {
            match self {
                Self::Baseline => {
                    return requested.with_file_name("three_d_instances_baseline.neo");
                }
                Self::Fast => return requested.with_file_name("three_d_instances_fast.neo"),
                Self::Culled => {}
                Self::Tiled => {
                    return requested.with_file_name(match layout {
                        StressInstanceLayout::AoSoA32 => "three_d_instances_tiled_aosoa32.neo",
                        StressInstanceLayout::AoSoA64 => "three_d_instances_tiled_aosoa64.neo",
                    });
                }
                Self::Macrocell => {
                    return requested.with_file_name("three_d_instances_macrocell_aosoa32.neo");
                }
            }
        }
        requested.to_path_buf()
    }
}

impl std::str::FromStr for InstanceStressVariant {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "fast" => Ok(Self::Fast),
            "culled" => Ok(Self::Culled),
            "tiled" => Ok(Self::Tiled),
            "macrocell" => Ok(Self::Macrocell),
            _ => bail!(
                "unknown instance stress variant `{value}`; expected baseline, fast, culled, tiled, or macrocell"
            ),
        }
    }
}

impl std::fmt::Display for InstanceStressVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline => f.write_str("baseline"),
            Self::Fast => f.write_str("fast"),
            Self::Culled => f.write_str("culled"),
            Self::Tiled => f.write_str("tiled"),
            Self::Macrocell => f.write_str("macrocell"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceDebugView {
    Off,
    TileRange,
    Iterations,
    HitMiss,
}

impl InstanceDebugView {
    fn code(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::TileRange => 1,
            Self::Iterations => 2,
            Self::HitMiss => 3,
        }
    }
}

impl std::str::FromStr for InstanceDebugView {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "tile-range" => Ok(Self::TileRange),
            "iterations" => Ok(Self::Iterations),
            "hit-miss" => Ok(Self::HitMiss),
            _ => bail!(
                "unknown instance debug view `{value}`; expected off, tile-range, iterations, or hit-miss"
            ),
        }
    }
}

impl std::fmt::Display for InstanceDebugView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::TileRange => f.write_str("tile-range"),
            Self::Iterations => f.write_str("iterations"),
            Self::HitMiss => f.write_str("hit-miss"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressInstanceLayout {
    AoSoA32,
    AoSoA64,
}

impl StressInstanceLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::AoSoA32 => "aosoa32",
            Self::AoSoA64 => "aosoa64",
        }
    }

    pub fn group_size(self) -> u32 {
        match self {
            Self::AoSoA32 => 32,
            Self::AoSoA64 => 64,
        }
    }

    pub fn data_layout(self) -> DataLayout {
        DataLayout::AoSoA {
            group_size: self.group_size(),
        }
    }
}

impl std::str::FromStr for StressInstanceLayout {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "aosoa32" => Ok(Self::AoSoA32),
            "aosoa64" => Ok(Self::AoSoA64),
            _ => bail!("unknown instance layout `{value}`; expected aosoa32 or aosoa64"),
        }
    }
}

impl std::fmt::Display for StressInstanceLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresenterKind {
    D3d12Interop,
    D3d12,
    D3d11,
    Gdi,
}

impl PresenterKind {
    fn uses_present_thread(self) -> bool {
        !matches!(self, Self::D3d12 | Self::D3d12Interop)
    }
}

impl std::str::FromStr for PresenterKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "d3d12-interop" => Ok(Self::D3d12Interop),
            "d3d12" => Ok(Self::D3d12),
            "d3d11" => Ok(Self::D3d11),
            "gdi" => Ok(Self::Gdi),
            _ => bail!("unknown presenter `{value}`; expected d3d12-interop, d3d12, d3d11, or gdi"),
        }
    }
}

impl std::fmt::Display for PresenterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D3d12Interop => f.write_str("d3d12-interop"),
            Self::D3d12 => f.write_str("d3d12-flip-upload"),
            Self::D3d11 => f.write_str("d3d11-flip-host"),
            Self::Gdi => f.write_str("win32-gdi"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteropFallback {
    NoInterop,
    Fail,
}

impl std::str::FromStr for InteropFallback {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "no-interop" => Ok(Self::NoInterop),
            "fail" => Ok(Self::Fail),
            _ => bail!("unknown interop fallback `{value}`; expected no-interop or fail"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum D3dUploadMode {
    MappedCopy,
    UpdateSubresource,
}

impl std::str::FromStr for D3dUploadMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "mapped-copy" => Ok(Self::MappedCopy),
            "update-subresource" => Ok(Self::UpdateSubresource),
            _ => bail!(
                "unknown D3D upload mode `{value}`; expected mapped-copy or update-subresource"
            ),
        }
    }
}

impl std::fmt::Display for D3dUploadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MappedCopy => f.write_str("mapped-copy"),
            Self::UpdateSubresource => f.write_str("update-subresource"),
        }
    }
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let value = args
        .next()
        .ok_or_else(|| anyhow!("missing value for {name}"))?;
    value
        .parse()
        .map_err(|err| anyhow!("invalid value for {name}: {err}"))
}

fn parse_min_projected_pixels(flag: &str, value: String) -> Result<u32> {
    let pixels = value
        .parse::<f32>()
        .map_err(|err| anyhow!("invalid value for {flag}: {err}"))?;
    if !pixels.is_finite() || pixels < 0.0 {
        bail!("{flag} must be finite and non-negative");
    }
    let millipixels = f64::from(pixels) * 1000.0;
    if millipixels > f64::from(u32::MAX) {
        bail!("{flag} is too large");
    }
    Ok(millipixels.round() as u32)
}

#[allow(deprecated)]
fn create_window(
    event_loop: &EventLoop<()>,
    title: &str,
    width: u32,
    height: u32,
) -> Result<Window> {
    let attrs = WindowAttributes::default()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(width.max(1), height.max(1)))
        .with_resizable(true);
    event_loop
        .create_window(attrs)
        .context("failed to create live window")
}

struct LiveKernel {
    kernel: Kernel,
    graph_kernel: Kernel,
}

impl LiveKernel {
    fn compile(ctx: &NeoContext, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        validate_live_kernel_abi(&source)?;
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &["image"])?;
        let graph_module =
            neo_runtime::Module::from_cuda_source(ctx, live_graph_cuda_source(&source)?)?;
        Ok(Self {
            kernel: module.kernel("image")?,
            graph_kernel: graph_module.kernel("image_graph")?,
        })
    }
}

fn live_graph_cuda_source(source: &str) -> Result<String> {
    let program = neo_lang::parse(source)?;
    let cuda = neo_lang::lower_program(&program);
    let signature = "extern \"C\" __global__ void image(unsigned char* pixels, unsigned int width, unsigned int height, float time, unsigned int frame)";
    let replacement = "__device__ __forceinline__ void neo_user_image(unsigned char* pixels, unsigned int width, unsigned int height, float time, unsigned int frame)";
    let device_source = cuda.replacen(signature, replacement, 1);
    if device_source == cuda {
        bail!("failed to build CUDA graph wrapper for live kernel ABI");
    }
    Ok(format!(
        "{device_source}\n\
struct NeoLiveGraphParams {{\n\
    float time;\n\
    unsigned int frame;\n\
}};\n\
\n\
extern \"C\" __global__ void image_graph(unsigned char* pixels, unsigned int width, unsigned int height, const NeoLiveGraphParams* params) {{\n\
    neo_user_image(pixels, width, height, params->time, params->frame);\n\
}}\n"
    ))
}

fn validate_live_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.kind == neo_lang::EntryPointKind::Kernel && kernel.name == "image")
        .ok_or_else(|| anyhow!("live kernel must define `kernel fn image(...)`"))?;
    let expected = [
        ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("width", None, TypeName::U32, 0),
        ("height", None, TypeName::U32, 0),
        ("time", None, TypeName::F32, 0),
        ("frame", None, TypeName::U32, 0),
    ];
    if kernel.params.len() != expected.len() {
        bail!(
            "live kernel `image` must have {} params: global u8* pixels, u32 width, u32 height, f32 time, u32 frame",
            expected.len()
        );
    }
    for (param, expected_param) in kernel.params.iter().zip(expected.iter()) {
        let (name, address_space, ty, pointer_depth) = expected_param;
        if param.name != *name
            || &param.address_space != address_space
            || &param.ty.base != ty
            || param.ty.pointer_depth != *pointer_depth
        {
            bail!(
                "invalid live kernel parameter `{}`; expected `{}` in ABI `global u8* pixels, u32 width, u32 height, f32 time, u32 frame`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct MeshKernel {
    kernel: Kernel,
}

impl MeshKernel {
    fn compile(ctx: &NeoContext, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        validate_mesh_kernel_abi(&source)?;
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &["raster"])?;
        Ok(Self {
            kernel: module.kernel("raster")?,
        })
    }
}

fn validate_mesh_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.kind == neo_lang::EntryPointKind::Kernel && kernel.name == "raster")
        .ok_or_else(|| anyhow!("mesh demo kernel must define `kernel fn raster(...)`"))?;
    let expected = [
        ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("width", None, TypeName::U32, 0),
        ("height", None, TypeName::U32, 0),
        ("mesh", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("time", None, TypeName::F32, 0),
        ("frame", None, TypeName::U32, 0),
    ];
    if kernel.params.len() != expected.len() {
        bail!(
            "mesh demo kernel `raster` must have {} params: global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame",
            expected.len()
        );
    }
    for (param, (name, address_space, ty, pointer_depth)) in kernel.params.iter().zip(expected) {
        if param.name != name
            || param.address_space != address_space
            || param.ty.base != ty
            || param.ty.pointer_depth != pointer_depth
        {
            bail!(
                "invalid mesh demo kernel parameter `{}`; expected `{}` in ABI `global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct InstanceKernel {
    raster: Kernel,
    cull: Option<Kernel>,
    tiled: bool,
    macrocell: bool,
}

impl InstanceKernel {
    fn compile(ctx: &NeoContext, path: &Path, variant: InstanceStressVariant) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        match variant {
            InstanceStressVariant::Baseline
            | InstanceStressVariant::Fast
            | InstanceStressVariant::Tiled => {
                validate_instance_kernel_abi(&source)?;
            }
            InstanceStressVariant::Culled => validate_culled_instance_kernel_abi(&source)?,
            InstanceStressVariant::Macrocell => validate_macrocell_instance_kernel_abi(&source)?,
        }
        let entrypoints = match variant {
            InstanceStressVariant::Baseline
            | InstanceStressVariant::Fast
            | InstanceStressVariant::Tiled
            | InstanceStressVariant::Macrocell => {
                vec!["instance_raster"]
            }
            InstanceStressVariant::Culled => vec!["instance_cull", "instance_raster"],
        };
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &entrypoints)?;
        Ok(Self {
            raster: module.kernel("instance_raster")?,
            cull: if variant == InstanceStressVariant::Culled {
                Some(module.kernel("instance_cull")?)
            } else {
                None
            },
            tiled: matches!(
                variant,
                InstanceStressVariant::Tiled | InstanceStressVariant::Macrocell
            ),
            macrocell: variant == InstanceStressVariant::Macrocell,
        })
    }
}

struct RasterStressKernel {
    cull_init: Kernel,
    cull: Kernel,
    graphics: neo_lang::GraphicsShaders,
    material: MaterialKernel,
}

impl RasterStressKernel {
    fn compile(ctx: &NeoContext, path: &Path, raster_plan: &HardwareRasterPlan) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let material = raster_plan.material_kernel();
        validate_raster_stress_abi_with_material(&source, material.abi())?;
        let graphics = neo_lang::lower_graphics_to_hlsl_for_entries_with_bindings(
            &source,
            &material.abi().vertex_entrypoint,
            &material.abi().fragment_entrypoint,
            graphics_bindings_for_material(material.abi())?,
        )?;
        let module = neo_runtime::Module::from_neo_source(
            ctx,
            &source,
            &["raster_cull_init", "raster_cull"],
        )?;
        Ok(Self {
            cull_init: module.kernel("raster_cull_init")?,
            cull: module.kernel("raster_cull")?,
            graphics,
            material,
        })
    }
}

fn raster_stress_source_path(requested: &Path) -> PathBuf {
    if requested.ends_with(Path::new("examples/live-window/live.neo")) {
        PathBuf::from("examples/stress-quads/hardware_raster.neo")
    } else {
        requested.to_path_buf()
    }
}

fn hardware_raster_visible_instances_for_log(
    policy: HardwareRasterDrawPolicy,
    grid: InstanceGrid,
) -> Option<u32> {
    match policy {
        HardwareRasterDrawPolicy::DrawAll => grid.count(),
        HardwareRasterDrawPolicy::ComputeCulled => None,
    }
}

#[cfg(test)]
fn validate_raster_stress_abi(source: &str) -> Result<()> {
    let material = hardware_raster_material();
    validate_raster_stress_abi_with_material(source, material.abi())
}

#[cfg(test)]
fn hardware_raster_material() -> MaterialKernel {
    MaterialKernel::from_stages("hardware-raster", "quad_vs", "quad_fs").with_abi(
        MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs"),
    )
}

fn validate_raster_stress_abi_with_material(
    source: &str,
    material: &MaterialKernelAbi,
) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "raster_cull_init",
        &[
            ("args", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ],
        "hardware raster cull init kernel",
        "global u8* args, global u8* camera",
    )?;
    validate_kernel_signature(
        &program,
        "raster_cull",
        &[
            ("args", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("visible", Some(AddressSpace::Global), TypeName::U8, 1usize),
            (
                "instances",
                Some(AddressSpace::Global),
                TypeName::U8,
                1usize,
            ),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("instance_count", None, TypeName::U32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "hardware raster cull kernel",
        "global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame",
    )?;
    let cull_init = program
        .kernels
        .iter()
        .find(|entry| {
            entry.kind == neo_lang::EntryPointKind::Kernel && entry.name == "raster_cull_init"
        })
        .ok_or_else(|| {
            anyhow!("hardware raster source must define `kernel fn raster_cull_init(...)`")
        })?;
    let cull_init_body = effective_shader_body(&cull_init.body);
    require_shader_body(
        &cull_init_body,
        "args",
        "hardware raster compute cull init kernel must write D3D12 indirect draw arguments",
    )?;
    let cull = program
        .kernels
        .iter()
        .find(|entry| entry.kind == neo_lang::EntryPointKind::Kernel && entry.name == "raster_cull")
        .ok_or_else(|| {
            anyhow!("hardware raster source must define `kernel fn raster_cull(...)`")
        })?;
    let cull_body = effective_shader_body(&cull.body);
    require_shader_body(
        &cull_body,
        "camera",
        "hardware raster compute cull kernel must read the explicit camera parameters",
    )?;
    require_shader_body(
        &cull_body,
        "visible",
        "hardware raster compute cull kernel must write the explicit visible InstanceStream",
    )?;
    require_shader_body(
        &cull_body,
        "args",
        "hardware raster compute cull kernel must write D3D12 indirect draw arguments",
    )?;
    validate_material_kernel_abi(&program, material)?;
    Ok(())
}

fn validate_material_kernel_abi(
    program: &neo_lang::Program,
    material: &MaterialKernelAbi,
) -> Result<()> {
    let vertex = program
        .kernels
        .iter()
        .find(|entry| {
            entry.kind == neo_lang::EntryPointKind::Vertex
                && entry.name == material.vertex_entrypoint
        })
        .ok_or_else(|| {
            anyhow!(
                "hardware raster source must define `vertex fn {}()`",
                material.vertex_entrypoint
            )
        })?;
    if !vertex.params.is_empty() {
        bail!(
            "hardware raster vertex stage must use ABI `vertex fn {}()`; use vertex_id(), instance_id(), and raster_* builtins for inputs",
            material.vertex_entrypoint
        );
    }
    let vertex_body = effective_shader_body(&vertex.body);
    for requirement in &material.vertex_requirements {
        validate_material_vertex_requirement(
            &vertex_body,
            *requirement,
            &material.vertex_entrypoint,
        )?;
    }
    let fragment = program
        .kernels
        .iter()
        .find(|entry| {
            entry.kind == neo_lang::EntryPointKind::Fragment
                && entry.name == material.fragment_entrypoint
        })
        .ok_or_else(|| {
            anyhow!(
                "hardware raster source must define `fragment fn {}()`",
                material.fragment_entrypoint
            )
        })?;
    if !fragment.params.is_empty() {
        bail!(
            "hardware raster fragment stage must use ABI `fragment fn {}()`; use input_color() for interpolated color",
            material.fragment_entrypoint
        );
    }
    let fragment_body = effective_shader_body(&fragment.body);
    for requirement in &material.fragment_requirements {
        validate_material_fragment_requirement(
            &fragment_body,
            *requirement,
            &material.fragment_entrypoint,
        )?;
    }
    Ok(())
}

fn validate_material_vertex_requirement(
    body: &str,
    requirement: MaterialVertexRequirement,
    entrypoint: &str,
) -> Result<()> {
    match requirement {
        MaterialVertexRequirement::VisibleInstanceStream => require_shader_body(
            body,
            "visible_instance_id(instance_id())",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must read the compute-culled InstanceStream with `visible_instance_id(instance_id())`"
            ),
        ),
        MaterialVertexRequirement::DirectInstanceId => require_shader_body(
            body,
            "instance_id()",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must read the explicit InstanceStream directly with `instance_id()`"
            ),
        ),
        MaterialVertexRequirement::InstancePosition => require_shader_body_any(
            body,
            &["neo_instance_position3f", "neo_stress_instance_position3f"],
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must read positions from the explicit InstanceStream"
            ),
        ),
        MaterialVertexRequirement::GeometryPosition => require_shader_body(
            body,
            "neo_geometry_position3f",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must read vertex positions from the explicit GeometryStream"
            ),
        ),
        MaterialVertexRequirement::ClipPositionOutput => require_shader_body(
            body,
            "set_position(",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must write a clip-space position with `set_position(...)`"
            ),
        ),
        MaterialVertexRequirement::VertexColorOutput => require_shader_body(
            body,
            "set_color(",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must write material color with `set_color(...)`"
            ),
        ),
    }
}

fn validate_material_fragment_requirement(
    body: &str,
    requirement: MaterialFragmentRequirement,
    entrypoint: &str,
) -> Result<()> {
    match requirement {
        MaterialFragmentRequirement::InterpolatedColorInput => require_shader_body(
            body,
            "input_color()",
            &format!(
                "hardware raster MaterialKernel `{entrypoint}` must return or consume `input_color()` from the vertex stage"
            ),
        ),
    }
}

fn require_shader_body(body: &str, required: &str, message: &str) -> Result<()> {
    if body.contains(required) {
        Ok(())
    } else {
        bail!("{message}")
    }
}

fn require_shader_body_any(body: &str, required: &[&str], message: &str) -> Result<()> {
    if required.iter().any(|required| body.contains(required)) {
        Ok(())
    } else {
        bail!("{message}")
    }
}

fn effective_shader_body(body: &str) -> String {
    remove_disabled_false_blocks(&strip_shader_comments(body))
}

fn strip_shader_comments(body: &str) -> String {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn remove_disabled_false_blocks(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut i = 0usize;
    while i < body.len() {
        if let Some(end) = disabled_if_false_block_end(body, i) {
            i = end;
        } else {
            out.push(body.as_bytes()[i] as char);
            i += 1;
        }
    }
    out
}

fn disabled_if_false_block_end(body: &str, start: usize) -> Option<usize> {
    let bytes = body.as_bytes();
    if start + 2 > bytes.len() || &bytes[start..start + 2] != b"if" {
        return None;
    }
    if start > 0 && is_ident_byte(bytes[start - 1]) {
        return None;
    }
    if start + 2 < bytes.len() && is_ident_byte(bytes[start + 2]) {
        return None;
    }
    let mut i = skip_ascii_ws(bytes, start + 2);
    if i >= bytes.len() || bytes[i] != b'(' {
        return None;
    }
    i = skip_ascii_ws(bytes, i + 1);
    let literal = b"false";
    if i + literal.len() > bytes.len() || &bytes[i..i + literal.len()] != literal {
        return None;
    }
    i = skip_ascii_ws(bytes, i + literal.len());
    if i >= bytes.len() || bytes[i] != b')' {
        return None;
    }
    i = skip_ascii_ws(bytes, i + 1);
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    let mut depth = 1usize;
    i += 1;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    Some(bytes.len())
}

fn skip_ascii_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn validate_instance_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "instance_raster",
        &[
            ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "instance stress kernel",
        "global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame",
    )
}

fn validate_culled_instance_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "instance_cull",
        &[
            ("cull", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "culled instance stress prepass kernel",
        "global u8* cull, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame",
    )?;
    validate_kernel_signature(
        &program,
        "instance_raster",
        &[
            ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("cull", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "culled instance stress raster kernel",
        "global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, global u8* cull, f32 time, u32 frame",
    )
}

fn validate_macrocell_instance_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "instance_raster",
        &[
            ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("visibility", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "macrocell instance stress kernel",
        "global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* visibility, global u8* camera, f32 time, u32 frame",
    )
}

fn validate_kernel_signature(
    program: &neo_lang::Program,
    name: &str,
    expected: &[(&str, Option<AddressSpace>, TypeName, usize)],
    label: &str,
    abi: &str,
) -> Result<()> {
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.kind == neo_lang::EntryPointKind::Kernel && kernel.name == name)
        .ok_or_else(|| anyhow!("{label} must define `kernel fn {name}(...)`"))?;
    if kernel.params.len() != expected.len() {
        bail!(
            "{label} `{name}` must have {} params: {abi}",
            expected.len(),
        );
    }
    for (param, (name, address_space, ty, pointer_depth)) in kernel.params.iter().zip(expected) {
        if param.name != *name
            || &param.address_space != address_space
            || &param.ty.base != ty
            || param.ty.pointer_depth != *pointer_depth
        {
            bail!(
                "invalid {label} parameter `{}`; expected `{}` in ABI `{abi}`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct ReloadState<T> {
    active: T,
    generation: u64,
    last_error: Option<String>,
}

impl<T> ReloadState<T> {
    fn new(active: T) -> Self {
        Self {
            active,
            generation: 0,
            last_error: None,
        }
    }

    fn try_replace<E: std::fmt::Display>(&mut self, result: Result<T, E>) -> bool {
        match result {
            Ok(next) => {
                self.active = next;
                self.generation += 1;
                self.last_error = None;
                true
            }
            Err(err) => {
                self.last_error = Some(err.to_string());
                false
            }
        }
    }
}

fn watch_source(
    path: &Path,
) -> Result<(
    notify::RecommendedWatcher,
    Receiver<notify::Result<NotifyEvent>>,
)> {
    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    })?;
    watcher.watch(path, RecursiveMode::NonRecursive)?;
    Ok((watcher, rx))
}

fn handle_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<LiveKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(LiveKernel::compile(ctx, source_path));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good kernel:\n{err}");
        }
    }
    Ok(())
}

fn handle_mesh_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<MeshKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(MeshKernel::compile(ctx, source_path));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good mesh kernel:\n{err}");
        }
    }
    Ok(())
}

fn handle_instance_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    variant: InstanceStressVariant,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<InstanceKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(InstanceKernel::compile(ctx, source_path, variant));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good instance kernel:\n{err}");
        }
    }
    Ok(())
}

fn handle_raster_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    raster_plan: &HardwareRasterPlan,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<RasterStressKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced =
            reload.try_replace(RasterStressKernel::compile(ctx, source_path, raster_plan));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good hardware raster source:\n{err}");
        }
    }
    Ok(())
}

struct FrameResources {
    width: u32,
    height: u32,
    device_pixels: DeviceBuffer<u8>,
    host_bgra: ReadablePinnedHostBuffer<u8>,
}

impl FrameResources {
    fn new(neo: &NeoContext, width: u32, height: u32) -> Result<Self> {
        let byte_len = frame_byte_len(width, height)?;
        Ok(Self {
            width,
            height,
            device_pixels: neo.alloc_zeros(byte_len)?,
            host_bgra: neo.alloc_readable_pinned(byte_len)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DemoVertex {
    position: [f32; 3],
    color_bgra: u32,
}

fn vertices_as_bytes(vertices: &[DemoVertex]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            vertices.as_ptr().cast::<u8>(),
            std::mem::size_of_val(vertices),
        )
        .to_vec()
    }
}

fn create_demo_mesh(neo: &NeoContext) -> Result<MeshBuffer> {
    let vertices = [
        DemoVertex {
            position: [-0.65, -0.65, 0.0],
            color_bgra: 0xffff_4040,
        },
        DemoVertex {
            position: [0.65, -0.65, 0.0],
            color_bgra: 0xff40_ff40,
        },
        DemoVertex {
            position: [0.65, 0.65, 0.0],
            color_bgra: 0xff40_40ff,
        },
        DemoVertex {
            position: [-0.65, 0.65, 0.0],
            color_bgra: 0xffff_ff40,
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    MeshBuffer::upload_typed(
        neo,
        MeshBufferDesc {
            vertex_count: vertices.len() as u32,
            vertex_layout: VertexLayout {
                stride: std::mem::size_of::<DemoVertex>() as u32,
                attributes: vec![
                    VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    },
                    VertexAttribute {
                        semantic: VertexSemantic::Color0,
                        format: VertexFormat::U8x4Unorm,
                        offset: 12,
                    },
                ],
            },
            index_format: IndexFormat::U16,
            index_count: indices.len() as u32,
            topology: PrimitiveTopology::TriangleList,
        },
        &vertices,
        &indices,
    )
    .context("failed to upload demo MeshBuffer")
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StressInstance {
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 2],
    color_bgra: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CameraParams {
    origin: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    grid: [u32; 4],
    view: [f32; 4],
    config: [u32; 4],
}

struct InstanceStressAssets {
    mesh: MeshBuffer,
    instances: InstanceBuffer,
    visibility: DeviceBuffer<u8>,
    raster_instances: Option<SharedInstanceStream>,
    camera_buffers: Vec<DeviceBuffer<u8>>,
    tile_cull_width: u32,
    tile_cull_height: u32,
    tile_cull_buffers: Vec<DeviceBuffer<u8>>,
}

fn create_instance_stress_assets(
    neo: &NeoContext,
    raster_interop: Option<&NeoD3d12InteropDevice>,
    grid: InstanceGrid,
    present_ring: usize,
    instance_layout: StressInstanceLayout,
) -> Result<InstanceStressAssets> {
    let mesh = create_demo_mesh(neo)?;
    let instance_count = grid
        .count()
        .ok_or_else(|| anyhow!("instance grid count overflow"))?;
    let instances = create_stress_instances(grid)?;
    let instance_desc = InstanceBufferDesc {
        instance_count,
        instance_layout: InstanceLayout {
            stride: std::mem::size_of::<StressInstance>() as u32,
            attributes: vec![
                InstanceAttribute {
                    semantic: InstanceSemantic::Position,
                    format: InstanceFormat::F32x3,
                    offset: 0,
                },
                InstanceAttribute {
                    semantic: InstanceSemantic::Rotation,
                    format: InstanceFormat::F32x4,
                    offset: 12,
                },
                InstanceAttribute {
                    semantic: InstanceSemantic::Scale,
                    format: InstanceFormat::F32x2,
                    offset: 28,
                },
                InstanceAttribute {
                    semantic: InstanceSemantic::Color0,
                    format: InstanceFormat::U8x4Unorm,
                    offset: 36,
                },
            ],
        },
    };
    let data_layout = instance_layout.data_layout();
    let visibility = DeviceBuffer::upload(neo, &create_visibility_grid_bytes(grid)?)
        .context("failed to upload macrocell visibility grid")?;
    let instance_buffer = InstanceBuffer::upload_typed_with_layout(
        neo,
        instance_desc.clone(),
        &instances,
        data_layout,
    )
    .context("failed to upload instance stress InstanceBuffer")?;
    let raster_instances = if let Some(interop) = raster_interop {
        Some(
            SharedInstanceStream::upload_typed(
                neo,
                interop,
                instance_desc.clone(),
                &instances,
                data_layout,
            )
            .context("failed to upload shared raster InstanceStream")?,
        )
    } else {
        None
    };
    let camera_len = std::mem::size_of::<CameraParams>();
    let mut camera_buffers = Vec::with_capacity(present_ring);
    for _ in 0..present_ring {
        camera_buffers.push(neo.alloc_zeros(camera_len)?);
    }
    Ok(InstanceStressAssets {
        mesh,
        instances: instance_buffer,
        visibility,
        raster_instances,
        camera_buffers,
        tile_cull_width: 0,
        tile_cull_height: 0,
        tile_cull_buffers: Vec::new(),
    })
}

fn tile_cull_grid_size(size: PhysicalSize<u32>) -> Result<(u32, u32)> {
    if size.width == 0 || size.height == 0 {
        bail!("tile cull size must be nonzero");
    }
    Ok((
        size.width.div_ceil(INSTANCE_CULL_TILE),
        size.height.div_ceil(INSTANCE_CULL_TILE),
    ))
}

fn tile_cull_byte_len(size: PhysicalSize<u32>) -> Result<usize> {
    let (tiles_x, tiles_y) = tile_cull_grid_size(size)?;
    tiles_x
        .checked_mul(tiles_y)
        .and_then(|tiles| tiles.checked_mul(TILE_CULL_RECORD_BYTES as u32))
        .map(|bytes| bytes as usize)
        .ok_or_else(|| {
            anyhow!(
                "tile cull buffer size overflow for {}x{}",
                size.width,
                size.height
            )
        })
}

fn macrocell_grid_size(grid: InstanceGrid) -> Result<(u32, u32, u32)> {
    grid.validate()?;
    Ok((
        grid.x.div_ceil(INSTANCE_MACROCELL_SIZE),
        grid.y.div_ceil(INSTANCE_MACROCELL_SIZE),
        grid.z.div_ceil(INSTANCE_MACROCELL_SIZE),
    ))
}

fn visibility_grid_u32_len(grid: InstanceGrid) -> Result<usize> {
    let (mx, my, mz) = macrocell_grid_size(grid)?;
    mx.checked_mul(my)
        .and_then(|xy| xy.checked_mul(mz))
        .and_then(|count| count.checked_mul(VISIBILITY_RECORD_U32S as u32))
        .and_then(|records| records.checked_add(VISIBILITY_HEADER_U32S as u32))
        .map(|values| values as usize)
        .ok_or_else(|| anyhow!("macrocell visibility grid size overflow"))
}

fn create_visibility_grid_bytes(grid: InstanceGrid) -> Result<Vec<u8>> {
    let (mx, my, mz) = macrocell_grid_size(grid)?;
    let mut values = vec![0u32; visibility_grid_u32_len(grid)?];
    values[0] = VISIBILITY_MAGIC;
    values[1] = INSTANCE_MACROCELL_SIZE;
    values[2] = mx;
    values[3] = my;
    values[4] = mz;
    values[5] = mx
        .checked_mul(my)
        .and_then(|xy| xy.checked_mul(mz))
        .ok_or_else(|| anyhow!("macrocell count overflow"))?;
    values[6] = 0;
    values[7] = 0;

    let mut record_index = VISIBILITY_HEADER_U32S;
    for z in 0..mz {
        for y in 0..my {
            for x in 0..mx {
                let min_x = x * INSTANCE_MACROCELL_SIZE;
                let min_y = y * INSTANCE_MACROCELL_SIZE;
                let min_z = z * INSTANCE_MACROCELL_SIZE;
                let max_x = (min_x + INSTANCE_MACROCELL_SIZE - 1).min(grid.x - 1);
                let max_y = (min_y + INSTANCE_MACROCELL_SIZE - 1).min(grid.y - 1);
                let max_z = (min_z + INSTANCE_MACROCELL_SIZE - 1).min(grid.z - 1);
                values[record_index..record_index + VISIBILITY_RECORD_U32S]
                    .copy_from_slice(&[min_x, max_x, min_y, max_y, min_z, max_z]);
                record_index += VISIBILITY_RECORD_U32S;
            }
        }
    }

    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn ensure_tile_cull_buffers(
    neo: &NeoContext,
    assets: &mut InstanceStressAssets,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let (tiles_x, tiles_y) = tile_cull_grid_size(size)?;
    if assets.tile_cull_width == tiles_x
        && assets.tile_cull_height == tiles_y
        && assets.tile_cull_buffers.len() == present_ring
    {
        return Ok(());
    }
    let byte_len = tile_cull_byte_len(size)?;
    let mut buffers = Vec::with_capacity(present_ring);
    for _ in 0..present_ring {
        buffers.push(neo.alloc_zeros(byte_len)?);
    }
    assets.tile_cull_width = tiles_x;
    assets.tile_cull_height = tiles_y;
    assets.tile_cull_buffers = buffers;
    Ok(())
}

fn create_stress_instances(grid: InstanceGrid) -> Result<Vec<StressInstance>> {
    let count = grid
        .count()
        .ok_or_else(|| anyhow!("instance grid count overflow"))? as usize;
    let mut instances = Vec::with_capacity(count);
    let spacing = 0.085f32;
    let ox = (grid.x as f32 - 1.0) * 0.5;
    let oy = (grid.y as f32 - 1.0) * 0.5;
    let oz = (grid.z as f32 - 1.0) * 0.5;
    for z in 0..grid.z {
        for y in 0..grid.y {
            for x in 0..grid.x {
                let id = z * grid.x * grid.y + y * grid.x + x;
                let px = (x as f32 - ox) * spacing;
                let py = (y as f32 - oy) * spacing;
                let pz = (z as f32 - oz) * spacing;
                let yaw = (x as f32 * 0.17 + z as f32 * 0.11).sin() * 0.45;
                let pitch = (y as f32 * 0.13 + z as f32 * 0.07).cos() * 0.35;
                let rotation = quat_from_euler(pitch, yaw, 0.0);
                let hash = id.wrapping_mul(747_796_405).wrapping_add(0x9e37_79b9);
                let r = (hash >> 16) & 255;
                let g = (hash >> 8) & 255;
                let b = hash & 255;
                instances.push(StressInstance {
                    position: [px, py, pz],
                    rotation,
                    scale: [spacing * 0.32, spacing * 0.32],
                    color_bgra: b | (g << 8) | (r << 16) | 0xff00_0000,
                });
            }
        }
    }
    Ok(instances)
}

fn quat_from_euler(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    let (sx, cx) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let (sz, cz) = (roll * 0.5).sin_cos();
    [
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    ]
}

fn camera_params_bytes(params: &CameraParams) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (params as *const CameraParams).cast::<u8>(),
            std::mem::size_of::<CameraParams>(),
        )
    }
}

fn instance_render_visibility(
    policy: RenderPolicy,
    window: WindowVisibilityState,
    camera: &CameraParams,
) -> RenderVisibility {
    if policy == RenderPolicy::ForceRender {
        return RenderVisibility::Visible;
    }
    let window_visibility = window.render_visibility();
    if window_visibility != RenderVisibility::Visible {
        return window_visibility;
    }
    if instance_lattice_visible(camera) {
        RenderVisibility::Visible
    } else {
        RenderVisibility::Empty
    }
}

fn instance_lattice_visible(camera: &CameraParams) -> bool {
    let grid_x = camera.grid[0] as f32;
    let grid_y = camera.grid[1] as f32;
    let grid_z = camera.grid[2] as f32;
    if grid_x <= 0.0 || grid_y <= 0.0 || grid_z <= 0.0 {
        return false;
    }
    let spacing = camera.view[2];
    if spacing <= 0.0 {
        return false;
    }
    let pad = spacing * 0.5;
    let extent_x = (grid_x - 1.0) * 0.5 * spacing + pad;
    let extent_y = (grid_y - 1.0) * 0.5 * spacing + pad;
    let extent_z = (grid_z - 1.0) * 0.5 * spacing + pad;
    let forward = [camera.forward[0], camera.forward[1], camera.forward[2]];
    let right = [camera.right[0], camera.right[1], camera.right[2]];
    let up = [camera.up[0], camera.up[1], camera.up[2]];
    let origin = [camera.origin[0], camera.origin[1], camera.origin[2]];
    let tan_x = camera.view[0].max(0.001);
    let tan_y = camera.view[1].max(0.001);
    let edge_slack = spacing * 3.0;

    let mut any_in_front = false;
    let mut outside_left = true;
    let mut outside_right = true;
    let mut outside_bottom = true;
    let mut outside_top = true;
    for x in [-extent_x, extent_x] {
        for y in [-extent_y, extent_y] {
            for z in [-extent_z, extent_z] {
                let rel = [x - origin[0], y - origin[1], z - origin[2]];
                let camera_x = dot3(rel, right);
                let camera_y = dot3(rel, up);
                let camera_z = dot3(rel, forward);
                any_in_front |= camera_z > 0.0;
                outside_right &= camera_z * tan_x - camera_x + edge_slack < 0.0;
                outside_left &= camera_z * tan_x + camera_x + edge_slack < 0.0;
                outside_top &= camera_z * tan_y - camera_y + edge_slack < 0.0;
                outside_bottom &= camera_z * tan_y + camera_y + edge_slack < 0.0;
            }
        }
    }

    any_in_front && !(outside_left || outside_right || outside_bottom || outside_top)
}

#[derive(Default)]
struct CameraController {
    auto: bool,
    origin: [f32; 3],
    yaw: f32,
    pitch: f32,
    moving_forward: bool,
    moving_backward: bool,
    moving_left: bool,
    moving_right: bool,
    moving_up: bool,
    moving_down: bool,
    right_mouse_down: bool,
    raw_mouse_seen_during_capture: bool,
    last_cursor: Option<(f64, f64)>,
    last_manual_time: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraWindowAction {
    None,
    CaptureMouse,
    ReleaseMouse,
    RecenterMouse,
}

impl CameraController {
    fn new() -> Self {
        Self {
            auto: true,
            origin: [0.0, -14.0, 5.2],
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: -0.32,
            ..Self::default()
        }
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> CameraWindowAction {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    return self.handle_key_code(code, pressed);
                }
            }
            WindowEvent::Focused(false) => {
                if self.right_mouse_down {
                    self.end_mouse_look();
                    return CameraWindowAction::ReleaseMouse;
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                if *state == ElementState::Pressed {
                    self.begin_mouse_look();
                    return CameraWindowAction::CaptureMouse;
                } else {
                    self.end_mouse_look();
                    return CameraWindowAction::ReleaseMouse;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.right_mouse_down && !self.raw_mouse_seen_during_capture {
                    if let Some((last_x, last_y)) = self.last_cursor {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.apply_mouse_delta(dx, dy);
                        self.last_cursor = None;
                        return CameraWindowAction::RecenterMouse;
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            _ => {}
        }
        CameraWindowAction::None
    }

    fn handle_key_code(&mut self, code: KeyCode, pressed: bool) -> CameraWindowAction {
        if pressed && code == KeyCode::Escape {
            self.end_mouse_look();
            return CameraWindowAction::ReleaseMouse;
        }
        match code {
            KeyCode::KeyW => self.moving_forward = pressed,
            KeyCode::KeyS => self.moving_backward = pressed,
            KeyCode::KeyA => self.moving_left = pressed,
            KeyCode::KeyD => self.moving_right = pressed,
            KeyCode::Space => self.moving_up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.moving_down = pressed,
            _ => {}
        }
        if pressed {
            self.auto = false;
        }
        CameraWindowAction::None
    }

    fn begin_mouse_look(&mut self) {
        self.right_mouse_down = true;
        self.raw_mouse_seen_during_capture = false;
        self.last_cursor = None;
        self.auto = false;
    }

    fn end_mouse_look(&mut self) {
        self.right_mouse_down = false;
        self.raw_mouse_seen_during_capture = false;
        self.last_cursor = None;
    }

    fn handle_raw_mouse_delta(&mut self, dx: f32, dy: f32) {
        if self.right_mouse_down {
            self.raw_mouse_seen_during_capture = true;
            self.apply_mouse_delta(dx, dy);
        }
    }

    fn apply_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.003;
        self.pitch = (self.pitch - dy * 0.003).clamp(-1.35, 1.35);
    }

    fn params(&mut self, size: PhysicalSize<u32>, time: f32, grid: InstanceGrid) -> CameraParams {
        let aspect = size.width as f32 / size.height as f32;
        let tan_y = (60.0f32.to_radians() * 0.5).tan();
        let tan_x = tan_y * aspect;
        let origin = if self.auto {
            self.last_manual_time = None;
            let radius = 14.5;
            let angle = time * 0.23;
            [angle.cos() * radius, angle.sin() * radius, 5.3]
        } else {
            self.update_manual(time);
            self.origin
        };
        let forward = if self.auto {
            normalize3([-origin[0], -origin[1], -origin[2] * 0.72])
        } else {
            [
                self.yaw.cos() * self.pitch.cos(),
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
            ]
        };
        let world_up = [0.0, 0.0, 1.0];
        let right = normalize3(cross3(forward, world_up));
        let up = normalize3(cross3(right, forward));
        CameraParams {
            origin: [origin[0], origin[1], origin[2], 0.0],
            right: [right[0], right[1], right[2], 0.0],
            up: [up[0], up[1], up[2], 0.0],
            forward: [forward[0], forward[1], forward[2], 0.0],
            grid: [grid.x, grid.y, grid.z, grid.count().unwrap_or(0)],
            view: [
                tan_x,
                tan_y,
                0.085,
                2.0 / size.width.min(size.height).max(1) as f32,
            ],
            config: [0, 32, 0, 0],
        }
    }

    fn update_manual(&mut self, time: f32) {
        let dt = self
            .last_manual_time
            .replace(time)
            .map(|last| (time - last).clamp(0.0, CAMERA_MAX_STEP_SECONDS))
            .unwrap_or(0.0);
        let step = CAMERA_MOVE_UNITS_PER_SEC * dt;
        let forward = [
            self.yaw.cos() * self.pitch.cos(),
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
        ];
        let right = normalize3(cross3(forward, [0.0, 0.0, 1.0]));
        let mut delta = [0.0, 0.0, 0.0];
        if self.moving_forward {
            add_scaled3(&mut delta, forward, step);
        }
        if self.moving_backward {
            add_scaled3(&mut delta, forward, -step);
        }
        if self.moving_right {
            add_scaled3(&mut delta, right, step);
        }
        if self.moving_left {
            add_scaled3(&mut delta, right, -step);
        }
        if self.moving_up {
            delta[2] += step;
        }
        if self.moving_down {
            delta[2] -= step;
        }
        self.origin[0] += delta[0];
        self.origin[1] += delta[1];
        self.origin[2] += delta[2];
    }
}

fn add_scaled3(dst: &mut [f32; 3], value: [f32; 3], scale: f32) {
    dst[0] += value[0] * scale;
    dst[1] += value[1] * scale;
    dst[2] += value[2] * scale;
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [1.0, 0.0, 0.0]
    }
}

struct ThroughputResources {
    width: u32,
    height: u32,
    slots: Vec<ThroughputSlot>,
}

struct ThroughputSlot {
    stream: CudaStream,
    device_pixels: DeviceBuffer<u8>,
    graph_params_device: DeviceBuffer<u8>,
    graph_params_host: ReadablePinnedHostBuffer<u8>,
    graph: Option<CudaGraph>,
    host_bgra: ReadablePinnedHostBuffer<u8>,
    ready: CudaFence,
    state: ThroughputSlotState,
    frame: u32,
    completed_kernels: u32,
    sample_started_at: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThroughputSlotState {
    Free,
    Pending { readback: bool },
}

impl ThroughputResources {
    fn new(neo: &NeoContext, width: u32, height: u32, slots: usize) -> Result<Self> {
        let byte_len = frame_byte_len(width, height)?;
        let mut ring = Vec::with_capacity(slots);
        for _ in 0..slots {
            let stream = neo.create_stream()?;
            ring.push(ThroughputSlot {
                device_pixels: DeviceBuffer::new_on_stream(&stream, byte_len)?,
                graph_params_device: DeviceBuffer::new_on_stream(
                    &stream,
                    std::mem::size_of::<LiveGraphParams>(),
                )?,
                graph_params_host: neo
                    .alloc_readable_pinned(std::mem::size_of::<LiveGraphParams>())?,
                graph: None,
                host_bgra: neo.alloc_readable_pinned(byte_len)?,
                ready: stream.create_fence()?,
                stream,
                state: ThroughputSlotState::Free,
                frame: 0,
                completed_kernels: 0,
                sample_started_at: None,
            });
        }
        Ok(Self {
            width,
            height,
            slots: ring,
        })
    }

    fn newest_completed_slot_index(&self) -> Result<Option<usize>> {
        let mut completed = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            if matches!(slot.state, ThroughputSlotState::Pending { readback: true })
                && slot.ready.is_complete()?
            {
                completed.push((index, slot.frame));
            }
        }
        Ok(choose_newest_completed_slot(completed))
    }

    fn drain_completed(
        &mut self,
        presenter: &mut PresentSink,
        size: PhysicalSize<u32>,
    ) -> Result<CompletedPresentation> {
        let present_index = self.newest_completed_slot_index()?;
        let mut stats = CompletedPresentation::default();

        for index in 0..self.slots.len() {
            let state = self.slots[index].state;
            if !matches!(state, ThroughputSlotState::Pending { .. })
                || !self.slots[index].ready.is_complete()?
            {
                continue;
            }

            stats.completed_kernels += self.slots[index].completed_kernels;
            if Some(index) == present_index {
                let slot = &mut self.slots[index];
                stats.sample_download += slot
                    .sample_started_at
                    .map(|started| started.elapsed())
                    .unwrap_or_default();
                let timings = presenter.present_sampled(
                    size,
                    slot.host_bgra.as_slice(),
                    stats.sample_download,
                )?;
                if let Some(timings) = timings {
                    stats.present = timings;
                    stats.sampled_frames = 1;
                    stats.presented_frames = 1;
                }
            }

            self.slots[index].state = ThroughputSlotState::Free;
            self.slots[index].completed_kernels = 0;
            self.slots[index].sample_started_at = None;
        }

        Ok(stats)
    }

    fn launch_graph_slot(&mut self, index: usize, input: GraphSlotLaunch<'_>) -> Result<()> {
        let slot = &mut self.slots[index];
        write_live_graph_params(
            &mut slot.graph_params_host,
            LiveGraphParams {
                time: input.time,
                frame: input.frame,
            },
        );
        if slot.graph.is_none() {
            let graph_kernel = input.kernel.on_stream(&slot.stream);
            slot.stream
                .synchronize()
                .context("failed to synchronize stream before CUDA graph capture")?;
            slot.stream
                .begin_graph_capture()
                .context("failed to begin CUDA graph capture")?;
            slot.graph_params_device
                .upload_from_readable_pinned_on_stream(&slot.stream, &slot.graph_params_host)
                .context("failed to capture CUDA graph parameter upload")?;
            {
                let mut launch = graph_kernel.launcher();
                launch
                    .arg_buffer_mut(&mut slot.device_pixels)
                    .arg(&input.width)
                    .arg(&input.height)
                    .arg_buffer(&slot.graph_params_device);
                unsafe {
                    launch
                        .launch(input.dims)
                        .context("failed to capture CUDA graph kernel launch")?;
                }
            }
            slot.graph = Some(
                slot.stream
                    .end_graph_capture()?
                    .ok_or_else(|| anyhow!("CUDA graph capture did not produce a graph"))
                    .context("failed to end CUDA graph capture")?,
            );
        }
        slot.graph
            .as_ref()
            .expect("graph was just initialized")
            .launch()
            .context("failed to launch CUDA graph")?;
        if input.readback {
            slot.device_pixels
                .download_into_readable_pinned_on_stream(&slot.stream, &mut slot.host_bgra)?;
            slot.sample_started_at = Some(Instant::now());
        } else {
            slot.sample_started_at = None;
        }
        slot.ready.record_on_stream(&slot.stream)?;
        slot.state = ThroughputSlotState::Pending {
            readback: input.readback,
        };
        slot.frame = input.frame;
        slot.completed_kernels = 1;
        Ok(())
    }
}

struct GraphSlotLaunch<'a> {
    kernel: &'a Kernel,
    dims: LaunchDims,
    width: u32,
    height: u32,
    time: f32,
    frame: u32,
    readback: bool,
}

struct InteropThroughputResources {
    width: u32,
    height: u32,
    ring: SharedFrameRing,
    slots: Vec<InteropThroughputSlot>,
}

struct InteropThroughputSlot {
    stream: CudaStream,
    state: InteropSlotState,
    frame: u32,
    cuda_done_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InteropSlotState {
    Free,
    Pending,
    Completed,
}

impl InteropThroughputResources {
    fn new(
        neo: &NeoContext,
        interop: &NeoD3d12InteropDevice,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<Self> {
        interop_trace("create shared frame ring");
        let ring = interop
            .create_shared_frame_ring(width, height, slots)
            .context("failed to create D3D12/CUDA shared frame ring")?;
        interop_trace("created shared frame ring");
        let mut aux = Vec::with_capacity(slots);
        for _ in 0..slots {
            aux.push(InteropThroughputSlot {
                stream: neo.create_stream()?,
                state: InteropSlotState::Free,
                frame: 0,
                cuda_done_value: 0,
            });
        }
        Ok(Self {
            width,
            height,
            ring,
            slots: aux,
        })
    }

    fn has_live_work(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.state != InteropSlotState::Free)
    }

    fn drain_completed(
        &mut self,
        presenter: &mut WindowPresenter,
        size: PhysicalSize<u32>,
        present_limiter: &mut PresentRateLimiter,
    ) -> Result<ThroughputBatchStats> {
        let mut stats = ThroughputBatchStats::default();
        for index in 0..self.slots.len() {
            if self.slots[index].state != InteropSlotState::Pending {
                continue;
            }
            let complete = self
                .ring
                .slot(index)
                .is_some_and(|shared| shared.is_fence_complete(self.slots[index].cuda_done_value));
            if !complete {
                continue;
            }
            stats.completed_kernels += 1;
            self.slots[index].state = InteropSlotState::Completed;
        }

        let newest = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == InteropSlotState::Completed)
            .max_by_key(|(_, slot)| slot.frame)
            .map(|(index, _)| index);

        if newest.is_some() && present_limiter.try_consume(Instant::now()) {
            if let Some(index) = newest {
                self.present_completed_slot(index, presenter, size, &mut stats)?;
            }
        }

        let newest_to_retain = if stats.presented_frames > 0 {
            None
        } else {
            newest
        };
        for index in 0..self.slots.len() {
            if self.slots[index].state == InteropSlotState::Completed
                && Some(index) != newest_to_retain
            {
                self.slots[index].state = InteropSlotState::Free;
            }
        }
        Ok(stats)
    }

    fn present_completed_slot(
        &mut self,
        index: usize,
        presenter: &mut WindowPresenter,
        size: PhysicalSize<u32>,
        stats: &mut ThroughputBatchStats,
    ) -> Result<()> {
        interop_trace("present shared frame");
        let timings = presenter.present_shared(
            size,
            self.ring.pitch_bytes(),
            self.ring
                .slot_mut(index)
                .context("missing shared interop slot")?,
            self.slots[index].cuda_done_value,
        )?;
        stats.present += timings.total;
        stats.draw += timings.draw;
        stats.swap_present += timings.swap_present;
        stats.sampled_frames += 1;
        stats.presented_frames += 1;
        self.slots[index].state = InteropSlotState::Free;
        interop_trace("presented shared frame");
        Ok(())
    }
}

struct RasterStressResources {
    slots: Vec<RasterStressSlot>,
    shaders: neo_lang::GraphicsShaders,
    visible_capacity: u32,
}

struct RasterStressSlot {
    stream: CudaStream,
    args: IndirectDrawBuffer,
    visible_ids: VisibleInstanceStream,
    draw_all_key: Option<DrawAllStaticKey>,
    camera: CameraParams,
    state: InteropSlotState,
    frame: u32,
    cuda_done_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DrawAllStaticKey {
    instance_count: u32,
    index_count: u32,
}

struct RasterDrainInput<'a> {
    presenter: &'a mut WindowPresenter,
    size: PhysicalSize<u32>,
    raster_instances: &'a mut SharedInstanceStream,
    material: &'a MaterialKernelAbi,
    geometry: &'a HardwareRasterGeometryStreamPlan,
    use_depth: bool,
    present_limiter: &'a mut PresentRateLimiter,
    grid: InstanceGrid,
}

impl RasterStressResources {
    fn new(
        neo: &NeoContext,
        interop: &NeoD3d12InteropDevice,
        slots: usize,
        shaders: neo_lang::GraphicsShaders,
        visible_capacity: u32,
    ) -> Result<Self> {
        if visible_capacity == 0 {
            bail!("hardware raster visible instance capacity must be greater than zero");
        }
        let mut ring = Vec::with_capacity(slots);
        for _ in 0..slots {
            ring.push(RasterStressSlot {
                stream: neo.create_stream()?,
                args: IndirectDrawBuffer::new(interop, 1)?,
                visible_ids: VisibleInstanceStream::new(interop, visible_capacity)?,
                draw_all_key: None,
                camera: CameraParams::default(),
                state: InteropSlotState::Free,
                frame: 0,
                cuda_done_value: 0,
            });
        }
        Ok(Self {
            slots: ring,
            shaders,
            visible_capacity,
        })
    }

    fn drain_completed(&mut self, input: RasterDrainInput<'_>) -> Result<ThroughputBatchStats> {
        let mut stats = ThroughputBatchStats::default();
        for index in 0..self.slots.len() {
            if self.slots[index].state != InteropSlotState::Pending {
                continue;
            }
            let complete = self.slots[index]
                .args
                .buffer()
                .is_fence_complete(self.slots[index].cuda_done_value);
            if !complete {
                continue;
            }
            stats.completed_kernels += 1;
            self.slots[index].state = InteropSlotState::Completed;
        }

        let newest = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == InteropSlotState::Completed)
            .max_by_key(|(_, slot)| slot.frame)
            .map(|(index, _)| index);
        if newest.is_some() && input.present_limiter.try_consume(Instant::now()) {
            let index = newest.expect("checked is_some");
            let slot = &mut self.slots[index];
            let cuda_done_value = slot.cuda_done_value;
            let frame = slot.frame;
            let timings = input
                .presenter
                .present_raster_indirect(RasterPresentInput {
                    size: input.size,
                    args: slot.args.buffer_mut(),
                    visible_ids: slot.visible_ids.buffer_mut(),
                    raster_instances: input.raster_instances.buffer_mut(),
                    material: input.material,
                    geometry: input.geometry,
                    cuda_done_value,
                    frame,
                    grid: input.grid,
                    camera: slot.camera,
                    shaders: &self.shaders,
                    use_depth: input.use_depth,
                })?;
            stats.present += timings.total;
            stats.draw += timings.draw;
            stats.swap_present += timings.swap_present;
            stats.sampled_frames += 1;
            stats.presented_frames += 1;
            slot.state = InteropSlotState::Free;
        }
        for index in 0..self.slots.len() {
            if self.slots[index].state == InteropSlotState::Completed && Some(index) != newest {
                self.slots[index].state = InteropSlotState::Free;
            }
        }
        Ok(stats)
    }
}

fn draw_all_identity_visible_bytes(instance_count: u32) -> Result<Vec<u8>> {
    let byte_len = usize::try_from(instance_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| anyhow!("draw-all visible stream size overflow"))?;
    let mut bytes = Vec::with_capacity(byte_len);
    for id in 0..instance_count {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Ok(bytes)
}

fn draw_all_indirect_command(index_count: u32, instance_count: u32) -> DrawIndexedIndirectCommand {
    DrawIndexedIndirectCommand {
        index_count_per_instance: index_count,
        instance_count,
        start_index_location: 0,
        base_vertex_location: 0,
        start_instance_location: 0,
    }
}

fn ensure_draw_all_slot_static_streams(
    slot: &mut RasterStressSlot,
    key: DrawAllStaticKey,
    needs_visible_stream: bool,
) -> Result<u64> {
    if slot.draw_all_key == Some(key) {
        return Ok(0);
    }
    slot.args
        .buffer()
        .wait_available_on_stream(&slot.stream)
        .context("failed to make CUDA wait for hardware raster argument buffer")?;
    if needs_visible_stream {
        let visible = draw_all_identity_visible_bytes(key.instance_count)?;
        slot.visible_ids
            .buffer_mut()
            .upload_bytes_on_stream(&slot.stream, &visible)
            .context("failed to upload draw-all identity visible stream")?;
    }
    let args = draw_all_indirect_command(key.index_count, key.instance_count);
    slot.args
        .buffer_mut()
        .upload_bytes_on_stream(&slot.stream, args.as_bytes())
        .context("failed to upload draw-all indirect args")?;
    let done = slot
        .args
        .buffer_mut()
        .signal_cuda_complete_on_stream(&slot.stream)
        .context("failed to signal draw-all static raster upload")?;
    slot.draw_all_key = Some(key);
    Ok(done)
}

fn ensure_raster_stress_resources(
    neo: &NeoContext,
    interop: &NeoD3d12InteropDevice,
    resources: &mut Option<RasterStressResources>,
    present_ring: usize,
    shaders: &neo_lang::GraphicsShaders,
    visible_capacity: u32,
) -> Result<()> {
    if resources.as_ref().is_none_or(|resources| {
        resources.slots.len() != present_ring
            || resources.shaders != *shaders
            || resources.visible_capacity != visible_capacity
    }) {
        *resources = Some(RasterStressResources::new(
            neo,
            interop,
            present_ring,
            shaders.clone(),
            visible_capacity,
        )?);
    }
    Ok(())
}

struct RasterStressBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<RasterStressResources>,
    kernel: &'a RasterStressKernel,
    assets: &'a mut InstanceStressAssets,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
    instance_grid: InstanceGrid,
    camera: CameraParams,
    raster_plan: &'a HardwareRasterPlan,
}

#[cfg(windows)]
fn hardware_raster_draw_recipe<'a>(
    assets: &'a InstanceStressAssets,
    material: &'a MaterialKernel,
    size: PhysicalSize<u32>,
    plan: &HardwareRasterPlan,
) -> Result<DrawExecution<'a>> {
    let target = Target::new(size.width.max(1), size.height.max(1))?;
    let builder = DrawExecution::builder(GeometryStream::from_mesh(&assets.mesh), material, target)
        .instance_stream(InstanceStream::from_instances(&assets.instances));
    let builder = match plan.draw_policy {
        HardwareRasterDrawPolicy::DrawAll => builder.draw_policy(DrawPolicy::DrawAll),
        HardwareRasterDrawPolicy::ComputeCulled => {
            builder.compute_culled(plan.cull_order.runtime_order())
        }
    };
    Ok(builder.try_build()?)
}

fn run_raster_stress_batch(input: RasterStressBatch<'_>) -> Result<ThroughputBatchStats> {
    let material_abi = input.kernel.material.abi().clone();
    {
        let draw = hardware_raster_draw_recipe(
            input.assets,
            &input.kernel.material,
            input.size,
            input.raster_plan,
        )?;
        debug_assert_eq!(
            draw.material().vertex_entrypoint(),
            input.raster_plan.material.vertex_entrypoint
        );
        debug_assert!(draw.instances().is_some());
    }
    let instance_count = input
        .instance_grid
        .count()
        .ok_or_else(|| anyhow!("instance grid count overflow"))?;
    ensure_raster_stress_resources(
        input.neo,
        input.interop,
        input.resources,
        input.present_ring,
        &input.kernel.graphics,
        instance_count,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("raster resources were just created");
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("draw-stress does not use the present thread"),
    };
    let raster_instances = input
        .assets
        .raster_instances
        .as_mut()
        .context("draw-stress is missing its shared InstanceStream")?;

    let _ = input.start;
    stats += resources.drain_completed(RasterDrainInput {
        presenter: direct_presenter,
        size: input.size,
        raster_instances,
        material: &material_abi,
        geometry: &input.raster_plan.geometry_stream,
        use_depth: input.raster_plan.uses_depth(),
        present_limiter: input.present_limiter,
        grid: input.instance_grid,
    })?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let mut camera = input.camera;
    camera.config[0] = input.raster_plan.visibility.code();
    camera.config[2] = input.raster_plan.geometry_stream.index_count();
    camera.config[3] = input.raster_plan.cull_order.code();
    let min_dimension = input.size.width.min(input.size.height).max(1) as f32;
    camera.view[3] =
        (input.raster_plan.min_projected_millipixels as f32 / 1000.0) * 2.0 / min_dimension;
    let camera_bytes = camera_params_bytes(&camera);
    let draw_all_key = DrawAllStaticKey {
        instance_count,
        index_count: input.raster_plan.geometry_stream.index_count(),
    };
    let mut launched = 0u32;
    for (index, slot) in resources.slots.iter_mut().enumerate() {
        if launched >= input.max_inflight {
            break;
        }
        if slot.state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let frame = *input.next_frame;
        let cuda_done_value = match input.raster_plan.draw_policy {
            HardwareRasterDrawPolicy::DrawAll => ensure_draw_all_slot_static_streams(
                slot,
                draw_all_key,
                input.raster_plan.material.kind.requires_visible_stream(),
            )?,
            HardwareRasterDrawPolicy::ComputeCulled => {
                slot.args
                    .buffer()
                    .wait_available_on_stream(&slot.stream)
                    .context("failed to make CUDA wait for hardware raster argument buffer")?;
                let arg_ptr = slot.args.buffer().device_ptr_arg();
                let visible_ptr = slot.visible_ids.buffer().device_ptr_arg();
                let instances_ptr = raster_instances.buffer().device_ptr_arg();
                input.assets.camera_buffers[index]
                    .upload_from_on_stream(&slot.stream, camera_bytes)
                    .context("failed to upload hardware raster camera params")?;
                let init_kernel = input.kernel.cull_init.on_stream(&slot.stream);
                let mut init_launch = init_kernel.launcher();
                init_launch
                    .arg_device_ptr(&arg_ptr)
                    .arg_buffer(&input.assets.camera_buffers[index]);
                unsafe {
                    init_launch
                        .launch(LaunchDims::for_2d(1, 1, (1, 1)))
                        .context("failed to launch hardware raster cull init kernel")?;
                }
                let stream_kernel = input.kernel.cull.on_stream(&slot.stream);
                let mut launch = stream_kernel.launcher();
                launch
                    .arg_device_ptr(&arg_ptr)
                    .arg_device_ptr(&visible_ptr)
                    .arg_device_ptr(&instances_ptr)
                    .arg_buffer(&input.assets.camera_buffers[index])
                    .arg(&instance_count)
                    .arg(&frame);
                unsafe {
                    launch
                        .launch(LaunchDims::for_2d(instance_count, 1, (256, 1)))
                        .context("failed to launch hardware raster cull/indirect kernel")?;
                }
                slot.args
                    .buffer_mut()
                    .signal_cuda_complete_on_stream(&slot.stream)
                    .context("failed to signal hardware raster indirect buffer completion")?
            }
        };
        slot.state = InteropSlotState::Pending;
        slot.frame = frame;
        slot.camera = camera;
        slot.cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

fn ensure_interop_throughput_resources(
    neo: &NeoContext,
    interop: &NeoD3d12InteropDevice,
    resources: &mut Option<InteropThroughputResources>,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let recreate = resources.as_ref().is_none_or(|resources| {
        resources.width != size.width
            || resources.height != size.height
            || resources.ring.len() != present_ring
    });
    if recreate {
        if let Some(resources) = resources.as_mut() {
            resources.ring.wait_idle(interop.queue())?;
        }
        *resources = Some(InteropThroughputResources::new(
            neo,
            interop,
            size.width,
            size.height,
            present_ring,
        )?);
    }
    Ok(())
}

struct InteropThroughputBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a Kernel,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_interop_throughput_batch(input: InteropThroughputBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, BLOCK);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("d3d12-interop does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        interop_trace("cuda wait slot available");
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        interop_trace("launch kernel into shared frame");
        unsafe {
            input.kernel.on_stream(stream).launch_image_raw_ptr(
                dims,
                shared.device_ptr_arg(),
                kernel_width,
                input.size.height,
                input.start.elapsed().as_secs_f32(),
                *input.next_frame,
            )
        }
        .context("failed to launch Neo kernel into D3D12 shared frame")?;
        interop_trace("cuda signal frame complete");
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = *input.next_frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

struct MeshDemoBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a Kernel,
    mesh: &'a MeshBuffer,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_mesh_demo_batch(input: MeshDemoBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, BLOCK);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("mesh-demo does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        let pixel_arg = shared.device_ptr_arg();
        let stream_kernel = input.kernel.on_stream(stream);
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        {
            let mut launch = stream_kernel.launcher();
            launch
                .arg_device_ptr(&pixel_arg)
                .arg(&kernel_width)
                .arg(&input.size.height)
                .arg_mesh(input.mesh)
                .arg(&time)
                .arg(&frame);
            unsafe {
                launch
                    .launch(dims)
                    .context("failed to launch mesh demo raster kernel")?;
            }
        }
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

struct InstanceStressBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a InstanceKernel,
    assets: &'a mut InstanceStressAssets,
    camera: CameraParams,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_instance_stress_batch(input: InstanceStressBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    if input.kernel.cull.is_some() {
        ensure_tile_cull_buffers(
            input.neo,
            input.assets,
            PhysicalSize::new(kernel_width, input.size.height),
            resources.slots.len(),
        )?;
    }
    let raster_block = if input.kernel.tiled { (8, 8) } else { BLOCK };
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, raster_block);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("instance-stress does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let camera_bytes = camera_params_bytes(&input.camera);
    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        input.assets.camera_buffers[index]
            .upload_from_on_stream(stream, camera_bytes)
            .context("failed to upload instance stress camera params")?;
        let pixel_arg = shared.device_ptr_arg();
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        if let Some(cull_kernel) = &input.kernel.cull {
            let (tiles_x, tiles_y) =
                tile_cull_grid_size(PhysicalSize::new(kernel_width, input.size.height))?;
            let cull_dims = LaunchDims::for_2d(tiles_x, tiles_y, BLOCK);
            let stream_kernel = cull_kernel.on_stream(stream);
            let mut launch = stream_kernel.launcher();
            launch
                .arg_buffer(&input.assets.tile_cull_buffers[index])
                .arg(&kernel_width)
                .arg(&input.size.height)
                .arg_mesh(&input.assets.mesh)
                .arg_instances(&input.assets.instances)
                .arg_buffer(&input.assets.camera_buffers[index])
                .arg(&time)
                .arg(&frame);
            unsafe {
                launch
                    .launch(cull_dims)
                    .context("failed to launch instance stress cull kernel")?;
            }
        }
        let stream_kernel = input.kernel.raster.on_stream(stream);
        {
            let mut launch = stream_kernel.launcher();
            if input.kernel.macrocell {
                launch
                    .arg_device_ptr(&pixel_arg)
                    .arg(&kernel_width)
                    .arg(&input.size.height)
                    .arg_mesh(&input.assets.mesh)
                    .arg_instances(&input.assets.instances)
                    .arg_buffer(&input.assets.visibility)
                    .arg_buffer(&input.assets.camera_buffers[index]);
            } else {
                launch
                    .arg_device_ptr(&pixel_arg)
                    .arg(&kernel_width)
                    .arg(&input.size.height)
                    .arg_mesh(&input.assets.mesh)
                    .arg_instances(&input.assets.instances)
                    .arg_buffer(&input.assets.camera_buffers[index]);
                if input.kernel.cull.is_some() {
                    launch.arg_buffer(&input.assets.tile_cull_buffers[index]);
                }
            }
            launch.arg(&time).arg(&frame);
            unsafe {
                launch
                    .launch(dims)
                    .context("failed to launch instance stress raster kernel")?;
            }
        }
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LiveGraphParams {
    time: f32,
    frame: u32,
}

fn write_live_graph_params(dst: &mut ReadablePinnedHostBuffer<u8>, params: LiveGraphParams) {
    debug_assert_eq!(dst.len(), std::mem::size_of::<LiveGraphParams>());
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&params as *const LiveGraphParams).cast::<u8>(),
            std::mem::size_of::<LiveGraphParams>(),
        )
    };
    dst.as_mut_slice().copy_from_slice(bytes);
}

#[derive(Default)]
struct CompletedPresentation {
    completed_kernels: u32,
    sample_download: Duration,
    present: PresentTimings,
    sampled_frames: u32,
    presented_frames: u32,
}

fn frame_byte_len(width: u32, height: u32) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .map(|bytes| bytes as usize)
        .ok_or_else(|| anyhow!("framebuffer size overflow for {width}x{height}"))
}

fn ensure_frame_resources(
    neo: &NeoContext,
    resources: &mut Option<FrameResources>,
    size: PhysicalSize<u32>,
) -> Result<()> {
    let recreate = resources
        .as_ref()
        .is_none_or(|resources| resources.width != size.width || resources.height != size.height);
    if recreate {
        *resources = Some(FrameResources::new(neo, size.width, size.height)?);
    }
    Ok(())
}

fn ensure_throughput_resources(
    neo: &NeoContext,
    resources: &mut Option<ThroughputResources>,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let recreate = resources.as_ref().is_none_or(|resources| {
        resources.width != size.width
            || resources.height != size.height
            || resources.slots.len() != present_ring
    });
    if recreate {
        *resources = Some(ThroughputResources::new(
            neo,
            size.width,
            size.height,
            present_ring,
        )?);
    }
    Ok(())
}

unsafe fn copy_bgra_rows(src: *const u8, dst: *mut u8, width: u32, height: u32, dst_pitch: usize) {
    let src_pitch = width as usize * 4;
    for y in 0..height as usize {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(y * src_pitch),
                dst.add(y * dst_pitch),
                src_pitch,
            );
        }
    }
}

fn choose_newest_completed_slot(slots: impl IntoIterator<Item = (usize, u32)>) -> Option<usize> {
    slots
        .into_iter()
        .max_by_key(|(_, frame)| *frame)
        .map(|(index, _)| index)
}

unsafe fn copy_bgra_to_mapped(
    src: &[u8],
    dst: *mut u8,
    width: u32,
    height: u32,
    dst_pitch: usize,
) -> bool {
    let src_pitch = width as usize * 4;
    if dst_pitch == src_pitch {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        true
    } else {
        unsafe {
            copy_bgra_rows(src.as_ptr(), dst, width, height, dst_pitch);
        }
        false
    }
}

struct RenderFrame<'a> {
    neo: &'a NeoContext,
    resources: &'a mut Option<FrameResources>,
    kernel: &'a Kernel,
    size: PhysicalSize<u32>,
    time: f32,
    frame: u32,
}

fn render_frame(input: RenderFrame<'_>) -> Result<FrameTimings> {
    ensure_frame_resources(input.neo, input.resources, input.size)?;
    let resources = input
        .resources
        .as_mut()
        .expect("frame resources were just created for nonzero size");
    let width = input.size.width;
    let height = input.size.height;
    let dims = LaunchDims::for_2d(width, height, BLOCK);
    let launch_start = Instant::now();
    {
        let mut launch = input.kernel.launcher();
        launch
            .arg_buffer_mut(&mut resources.device_pixels)
            .arg(&width)
            .arg(&height)
            .arg(&input.time)
            .arg(&input.frame);
        unsafe {
            launch.launch(dims)?;
        }
    }
    let download_start = Instant::now();
    resources
        .device_pixels
        .download_into_readable_pinned(&mut resources.host_bgra)?;
    let sync_start = Instant::now();
    input.neo.synchronize()?;
    Ok(FrameTimings {
        launch: download_start - launch_start,
        download: sync_start - download_start,
        sync: sync_start.elapsed(),
        ..FrameTimings::default()
    })
}

struct ThroughputBatch<'a> {
    neo: &'a NeoContext,
    resources: &'a mut Option<ThroughputResources>,
    graph_kernel: &'a Kernel,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    next_sample_at: &'a mut u64,
    next_present_at: &'a mut Option<Instant>,
    sample_every: u32,
    present_interval: Option<Duration>,
    max_inflight: u32,
    present_ring: usize,
}

fn run_kernel_throughput_batch(input: ThroughputBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_throughput_resources(input.neo, input.resources, input.size, input.present_ring)?;
    let resources = input
        .resources
        .as_mut()
        .expect("frame resources were just created for nonzero size");
    let width = input.size.width;
    let height = input.size.height;
    let dims = LaunchDims::for_2d(width, height, BLOCK);
    let mut stats = ThroughputBatchStats::default();

    stats += input.presenter.drain_results()?;
    let completed = resources.drain_completed(input.presenter, input.size)?;
    if completed.completed_kernels > 0 {
        stats.completed_kernels += completed.completed_kernels;
        *input.completed_kernels += u64::from(completed.completed_kernels);
    }
    if completed.presented_frames > 0 {
        stats.sample_download += completed.sample_download;
        stats.present += completed.present.total;
        stats.map_copy += completed.present.map_copy;
        stats.draw += completed.present.draw;
        stats.swap_present += completed.present.swap_present;
        stats.sampled_frames += completed.sampled_frames;
        stats.presented_frames += completed.presented_frames;
        stats.sampled_bytes += frame_byte_len(width, height)?;
        stats.uploaded_bytes += frame_byte_len(width, height)?;
    }

    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let should_sample_by_count =
        input.present_interval.is_none() && *input.completed_kernels >= *input.next_sample_at;
    let should_sample_by_time = input
        .next_present_at
        .is_some_and(|next_present_at| Instant::now() >= next_present_at);
    let mut should_readback = should_sample_by_count || should_sample_by_time;
    let mut launched = 0u32;
    for launch_slot_index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[launch_slot_index].state != ThroughputSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        resources
            .launch_graph_slot(
                launch_slot_index,
                GraphSlotLaunch {
                    kernel: input.graph_kernel,
                    dims,
                    width,
                    height,
                    time,
                    frame,
                    readback: should_readback,
                },
            )
            .context("failed to submit CUDA graph slot")?;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
        should_readback = false;
    }

    if launched == 0 {
        return Ok(stats);
    }

    let wait_start = Instant::now();
    stats.completion_wait += wait_start.elapsed();

    if should_sample_by_count || should_sample_by_time {
        if let Some(interval) = input.present_interval {
            let now = Instant::now();
            let next_present_at = input.next_present_at.get_or_insert(now + interval);
            while *next_present_at <= now {
                *next_present_at += interval;
            }
        } else {
            while *input.completed_kernels >= *input.next_sample_at {
                *input.next_sample_at += u64::from(input.sample_every);
            }
        }
    }

    Ok(stats)
}

#[derive(Clone, Copy, Debug, Default)]
struct ThroughputBatchStats {
    completed_kernels: u32,
    sampled_frames: u32,
    presented_frames: u32,
    sampled_bytes: usize,
    uploaded_bytes: usize,
    launch: Duration,
    completion_wait: Duration,
    sample_download: Duration,
    present: Duration,
    map_copy: Duration,
    draw: Duration,
    swap_present: Duration,
}

impl std::ops::AddAssign for ThroughputBatchStats {
    fn add_assign(&mut self, rhs: Self) {
        self.completed_kernels += rhs.completed_kernels;
        self.sampled_frames += rhs.sampled_frames;
        self.presented_frames += rhs.presented_frames;
        self.sampled_bytes += rhs.sampled_bytes;
        self.uploaded_bytes += rhs.uploaded_bytes;
        self.launch += rhs.launch;
        self.completion_wait += rhs.completion_wait;
        self.sample_download += rhs.sample_download;
        self.present += rhs.present;
        self.map_copy += rhs.map_copy;
        self.draw += rhs.draw;
        self.swap_present += rhs.swap_present;
    }
}

struct KernelRateLimiter {
    target_fps: Option<f64>,
    tokens: f64,
    max_burst: f64,
    last_refill: Instant,
}

impl KernelRateLimiter {
    fn new(target_fps: Option<f32>, max_inflight: u32, now: Instant) -> Self {
        Self {
            target_fps: target_fps.map(f64::from),
            tokens: 0.0,
            max_burst: f64::from(max_inflight.max(1).saturating_mul(4)),
            last_refill: now,
        }
    }

    fn grant(&mut self, now: Instant, max_inflight: u32) -> u32 {
        let granted = self.available(now, max_inflight);
        self.consume(granted);
        granted
    }

    fn available(&mut self, now: Instant, max_inflight: u32) -> u32 {
        if self.target_fps.is_none() {
            return max_inflight;
        }
        self.refill(now);
        self.tokens.floor().min(f64::from(max_inflight)) as u32
    }

    fn consume(&mut self, granted: u32) {
        if self.target_fps.is_some() {
            self.tokens -= f64::from(granted);
        }
    }

    fn next_token_at(&self, now: Instant) -> Option<Instant> {
        let target_fps = self.target_fps?;
        if self.tokens >= 1.0 {
            return Some(now);
        }
        let seconds_until_token = (1.0 - self.tokens) / target_fps;
        Some(now + Duration::from_secs_f64(seconds_until_token))
    }

    fn refill(&mut self, now: Instant) {
        let Some(target_fps) = self.target_fps else {
            return;
        };
        let elapsed = now.saturating_duration_since(self.last_refill);
        self.tokens = (self.tokens + elapsed.as_secs_f64() * target_fps).min(self.max_burst);
        self.last_refill = now;
    }
}

struct PresentRateLimiter {
    target_fps: Option<f64>,
    tokens: f64,
    last_refill: Instant,
}

impl PresentRateLimiter {
    fn new(target_fps: Option<f32>, now: Instant) -> Self {
        Self {
            target_fps: target_fps.map(f64::from),
            tokens: 0.0,
            last_refill: now,
        }
    }

    fn try_consume(&mut self, now: Instant) -> bool {
        let Some(target_fps) = self.target_fps else {
            return true;
        };
        let elapsed = now.saturating_duration_since(self.last_refill);
        self.tokens = (self.tokens + elapsed.as_secs_f64() * target_fps).min(1.0);
        self.last_refill = now;
        if self.tokens + 1.0e-9 >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

struct ThroughputCounter {
    last_log: Instant,
    completed_since_log: u64,
    sampled_since_log: u64,
    presented_since_log: u64,
    total_completed: u64,
    total_sampled: u64,
    total_presented: u64,
    sampled_bytes_since_log: u64,
    uploaded_bytes_since_log: u64,
    launch_accum: Duration,
    completion_wait_accum: Duration,
    sample_download_accum: Duration,
    present_accum: Duration,
    map_copy_accum: Duration,
    draw_accum: Duration,
    swap_present_accum: Duration,
}

struct ThroughputLogContext<'a> {
    size: PhysicalSize<u32>,
    frame: u32,
    presenter: PresenterKind,
    reload_error: Option<&'a str>,
    kernel_cap: Option<f32>,
    instance_variant: Option<InstanceStressVariant>,
    instance_layout: Option<String>,
    instance_debug_view: Option<InstanceDebugView>,
    renderer: Option<DrawBackend>,
    draw_policy: Option<HardwareRasterDrawPolicy>,
    draw_depth: Option<DrawDepthMode>,
    uses_depth: Option<bool>,
    cull_order: Option<HardwareRasterCullOrder>,
    draw_visibility: Option<HardwareRasterVisibilityMode>,
    min_projected_millipixels: Option<u32>,
    visible_instances: Option<u32>,
    indirect_draws: Option<u32>,
    render_policy: RenderPolicy,
    visibility: RenderVisibility,
}

#[derive(Clone, Copy, Debug, Default)]
struct DrawPolicyLogFields {
    draw_policy: Option<HardwareRasterDrawPolicy>,
    draw_depth: Option<DrawDepthMode>,
    uses_depth: Option<bool>,
    cull_order: Option<HardwareRasterCullOrder>,
    draw_visibility: Option<HardwareRasterVisibilityMode>,
    min_projected_millipixels: Option<u32>,
    visible_instances: Option<u32>,
    indirect_draws: Option<u32>,
}

fn draw_policy_log_markers(fields: DrawPolicyLogFields) -> String {
    let draw_policy_marker = fields
        .draw_policy
        .map(|policy| format!(" | draw_policy {policy}"))
        .unwrap_or_default();
    let draw_depth_marker = fields
        .draw_depth
        .map(|mode| format!(" | draw_depth {mode}"))
        .unwrap_or_default();
    let depth_resolved_marker = fields
        .uses_depth
        .map(|enabled| format!(" | depth {}", if enabled { "on" } else { "off" }))
        .unwrap_or_default();
    let cull_order_marker = fields
        .cull_order
        .map(|order| format!(" | cull_order {order}"))
        .unwrap_or_default();
    let draw_visibility_marker = fields
        .draw_visibility
        .map(|mode| format!(" | draw_visibility {mode}"))
        .unwrap_or_default();
    let min_projected_marker = fields
        .min_projected_millipixels
        .map(|millipixels| format!(" | min_projected_px {:.3}", millipixels as f32 / 1000.0))
        .unwrap_or_default();
    let visible_marker = fields
        .visible_instances
        .map(|count| format!(" | visible_instances {count}"))
        .unwrap_or_default();
    let indirect_marker = fields
        .indirect_draws
        .map(|count| format!(" | indirect_draws {count}"))
        .unwrap_or_default();
    format!(
        "{draw_policy_marker}{draw_depth_marker}{depth_resolved_marker}{cull_order_marker}{draw_visibility_marker}{min_projected_marker}{visible_marker}{indirect_marker}"
    )
}

impl ThroughputCounter {
    fn new() -> Self {
        Self {
            last_log: Instant::now(),
            completed_since_log: 0,
            sampled_since_log: 0,
            presented_since_log: 0,
            total_completed: 0,
            total_sampled: 0,
            total_presented: 0,
            sampled_bytes_since_log: 0,
            uploaded_bytes_since_log: 0,
            launch_accum: Duration::ZERO,
            completion_wait_accum: Duration::ZERO,
            sample_download_accum: Duration::ZERO,
            present_accum: Duration::ZERO,
            map_copy_accum: Duration::ZERO,
            draw_accum: Duration::ZERO,
            swap_present_accum: Duration::ZERO,
        }
    }

    fn record(&mut self, batch: ThroughputBatchStats) {
        let completed = u64::from(batch.completed_kernels);
        let sampled = u64::from(batch.sampled_frames);
        let presented = u64::from(batch.presented_frames);
        self.completed_since_log += completed;
        self.sampled_since_log += sampled;
        self.presented_since_log += presented;
        self.total_completed += completed;
        self.total_sampled += sampled;
        self.total_presented += presented;
        self.sampled_bytes_since_log += batch.sampled_bytes as u64;
        self.uploaded_bytes_since_log += batch.uploaded_bytes as u64;
        self.launch_accum += batch.launch;
        self.completion_wait_accum += batch.completion_wait;
        self.sample_download_accum += batch.sample_download;
        self.present_accum += batch.present;
        self.map_copy_accum += batch.map_copy;
        self.draw_accum += batch.draw;
        self.swap_present_accum += batch.swap_present;
    }

    fn log_if_due(&mut self, context: ThroughputLogContext<'_>) {
        let elapsed = self.last_log.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let secs = elapsed.as_secs_f64();
        let kernel_fps = self.completed_since_log as f64 / secs;
        let sample_fps = self.sampled_since_log as f64 / secs;
        let present_fps = self.presented_since_log as f64 / secs;
        let completed = self.completed_since_log.max(1) as f64;
        let launch_us = self.launch_accum.as_secs_f64() * 1_000_000.0 / completed;
        let wait_us = self.completion_wait_accum.as_secs_f64() * 1_000_000.0 / completed;
        let sample_us = if self.sampled_since_log == 0 {
            0.0
        } else {
            self.sample_download_accum.as_secs_f64() * 1_000_000.0 / self.sampled_since_log as f64
        };
        let present_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.present_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let map_copy_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.map_copy_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let draw_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.draw_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let swap_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.swap_present_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let mb_frame = frame_byte_len(context.size.width, context.size.height)
            .map(|bytes| bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let dtoh_gbps = self.sampled_bytes_since_log as f64 / secs / 1_000_000_000.0;
        let upload_gbps = self.uploaded_bytes_since_log as f64 / secs / 1_000_000_000.0;
        let reload_state = if context.reload_error.is_some() {
            "last-good"
        } else {
            "current"
        };
        let kernel_cap = context
            .kernel_cap
            .map(|fps| format!("{fps:.1} fps"))
            .unwrap_or_else(|| "uncapped".to_string());
        let interop_marker = if context.presenter == PresenterKind::D3d12Interop {
            " | interop d3d12-external-memory"
        } else {
            ""
        };
        let variant_marker = context
            .instance_variant
            .map(|variant| format!(" | instance_variant {variant}"))
            .unwrap_or_default();
        let layout_marker = context
            .instance_layout
            .as_deref()
            .map(|layout| format!(" | instance_layout {layout}"))
            .unwrap_or_default();
        let debug_marker = context
            .instance_debug_view
            .map(|view| format!(" | instance_debug {view}"))
            .unwrap_or_default();
        let renderer_marker = context
            .renderer
            .map(|renderer| format!(" | renderer {renderer}"))
            .unwrap_or_default();
        let draw_markers = draw_policy_log_markers(DrawPolicyLogFields {
            draw_policy: context.draw_policy,
            draw_depth: context.draw_depth,
            uses_depth: context.uses_depth,
            cull_order: context.cull_order,
            draw_visibility: context.draw_visibility,
            min_projected_millipixels: context.min_projected_millipixels,
            visible_instances: context.visible_instances,
            indirect_draws: context.indirect_draws,
        });
        let presenter = context.presenter;
        let render_policy = context.render_policy;
        let visibility = context.visibility;
        let frame = context.frame;
        println!(
            "kernel_fps {kernel_fps:>9.1} | sample_fps {sample_fps:>6.1} | present_fps {present_fps:>6.1} | completed {:>10} | frame {frame:>8} | {}x{} | {mb_frame:>5.1} MB/frame | dtoh {dtoh_gbps:>5.1} GB/s {sample_us:>6.1} us | upload {upload_gbps:>5.1} GB/s map_copy {map_copy_us:>6.1} us | gpu_copy {draw_us:>6.1} us | swap {swap_us:>6.1} us | present {present_us:>6.1} us | launch {launch_us:>5.1} us/k | wait {wait_us:>5.1} us/k | presenter {presenter} | kernel_cap {kernel_cap} | render_policy {render_policy} | visibility {visibility} | kernel {reload_state}{interop_marker}{variant_marker}{layout_marker}{debug_marker}{renderer_marker}{draw_markers}",
            self.total_completed, context.size.width, context.size.height
        );
        self.completed_since_log = 0;
        self.sampled_since_log = 0;
        self.presented_since_log = 0;
        self.sampled_bytes_since_log = 0;
        self.uploaded_bytes_since_log = 0;
        self.launch_accum = Duration::ZERO;
        self.completion_wait_accum = Duration::ZERO;
        self.sample_download_accum = Duration::ZERO;
        self.present_accum = Duration::ZERO;
        self.map_copy_accum = Duration::ZERO;
        self.draw_accum = Duration::ZERO;
        self.swap_present_accum = Duration::ZERO;
        self.last_log = Instant::now();
    }
}

struct FpsCounter {
    last_log: Instant,
    frames_since_log: u32,
    launch_accum: Duration,
    download_accum: Duration,
    sync_accum: Duration,
    render_accum: Duration,
    present_accum: Duration,
    total_accum: Duration,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last_log: Instant::now(),
            frames_since_log: 0,
            launch_accum: Duration::ZERO,
            download_accum: Duration::ZERO,
            sync_accum: Duration::ZERO,
            render_accum: Duration::ZERO,
            present_accum: Duration::ZERO,
            total_accum: Duration::ZERO,
        }
    }

    fn tick(
        &mut self,
        size: PhysicalSize<u32>,
        frame: u32,
        reload_error: Option<&str>,
        presenter: PresenterKind,
        timings: FrameTimings,
    ) {
        self.frames_since_log += 1;
        self.launch_accum += timings.launch;
        self.download_accum += timings.download;
        self.sync_accum += timings.sync;
        self.render_accum += timings.render;
        self.present_accum += timings.present;
        self.total_accum += timings.total;
        let elapsed = self.last_log.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let fps = self.frames_since_log as f32 / elapsed.as_secs_f32();
        let frame_count = self.frames_since_log as f64;
        let launch_us = self.launch_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let download_us = self.download_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let sync_us = self.sync_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let render_us = self.render_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let present_us = self.present_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let total_us = self.total_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let reload_state = if reload_error.is_some() {
            "last-good"
        } else {
            "current"
        };
        println!(
            "fps {fps:>7.1} | frame {frame:>8} | {}x{} | launch {launch_us:>5.1} us | dtoh {download_us:>5.1} us | sync {sync_us:>5.1} us | render {render_us:>6.1} us | present {present_us:>6.1} us | total {total_us:>6.1} us | presenter {presenter} | kernel {reload_state}",
            size.width, size.height
        );
        self.frames_since_log = 0;
        self.launch_accum = Duration::ZERO;
        self.download_accum = Duration::ZERO;
        self.sync_accum = Duration::ZERO;
        self.render_accum = Duration::ZERO;
        self.present_accum = Duration::ZERO;
        self.total_accum = Duration::ZERO;
        self.last_log = Instant::now();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameTimings {
    launch: Duration,
    download: Duration,
    sync: Duration,
    render: Duration,
    present: Duration,
    total: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct PresentTimings {
    map_copy: Duration,
    draw: Duration,
    swap_present: Duration,
    total: Duration,
}

enum PresentSink {
    Direct(WindowPresenter),
    Threaded(ThreadedPresenter),
}

impl PresentSink {
    fn threaded(presenter: WindowPresenter) -> Self {
        Self::Threaded(ThreadedPresenter::new(presenter))
    }

    fn present_sync(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        match self {
            Self::Direct(presenter) => presenter.present(size, bgra),
            Self::Threaded(_) => {
                bail!("synchronous presentation is not available on present thread")
            }
        }
    }

    fn present_sampled(
        &mut self,
        size: PhysicalSize<u32>,
        bgra: &[u8],
        sample_download: Duration,
    ) -> Result<Option<PresentTimings>> {
        match self {
            Self::Direct(presenter) => Ok(Some(presenter.present(size, bgra)?)),
            Self::Threaded(presenter) => {
                presenter.submit(PresentFrame {
                    size,
                    bgra: bgra.to_vec(),
                    sample_download,
                })?;
                Ok(None)
            }
        }
    }

    fn drain_results(&mut self) -> Result<ThroughputBatchStats> {
        match self {
            Self::Direct(_) => Ok(ThroughputBatchStats::default()),
            Self::Threaded(presenter) => presenter.drain_results(),
        }
    }

    fn kind(&self) -> PresenterKind {
        match self {
            Self::Direct(presenter) => presenter.kind(),
            Self::Threaded(presenter) => presenter.kind(),
        }
    }
}

struct PresentFrame {
    size: PhysicalSize<u32>,
    bgra: Vec<u8>,
    sample_download: Duration,
}

struct PresentThreadResult {
    size: PhysicalSize<u32>,
    sample_download: Duration,
    timings: PresentTimings,
}

struct ThreadedPresenter {
    kind: PresenterKind,
    shared: Arc<ThreadedPresenterShared>,
    results: Receiver<std::result::Result<PresentThreadResult, String>>,
    worker: Option<thread::JoinHandle<()>>,
}

struct ThreadedPresenterShared {
    latest: Mutex<Option<PresentFrame>>,
    available: Condvar,
    shutdown: AtomicBool,
}

impl ThreadedPresenter {
    fn new(mut presenter: WindowPresenter) -> Self {
        let kind = presenter.kind();
        let shared = Arc::new(ThreadedPresenterShared {
            latest: Mutex::new(None),
            available: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let worker_shared = shared.clone();
        let (result_tx, results) = mpsc::channel();
        let worker = thread::spawn(move || {
            loop {
                let frame = {
                    let mut latest = worker_shared
                        .latest
                        .lock()
                        .expect("present thread queue mutex poisoned");
                    while latest.is_none() && !worker_shared.shutdown.load(Ordering::Acquire) {
                        latest = worker_shared
                            .available
                            .wait(latest)
                            .expect("present thread queue mutex poisoned");
                    }
                    if latest.is_none() && worker_shared.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    latest.take()
                };

                let Some(frame) = frame else {
                    continue;
                };
                let result = presenter
                    .present(frame.size, &frame.bgra)
                    .map(|timings| PresentThreadResult {
                        size: frame.size,
                        sample_download: frame.sample_download,
                        timings,
                    })
                    .map_err(|err| format!("{err:#}"));
                if result_tx.send(result).is_err() {
                    break;
                }
            }
        });
        Self {
            kind,
            shared,
            results,
            worker: Some(worker),
        }
    }

    fn submit(&self, frame: PresentFrame) -> Result<()> {
        let mut latest = self
            .shared
            .latest
            .lock()
            .map_err(|_| anyhow!("present thread queue mutex poisoned"))?;
        *latest = Some(frame);
        self.shared.available.notify_one();
        Ok(())
    }

    fn drain_results(&mut self) -> Result<ThroughputBatchStats> {
        let mut stats = ThroughputBatchStats::default();
        for result in self.results.try_iter() {
            let result = result.map_err(|err| anyhow!("{err}"))?;
            stats.sample_download += result.sample_download;
            stats.present += result.timings.total;
            stats.map_copy += result.timings.map_copy;
            stats.draw += result.timings.draw;
            stats.swap_present += result.timings.swap_present;
            stats.sampled_frames += 1;
            stats.presented_frames += 1;
            let bytes = frame_byte_len(result.size.width, result.size.height)?;
            stats.sampled_bytes += bytes;
            stats.uploaded_bytes += bytes;
        }
        Ok(stats)
    }

    fn kind(&self) -> PresenterKind {
        self.kind
    }
}

impl Drop for ThreadedPresenter {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.available.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(windows)]
struct WindowPresenter {
    inner: PresenterImpl,
}

// The presenter is moved once onto the present worker and then used only there.
// The wrapped Win32/D3D handles are not accessed concurrently after the move.
#[cfg(windows)]
unsafe impl Send for WindowPresenter {}

#[cfg(windows)]
enum PresenterImpl {
    D3d12Interop(D3d12InteropPresenter),
    D3d12(D3d12Presenter),
    D3d11(D3d11Presenter),
    Gdi(GdiPresenter),
}

#[cfg(windows)]
struct RasterPresentInput<'a> {
    size: PhysicalSize<u32>,
    args: &'a mut SharedGpuBuffer,
    visible_ids: &'a mut SharedGpuBuffer,
    raster_instances: &'a mut SharedGpuBuffer,
    material: &'a MaterialKernelAbi,
    geometry: &'a HardwareRasterGeometryStreamPlan,
    cuda_done_value: u64,
    frame: u32,
    grid: InstanceGrid,
    camera: CameraParams,
    shaders: &'a neo_lang::GraphicsShaders,
    use_depth: bool,
}

#[cfg(windows)]
impl WindowPresenter {
    fn new(
        window: &Window,
        kind: PresenterKind,
        upload_ring: usize,
        d3d_upload: D3dUploadMode,
        interop_device: Option<&NeoD3d12InteropDevice>,
    ) -> Result<Self> {
        let inner = match kind {
            PresenterKind::D3d12Interop => {
                let device = interop_device
                    .context("d3d12-interop presenter requires a Neo D3D12 interop device")?;
                PresenterImpl::D3d12Interop(D3d12InteropPresenter::new(window, device)?)
            }
            PresenterKind::D3d12 => PresenterImpl::D3d12(D3d12Presenter::new(window, upload_ring)?),
            PresenterKind::D3d11 => {
                PresenterImpl::D3d11(D3d11Presenter::new(window, upload_ring, d3d_upload)?)
            }
            PresenterKind::Gdi => PresenterImpl::Gdi(GdiPresenter::new(window)?),
        };
        Ok(Self { inner })
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(_) => {
                bail!("d3d12-interop presenter requires a shared Neo frame slot")
            }
            PresenterImpl::D3d12(presenter) => presenter.present(size, bgra),
            PresenterImpl::D3d11(presenter) => presenter.present(size, bgra),
            PresenterImpl::Gdi(presenter) => presenter.present(size, bgra),
        }
    }

    #[cfg(windows)]
    fn present_shared(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        slot: &mut neo_runtime::SharedFrameSlot,
        cuda_done_value: u64,
    ) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(presenter) => {
                presenter.present_shared(size, pitch_bytes, slot, cuda_done_value)
            }
            _ => bail!("shared frame presentation requires d3d12-interop presenter"),
        }
    }

    #[cfg(windows)]
    fn present_raster_indirect(&mut self, input: RasterPresentInput<'_>) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(presenter) => presenter.present_raster_indirect(input),
            _ => bail!("hardware raster presentation requires d3d12-interop presenter"),
        }
    }

    fn kind(&self) -> PresenterKind {
        match self.inner {
            PresenterImpl::D3d12Interop(_) => PresenterKind::D3d12Interop,
            PresenterImpl::D3d12(_) => PresenterKind::D3d12,
            PresenterImpl::D3d11(_) => PresenterKind::D3d11,
            PresenterImpl::Gdi(_) => PresenterKind::Gdi,
        }
    }
}

#[cfg(windows)]
struct GdiPresenter {
    hwnd: windows_sys::Win32::Foundation::HWND,
    hdc: windows_sys::Win32::Graphics::Gdi::HDC,
    width: u32,
    height: u32,
    bitmap_info: windows_sys::Win32::Graphics::Gdi::BITMAPINFO,
}

#[cfg(windows)]
impl GdiPresenter {
    fn new(window: &Window) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("no-interop presenter requires a Win32 window handle");
        };

        let hwnd = handle.hwnd.get() as windows_sys::Win32::Foundation::HWND;
        let hdc = unsafe { windows_sys::Win32::Graphics::Gdi::GetDC(hwnd) };
        if hdc.is_null() {
            bail!("GetDC failed for live window");
        }

        Ok(Self {
            hwnd,
            hdc,
            width: 0,
            height: 0,
            bitmap_info: bitmap_info(1, 1),
        })
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        if self.width != size.width || self.height != size.height {
            self.width = size.width;
            self.height = size.height;
            self.bitmap_info = bitmap_info(size.width, size.height);
        }

        unsafe {
            use windows_sys::Win32::Graphics::Gdi::{DIB_RGB_COLORS, SetDIBitsToDevice};

            let scanlines = SetDIBitsToDevice(
                self.hdc,
                0,
                0,
                size.width,
                size.height,
                0,
                0,
                0,
                size.height,
                bgra.as_ptr().cast(),
                &self.bitmap_info,
                DIB_RGB_COLORS,
            );
            if scanlines == 0 {
                bail!("SetDIBitsToDevice failed for live frame");
            }
        }
        let elapsed = start.elapsed();
        Ok(PresentTimings {
            swap_present: elapsed,
            total: elapsed,
            ..PresentTimings::default()
        })
    }
}

#[cfg(windows)]
impl Drop for GdiPresenter {
    fn drop(&mut self) {
        if !self.hdc.is_null() {
            unsafe {
                windows_sys::Win32::Graphics::Gdi::ReleaseDC(self.hwnd, self.hdc);
            }
        }
    }
}

#[cfg(windows)]
const D3D12_SWAPCHAIN_BUFFER_COUNT: usize = 3;

#[cfg(windows)]
const RASTER_ROOT_CONSTANT_DWORDS: u32 = 28;

#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HardwareRasterSubmitKind {
    DirectIndexedInstanced,
    IndirectIndexedInstanced,
}

#[cfg(windows)]
fn hardware_raster_submit_kind(material: &MaterialKernelAbi) -> HardwareRasterSubmitKind {
    if material.requires_compute_culling() {
        HardwareRasterSubmitKind::IndirectIndexedInstanced
    } else {
        HardwareRasterSubmitKind::DirectIndexedInstanced
    }
}

#[cfg(windows)]
fn raster_root_constants(
    grid: InstanceGrid,
    size: PhysicalSize<u32>,
    camera: CameraParams,
    geometry: &HardwareRasterGeometryStreamPlan,
    frame: u32,
) -> [u32; RASTER_ROOT_CONSTANT_DWORDS as usize] {
    [
        grid.x,
        grid.y,
        grid.z,
        frame,
        size.width,
        size.height,
        geometry.vertex_stride,
        geometry.color_offset,
        camera.origin[0].to_bits(),
        camera.origin[1].to_bits(),
        camera.origin[2].to_bits(),
        camera.origin[3].to_bits(),
        camera.right[0].to_bits(),
        camera.right[1].to_bits(),
        camera.right[2].to_bits(),
        camera.right[3].to_bits(),
        camera.up[0].to_bits(),
        camera.up[1].to_bits(),
        camera.up[2].to_bits(),
        camera.up[3].to_bits(),
        camera.forward[0].to_bits(),
        camera.forward[1].to_bits(),
        camera.forward[2].to_bits(),
        camera.forward[3].to_bits(),
        camera.view[0].to_bits(),
        camera.view[1].to_bits(),
        camera.view[2].to_bits(),
        camera.view[3].to_bits(),
    ]
}

#[cfg(windows)]
const D3D12_RASTER_VS_TARGET: &[u8; 7] = b"vs_5_1\0";
#[cfg(windows)]
const D3D12_RASTER_PS_TARGET: &[u8; 7] = b"ps_5_1\0";

#[cfg(windows)]
struct D3d12RasterState {
    root_signature: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pipeline: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    command_signature: windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature,
    geometry_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    index_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    index_view: windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_VIEW,
    shader_hash: u64,
}

#[cfg(windows)]
impl D3d12RasterState {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        shaders: &neo_lang::GraphicsShaders,
        material: &MaterialKernelAbi,
        geometry: &HardwareRasterGeometryStreamPlan,
        use_depth: bool,
        shader_hash: u64,
    ) -> Result<Self> {
        use windows::{
            Win32::Graphics::{
                Direct3D::ID3DBlob,
                Direct3D12::{
                    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_COLOR_WRITE_ENABLE_ALL,
                    D3D12_COMMAND_SIGNATURE_DESC, D3D12_COMPARISON_FUNC_LESS,
                    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    D3D12_CULL_MODE_NONE, D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, D3D12_DEPTH_STENCIL_DESC,
                    D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_FILL_MODE_SOLID,
                    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_HEAP_FLAG_NONE,
                    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_UPLOAD, D3D12_INDEX_BUFFER_VIEW,
                    D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
                    D3D12_INPUT_LAYOUT_DESC, D3D12_LOGIC_OP_NOOP, D3D12_MEMORY_POOL_UNKNOWN,
                    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC,
                    D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOURCE_DESC,
                    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR,
                    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
                    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
                    D3D12_ROOT_SIGNATURE_DESC,
                    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                    D3D12_SHADER_BYTECODE, D3D12_SO_DECLARATION_ENTRY, D3D12_STATIC_SAMPLER_DESC,
                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12SerializeRootSignature,
                },
                Dxgi::Common::{
                    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16_UINT,
                    DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
                },
            },
            core::PCSTR,
        };

        validate_contiguous_material_root_bindings(material)?;

        let mut root_params = Vec::with_capacity(material.bindings.len());
        for binding in &material.bindings {
            let root_param = match binding.kind {
                MaterialBindingKind::DrawParams | MaterialBindingKind::RasterParams => {
                    D3D12_ROOT_PARAMETER {
                        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                        Anonymous: D3D12_ROOT_PARAMETER_0 {
                            Constants: D3D12_ROOT_CONSTANTS {
                                ShaderRegister: binding.shader_register,
                                RegisterSpace: binding.register_space,
                                Num32BitValues: RASTER_ROOT_CONSTANT_DWORDS,
                            },
                        },
                        ShaderVisibility:
                            windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
                    }
                }
                MaterialBindingKind::VisibleInstanceStream
                | MaterialBindingKind::InstanceStream
                | MaterialBindingKind::GeometryStream => D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: binding.shader_register,
                            RegisterSpace: binding.register_space,
                        },
                    },
                    ShaderVisibility:
                        windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
                },
            };
            root_params.push(root_param);
        }
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: root_params.len() as u32,
            pParameters: root_params.as_mut_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null::<D3D12_STATIC_SAMPLER_DESC>(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        let mut signature_blob: Option<ID3DBlob> = None;
        let mut signature_error: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut signature_blob,
                Some(&mut signature_error),
            )
            .map_err(|err| anyhow!("failed to serialize D3D12 root signature: {err:?}"))?;
        }
        let signature_blob =
            signature_blob.context("D3D12 root signature serialization returned no blob")?;
        let root_signature: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    signature_blob.GetBufferPointer().cast::<u8>(),
                    signature_blob.GetBufferSize(),
                ),
            )?
        };

        let vertex_entrypoint = std::ffi::CString::new(material.vertex_entrypoint.as_str())
            .context("hardware raster vertex entrypoint contains an interior NUL byte")?;
        let fragment_entrypoint = std::ffi::CString::new(material.fragment_entrypoint.as_str())
            .context("hardware raster fragment entrypoint contains an interior NUL byte")?;
        let vs = compile_hlsl(
            &shaders.vertex_source,
            PCSTR(vertex_entrypoint.as_ptr().cast()),
            PCSTR(D3D12_RASTER_VS_TARGET.as_ptr()),
        )?;
        let ps = compile_hlsl(
            &shaders.fragment_source,
            PCSTR(fragment_entrypoint.as_ptr().cast()),
            PCSTR(D3D12_RASTER_PS_TARGET.as_ptr()),
        )?;
        let blend_desc = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                LogicOpEnable: false.into(),
                SrcBlend: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ONE,
                DestBlend: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ZERO,
                BlendOp: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ONE,
                DestBlendAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ZERO,
                BlendOpAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            }; 8],
        };
        let rasterizer_desc = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: false.into(),
            DepthBias: D3D12_DEFAULT_DEPTH_BIAS,
            DepthBiasClamp: D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
            SlopeScaledDepthBias: D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
            DepthClipEnable: true.into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        };
        let depth_desc = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: use_depth.into(),
            DepthWriteMask: if use_depth {
                D3D12_DEPTH_WRITE_MASK_ALL
            } else {
                D3D12_DEPTH_WRITE_MASK_ZERO
            },
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: Default::default(),
            BackFace: Default::default(),
        };
        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rtv_formats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        let (vs_ptr, vs_len, ps_ptr, ps_len) = unsafe {
            (
                vs.GetBufferPointer(),
                vs.GetBufferSize(),
                ps.GetBufferPointer(),
                ps.GetBufferSize(),
            )
        };
        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_ptr,
                BytecodeLength: vs_len,
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps_ptr,
                BytecodeLength: ps_len,
            },
            DS: D3D12_SHADER_BYTECODE::default(),
            HS: D3D12_SHADER_BYTECODE::default(),
            GS: D3D12_SHADER_BYTECODE::default(),
            StreamOutput: windows::Win32::Graphics::Direct3D12::D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: std::ptr::null::<D3D12_SO_DECLARATION_ENTRY>(),
                NumEntries: 0,
                pBufferStrides: std::ptr::null(),
                NumStrides: 0,
                RasterizedStream: 0,
            },
            BlendState: blend_desc,
            SampleMask: u32::MAX,
            RasterizerState: rasterizer_desc,
            DepthStencilState: depth_desc,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: std::ptr::null(),
                NumElements: 0,
            },
            IBStripCutValue:
                windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: rtv_formats,
            DSVFormat: if use_depth {
                DXGI_FORMAT_D32_FLOAT
            } else {
                DXGI_FORMAT_UNKNOWN
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: Default::default(),
            Flags: windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pipeline = unsafe { device.CreateGraphicsPipelineState(&pso_desc)? };

        let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            Anonymous: Default::default(),
        };
        let command_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: DrawIndexedIndirectCommand::BYTE_LEN as u32,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg_desc,
            NodeMask: 0,
        };
        let mut command_signature = None;
        unsafe {
            device.CreateCommandSignature(
                &command_desc,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12RootSignature>,
                &mut command_signature,
            )?;
        }
        let command_signature =
            command_signature.context("D3D12 returned no indirect command signature")?;

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let geometry_bytes = geometry.vertex_bytes.len() as u64;
        let mut geometry_buffer = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Alignment: 0,
                    Width: geometry_bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut geometry_buffer,
            )?;
        }
        let geometry_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource =
            geometry_buffer.context("D3D12 returned no geometry buffer")?;
        unsafe {
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            geometry_buffer.Map(0, Some(&read_range), Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                geometry.vertex_bytes.as_ptr(),
                mapped.cast::<u8>(),
                geometry.vertex_bytes.len(),
            );
            geometry_buffer.Unmap(0, None);
        }

        let indices = &geometry.indices_u16;
        let index_bytes = std::mem::size_of_val(indices.as_slice()) as u64;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: index_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut index_buffer = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut index_buffer,
            )?;
        }
        let index_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource =
            index_buffer.context("D3D12 returned no index buffer")?;
        unsafe {
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            index_buffer.Map(0, Some(&read_range), Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                indices.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                index_bytes as usize,
            );
            index_buffer.Unmap(0, None);
        }
        let index_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { index_buffer.GetGPUVirtualAddress() },
            SizeInBytes: index_bytes as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };
        Ok(Self {
            root_signature,
            pipeline,
            command_signature,
            geometry_buffer,
            index_buffer,
            index_view,
            shader_hash,
        })
    }
}

#[cfg(windows)]
fn compile_hlsl(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    use windows::Win32::Graphics::{Direct3D::Fxc::D3DCompile, Direct3D::ID3DBlob};
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::PCSTR::null(),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
        .map_err(|err| {
            let message = errors
                .as_ref()
                .map(|blob| {
                    let bytes = std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    );
                    String::from_utf8_lossy(bytes).to_string()
                })
                .unwrap_or_default();
            anyhow!("failed to compile D3D12 raster HLSL: {err:?}\n{message}")
        })?;
    }
    code.context("D3DCompile returned no shader bytecode")
}

#[cfg(windows)]
fn raster_shader_hash(
    shaders: &neo_lang::GraphicsShaders,
    material: &MaterialKernelAbi,
    geometry: &HardwareRasterGeometryStreamPlan,
    use_depth: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    shaders.vertex_source.hash(&mut hasher);
    shaders.fragment_source.hash(&mut hasher);
    material.hash(&mut hasher);
    use_depth.hash(&mut hasher);
    geometry.vertex_bytes.hash(&mut hasher);
    geometry.vertex_stride.hash(&mut hasher);
    geometry.color_offset.hash(&mut hasher);
    geometry.indices_u16.hash(&mut hasher);
    hasher.finish()
}

#[cfg(windows)]
fn material_binding<'a>(
    material: &'a MaterialKernelAbi,
    kind: MaterialBindingKind,
    label: &str,
) -> Result<&'a neo_runtime::MaterialBinding> {
    material.binding(kind).with_context(|| {
        format!(
            "hardware raster MaterialKernel `{}`/`{}` is missing its {label} binding",
            material.vertex_entrypoint, material.fragment_entrypoint
        )
    })
}

#[cfg(windows)]
fn validate_contiguous_material_root_bindings(material: &MaterialKernelAbi) -> Result<()> {
    let mut expected = vec![(MaterialBindingKind::DrawParams, "draw params")];
    if material.requires_compute_culling() {
        expected.push((
            MaterialBindingKind::VisibleInstanceStream,
            "visible InstanceStream",
        ));
    }
    if material.requires_instance_stream() {
        expected.push((MaterialBindingKind::InstanceStream, "InstanceStream"));
    }
    if material
        .vertex_requirements
        .contains(&MaterialVertexRequirement::GeometryPosition)
    {
        expected.push((MaterialBindingKind::GeometryStream, "GeometryStream"));
    }

    for (expected_root, (kind, label)) in expected.into_iter().enumerate() {
        let binding = material_binding(material, kind, label)?;
        if binding.root_parameter_index != expected_root as u32 {
            bail!(
                "hardware raster MaterialKernel `{}`/`{}` binding `{label}` must use root parameter {expected_root}, got {}",
                material.vertex_entrypoint,
                material.fragment_entrypoint,
                binding.root_parameter_index
            );
        }
    }
    Ok(())
}

#[cfg(windows)]
fn graphics_bindings_for_material(
    material: &MaterialKernelAbi,
) -> Result<neo_lang::GraphicsBindings> {
    let draw_params = material_binding(material, MaterialBindingKind::DrawParams, "draw params")?;
    let visible_instances = material
        .binding(MaterialBindingKind::VisibleInstanceStream)
        .map(|binding| neo_lang::HlslRegister::new(binding.shader_register, binding.register_space))
        .unwrap_or_else(|| neo_lang::HlslRegister::new(0, 0));
    let instances = material_binding(
        material,
        MaterialBindingKind::InstanceStream,
        "InstanceStream",
    )?;
    let geometry = material_binding(
        material,
        MaterialBindingKind::GeometryStream,
        "GeometryStream",
    )?;
    Ok(neo_lang::GraphicsBindings {
        raster_params: neo_lang::HlslRegister::new(
            draw_params.shader_register,
            draw_params.register_space,
        ),
        visible_instances,
        instances: neo_lang::HlslRegister::new(instances.shader_register, instances.register_space),
        geometry: neo_lang::HlslRegister::new(geometry.shader_register, geometry.register_space),
    })
}

#[cfg(windows)]
struct D3d12InteropPresenter {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    command_frames: Vec<D3d12InteropCommandFrame>,
    command_frame_index: usize,
    command_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    back_buffers: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    rtv_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    dsv_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    depth_buffer: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    rtv_descriptor_size: u32,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    fence_value: u64,
    fence_event: windows::Win32::Foundation::HANDLE,
    width: u32,
    height: u32,
    tearing_supported: bool,
    raster_state: Option<D3d12RasterState>,
}

#[cfg(windows)]
struct D3d12InteropCommandFrame {
    command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    fence_value: u64,
}

#[cfg(windows)]
impl D3d12InteropPresenter {
    fn new(window: &Window, interop: &NeoD3d12InteropDevice) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::HWND,
                Graphics::{
                    Direct3D12::{
                        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_DESCRIPTOR_HEAP_DESC,
                        D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_FENCE_FLAG_NONE,
                        ID3D12CommandAllocator, ID3D12DescriptorHeap, ID3D12Fence,
                        ID3D12GraphicsCommandList,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_NONE,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2, IDXGISwapChain3,
                    },
                },
                System::Threading::CreateEventW,
            },
            core::{BOOL, Interface as _, PCWSTR},
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D12 interop presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let device = interop.device().clone();
        let queue = interop.queue().clone();
        let mut command_frames = Vec::with_capacity(D3D12_SWAPCHAIN_BUFFER_COUNT);
        for _ in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let command_allocator: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)? };
            command_frames.push(D3d12InteropCommandFrame {
                command_allocator,
                fence_value: 0,
            });
        }
        let command_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &command_frames[0].command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?
        };
        unsafe {
            command_list.Close()?;
        }

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D12 interop presenter tearing support: {tearing_supported}");
        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Scaling: DXGI_SCALING_NONE,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(&queue, HWND(handle.hwnd.get() as _), &desc, None, None)
        }?
        .cast::<IDXGISwapChain3>()?;
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let fence_event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }?;
        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&rtv_heap_desc)? };
        let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let dsv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&dsv_heap_desc)? };
        let rtv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
        let mut presenter = Self {
            device,
            queue,
            command_frames,
            command_frame_index: 0,
            command_list,
            swap_chain,
            back_buffers: Vec::new(),
            rtv_heap,
            dsv_heap,
            depth_buffer: None,
            rtv_descriptor_size,
            fence,
            fence_value: 0,
            fence_event,
            width,
            height,
            tearing_supported,
            raster_state: None,
        };
        presenter.recreate_backbuffers()?;
        Ok(presenter)
    }

    fn present_shared(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        slot: &mut neo_runtime::SharedFrameSlot,
        cuda_done_value: u64,
    ) -> Result<PresentTimings> {
        let total_start = Instant::now();
        self.ensure_size(size)?;
        let copy_start = Instant::now();
        slot.wait_d3d_for_value(&self.queue, cuda_done_value)?;
        self.copy_shared_to_backbuffer(size, pitch_bytes, slot.resource())?;
        let draw = copy_start.elapsed();
        let _ = slot.signal_available_on_d3d(&self.queue)?;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            draw,
            swap_present,
            total: total_start.elapsed(),
            ..PresentTimings::default()
        })
    }

    fn present_raster_indirect(&mut self, input: RasterPresentInput<'_>) -> Result<PresentTimings> {
        let RasterPresentInput {
            size,
            args,
            visible_ids,
            raster_instances,
            material,
            geometry,
            cuda_done_value,
            frame,
            grid,
            camera,
            shaders,
            use_depth,
        } = input;
        use windows::{
            Win32::Foundation::RECT,
            Win32::Graphics::{
                Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                Direct3D12::{
                    D3D12_CLEAR_FLAG_DEPTH, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_VIEWPORT, ID3D12CommandList,
                },
                Dxgi::DXGI_PRESENT,
            },
            core::Interface as _,
        };

        let total_start = Instant::now();
        self.ensure_size(size)?;
        let shader_hash = raster_shader_hash(shaders, material, geometry, use_depth);
        if self
            .raster_state
            .as_ref()
            .is_none_or(|state| state.shader_hash != shader_hash)
        {
            self.raster_state = Some(D3d12RasterState::new(
                &self.device,
                shaders,
                material,
                geometry,
                use_depth,
                shader_hash,
            )?);
        }
        let (command_frame, command_allocator) = self.acquire_command_frame()?;
        let raster = self
            .raster_state
            .as_ref()
            .context("missing D3D12 raster state")?;
        let _keep_index_buffer_alive = &raster.index_buffer;
        let draw_start = Instant::now();
        let submit_kind = hardware_raster_submit_kind(material);
        let uses_indirect_args = submit_kind == HardwareRasterSubmitKind::IndirectIndexedInstanced;
        if cuda_done_value != 0 {
            args.wait_d3d_for_value(&self.queue, cuda_done_value)?;
        }
        let direct_instance_count = grid
            .count()
            .context("hardware raster direct draw instance count overflow")?;
        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 interop backbuffer is not available")?;
        let rtv_handle = windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
                + back_index * self.rtv_descriptor_size as usize,
        };
        let _keep_depth_buffer_alive = if use_depth {
            Some(
                self.depth_buffer
                    .as_ref()
                    .context("D3D12 raster depth buffer is not available")?,
            )
        } else {
            None
        };
        let dsv_handle = unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe {
            command_allocator.Reset()?;
            self.command_list
                .Reset(&command_allocator, Some(&raster.pipeline))?;

            let mut back_to_rt = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_rt));
            drop_d3d12_transition_barrier(&mut back_to_rt);
            let mut args_to_indirect = if uses_indirect_args {
                Some(d3d12_transition(
                    args.resource(),
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ))
            } else {
                None
            };
            if let Some(barrier) = args_to_indirect.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = args_to_indirect.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut visible_to_srv = if material.requires_compute_culling() {
                Some(d3d12_transition(
                    visible_ids.resource(),
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ))
            } else {
                None
            };
            if let Some(barrier) = visible_to_srv.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = visible_to_srv.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut instances_to_srv = d3d12_transition(
                raster_instances.resource(),
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&instances_to_srv));
            drop_d3d12_transition_barrier(&mut instances_to_srv);

            self.command_list
                .SetGraphicsRootSignature(&raster.root_signature);
            self.command_list.SetPipelineState(&raster.pipeline);
            self.command_list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.command_list.IASetIndexBuffer(Some(&raster.index_view));
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size.width as f32,
                Height: size.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            self.command_list.RSSetViewports(&[viewport]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: size.width as i32,
                bottom: size.height as i32,
            };
            self.command_list.RSSetScissorRects(&[scissor]);
            self.command_list.OMSetRenderTargets(
                1,
                Some(&rtv_handle),
                false,
                use_depth.then_some(&dsv_handle),
            );
            let clear = [0.005, 0.006, 0.009, 1.0];
            self.command_list
                .ClearRenderTargetView(rtv_handle, &clear, None);
            if use_depth {
                self.command_list.ClearDepthStencilView(
                    dsv_handle,
                    D3D12_CLEAR_FLAG_DEPTH,
                    1.0,
                    0,
                    None,
                );
            }
            let constants = raster_root_constants(grid, size, camera, geometry, frame);
            for binding in &material.bindings {
                match binding.kind {
                    MaterialBindingKind::DrawParams | MaterialBindingKind::RasterParams => {
                        self.command_list.SetGraphicsRoot32BitConstants(
                            binding.root_parameter_index,
                            constants.len() as u32,
                            constants.as_ptr().cast(),
                            0,
                        );
                    }
                    MaterialBindingKind::VisibleInstanceStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            visible_ids.resource().GetGPUVirtualAddress(),
                        );
                    }
                    MaterialBindingKind::InstanceStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            raster_instances.resource().GetGPUVirtualAddress(),
                        );
                    }
                    MaterialBindingKind::GeometryStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            raster.geometry_buffer.GetGPUVirtualAddress(),
                        );
                    }
                }
            }
            if uses_indirect_args {
                self.command_list.ExecuteIndirect(
                    &raster.command_signature,
                    1,
                    args.resource(),
                    0,
                    None,
                    0,
                );
            } else {
                self.command_list.DrawIndexedInstanced(
                    geometry.index_count(),
                    direct_instance_count,
                    0,
                    0,
                    0,
                );
            }

            let mut args_to_common = if uses_indirect_args {
                Some(d3d12_transition(
                    args.resource(),
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                ))
            } else {
                None
            };
            if let Some(barrier) = args_to_common.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = args_to_common.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut visible_to_common = if material.requires_compute_culling() {
                Some(d3d12_transition(
                    visible_ids.resource(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                ))
            } else {
                None
            };
            if let Some(barrier) = visible_to_common.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = visible_to_common.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut instances_to_common = d3d12_transition(
                raster_instances.resource(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&instances_to_common));
            drop_d3d12_transition_barrier(&mut instances_to_common);
            let mut back_to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_present));
            drop_d3d12_transition_barrier(&mut back_to_present);
            self.command_list.Close()?;
            let list: ID3D12CommandList = self.command_list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
        }
        self.signal_command_frame(command_frame)?;
        let draw = draw_start.elapsed();
        if uses_indirect_args {
            let _ = args.signal_available_on_d3d(&self.queue)?;
        }
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            draw,
            swap_present,
            total: total_start.elapsed(),
            ..PresentTimings::default()
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.wait_for_gpu()?;
        self.back_buffers.clear();
        self.depth_buffer = None;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain.ResizeBuffers(
                D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
                width,
                height,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                flags,
            )?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffers()
    }

    fn recreate_backbuffers(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

        self.back_buffers.clear();
        let base = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for index in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let back_buffer: ID3D12Resource = unsafe { self.swap_chain.GetBuffer(index as u32)? };
            let handle = windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + index * self.rtv_descriptor_size as usize,
            };
            unsafe {
                self.device
                    .CreateRenderTargetView(&back_buffer, None, handle);
            }
            self.back_buffers.push(back_buffer);
        }
        self.recreate_depth_target()?;
        Ok(())
    }

    fn recreate_depth_target(&mut self) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                D3D12_DEPTH_STENCIL_VALUE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_SAMPLE_DESC},
        };

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: self.width as u64,
            Height: self.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };
        let mut depth_buffer: Option<ID3D12Resource> = None;
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&clear),
                &mut depth_buffer,
            )?;
        }
        let depth_buffer = depth_buffer.context("D3D12 returned no raster depth buffer")?;
        let dsv_handle = unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe {
            self.device
                .CreateDepthStencilView(&depth_buffer, None, dsv_handle);
        }
        self.depth_buffer = Some(depth_buffer);
        Ok(())
    }

    fn copy_shared_to_backbuffer(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        shared: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) -> Result<()> {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PRESENT, D3D12_SUBRESOURCE_FOOTPRINT,
                D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ID3D12CommandList,
            },
            core::Interface as _,
        };

        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let (command_frame, command_allocator) = self.acquire_command_frame()?;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 interop backbuffer is not available")?;
        unsafe {
            command_allocator.Reset()?;
            self.command_list.Reset(
                &command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            let mut shared_to_copy = d3d12_transition(
                shared,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&shared_to_copy));
            drop_d3d12_transition_barrier(&mut shared_to_copy);
            let mut back_to_copy = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_copy));
            drop_d3d12_transition_barrier(&mut back_to_copy);

            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                    Width: size.width,
                    Height: size.height,
                    Depth: 1,
                    RowPitch: pitch_bytes,
                },
            };
            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(shared.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(back_buffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            self.command_list
                .CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop_d3d12_texture_copy_location(&mut src);
            drop_d3d12_texture_copy_location(&mut dst);

            let mut shared_to_common = d3d12_transition(
                shared,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&shared_to_common));
            drop_d3d12_transition_barrier(&mut shared_to_common);
            let mut back_to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_present));
            drop_d3d12_transition_barrier(&mut back_to_present);
            self.command_list.Close()?;
            let list: ID3D12CommandList = self.command_list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
        }
        self.signal_command_frame(command_frame)?;
        Ok(())
    }

    fn acquire_command_frame(
        &mut self,
    ) -> Result<(
        usize,
        windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    )> {
        let index = self.command_frame_index;
        self.command_frame_index = (self.command_frame_index + 1) % self.command_frames.len();
        self.wait_for_fence(self.command_frames[index].fence_value)?;
        Ok((index, self.command_frames[index].command_allocator.clone()))
    }

    fn signal_command_frame(&mut self, index: usize) -> Result<u64> {
        let fence_value = self.signal_queue()?;
        self.command_frames[index].fence_value = fence_value;
        Ok(fence_value)
    }

    fn signal_queue(&mut self) -> Result<u64> {
        self.fence_value += 1;
        unsafe {
            self.queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    fn wait_for_fence(&self, fence_value: u64) -> Result<()> {
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        if fence_value == 0 {
            return Ok(());
        }
        unsafe {
            if self.fence.GetCompletedValue() < fence_value {
                self.fence
                    .SetEventOnCompletion(fence_value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<()> {
        let fence_value = self.signal_queue()?;
        self.wait_for_fence(fence_value)
    }
}

#[cfg(windows)]
impl Drop for D3d12InteropPresenter {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.fence_event) };
        }
    }
}

#[cfg(windows)]
struct D3d12Presenter {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    command_queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    command_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    back_buffers: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    upload_slots: Vec<D3d12UploadSlot>,
    upload_index: usize,
    upload_ring: usize,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    fence_value: u64,
    fence_event: windows::Win32::Foundation::HANDLE,
    width: u32,
    height: u32,
    tearing_supported: bool,
}

#[cfg(windows)]
struct D3d12UploadSlot {
    command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    mapped: *mut u8,
    layout: windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    fence_value: u64,
}

#[cfg(windows)]
impl Drop for D3d12UploadSlot {
    fn drop(&mut self) {
        unsafe {
            self.resource.Unmap(0, None);
        }
    }
}

#[cfg(windows)]
impl D3d12Presenter {
    fn new(window: &Window, upload_ring: usize) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::HWND,
                Graphics::{
                    Direct3D::D3D_FEATURE_LEVEL_11_0,
                    Direct3D12::{
                        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                        D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                        D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandQueue,
                        ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_NONE,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2, IDXGISwapChain3,
                    },
                },
                System::Threading::CreateEventW,
            },
            core::{BOOL, Interface as _, PCWSTR},
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D12 presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut device: Option<ID3D12Device> = None;
        unsafe {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device)
                .context("failed to create D3D12 device")?;
        }
        let device = device.context("D3D12 did not return a device")?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let command_queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&queue_desc)
                .context("failed to create D3D12 command queue")?
        };
        let bootstrap_allocator: ID3D12CommandAllocator = unsafe {
            device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .context("failed to create D3D12 command allocator")?
        };
        let command_list: ID3D12GraphicsCommandList = unsafe {
            device
                .CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &bootstrap_allocator,
                    None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                )
                .context("failed to create D3D12 command list")?
        };
        unsafe {
            command_list.Close()?;
        }

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("failed to create DXGI factory")?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D12 flip presenter tearing support: {tearing_supported}");
        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Scaling: DXGI_SCALING_NONE,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(
                &command_queue,
                HWND(handle.hwnd.get() as _),
                &desc,
                None,
                None,
            )
        }
        .context("failed to create D3D12 flip-model swapchain")?
        .cast::<IDXGISwapChain3>()
        .context("failed to cast D3D12 swapchain to IDXGISwapChain3")?;

        let fence: ID3D12Fence = unsafe {
            device.CreateFence(
                0,
                windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE,
            )
        }
        .context("failed to create D3D12 fence")?;
        let fence_event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
            .context("failed to create D3D12 fence event")?;

        let mut presenter = Self {
            device,
            command_queue,
            command_list,
            swap_chain,
            back_buffers: Vec::new(),
            upload_slots: Vec::new(),
            upload_index: 0,
            upload_ring,
            fence,
            fence_value: 0,
            fence_event,
            width,
            height,
            tearing_supported,
        };
        presenter.recreate_backbuffers()?;
        presenter.recreate_upload_buffers(width, height, upload_ring)?;
        Ok(presenter)
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let total_start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        self.ensure_size(size)?;
        let upload_index = self.upload_index;
        self.upload_index = (self.upload_index + 1) % self.upload_slots.len();
        self.wait_for_upload_slot(upload_index)?;

        let map_copy_start = Instant::now();
        self.upload_bgra(upload_index, size.width, size.height, bgra)?;
        let map_copy = map_copy_start.elapsed();

        let copy_start = Instant::now();
        let fence_value = self.copy_upload_to_backbuffer(upload_index)?;
        self.upload_slots[upload_index].fence_value = fence_value;
        let draw = copy_start.elapsed();

        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            map_copy,
            draw,
            swap_present,
            total: total_start.elapsed(),
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.release_resize_references()?;
        self.back_buffers.clear();
        self.upload_slots.clear();
        self.upload_index = 0;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain
                .ResizeBuffers(
                    D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
                    width,
                    height,
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                    flags,
                )
                .context("failed to resize D3D12 swapchain buffers")?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffers()?;
        self.recreate_upload_buffers(width, height, self.upload_ring)?;
        Ok(())
    }

    fn release_resize_references(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT;

        self.wait_for_gpu()?;
        for slot in &self.upload_slots {
            self.wait_for_fence(slot.fence_value)?;
        }

        if let Some(slot) = self.upload_slots.first() {
            unsafe {
                slot.command_allocator
                    .Reset()
                    .context("failed to reset D3D12 command allocator before resize")?;
                self.command_list
                    .Reset(
                        &slot.command_allocator,
                        None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                    )
                    .context("failed to reset D3D12 command list before resize")?;
                self.command_list
                    .Close()
                    .context("failed to close D3D12 command list before resize")?;
            }
        } else {
            let command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .context("failed to create D3D12 command allocator before resize")?
            };
            unsafe {
                self.command_list
                    .Reset(
                        &command_allocator,
                        None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                    )
                    .context("failed to reset D3D12 command list before resize")?;
                self.command_list
                    .Close()
                    .context("failed to close D3D12 command list before resize")?;
            }
        }

        Ok(())
    }

    fn recreate_backbuffers(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

        self.back_buffers.clear();
        self.back_buffers.reserve(D3D12_SWAPCHAIN_BUFFER_COUNT);
        for index in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let back_buffer: ID3D12Resource = unsafe { self.swap_chain.GetBuffer(index as u32) }
                .with_context(|| format!("failed to get D3D12 swapchain backbuffer {index}"))?;
            self.back_buffers.push(back_buffer);
        }
        Ok(())
    }

    fn recreate_upload_buffers(
        &mut self,
        width: u32,
        height: u32,
        upload_ring: usize,
    ) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_UPLOAD,
                D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator, ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        };

        let texture_desc = d3d12_texture_desc(width, height);
        let mut layout =
            windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut row_count = 0;
        let mut _row_size_bytes = 0;
        let mut total_bytes = 0;
        unsafe {
            self.device.GetCopyableFootprints(
                &texture_desc,
                0,
                1,
                0,
                Some(&mut layout),
                Some(&mut row_count),
                Some(&mut _row_size_bytes),
                Some(&mut total_bytes),
            );
        }

        let upload_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: total_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };

        self.upload_slots.clear();
        self.upload_slots.reserve(upload_ring);
        for _ in 0..upload_ring {
            let command_allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .context("failed to create D3D12 upload command allocator")?
            };
            let mut resource: Option<ID3D12Resource> = None;
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &upload_desc,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                        None,
                        &mut resource,
                    )
                    .context("failed to create D3D12 upload resource")?;
            }
            let resource = resource.context("D3D12 did not return an upload resource")?;
            let mut mapped = std::ptr::null_mut();
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            unsafe {
                resource
                    .Map(0, Some(&read_range), Some(&mut mapped))
                    .context("failed to persistently map D3D12 upload resource")?;
            }
            self.upload_slots.push(D3d12UploadSlot {
                command_allocator,
                resource,
                mapped: mapped.cast(),
                layout,
                fence_value: 0,
            });
        }
        Ok(())
    }

    fn upload_bgra(&self, slot_index: usize, width: u32, height: u32, bgra: &[u8]) -> Result<()> {
        let slot = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?;
        if slot.mapped.is_null() {
            bail!("D3D12 upload resource is not mapped");
        }
        unsafe {
            let dst_pitch = slot.layout.Footprint.RowPitch as usize;
            let _used_fast_path = copy_bgra_to_mapped(bgra, slot.mapped, width, height, dst_pitch);
        }
        Ok(())
    }

    fn copy_upload_to_backbuffer(&mut self, slot_index: usize) -> Result<u64> {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT,
                D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ID3D12CommandList,
            },
            core::Interface as _,
        };

        let slot = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?;
        let command_allocator = slot.command_allocator.clone();
        let upload_resource = slot.resource.clone();
        let upload_layout = slot.layout;
        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 backbuffer is not available")?;

        unsafe {
            command_allocator.Reset()?;
            self.command_list.Reset(
                &command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            let mut to_copy = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&to_copy));
            drop_d3d12_transition_barrier(&mut to_copy);

            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(upload_resource)),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: upload_layout,
                },
            };
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(back_buffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            self.command_list
                .CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop_d3d12_texture_copy_location(&mut src);
            drop_d3d12_texture_copy_location(&mut dst);

            let mut to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&to_present));
            drop_d3d12_transition_barrier(&mut to_present);
            self.command_list.Close()?;
            let command_list: ID3D12CommandList = self.command_list.cast()?;
            self.command_queue
                .ExecuteCommandLists(&[Some(command_list)]);
        }
        self.signal_queue()
    }

    fn signal_queue(&mut self) -> Result<u64> {
        self.fence_value += 1;
        unsafe {
            self.command_queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    fn wait_for_upload_slot(&self, slot_index: usize) -> Result<()> {
        let fence_value = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?
            .fence_value;
        self.wait_for_fence(fence_value)
    }

    fn wait_for_fence(&self, fence_value: u64) -> Result<()> {
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        if fence_value == 0 {
            return Ok(());
        }
        unsafe {
            if self.fence.GetCompletedValue() < fence_value {
                self.fence
                    .SetEventOnCompletion(fence_value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<()> {
        let fence_value = self.signal_queue()?;
        self.wait_for_fence(fence_value)
    }
}

#[cfg(windows)]
impl Drop for D3d12Presenter {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.fence_event) };
        }
    }
}

#[cfg(windows)]
fn d3d12_texture_desc(
    width: u32,
    height: u32,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC {
    use windows::Win32::Graphics::{
        Direct3D12::{
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
        },
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    };

    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

#[cfg(windows)]
fn d3d12_transition(
    resource: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_TRANSITION_BARRIER,
    };

    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[cfg(windows)]
fn drop_d3d12_texture_copy_location(
    location: &mut windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_COPY_LOCATION,
) {
    unsafe {
        std::mem::ManuallyDrop::drop(&mut location.pResource);
    }
}

#[cfg(windows)]
fn drop_d3d12_transition_barrier(
    barrier: &mut windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER,
) {
    unsafe {
        let transition = &mut *barrier.Anonymous.Transition;
        std::mem::ManuallyDrop::drop(&mut transition.pResource);
    }
}

#[cfg(windows)]
struct D3d11Presenter {
    device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain1,
    back_buffer: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    upload_slots: Vec<D3d11UploadSlot>,
    upload_index: usize,
    upload_mode: D3dUploadMode,
    width: u32,
    height: u32,
    tearing_supported: bool,
}

#[cfg(windows)]
struct D3d11UploadSlot {
    texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
}

#[cfg(windows)]
impl D3d11Presenter {
    fn new(window: &Window, upload_ring: usize, upload_mode: D3dUploadMode) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::{HMODULE, HWND},
                Graphics::{
                    Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0},
                    Direct3D11::{
                        D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice,
                        ID3D11Device, ID3D11DeviceContext,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_STRETCH,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2,
                    },
                },
            },
            core::BOOL,
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D11 presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .context("failed to create D3D11 device")?;
        }
        let device = device.context("D3D11 did not return a device")?;
        let context = context.context("D3D11 did not return an immediate context")?;

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("failed to create DXGI factory")?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D11 flip presenter tearing support: {tearing_supported}");
        eprintln!("D3D11 upload mode: {upload_mode}");

        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 3,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(&device, HWND(handle.hwnd.get() as _), &desc, None, None)
        }
        .context("failed to create D3D11 flip-model swapchain")?;

        let mut presenter = Self {
            device,
            context,
            swap_chain,
            back_buffer: None,
            upload_slots: Vec::new(),
            upload_index: 0,
            upload_mode,
            width: size.width.max(1),
            height: size.height.max(1),
            tearing_supported,
        };
        presenter.recreate_backbuffer()?;
        if upload_mode == D3dUploadMode::MappedCopy {
            presenter.recreate_upload_textures(width, height, upload_ring)?;
        }
        Ok(presenter)
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let total_start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        self.ensure_size(size)?;
        let (map_copy, draw) = match self.upload_mode {
            D3dUploadMode::MappedCopy => {
                let upload_index = self.upload_index;
                self.upload_index = (self.upload_index + 1) % self.upload_slots.len();
                let map_copy_start = Instant::now();
                self.upload_bgra(upload_index, size.width, size.height, bgra)?;
                let map_copy = map_copy_start.elapsed();
                let copy_start = Instant::now();
                self.copy_upload_to_backbuffer(upload_index)?;
                (map_copy, copy_start.elapsed())
            }
            D3dUploadMode::UpdateSubresource => {
                let map_copy_start = Instant::now();
                self.update_backbuffer(size.width, bgra)?;
                (map_copy_start.elapsed(), Duration::ZERO)
            }
        };
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            map_copy,
            draw,
            swap_present,
            total: total_start.elapsed(),
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.back_buffer = None;
        let upload_ring = self.upload_slots.len();
        self.upload_slots.clear();
        self.upload_index = 0;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain.ResizeBuffers(
                0,
                width,
                height,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                flags,
            )?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffer()?;
        if self.upload_mode == D3dUploadMode::MappedCopy {
            self.recreate_upload_textures(width, height, upload_ring)?;
        }
        Ok(())
    }

    fn recreate_backbuffer(&mut self) -> Result<()> {
        self.back_buffer = Some(unsafe { self.swap_chain.GetBuffer(0)? });
        Ok(())
    }

    fn recreate_upload_textures(
        &mut self,
        width: u32,
        height: u32,
        upload_ring: usize,
    ) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D11::{
                D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_DYNAMIC, ID3D11Texture2D,
            },
            Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
        };

        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
        };
        self.upload_slots.clear();
        self.upload_slots.reserve(upload_ring);
        for _ in 0..upload_ring {
            let mut texture: Option<ID3D11Texture2D> = None;
            unsafe {
                self.device
                    .CreateTexture2D(&desc, None, Some(&mut texture))?;
            }
            let texture = texture.context("D3D11 did not return an upload texture")?;
            self.upload_slots.push(D3d11UploadSlot { texture });
        }
        self.upload_index = 0;
        Ok(())
    }

    fn upload_bgra(&self, slot_index: usize, width: u32, height: u32, bgra: &[u8]) -> Result<()> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE,
        };

        let texture = &self
            .upload_slots
            .get(slot_index)
            .context("D3D11 upload texture is not available")?
            .texture;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(texture, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))?;
            let dst_pitch = mapped.RowPitch as usize;
            let dst_base = mapped.pData.cast::<u8>();
            let _used_fast_path = copy_bgra_to_mapped(bgra, dst_base, width, height, dst_pitch);
            self.context.Unmap(texture, 0);
        }
        Ok(())
    }

    fn copy_upload_to_backbuffer(&self, slot_index: usize) -> Result<()> {
        let upload_texture = self
            .upload_slots
            .get(slot_index)
            .context("D3D11 upload texture is not available")?
            .texture
            .clone();
        let back_buffer = self
            .back_buffer
            .as_ref()
            .context("D3D11 backbuffer is not available")?;
        unsafe {
            self.context.CopyResource(back_buffer, &upload_texture);
        }
        Ok(())
    }

    fn update_backbuffer(&self, width: u32, bgra: &[u8]) -> Result<()> {
        let back_buffer = self
            .back_buffer
            .as_ref()
            .context("D3D11 backbuffer is not available")?;
        unsafe {
            self.context.UpdateSubresource(
                back_buffer,
                0,
                None,
                bgra.as_ptr().cast(),
                width * 4,
                0,
            );
        }
        Ok(())
    }
}

#[cfg(windows)]
fn dxgi_tearing_supported(factory: &windows::Win32::Graphics::Dxgi::IDXGIFactory2) -> bool {
    use windows::{
        Win32::Graphics::Dxgi::{DXGI_FEATURE_PRESENT_ALLOW_TEARING, IDXGIFactory5},
        core::BOOL,
        core::Interface as _,
    };

    let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allow_tearing = BOOL(0);
    unsafe {
        factory5
            .CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                (&mut allow_tearing as *mut BOOL).cast(),
                std::mem::size_of::<BOOL>() as u32,
            )
            .is_ok()
            && allow_tearing.as_bool()
    }
}

#[cfg(windows)]
fn bitmap_info(width: u32, height: u32) -> windows_sys::Win32::Graphics::Gdi::BITMAPINFO {
    use windows_sys::Win32::Graphics::Gdi::{BI_RGB, BITMAPINFO, BITMAPINFOHEADER, RGBQUAD};

    BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: width.saturating_mul(height).saturating_mul(4),
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [RGBQUAD {
            rgbBlue: 0,
            rgbGreen: 0,
            rgbRed: 0,
            rgbReserved: 0,
        }],
    }
}

#[cfg(not(windows))]
struct WindowPresenter;

#[cfg(not(windows))]
impl WindowPresenter {
    fn new(
        _window: &Window,
        _kind: PresenterKind,
        _upload_ring: usize,
        _d3d_upload: D3dUploadMode,
        _interop_device: Option<&NeoD3d12InteropDevice>,
    ) -> Result<Self> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    fn present(&mut self, _size: PhysicalSize<u32>, _bgra: &[u8]) -> Result<PresentTimings> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    fn kind(&self) -> PresenterKind {
        PresenterKind::Gdi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_kernel_abi_accepts_expected_signature() {
        let source =
            "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        validate_live_kernel_abi(source).unwrap();
    }

    #[test]
    fn live_graph_cuda_source_wraps_image_kernel() {
        let source =
            "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        let cuda = live_graph_cuda_source(source).unwrap();
        assert!(cuda.contains("__device__ __forceinline__ void neo_user_image"));
        assert!(cuda.contains("extern \"C\" __global__ void image_graph"));
        assert!(cuda.contains("params->time"));
        assert!(cuda.contains("params->frame"));
    }

    #[test]
    fn live_kernel_abi_rejects_missing_frame() {
        let source = "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time) {}";
        let err = validate_live_kernel_abi(source).unwrap_err().to_string();
        assert!(err.contains("must have 5 params"));
    }

    #[test]
    fn quad_stress_kernel_keeps_live_abi() {
        validate_live_kernel_abi(include_str!("../../stress-quads/million_quads.neo")).unwrap();
    }

    #[test]
    fn mesh_kernel_abi_accepts_expected_signature() {
        let source = "kernel fn raster(global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame) {}";
        validate_mesh_kernel_abi(source).unwrap();
    }

    #[test]
    fn mesh_kernel_abi_rejects_missing_mesh_pointer() {
        let source =
            "kernel fn raster(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        let err = validate_mesh_kernel_abi(source).unwrap_err().to_string();
        assert!(err.contains("must have 6 params"));
    }

    #[test]
    fn instance_kernel_abi_accepts_expected_signature() {
        let source = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame) {}";
        validate_instance_kernel_abi(source).unwrap();
    }

    #[test]
    fn instance_kernel_abi_rejects_missing_camera_pointer() {
        let source = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, f32 time, u32 frame) {}";
        let err = validate_instance_kernel_abi(source)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must have 8 params"));
    }

    #[test]
    fn macrocell_instance_kernel_abi_accepts_visibility_pointer() {
        let source = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* visibility, global u8* camera, f32 time, u32 frame) {}";
        validate_macrocell_instance_kernel_abi(source).unwrap();
        let missing_visibility = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame) {}";
        let err = validate_macrocell_instance_kernel_abi(missing_visibility)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must have 9 params"));
    }

    #[test]
    fn instance_stress_kernel_keeps_expected_abi() {
        validate_culled_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_fast.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_baseline.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_tiled.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_tiled_aosoa32.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_tiled_aosoa64.neo"
        ))
        .unwrap();
        validate_macrocell_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_macrocell_aosoa32.neo"
        ))
        .unwrap();
        assert!(
            !include_str!("../../stress-quads/three_d_instances_tiled.neo")
                .contains("instance_cull")
        );
    }

    #[test]
    fn hardware_raster_source_keeps_expected_abi() {
        let source = include_str!("../../stress-quads/hardware_raster.neo");
        validate_raster_stress_abi(source).unwrap();
        assert!(source.contains("kernel fn raster_cull_init(global u8* args, global u8* camera)"));
        assert!(source.contains("((unsigned int*)camera)[26]"));
        assert!(source.contains("global u8* camera"));
        assert!(source.contains("corner_x * s.x"));
        assert!(!source.contains("* s.x * quad_scale"));
        assert!(source.contains("atomicAdd"));
        assert!(source.contains("shared u32 block_prefix[256]"));
        assert!(source.contains("((unsigned int*)camera)[24]"));
        assert!(source.contains("((unsigned int*)camera)[27]"));
        assert!(source.contains("projected_x"));
        assert!(source.contains("let min_projected: f32 = cam->view.w;"));
        assert!(!source.contains("* 0.85f"));
        assert!(source.contains("vertex fn quad_vs_direct()"));
        validate_raster_stress_abi_with_material(
            source,
            HardwareRasterPlan::stock().material_kernel().abi(),
        )
        .unwrap();
        let shaders = neo_lang::lower_graphics_to_hlsl(source).unwrap();
        assert!(shaders.vertex_source.contains("quad_vs"));
        assert!(shaders.vertex_source.contains("visible_instance_id"));
        assert!(shaders.vertex_source.contains("raster_camera_origin"));
        assert!(shaders.vertex_source.contains("raster_camera_tan_x"));
        assert!(shaders.fragment_source.contains("quad_fs"));
        let cuda = neo_lang::lower_to_cuda(source).unwrap();
        assert!(cuda.contains("__shared__ unsigned int block_prefix[256];"));
        assert!(cuda.contains("__syncthreads();"));
        assert!(cuda.contains("block_base[0] + local_out"));
        assert!(cuda.contains("unsigned int visibility_mode"));
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_stock_hlsl_compiles() {
        let source = include_str!("../../stress-quads/hardware_raster.neo");
        let material = hardware_raster_material();
        validate_raster_stress_abi_with_material(source, material.abi()).unwrap();
        let shaders = neo_lang::lower_graphics_to_hlsl_for_entries_with_bindings(
            source,
            material.vertex_entrypoint(),
            material.fragment_entrypoint(),
            graphics_bindings_for_material(material.abi()).unwrap(),
        )
        .unwrap();
        let vertex_entrypoint = std::ffi::CString::new(material.vertex_entrypoint()).unwrap();
        let fragment_entrypoint = std::ffi::CString::new(material.fragment_entrypoint()).unwrap();
        compile_hlsl(
            &shaders.vertex_source,
            windows::core::PCSTR(vertex_entrypoint.as_ptr().cast()),
            windows::core::PCSTR(D3D12_RASTER_VS_TARGET.as_ptr()),
        )
        .unwrap();
        compile_hlsl(
            &shaders.fragment_source,
            windows::core::PCSTR(fragment_entrypoint.as_ptr().cast()),
            windows::core::PCSTR(D3D12_RASTER_PS_TARGET.as_ptr()),
        )
        .unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_draw_all_stock_hlsl_compiles_without_visible_stream_binding() {
        let source = include_str!("../../stress-quads/hardware_raster.neo");
        let material = HardwareRasterPlan::stock().material_kernel();
        assert_eq!(material.vertex_entrypoint(), "quad_vs_direct");
        assert_eq!(material.abi().bindings.len(), 3);
        assert!(
            material
                .abi()
                .binding(MaterialBindingKind::VisibleInstanceStream)
                .is_none()
        );
        validate_raster_stress_abi_with_material(source, material.abi()).unwrap();
        let bindings = graphics_bindings_for_material(material.abi()).unwrap();
        assert_eq!(bindings.instances, neo_lang::HlslRegister::new(1, 0));
        assert_eq!(bindings.geometry, neo_lang::HlslRegister::new(2, 0));
        let shaders = neo_lang::lower_graphics_to_hlsl_for_entries_with_bindings(
            source,
            material.vertex_entrypoint(),
            material.fragment_entrypoint(),
            bindings,
        )
        .unwrap();
        assert!(shaders.vertex_source.contains("quad_vs_direct("));
        assert!(
            !shaders
                .vertex_source
                .contains("visible_instance_id(instance_id")
        );
        let vertex_entrypoint = std::ffi::CString::new(material.vertex_entrypoint()).unwrap();
        let fragment_entrypoint = std::ffi::CString::new(material.fragment_entrypoint()).unwrap();
        compile_hlsl(
            &shaders.vertex_source,
            windows::core::PCSTR(vertex_entrypoint.as_ptr().cast()),
            windows::core::PCSTR(D3D12_RASTER_VS_TARGET.as_ptr()),
        )
        .unwrap();
        compile_hlsl(
            &shaders.fragment_source,
            windows::core::PCSTR(fragment_entrypoint.as_ptr().cast()),
            windows::core::PCSTR(D3D12_RASTER_PS_TARGET.as_ptr()),
        )
        .unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_material_bindings_are_explicit() {
        let material = hardware_raster_material();
        let abi = material.abi();
        let draw_params =
            material_binding(abi, MaterialBindingKind::DrawParams, "draw params").unwrap();
        let legacy_raster_params =
            material_binding(abi, MaterialBindingKind::RasterParams, "raster params").unwrap();
        let visible = material_binding(
            abi,
            MaterialBindingKind::VisibleInstanceStream,
            "visible InstanceStream",
        )
        .unwrap();
        let instances =
            material_binding(abi, MaterialBindingKind::InstanceStream, "InstanceStream").unwrap();
        assert_eq!(draw_params.kind, MaterialBindingKind::DrawParams);
        assert_eq!(draw_params.root_parameter_index, 0);
        assert_eq!(draw_params.shader_register, 0);
        assert_eq!(legacy_raster_params.root_parameter_index, 0);
        assert_eq!(visible.root_parameter_index, 1);
        assert_eq!(visible.shader_register, 0);
        assert_eq!(instances.root_parameter_index, 2);
        assert_eq!(instances.shader_register, 1);
        let geometry =
            material_binding(abi, MaterialBindingKind::GeometryStream, "GeometryStream").unwrap();
        assert_eq!(geometry.root_parameter_index, 3);
        assert_eq!(geometry.shader_register, 2);
        validate_contiguous_material_root_bindings(abi).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_direct_material_bindings_are_explicit() {
        let material =
            MaterialKernel::from_stages("hardware-raster", "quad_vs_direct", "quad_fs").with_abi(
                MaterialKernelAbi::direct_instance_color("quad_vs_direct", "quad_fs"),
            );
        let abi = material.abi();
        let draw_params =
            material_binding(abi, MaterialBindingKind::DrawParams, "draw params").unwrap();
        let legacy_raster_params =
            material_binding(abi, MaterialBindingKind::RasterParams, "raster params").unwrap();
        assert!(
            abi.binding(MaterialBindingKind::VisibleInstanceStream)
                .is_none()
        );
        let instances =
            material_binding(abi, MaterialBindingKind::InstanceStream, "InstanceStream").unwrap();
        let geometry =
            material_binding(abi, MaterialBindingKind::GeometryStream, "GeometryStream").unwrap();
        assert_eq!(draw_params.kind, MaterialBindingKind::DrawParams);
        assert_eq!(draw_params.root_parameter_index, 0);
        assert_eq!(legacy_raster_params.root_parameter_index, 0);
        assert_eq!(instances.root_parameter_index, 1);
        assert_eq!(geometry.root_parameter_index, 2);
        validate_contiguous_material_root_bindings(abi).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_submit_kind_follows_material_abi() {
        let direct = MaterialKernelAbi::direct_instance_color("quad_vs_direct", "quad_fs");
        assert_eq!(
            hardware_raster_submit_kind(&direct),
            HardwareRasterSubmitKind::DirectIndexedInstanced
        );
        let culled = MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs");
        assert_eq!(
            hardware_raster_submit_kind(&culled),
            HardwareRasterSubmitKind::IndirectIndexedInstanced
        );
    }

    #[test]
    fn hardware_raster_stock_plan_syncs_material_to_draw_policy() {
        let mut plan = HardwareRasterPlan::stock();
        assert_eq!(plan.draw_policy, DrawPolicyPlan::DrawAll);
        assert_eq!(plan.material.vertex_entrypoint, "quad_vs_direct");
        assert_eq!(
            plan.material.kind,
            MaterialKernelPlanKind::DirectInstanceColor
        );

        plan.draw_policy = DrawPolicyPlan::ComputeCulled;
        plan.sync_stock_material_to_draw_policy();
        assert_eq!(plan.material.vertex_entrypoint, "quad_vs");
        assert_eq!(
            plan.material.kind,
            MaterialKernelPlanKind::ComputeCulledInstanceColor
        );
    }

    #[test]
    fn hardware_raster_plan_exposes_neutral_draw_contract() {
        let mut plan = DrawExecutionPlan::stock();
        plan.draw_policy = DrawPolicyPlan::ComputeCulled;
        plan.cull_order = DrawCullOrder::StableDense;
        plan.visibility = DrawVisibilityMode::ProjectedSize;
        plan.min_projected_millipixels = 850;
        plan.sync_stock_material_to_draw_policy();

        let _geometry: &GeometryStreamPlan = plan.geometry_stream();
        let _instances: &InstanceStreamPlan = plan.instance_stream();
        let _target: &TargetPlan = plan.target();
        let _material: &MaterialKernelPlan = plan.material();
        let _policy: DrawPolicyPlan = plan.draw_policy();
        let _cull_order: DrawCullOrder = plan.cull_order;
        let _visibility: DrawVisibilityMode = plan.visibility;

        assert_eq!(plan.draw_name(), "main");
        assert_eq!(plan.backend(), DrawBackend::HardwareRaster);
        assert_eq!(plan.backend().label(), "hardware-raster");
        assert_eq!(plan.geometry_stream().name, "quad");
        assert_eq!(plan.instance_stream().name, "instances");
        assert_eq!(plan.target().name, "window");
        assert_eq!(plan.material().name, "hardware-raster");
        assert_eq!(plan.draw_policy(), DrawPolicyPlan::ComputeCulled);
        assert_eq!(plan.depth(), DrawDepthMode::Auto);
        assert!(plan.uses_depth());
        assert_eq!(plan.draw_policy().label(), "compute-culled");
        assert_eq!(plan.policy(), DrawPolicy::ComputeCulled);
        assert_eq!(plan.cull_order(), DrawCullOrder::StableDense);
        assert_eq!(plan.cull_order().label(), "stable-dense");
        assert_eq!(plan.visibility(), DrawVisibilityMode::ProjectedSize);
        assert_eq!(plan.visibility().label(), "projected-size");
        assert_eq!(plan.min_projected_pixels(), 0.85);
        assert_eq!(
            plan.policy_config(),
            DrawPolicyConfig::compute_culled_with_visibility(
                RuntimeRasterCullOrder::StableDense,
                RuntimeRasterVisibilityMode::ProjectedSize
            )
            .with_min_projected_millipixels(850)
        );

        let contract = plan.contract();
        let _contract: DrawExecutionContract = contract.clone();
        assert_eq!(contract.draw_name, "main");
        assert_eq!(contract.geometry_stream, "quad");
        assert_eq!(contract.instance_stream, "instances");
        assert_eq!(contract.instance_layout, StressInstanceLayout::AoSoA32);
        assert_eq!(contract.material, "hardware-raster");
        assert_eq!(contract.target, "window");
        assert_eq!(contract.target_width, DEFAULT_WIDTH);
        assert_eq!(contract.target_height, DEFAULT_HEIGHT);
        assert_eq!(
            contract.target_dimensions(),
            (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        );
        assert_eq!(contract.policy(), DrawPolicy::ComputeCulled);
        assert_eq!(contract.policy_label(), "compute-culled");
        assert_eq!(contract.depth_label(), "auto");
        assert!(contract.uses_depth());
        assert_eq!(contract.cull_order_label(), "stable-dense");
        assert_eq!(contract.visibility_label(), "projected-size");
        assert_eq!(contract.min_projected_pixels(), 0.85);
        assert_eq!(contract.instance_layout_label(), "aosoa32");
        assert_eq!(contract.backend, DrawBackend::HardwareRaster);
        assert_eq!(contract.backend_label(), "hardware-raster");
        assert_eq!(contract.policy_config, plan.policy_config());
    }

    #[test]
    fn draw_depth_auto_keeps_culled_stable_and_draw_all_fast() {
        assert!(!DrawDepthMode::Auto.uses_depth(DrawPolicyPlan::DrawAll));
        assert!(DrawDepthMode::Auto.uses_depth(DrawPolicyPlan::ComputeCulled));
        assert!(DrawDepthMode::On.uses_depth(DrawPolicyPlan::DrawAll));
        assert!(!DrawDepthMode::Off.uses_depth(DrawPolicyPlan::ComputeCulled));
        assert_eq!(
            "auto".parse::<DrawDepthMode>().unwrap(),
            DrawDepthMode::Auto
        );
        assert_eq!("on".parse::<DrawDepthMode>().unwrap(), DrawDepthMode::On);
        assert_eq!("off".parse::<DrawDepthMode>().unwrap(), DrawDepthMode::Off);
        let err = "maybe".parse::<DrawDepthMode>().unwrap_err().to_string();
        assert!(err.contains("expected auto, on, or off"));
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_uses_shader_model_5_1_for_register_spaces() {
        assert_eq!(D3D12_RASTER_VS_TARGET, b"vs_5_1\0");
        assert_eq!(D3D12_RASTER_PS_TARGET, b"ps_5_1\0");
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_material_bindings_drive_hlsl_contract() {
        let mut abi = MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs");
        abi.bindings[0].shader_register = 4;
        abi.bindings[0].register_space = 2;
        abi.bindings[1].shader_register = 7;
        abi.bindings[1].register_space = 3;
        abi.bindings[2].shader_register = 8;
        abi.bindings[2].register_space = 3;
        abi.bindings[3].shader_register = 9;
        abi.bindings[3].register_space = 4;

        let bindings = graphics_bindings_for_material(&abi).unwrap();
        assert_eq!(bindings.raster_params, neo_lang::HlslRegister::new(4, 2));
        assert_eq!(
            bindings.visible_instances,
            neo_lang::HlslRegister::new(7, 3)
        );
        assert_eq!(bindings.instances, neo_lang::HlslRegister::new(8, 3));
        assert_eq!(bindings.geometry, neo_lang::HlslRegister::new(9, 4));

        let source = include_str!("../../stress-quads/hardware_raster.neo");
        let shaders = neo_lang::lower_graphics_to_hlsl_with_bindings(source, bindings).unwrap();
        assert!(
            shaders
                .vertex_source
                .contains("RasterParams : register(b4, space2)")
        );
        assert!(
            shaders
                .vertex_source
                .contains("neo_visible_instances : register(t7, space3)")
        );
        assert!(
            shaders
                .vertex_source
                .contains("neo_instances : register(t8, space3)")
        );
        assert!(
            shaders
                .vertex_source
                .contains("neo_geometry : register(t9, space4)")
        );
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_material_entrypoints_drive_hlsl_selection() {
        let abi = MaterialKernelAbi::compute_culled_instance_color("material_vs", "material_fs");
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn fallback_vs() {
    set_position(vec4f(0.0f, 0.0f, 0.0f, 1.0f));
    set_color(vec4f(1.0f, 0.0f, 0.0f, 1.0f));
}
fragment fn fallback_fs() {
    return input_color();
}
vertex fn material_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    let corner: vec3f = neo_geometry_position3f(vertex_id());
    set_position(vec4f(p.x + corner.x, p.y + corner.y, p.z + corner.z, 1.0f));
    set_color(vec4f(0.0f, 1.0f, 0.0f, 1.0f));
}
fragment fn material_fs() {
    return input_color();
}
"#;
        validate_raster_stress_abi_with_material(source, &abi).unwrap();
        let shaders = neo_lang::lower_graphics_to_hlsl_for_entries_with_bindings(
            source,
            &abi.vertex_entrypoint,
            &abi.fragment_entrypoint,
            graphics_bindings_for_material(&abi).unwrap(),
        )
        .unwrap();
        assert!(shaders.vertex_source.contains("material_vs("));
        assert!(!shaders.vertex_source.contains("fallback_vs("));
        assert!(shaders.fragment_source.contains("material_fs("));
        assert!(!shaders.fragment_source.contains("fallback_fs("));
    }

    #[test]
    fn hardware_raster_plan_accepts_stock_executor_contract_with_custom_material() {
        let plan = HardwareRasterPlan {
            draw_name: "main".to_string(),
            geometry_stream: HardwareRasterGeometryStreamPlan::stock_quad(),
            instance_stream: HardwareRasterInstanceStreamPlan::stock_instances(
                InstanceGrid::new(16, 16, 4),
                StressInstanceLayout::AoSoA64,
            ),
            target: HardwareRasterTargetPlan::window(1280, 720),
            material: HardwareRasterMaterialPlan::compute_culled_instance_color(
                "lit-quads",
                "lit_quad_vs",
                "lit_quad_fs",
            ),
            draw_policy: HardwareRasterDrawPolicy::ComputeCulled,
            depth: DrawDepthMode::Auto,
            cull_order: HardwareRasterCullOrder::AtomicCompact,
            visibility: HardwareRasterVisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_MIN_PROJECTED_MILLIPIXELS,
        };

        plan.validate_executor_contract().unwrap();
        assert_eq!(plan.geometry_stream.index_count(), 6);
        assert_eq!(plan.geometry_stream.indices_u16, vec![0, 1, 2, 2, 1, 3]);
        assert_eq!(plan.instance_stream.grid.count(), Some(1024));
        assert_eq!(plan.instance_stream.layout, StressInstanceLayout::AoSoA64);
        assert_eq!(plan.target.width, 1280);
        assert_eq!(plan.target.height, 720);
        let material = plan.material_kernel();
        assert_eq!(material.label(), "lit-quads");
        assert_eq!(material.vertex_entrypoint(), "lit_quad_vs");
        assert_eq!(material.fragment_entrypoint(), "lit_quad_fs");
    }

    #[test]
    fn hardware_raster_plan_rejects_unsupported_geometry_and_invalid_resources() {
        let mut plan = HardwareRasterPlan::stock();
        plan.geometry_stream.indices_u16.clear();
        let err = plan.validate_executor_contract().unwrap_err().to_string();
        assert!(err.contains("GeometryStream `quad` with no indices"));

        let mut plan = HardwareRasterPlan::stock();
        plan.instance_stream.grid = InstanceGrid::new(4, 0, 4);
        let err = plan.validate_executor_contract().unwrap_err().to_string();
        assert!(err.contains("--instance-grid dimensions"));

        let mut plan = HardwareRasterPlan::stock();
        plan.target.width = 0;
        let err = plan.validate_executor_contract().unwrap_err().to_string();
        assert!(err.contains("Target `window` with zero size"));
    }

    #[cfg(windows)]
    #[test]
    fn hardware_raster_material_bindings_reject_mismatched_root_contract() {
        let mut abi = MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs");
        abi.bindings[1].root_parameter_index = 2;
        let err = validate_contiguous_material_root_bindings(&abi)
            .unwrap_err()
            .to_string();
        assert!(err.contains("root parameter 1"));
    }

    #[cfg(windows)]
    #[test]
    fn raster_root_constants_pack_camera_contract() {
        let camera = CameraParams {
            origin: [1.0, 2.0, 3.0, 0.0],
            right: [4.0, 5.0, 6.0, 0.0],
            up: [7.0, 8.0, 9.0, 0.0],
            forward: [10.0, 11.0, 12.0, 0.0],
            grid: [13, 14, 15, 0],
            view: [16.0, 17.0, 18.0, 0.0],
            config: [19, 20, 21, 22],
        };
        let constants = raster_root_constants(
            InstanceGrid { x: 2, y: 3, z: 4 },
            PhysicalSize::new(640, 360),
            camera,
            &HardwareRasterGeometryStreamPlan::indexed_u16(
                "test",
                vec![0; 64],
                16,
                12,
                vec![0, 1, 2],
            ),
            99,
        );
        assert_eq!(constants.len(), RASTER_ROOT_CONSTANT_DWORDS as usize);
        assert_eq!(&constants[0..8], &[2, 3, 4, 99, 640, 360, 16, 12]);
        assert_eq!(constants[8], 1.0f32.to_bits());
        assert_eq!(constants[12], 4.0f32.to_bits());
        assert_eq!(constants[16], 7.0f32.to_bits());
        assert_eq!(constants[20], 10.0f32.to_bits());
        assert_eq!(constants[24], 16.0f32.to_bits());
        assert_eq!(constants[26], 18.0f32.to_bits());
    }

    #[test]
    fn hardware_raster_abi_rejects_missing_cull_init_kernel() {
        let source = r#"
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    set_position(vec4f(p.x, p.y, p.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("raster_cull_init"));
    }

    #[test]
    fn hardware_raster_abi_rejects_cull_stage_without_camera() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, u32 instance_count, u32 frame) {
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    set_position(vec4f(p.x, p.y, p.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("camera"));
    }

    #[test]
    fn hardware_raster_abi_rejects_vertex_stage_without_instance_stream() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    set_position(vec4f(0.0f, 0.0f, 0.0f, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("compute-culled InstanceStream"));
    }

    #[test]
    fn hardware_raster_abi_rejects_cull_stage_without_visible_stream() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    set_position(vec4f(p.x, p.y, p.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("visible InstanceStream"));
    }

    #[test]
    fn hardware_raster_abi_ignores_commented_contract_tokens() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    // ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    // let id: u32 = visible_instance_id(instance_id());
    let id: u32 = instance_id();
    let p: vec3f = neo_instance_position3f(id);
    set_position(vec4f(p.x, p.y, p.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(
            err.contains("visible InstanceStream") || err.contains("compute-culled InstanceStream")
        );
    }

    #[test]
    fn hardware_raster_abi_ignores_disabled_false_blocks() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    if (false) {
        ((unsigned int*)visible)[0] = 0u;
    }
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    if (false) {
        let dead_id: u32 = visible_instance_id(instance_id());
        set_position(vec4f(0.0f, 0.0f, 0.0f, 1.0f));
    }
    let id: u32 = instance_id();
    let p: vec3f = neo_instance_position3f(id);
    set_position(vec4f(p.x, p.y, p.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(
            err.contains("visible InstanceStream") || err.contains("compute-culled InstanceStream")
        );
    }

    #[test]
    fn hardware_raster_abi_rejects_vertex_stage_without_position_output() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    let corner: vec3f = neo_geometry_position3f(vertex_id());
    let q: vec3f = vec3f(p.x + corner.x, p.y + corner.y, p.z + corner.z);
    set_color(vec4f(p.x, p.y, p.z, 1.0f));
}
fragment fn quad_fs() {
    return input_color();
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("clip-space position"));
    }

    #[test]
    fn hardware_raster_abi_rejects_fragment_stage_without_interpolated_color() {
        let source = r#"
kernel fn raster_cull_init(global u8* args, global u8* camera) {
    ((unsigned int*)args)[0] = 6u;
}
kernel fn raster_cull(global u8* args, global u8* visible, global u8* instances, global u8* camera, u32 instance_count, u32 frame) {
    let cam: u8* = camera;
    ((unsigned int*)visible)[0] = 0u;
    ((unsigned int*)args)[0] = 6u;
}
vertex fn quad_vs() {
    let id: u32 = visible_instance_id(instance_id());
    let p: vec3f = neo_instance_position3f(id);
    let corner: vec3f = neo_geometry_position3f(vertex_id());
    set_position(vec4f(p.x + corner.x, p.y + corner.y, p.z + corner.z, 1.0f));
    set_color(vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}
fragment fn quad_fs() {
    return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
}
"#;
        let err = validate_raster_stress_abi(source).unwrap_err().to_string();
        assert!(err.contains("input_color"));
    }

    #[test]
    fn reload_state_keeps_last_good_after_failure() {
        let mut reload = ReloadState::new(7);
        assert!(!reload.try_replace::<&str>(Err("compile failed")));
        assert_eq!(reload.active, 7);
        assert_eq!(reload.generation, 0);
        assert!(
            reload
                .last_error
                .as_deref()
                .unwrap()
                .contains("compile failed")
        );
        assert!(reload.try_replace::<&str>(Ok(9)));
        assert_eq!(reload.active, 9);
        assert_eq!(reload.generation, 1);
        assert!(reload.last_error.is_none());
    }

    #[test]
    fn live_options_accept_bounded_run_controls() {
        let options = LiveOptions::parse(
            [
                "custom.neo",
                "--title",
                "Tiny Stress",
                "--width",
                "320",
                "--height",
                "180",
                "--frames",
                "12",
                "--seconds",
                "0.5",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.source_path, PathBuf::from("custom.neo"));
        assert_eq!(options.title, "Tiny Stress");
        assert_eq!(options.width, 320);
        assert_eq!(options.height, 180);
        assert_eq!(options.max_frames, Some(12));
        assert_eq!(options.max_seconds, Some(0.5));
        assert!(options.should_stop(12, Duration::from_millis(1)));
        assert!(options.should_stop(1, Duration::from_millis(500)));
    }

    #[test]
    fn live_options_accept_kernel_throughput_controls() {
        let options = LiveOptions::parse(
            [
                "custom.neo",
                "--mode",
                "kernel-throughput",
                "--sample-every",
                "128",
                "--max-inflight",
                "8",
                "--presenter",
                "gdi",
                "--present-target-fps",
                "60",
                "--kernel-target-fps",
                "240",
                "--present-ring",
                "5",
                "--d3d-upload",
                "update-subresource",
                "--interop-fallback",
                "fail",
                "--no-hot-reload",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::KernelThroughput);
        assert_eq!(options.sample_every, 128);
        assert_eq!(options.max_inflight, 8);
        assert_eq!(options.presenter, PresenterKind::Gdi);
        assert_eq!(options.present_target_fps, Some(60.0));
        assert_eq!(options.kernel_target_fps, Some(240.0));
        assert_eq!(options.kernel_cap(), Some(240.0));
        assert_eq!(options.present_ring, 5);
        assert_eq!(options.d3d_upload, D3dUploadMode::UpdateSubresource);
        assert_eq!(options.interop_fallback, InteropFallback::Fail);
        assert!(!options.hot_reload);
        assert_eq!(
            options.present_interval(),
            Some(Duration::from_secs_f32(1.0 / 60.0))
        );
    }

    #[test]
    fn live_options_accept_mesh_demo_mode() {
        let options = LiveOptions::parse(
            [
                "examples/mesh-buffer/raster.neo",
                "--mode",
                "mesh-demo",
                "--presenter",
                "d3d12-interop",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::MeshDemo);
        assert_eq!(options.presenter, PresenterKind::D3d12Interop);
    }

    #[test]
    fn live_options_accept_instance_stress_mode_and_grid() {
        let options = LiveOptions::parse(
            [
                "examples/stress-quads/three_d_instances.neo",
                "--mode",
                "instance-stress",
                "--presenter",
                "d3d12-interop",
                "--instance-grid",
                "256x256x128",
                "--instance-stress-variant",
                "baseline",
                "--instance-debug-view",
                "iterations",
                "--instance-layout",
                "aosoa64",
                "--render-policy",
                "pause-when-empty",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::InstanceStress);
        assert_eq!(options.presenter, PresenterKind::D3d12Interop);
        assert_eq!(options.instance_grid, DEFAULT_INSTANCE_GRID);
        assert_eq!(options.instance_grid.count(), Some(8_388_608));
        assert_eq!(
            options.instance_stress_variant,
            InstanceStressVariant::Baseline
        );
        assert_eq!(options.instance_debug_view, InstanceDebugView::Iterations);
        assert_eq!(options.instance_layout, StressInstanceLayout::AoSoA64);
        assert_eq!(options.render_policy, RenderPolicy::PauseWhenEmpty);
    }

    #[test]
    fn live_options_accept_draw_stress_mode() {
        let options = LiveOptions::parse(
            [
                "examples/stress-quads/hardware_raster.neo",
                "--mode",
                "draw-stress",
                "--presenter",
                "d3d12-interop",
                "--instance-grid",
                "256x256x128",
                "--draw-policy",
                "compute-culled",
                "--draw-depth",
                "off",
                "--cull-order",
                "atomic-compact",
                "--visibility",
                "projected-size",
                "--min-projected-pixels",
                "1.25",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::DrawStress);
        assert_eq!(options.mode.to_string(), "draw-stress");
        assert_eq!(options.presenter, PresenterKind::D3d12Interop);
        assert_eq!(options.instance_grid.count(), Some(8_388_608));
        assert_eq!(
            options.raster_plan.draw_policy,
            HardwareRasterDrawPolicy::ComputeCulled
        );
        assert_eq!(options.raster_plan.depth, DrawDepthMode::Off);
        assert!(!options.raster_plan.uses_depth());
        assert_eq!(
            options.raster_plan.cull_order,
            HardwareRasterCullOrder::AtomicCompact
        );
        assert_eq!(
            options.raster_plan.visibility,
            HardwareRasterVisibilityMode::ProjectedSize
        );
        assert_eq!(options.raster_plan.min_projected_millipixels, 1250);
        assert_eq!(
            raster_stress_source_path(Path::new("examples/live-window/live.neo")),
            PathBuf::from("examples/stress-quads/hardware_raster.neo")
        );
    }

    #[test]
    fn live_options_accept_legacy_raster_stress_and_draw_policy_aliases() {
        let options = LiveOptions::parse(
            [
                "--mode",
                "raster-stress",
                "--raster-draw-policy",
                "compute-culled",
                "--raster-depth",
                "on",
                "--raster-cull-order",
                "stable-dense",
                "--raster-visibility",
                "projected-size",
                "--raster-min-projected-pixels",
                "0.85",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(options.mode, RunMode::DrawStress);
        assert_eq!(
            options.raster_plan.draw_policy,
            HardwareRasterDrawPolicy::ComputeCulled
        );
        assert_eq!(options.raster_plan.depth, DrawDepthMode::On);
        assert!(options.raster_plan.uses_depth());
        assert_eq!(
            options.raster_plan.cull_order,
            HardwareRasterCullOrder::StableDense
        );
        assert_eq!(
            options.raster_plan.visibility,
            HardwareRasterVisibilityMode::ProjectedSize
        );
        assert_eq!(options.raster_plan.min_projected_millipixels, 850);
    }

    #[test]
    fn live_options_reject_invalid_min_projected_pixels() {
        for value in ["-1", "NaN", "inf"] {
            let err = LiveOptions::parse(
                ["--min-projected-pixels", value]
                    .into_iter()
                    .map(String::from),
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("--min-projected-pixels must be finite and non-negative"));
        }
    }

    #[test]
    fn instance_debug_view_accepts_expected_values() {
        assert_eq!(
            "off".parse::<InstanceDebugView>().unwrap(),
            InstanceDebugView::Off
        );
        assert_eq!(
            "tile-range".parse::<InstanceDebugView>().unwrap(),
            InstanceDebugView::TileRange
        );
        assert_eq!(
            "iterations".parse::<InstanceDebugView>().unwrap(),
            InstanceDebugView::Iterations
        );
        assert_eq!(
            "hit-miss".parse::<InstanceDebugView>().unwrap(),
            InstanceDebugView::HitMiss
        );
        let err = "cost".parse::<InstanceDebugView>().unwrap_err().to_string();
        assert!(err.contains("expected off, tile-range, iterations, or hit-miss"));
    }

    #[test]
    fn instance_layout_accepts_aosoa32_and_aosoa64() {
        assert_eq!(
            "aosoa32".parse::<StressInstanceLayout>().unwrap(),
            StressInstanceLayout::AoSoA32
        );
        assert_eq!(
            "aosoa64".parse::<StressInstanceLayout>().unwrap(),
            StressInstanceLayout::AoSoA64
        );
        assert_eq!(StressInstanceLayout::AoSoA32.label(), "aosoa32");
        assert_eq!(StressInstanceLayout::AoSoA32.to_string(), "aosoa32");
        assert_eq!(StressInstanceLayout::AoSoA64.label(), "aosoa64");
        assert_eq!(StressInstanceLayout::AoSoA64.to_string(), "aosoa64");
        assert_eq!(StressInstanceLayout::AoSoA64.group_size(), 64);
        let err = "soa"
            .parse::<StressInstanceLayout>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected aosoa32 or aosoa64"));
    }

    #[test]
    fn render_policy_accepts_expected_values() {
        assert_eq!("auto".parse::<RenderPolicy>().unwrap(), RenderPolicy::Auto);
        assert_eq!(
            "force-render".parse::<RenderPolicy>().unwrap(),
            RenderPolicy::ForceRender
        );
        assert_eq!(
            "pause-when-empty".parse::<RenderPolicy>().unwrap(),
            RenderPolicy::PauseWhenEmpty
        );
        let err = "maybe".parse::<RenderPolicy>().unwrap_err().to_string();
        assert!(err.contains("expected auto, force-render, or pause-when-empty"));
    }

    #[test]
    fn hardware_cull_order_accepts_expected_values() {
        assert_eq!(
            "atomic-compact".parse::<HardwareRasterCullOrder>().unwrap(),
            HardwareRasterCullOrder::AtomicCompact
        );
        assert_eq!(
            HardwareRasterCullOrder::AtomicCompact.label(),
            "atomic-compact"
        );
        assert_eq!(
            "atomic".parse::<HardwareRasterCullOrder>().unwrap(),
            HardwareRasterCullOrder::AtomicCompact
        );
        assert_eq!(
            "stable-dense".parse::<HardwareRasterCullOrder>().unwrap(),
            HardwareRasterCullOrder::StableDense
        );
        assert_eq!(HardwareRasterCullOrder::StableDense.label(), "stable-dense");
        assert_eq!(
            "stable".parse::<HardwareRasterCullOrder>().unwrap(),
            HardwareRasterCullOrder::StableDense
        );
        let err = "random"
            .parse::<HardwareRasterCullOrder>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected atomic-compact or stable-dense"));
    }

    #[test]
    fn hardware_draw_visibility_accepts_expected_values() {
        assert_eq!(
            "frustum".parse::<HardwareRasterVisibilityMode>().unwrap(),
            HardwareRasterVisibilityMode::Frustum
        );
        assert_eq!(HardwareRasterVisibilityMode::Frustum.label(), "frustum");
        assert_eq!(
            "projected-size"
                .parse::<HardwareRasterVisibilityMode>()
                .unwrap(),
            HardwareRasterVisibilityMode::ProjectedSize
        );
        assert_eq!(
            HardwareRasterVisibilityMode::ProjectedSize.label(),
            "projected-size"
        );
        assert_eq!(
            "pixel-size"
                .parse::<HardwareRasterVisibilityMode>()
                .unwrap(),
            HardwareRasterVisibilityMode::ProjectedSize
        );
        let err = "portal"
            .parse::<HardwareRasterVisibilityMode>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected frustum or projected-size"));
    }

    #[test]
    fn hardware_draw_policy_accepts_expected_values() {
        assert_eq!(
            "draw-all".parse::<HardwareRasterDrawPolicy>().unwrap(),
            HardwareRasterDrawPolicy::DrawAll
        );
        assert_eq!(HardwareRasterDrawPolicy::DrawAll.label(), "draw-all");
        assert_eq!(
            "all".parse::<HardwareRasterDrawPolicy>().unwrap(),
            HardwareRasterDrawPolicy::DrawAll
        );
        assert_eq!(
            "compute-culled"
                .parse::<HardwareRasterDrawPolicy>()
                .unwrap(),
            HardwareRasterDrawPolicy::ComputeCulled
        );
        assert_eq!(
            HardwareRasterDrawPolicy::ComputeCulled.label(),
            "compute-culled"
        );
        assert_eq!(
            "culled".parse::<HardwareRasterDrawPolicy>().unwrap(),
            HardwareRasterDrawPolicy::ComputeCulled
        );
        let err = "indirect"
            .parse::<HardwareRasterDrawPolicy>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected draw-all or compute-culled"));
    }

    #[test]
    fn hardware_raster_visible_instances_log_is_honest_for_policy() {
        let grid = InstanceGrid {
            x: 256,
            y: 256,
            z: 128,
        };
        assert_eq!(
            hardware_raster_visible_instances_for_log(HardwareRasterDrawPolicy::DrawAll, grid),
            Some(8_388_608)
        );
        assert_eq!(
            hardware_raster_visible_instances_for_log(
                HardwareRasterDrawPolicy::ComputeCulled,
                grid
            ),
            None
        );
    }

    #[test]
    fn throughput_log_uses_neutral_draw_policy_markers() {
        let markers = draw_policy_log_markers(DrawPolicyLogFields {
            draw_policy: Some(HardwareRasterDrawPolicy::ComputeCulled),
            draw_depth: Some(DrawDepthMode::Auto),
            uses_depth: Some(true),
            cull_order: Some(HardwareRasterCullOrder::StableDense),
            draw_visibility: Some(HardwareRasterVisibilityMode::ProjectedSize),
            min_projected_millipixels: Some(850),
            visible_instances: None,
            indirect_draws: Some(1),
        });

        assert!(markers.contains("draw_policy compute-culled"));
        assert!(markers.contains("draw_depth auto"));
        assert!(markers.contains("depth on"));
        assert!(markers.contains("cull_order stable-dense"));
        assert!(markers.contains("draw_visibility projected-size"));
        assert!(markers.contains("min_projected_px 0.850"));
        assert!(markers.contains("indirect_draws 1"));
        assert!(!markers.contains("raster_draw_policy"));
        assert!(!markers.contains("raster_cull_order"));
        assert!(!markers.contains("raster_visibility"));
        assert!(!markers.contains("raster_min_projected"));
    }

    #[test]
    fn draw_all_static_stream_helpers_pack_identity_and_indirect_args() {
        let visible = draw_all_identity_visible_bytes(4).unwrap();
        assert_eq!(
            visible,
            [
                0u32.to_le_bytes(),
                1u32.to_le_bytes(),
                2u32.to_le_bytes(),
                3u32.to_le_bytes(),
            ]
            .concat()
        );
        let command = draw_all_indirect_command(12, 4);
        assert_eq!(command.index_count_per_instance, 12);
        assert_eq!(command.instance_count, 4);
        assert_eq!(command.start_index_location, 0);
        assert_eq!(command.base_vertex_location, 0);
        assert_eq!(command.start_instance_location, 0);
    }

    #[test]
    fn instance_stress_variant_accepts_baseline_fast_culled_tiled_and_macrocell() {
        assert_eq!(
            "baseline".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Baseline
        );
        assert_eq!(
            "fast".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Fast
        );
        assert_eq!(
            "culled".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Culled
        );
        assert_eq!(
            "tiled".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Tiled
        );
        assert_eq!(
            "macrocell".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Macrocell
        );
        assert_eq!(InstanceStressVariant::Macrocell.to_string(), "macrocell");
        let err = "turbo"
            .parse::<InstanceStressVariant>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected baseline, fast, culled, tiled, or macrocell"));
    }

    #[test]
    fn instance_stress_variant_resolves_stock_sources() {
        let requested = Path::new("D:/Neo/examples/stress-quads/three_d_instances.neo");
        assert_eq!(
            InstanceStressVariant::Baseline.source_path(requested, StressInstanceLayout::AoSoA32),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_baseline.neo")
        );
        assert_eq!(
            InstanceStressVariant::Fast.source_path(requested, StressInstanceLayout::AoSoA32),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_fast.neo")
        );
        assert_eq!(
            InstanceStressVariant::Culled.source_path(requested, StressInstanceLayout::AoSoA32),
            requested.to_path_buf()
        );
        assert_eq!(
            InstanceStressVariant::Tiled.source_path(requested, StressInstanceLayout::AoSoA32),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_tiled_aosoa32.neo")
        );
        assert_eq!(
            InstanceStressVariant::Tiled.source_path(requested, StressInstanceLayout::AoSoA64),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_tiled_aosoa64.neo")
        );
        assert_eq!(
            InstanceStressVariant::Macrocell.source_path(requested, StressInstanceLayout::AoSoA32),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_macrocell_aosoa32.neo")
        );
        assert_eq!(
            InstanceStressVariant::Macrocell.source_path(requested, StressInstanceLayout::AoSoA64),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_macrocell_aosoa32.neo")
        );
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn macrocell_visibility_grid_packs_stable_records() {
        let grid = InstanceGrid::new(256, 256, 128);
        assert_eq!(macrocell_grid_size(grid).unwrap(), (32, 32, 16));
        assert_eq!(
            visibility_grid_u32_len(grid).unwrap(),
            VISIBILITY_HEADER_U32S + 32 * 32 * 16 * VISIBILITY_RECORD_U32S
        );
        let bytes = create_visibility_grid_bytes(grid).unwrap();
        assert_eq!(read_u32_le(&bytes, 0), VISIBILITY_MAGIC);
        assert_eq!(read_u32_le(&bytes, 4), INSTANCE_MACROCELL_SIZE);
        assert_eq!(read_u32_le(&bytes, 8), 32);
        assert_eq!(read_u32_le(&bytes, 12), 32);
        assert_eq!(read_u32_le(&bytes, 16), 16);
        assert_eq!(read_u32_le(&bytes, 20), 32 * 32 * 16);
        let first = VISIBILITY_HEADER_U32S * 4;
        assert_eq!(
            [
                read_u32_le(&bytes, first),
                read_u32_le(&bytes, first + 4),
                read_u32_le(&bytes, first + 8),
                read_u32_le(&bytes, first + 12),
                read_u32_le(&bytes, first + 16),
                read_u32_le(&bytes, first + 20),
            ],
            [0, 7, 0, 7, 0, 7]
        );
        let last = bytes.len() - VISIBILITY_RECORD_U32S * 4;
        assert_eq!(
            [
                read_u32_le(&bytes, last),
                read_u32_le(&bytes, last + 4),
                read_u32_le(&bytes, last + 8),
                read_u32_le(&bytes, last + 12),
                read_u32_le(&bytes, last + 16),
                read_u32_le(&bytes, last + 20),
            ],
            [248, 255, 248, 255, 120, 127]
        );
    }

    #[test]
    fn macrocell_visibility_grid_rounds_up_and_rejects_invalid_sizes() {
        let grid = InstanceGrid::new(17, 9, 1);
        assert_eq!(macrocell_grid_size(grid).unwrap(), (3, 2, 1));
        let bytes = create_visibility_grid_bytes(grid).unwrap();
        let last = bytes.len() - VISIBILITY_RECORD_U32S * 4;
        assert_eq!(
            [
                read_u32_le(&bytes, last),
                read_u32_le(&bytes, last + 4),
                read_u32_le(&bytes, last + 8),
                read_u32_le(&bytes, last + 12),
                read_u32_le(&bytes, last + 16),
                read_u32_le(&bytes, last + 20),
            ],
            [16, 16, 8, 8, 0, 0]
        );
        assert!(macrocell_grid_size(InstanceGrid::new(0, 1, 1)).is_err());
        assert!(visibility_grid_u32_len(InstanceGrid::new(u32::MAX, u32::MAX, u32::MAX)).is_err());
    }

    #[test]
    fn tile_cull_size_covers_non_multiple_framebuffers() {
        assert_eq!(
            tile_cull_grid_size(PhysicalSize::new(16, 8)).unwrap(),
            (2, 1)
        );
        assert_eq!(
            tile_cull_grid_size(PhysicalSize::new(17, 9)).unwrap(),
            (3, 2)
        );
        assert_eq!(
            tile_cull_byte_len(PhysicalSize::new(17, 9)).unwrap(),
            3 * 2 * TILE_CULL_RECORD_BYTES
        );
    }

    #[test]
    fn tile_cull_size_rejects_zero_and_overflow() {
        let err = tile_cull_byte_len(PhysicalSize::new(0, 8))
            .unwrap_err()
            .to_string();
        assert!(err.contains("nonzero"));
        let err = tile_cull_byte_len(PhysicalSize::new(u32::MAX, u32::MAX))
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"));
    }

    #[test]
    fn instance_grid_rejects_zero_and_malformed_values() {
        let err = "128x0x64".parse::<InstanceGrid>().unwrap_err().to_string();
        assert!(err.contains("greater than zero"));
        let err = "128,128,64"
            .parse::<InstanceGrid>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected XxYxZ"));
    }

    #[test]
    fn camera_defaults_to_auto_orbit() {
        let mut camera = CameraController::new();
        let params = camera.params(PhysicalSize::new(1280, 720), 1.0, DEFAULT_INSTANCE_GRID);
        assert!(camera.auto);
        assert_eq!(params.grid, [256, 256, 128, 8_388_608]);
        assert!(params.view[0] > params.view[1]);
    }

    #[test]
    fn window_visibility_state_prioritizes_minimized_and_occluded() {
        let mut state = WindowVisibilityState::default();
        assert_eq!(state.render_visibility(), RenderVisibility::Visible);
        state.occluded = true;
        assert_eq!(state.render_visibility(), RenderVisibility::Occluded);
        state.minimized = true;
        assert_eq!(state.render_visibility(), RenderVisibility::Minimized);
    }

    #[test]
    fn instance_lattice_visibility_detects_visible_and_empty_views() {
        let mut camera = CameraController::new();
        let visible = camera.params(PhysicalSize::new(1280, 720), 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&visible));

        let mut empty = visible;
        empty.forward = [
            -visible.forward[0],
            -visible.forward[1],
            -visible.forward[2],
            0.0,
        ];
        assert!(!instance_lattice_visible(&empty));
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Empty
        );
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::ForceRender,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Visible
        );
    }

    #[test]
    fn instance_lattice_visibility_stays_stable_at_edges() {
        let mut camera = CameraController::new();
        camera.auto = false;
        let size = PhysicalSize::new(1280, 720);
        camera.yaw = std::f32::consts::FRAC_PI_2 + 0.9;
        let edge_visible = camera.params(size, 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&edge_visible));

        camera.yaw = std::f32::consts::FRAC_PI_2 + 3.1;
        let fully_empty = camera.params(size, 1.01, DEFAULT_INSTANCE_GRID);
        assert!(!instance_lattice_visible(&fully_empty));
    }

    #[test]
    fn empty_idle_ticks_skip_gpu_work_without_live_interop_slots() {
        assert!(should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::Auto,
            0,
            false
        ));
        assert!(should_skip_idle_instance_gpu_work(
            RenderVisibility::Occluded,
            RenderPolicy::PauseWhenEmpty,
            0,
            false
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::Auto,
            0,
            true
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::ForceRender,
            0,
            false
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Visible,
            RenderPolicy::Auto,
            0,
            false
        ));
    }

    #[test]
    fn mouse_look_capture_transitions_are_explicit() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        assert!(camera.right_mouse_down);
        assert!(!camera.auto);
        assert!(camera.last_cursor.is_none());

        camera.end_mouse_look();
        assert!(!camera.right_mouse_down);
        assert!(camera.last_cursor.is_none());
    }

    #[test]
    fn escape_exits_mouse_look_capture() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        let action = camera.handle_key_code(KeyCode::Escape, true);
        assert_eq!(action, CameraWindowAction::ReleaseMouse);
        assert!(!camera.right_mouse_down);
    }

    #[test]
    fn raw_mouse_delta_can_turn_beyond_full_rotation() {
        let mut camera = CameraController::new();
        let start_yaw = camera.yaw;
        camera.handle_raw_mouse_delta(10_000.0, 0.0);
        assert_eq!(camera.yaw, start_yaw);

        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta((std::f32::consts::TAU / 0.003) + 100.0, 0.0);
        assert!(camera.raw_mouse_seen_during_capture);
        assert!(camera.yaw - start_yaw > std::f32::consts::TAU);
    }

    #[test]
    fn raw_mouse_delta_can_turn_both_directions() {
        let mut camera = CameraController::new();
        let start_yaw = camera.yaw;
        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta(250.0, 0.0);
        assert!(camera.yaw > start_yaw);
        camera.handle_raw_mouse_delta(-500.0, 0.0);
        assert!(camera.yaw < start_yaw);
    }

    #[test]
    fn cursor_fallback_recenters_after_bounded_mouse_delta() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        let first = WindowEvent::CursorMoved {
            device_id: winit::event::DeviceId::dummy(),
            position: PhysicalPosition::new(100.0, 100.0),
        };
        assert_eq!(camera.handle_window_event(&first), CameraWindowAction::None);
        let second = WindowEvent::CursorMoved {
            device_id: winit::event::DeviceId::dummy(),
            position: PhysicalPosition::new(90.0, 100.0),
        };
        assert_eq!(
            camera.handle_window_event(&second),
            CameraWindowAction::RecenterMouse
        );
        assert!(camera.last_cursor.is_none());
    }

    #[test]
    fn empty_visibility_keeps_input_alive_until_view_returns() {
        let mut camera = CameraController::new();
        camera.auto = false;
        let size = PhysicalSize::new(1280, 720);
        let visible = camera.params(size, 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&visible));

        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta(std::f32::consts::PI / 0.003, 0.0);
        let empty = camera.params(size, 1.01, DEFAULT_INSTANCE_GRID);
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Empty
        );
        assert_eq!(RenderVisibility::Empty.limit_max_inflight(8), 0);

        camera.handle_raw_mouse_delta(std::f32::consts::PI / 0.003, 0.0);
        let visible_again = camera.params(size, 1.02, DEFAULT_INSTANCE_GRID);
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &visible_again,
            ),
            RenderVisibility::Visible
        );
    }

    #[test]
    fn empty_or_occluded_visibility_prevents_kernel_launches() {
        assert_eq!(RenderVisibility::Empty.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Occluded.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Minimized.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Visible.limit_max_inflight(8), 8);
    }

    #[test]
    fn unfocused_pacing_does_not_consume_main_tokens_until_soft_limiter_is_ready() {
        let start = Instant::now();
        let mut kernel = KernelRateLimiter::new(Some(1000.0), 8, start);
        let mut unfocused = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
        let first = start + Duration::from_millis(1);
        assert_eq!(
            grant_instance_inflight(
                first,
                RenderVisibility::Visible,
                RenderPolicy::Auto,
                false,
                8,
                &mut kernel,
                &mut unfocused,
            ),
            0
        );
        let ready = start + Duration::from_millis(67);
        assert_eq!(
            grant_instance_inflight(
                ready,
                RenderVisibility::Visible,
                RenderPolicy::Auto,
                false,
                8,
                &mut kernel,
                &mut unfocused,
            ),
            1
        );
    }

    #[test]
    fn unfocused_next_tick_waits_for_both_limiters() {
        let start = Instant::now();
        let kernel = KernelRateLimiter::new(Some(1000.0), 8, start);
        let unfocused = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
        let next = next_instance_tick_at(InstanceTickPacing {
            now: start,
            visibility: RenderVisibility::Visible,
            kernel_limiter: &kernel,
            unfocused_limiter: &unfocused,
            render_policy: RenderPolicy::Auto,
            focused: false,
            wait_for_kernel_token: false,
        })
        .unwrap();
        assert!(next.duration_since(start) >= Duration::from_millis(66));
    }

    #[test]
    fn lattice_traversal_math_handles_positive_negative_and_miss() {
        fn range(origin_z: f32, dir_z: f32, spacing: f32, grid_z: u32) -> Option<(i32, i32, i32)> {
            if dir_z.abs() <= 0.00001 {
                return None;
            }
            let half_z = (grid_z as f32 - 1.0) * 0.5;
            let min_z = -half_z * spacing;
            let max_z = half_z * spacing;
            let t0 = (min_z - origin_z) / dir_z;
            let t1 = (max_z - origin_z) / dir_z;
            let enter = t0.min(t1);
            let exit = t0.max(t1);
            if exit < 0.0 {
                return None;
            }
            let begin = enter.max(0.0);
            if begin > exit {
                return None;
            }
            let to_layer =
                |z: f32| ((z / spacing + half_z + 0.5).floor() as i32).clamp(0, grid_z as i32 - 1);
            let start = to_layer(origin_z + dir_z * begin);
            let end = to_layer(origin_z + dir_z * exit);
            let step = if start > end { -1 } else { 1 };
            Some((start, end, step))
        }

        assert_eq!(range(2.0, -1.0, 0.5, 8), Some((7, 0, -1)));
        assert_eq!(range(-2.0, 1.0, 0.5, 8), Some((0, 7, 1)));
        assert_eq!(range(2.0, 1.0, 0.5, 8), None);
    }

    #[test]
    fn manual_camera_movement_is_delta_time_based_and_clamped() {
        let mut camera = CameraController::new();
        camera.auto = false;
        camera.moving_forward = true;
        let start_y = camera.origin[1];
        camera.params(PhysicalSize::new(1280, 720), 10.0, DEFAULT_INSTANCE_GRID);
        assert_eq!(camera.origin[1], start_y);

        camera.params(PhysicalSize::new(1280, 720), 10.016, DEFAULT_INSTANCE_GRID);
        let small_step = camera.origin[1] - start_y;
        assert!(small_step > 0.05);
        assert!(small_step < 0.08);

        camera.params(PhysicalSize::new(1280, 720), 11.0, DEFAULT_INSTANCE_GRID);
        let clamped_step = camera.origin[1] - start_y - small_step;
        assert!(clamped_step > 0.12);
        assert!(clamped_step < 0.14);
    }

    #[test]
    fn live_options_enable_hot_reload_by_default() {
        let options = LiveOptions::parse(std::iter::empty::<String>()).unwrap();
        assert!(options.hot_reload);
        assert_eq!(options.present_ring, 6);

        let options =
            LiveOptions::parse(["--no-hot-reload"].into_iter().map(String::from)).unwrap();
        assert!(!options.hot_reload);

        let options = LiveOptions::parse(
            ["--no-hot-reload", "--hot-reload"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(options.hot_reload);
    }

    #[test]
    fn presenter_kind_accepts_d3d11_and_gdi() {
        assert_eq!(
            "d3d12-interop".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d12Interop
        );
        assert_eq!(
            "d3d12".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d12
        );
        assert_eq!(
            "d3d11".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d11
        );
        assert_eq!("gdi".parse::<PresenterKind>().unwrap(), PresenterKind::Gdi);
        assert_eq!(PresenterKind::D3d12Interop.to_string(), "d3d12-interop");
        assert_eq!(PresenterKind::D3d12.to_string(), "d3d12-flip-upload");
        assert_eq!(PresenterKind::D3d11.to_string(), "d3d11-flip-host");
        assert_eq!(PresenterKind::Gdi.to_string(), "win32-gdi");
        assert!(!PresenterKind::D3d12Interop.uses_present_thread());
        assert!(!PresenterKind::D3d12.uses_present_thread());
        assert!(PresenterKind::D3d11.uses_present_thread());
        assert!(PresenterKind::Gdi.uses_present_thread());
    }

    #[test]
    fn interop_fallback_accepts_no_interop_and_fail() {
        assert_eq!(
            "no-interop".parse::<InteropFallback>().unwrap(),
            InteropFallback::NoInterop
        );
        assert_eq!(
            "fail".parse::<InteropFallback>().unwrap(),
            InteropFallback::Fail
        );
        let err = "maybe".parse::<InteropFallback>().unwrap_err().to_string();
        assert!(err.contains("expected no-interop or fail"));
    }

    #[test]
    fn interop_stats_do_not_count_host_transfer_bytes() {
        let mut counter = ThroughputCounter::new();
        counter.record(ThroughputBatchStats {
            completed_kernels: 3,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 0,
            uploaded_bytes: 0,
            launch: Duration::from_micros(30),
            completion_wait: Duration::ZERO,
            sample_download: Duration::ZERO,
            present: Duration::from_micros(50),
            map_copy: Duration::ZERO,
            draw: Duration::from_micros(20),
            swap_present: Duration::from_micros(30),
        });
        assert_eq!(counter.sampled_bytes_since_log, 0);
        assert_eq!(counter.uploaded_bytes_since_log, 0);
        assert_eq!(counter.total_presented, 1);
    }

    #[test]
    fn interop_high_rate_presenter_does_not_sleep_on_kernel_tokens() {
        assert!(!should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(1000.0)
        ));
        assert!(!should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(240.0)
        ));
        assert!(should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(60.0)
        ));
        assert!(should_wait_for_kernel_token(
            PresenterKind::D3d12,
            Some(1000.0)
        ));
    }

    #[test]
    fn present_rate_limiter_caps_average_rate_without_sleeping() {
        let start = Instant::now();
        let mut limiter = PresentRateLimiter::new(Some(1000.0), start);
        let mut presented = 0;
        for tick in 0..10_000 {
            let now = start + Duration::from_micros(tick * 100);
            if limiter.try_consume(now) {
                presented += 1;
            }
        }
        assert!((999..=1000).contains(&presented));
    }

    #[test]
    fn d3d_upload_mode_accepts_mapped_copy_and_update_subresource() {
        assert_eq!(
            "mapped-copy".parse::<D3dUploadMode>().unwrap(),
            D3dUploadMode::MappedCopy
        );
        assert_eq!(
            "update-subresource".parse::<D3dUploadMode>().unwrap(),
            D3dUploadMode::UpdateSubresource
        );
        let err = "wat".parse::<D3dUploadMode>().unwrap_err().to_string();
        assert!(err.contains("expected mapped-copy or update-subresource"));
    }

    #[test]
    fn frame_byte_len_validates_overflow() {
        assert_eq!(frame_byte_len(2, 3).unwrap(), 24);
        assert!(frame_byte_len(u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn live_options_accept_zero_kernel_target_as_uncapped() {
        let options =
            LiveOptions::parse(["--kernel-target-fps", "0"].into_iter().map(String::from)).unwrap();
        assert_eq!(options.kernel_target_fps, Some(0.0));
        assert_eq!(options.kernel_cap(), None);
    }

    #[test]
    fn live_options_reject_zero_throughput_values() {
        let err = LiveOptions::parse(["--sample-every", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sample-every must be greater than zero"));

        let err = LiveOptions::parse(["--max-inflight", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--max-inflight must be greater than zero"));

        let err = LiveOptions::parse(["--present-target-fps", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--present-target-fps must be greater than zero"));

        let err = LiveOptions::parse(["--present-ring", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--present-ring must be greater than zero"));
    }

    #[test]
    fn live_options_reject_invalid_kernel_target_fps() {
        for value in ["-1", "NaN", "inf"] {
            let err =
                LiveOptions::parse(["--kernel-target-fps", value].into_iter().map(String::from))
                    .unwrap_err()
                    .to_string();
            assert!(err.contains("--kernel-target-fps must be finite and non-negative"));
        }
    }

    #[test]
    fn kernel_rate_limiter_grants_target_rate_over_time() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(240.0), 8, start);
        let mut granted = 0;
        for tick in 1..=100 {
            granted += limiter.grant(start + Duration::from_millis(tick * 10), 8);
        }
        assert!((239..=241).contains(&granted));
    }

    #[test]
    fn kernel_rate_limiter_clamps_burst() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(10.0), 4, start);
        let granted = limiter.grant(start + Duration::from_secs(10), 100);
        assert_eq!(granted, 16);
    }

    #[test]
    fn kernel_rate_limiter_returns_zero_without_tokens() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(240.0), 8, start);
        assert_eq!(limiter.grant(start, 8), 0);
        assert!(
            limiter
                .next_token_at(start)
                .is_some_and(|wake| wake > start)
        );
    }

    #[test]
    fn presentation_ring_chooses_newest_completed_slot() {
        let newest = choose_newest_completed_slot([(0, 10), (1, 12), (2, 11)]);
        assert_eq!(newest, Some(1));
        assert_eq!(choose_newest_completed_slot([]), None);
    }

    #[test]
    fn bgra_copy_reports_fast_path_only_for_matching_pitch() {
        let src: Vec<u8> = (0..16).collect();
        let mut tight = vec![0u8; 16];
        let used_fast_path = unsafe { copy_bgra_to_mapped(&src, tight.as_mut_ptr(), 2, 2, 8) };
        assert!(used_fast_path);
        assert_eq!(tight, src);

        let mut padded = vec![0u8; 24];
        let used_fast_path = unsafe { copy_bgra_to_mapped(&src, padded.as_mut_ptr(), 2, 2, 12) };
        assert!(!used_fast_path);
        assert_eq!(&padded[0..8], &src[0..8]);
        assert_eq!(&padded[12..20], &src[8..16]);
    }

    #[test]
    fn throughput_counter_tracks_completed_separately_from_presented() {
        let mut counter = ThroughputCounter::new();
        counter.record(ThroughputBatchStats {
            completed_kernels: 256,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 1024,
            uploaded_bytes: 1024,
            launch: Duration::from_micros(256),
            completion_wait: Duration::from_micros(512),
            sample_download: Duration::from_micros(200),
            present: Duration::from_micros(100),
            map_copy: Duration::from_micros(40),
            draw: Duration::from_micros(30),
            swap_present: Duration::from_micros(30),
        });
        assert_eq!(counter.total_completed, 256);
        assert_eq!(counter.total_sampled, 1);
        assert_eq!(counter.total_presented, 1);
        assert_ne!(counter.total_completed, counter.total_presented);
    }

    #[test]
    fn throughput_batch_stats_accumulate_threaded_present_results() {
        let mut stats = ThroughputBatchStats {
            completed_kernels: 4,
            sampled_frames: 0,
            presented_frames: 0,
            sampled_bytes: 0,
            uploaded_bytes: 0,
            launch: Duration::from_micros(4),
            completion_wait: Duration::ZERO,
            sample_download: Duration::ZERO,
            present: Duration::ZERO,
            map_copy: Duration::ZERO,
            draw: Duration::ZERO,
            swap_present: Duration::ZERO,
        };
        stats += ThroughputBatchStats {
            completed_kernels: 0,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 16,
            uploaded_bytes: 16,
            launch: Duration::ZERO,
            completion_wait: Duration::ZERO,
            sample_download: Duration::from_micros(10),
            present: Duration::from_micros(20),
            map_copy: Duration::from_micros(6),
            draw: Duration::from_micros(7),
            swap_present: Duration::from_micros(8),
        };
        assert_eq!(stats.completed_kernels, 4);
        assert_eq!(stats.sampled_frames, 1);
        assert_eq!(stats.presented_frames, 1);
        assert_eq!(stats.sampled_bytes, 16);
        assert_eq!(stats.uploaded_bytes, 16);
        assert_eq!(stats.sample_download, Duration::from_micros(10));
        assert_eq!(stats.present, Duration::from_micros(20));
    }
}
