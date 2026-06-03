use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};
use neo_runtime::{
    Context, IndexFormat, MeshBuffer, MeshBufferDesc, PrimitiveTopology, VertexAttribute,
    VertexFormat, VertexLayout, VertexSemantic,
};
#[cfg(windows)]
use neo_runtime::{
    CudaDraw as RuntimeCudaDraw, CullOrder as RuntimeCullOrder, DataLayout as RuntimeDataLayout,
    DrawBackend as RuntimeDrawBackend, DrawContract as RuntimeDrawContract,
    DrawDepthMode as RuntimeDrawDepthMode, DrawExecution as RuntimeDrawExecution,
    DrawPolicy as RuntimeDrawPolicy, DrawPolicyConfig as RuntimeDrawPolicyConfig,
    DrawRecipe as RuntimeDrawRecipe, GeometryStream as RuntimeGeometryStream,
    InstanceAttribute as RuntimeInstanceAttribute, InstanceBuffer, InstanceBufferDesc,
    InstanceFormat as RuntimeInstanceFormat, InstanceLayout as RuntimeInstanceLayout,
    InstanceSemantic as RuntimeInstanceSemantic, InstanceStream as RuntimeInstanceStream,
    MaterialKernel as RuntimeMaterialKernel, MaterialKernelAbi as RuntimeMaterialKernelAbi,
    Target as RuntimeTarget, VisibilityMode as RuntimeVisibilityMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presenter {
    D3d12Interop,
    D3d12Upload,
    D3d11Upload,
    Gdi,
}

impl Presenter {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::D3d12Interop => "d3d12-interop",
            Self::D3d12Upload => "d3d12",
            Self::D3d11Upload => "d3d11",
            Self::Gdi => "gdi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteropFallback {
    NoInterop,
    Fail,
}

impl InteropFallback {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::NoInterop => "no-interop",
            Self::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPolicy {
    Auto,
    ForceRender,
    PauseWhenEmpty,
}

impl RenderPolicy {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceRender => "force-render",
            Self::PauseWhenEmpty => "pause-when-empty",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FpsCap {
    pub kernel: Option<f32>,
    pub present: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSpec {
    pub entrypoint: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryStreamSpec {
    pub mesh: String,
    pub vertex_bytes: Vec<u8>,
    pub vertex_stride: u32,
    pub color_offset: u32,
    pub indices_u16: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceStreamSpec {
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceLayout {
    AoSoA32,
    AoSoA64,
}

impl InstanceLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::AoSoA32 => "aosoa32",
            Self::AoSoA64 => "aosoa64",
        }
    }

    fn live_window_name(self) -> &'static str {
        self.label()
    }
}

impl fmt::Display for InstanceLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceGrid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl InstanceGrid {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn count(self) -> Option<u32> {
        self.x.checked_mul(self.y)?.checked_mul(self.z)
    }

    fn validate(self) -> Result<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            bail!("instance stream grid dimensions must be greater than zero");
        }
        if self.count().is_none() {
            bail!("instance stream grid instance count overflowed u32");
        }
        Ok(())
    }

    fn live_window_name(self) -> String {
        format!("{}x{}x{}", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceStreamConfig {
    pub name: String,
    pub grid: InstanceGrid,
    pub layout: InstanceLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTextureConfig {
    pub name: String,
    pub virtual_width: u32,
    pub virtual_height: u32,
    pub page_size: u32,
    pub physical_pages: u32,
    pub checker_pages: bool,
    pub feedback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialStreamConfig {
    pub name: String,
    pub ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTextureBuilder {
    virtual_width: u32,
    virtual_height: u32,
    page_size: u32,
    physical_pages: u32,
    checker_pages: bool,
    feedback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialStreamBuilder {
    ids: Vec<u32>,
}

impl Default for SparseTextureBuilder {
    fn default() -> Self {
        Self {
            virtual_width: 2048,
            virtual_height: 2048,
            page_size: 128,
            physical_pages: 256,
            checker_pages: false,
            feedback: false,
        }
    }
}

impl SparseTextureBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn virtual_size(mut self, width: u32, height: u32) -> Self {
        self.virtual_width = width;
        self.virtual_height = height;
        self
    }

    pub fn page_size(mut self, page_size: u32) -> Self {
        self.page_size = page_size;
        self
    }

    pub fn checker_pages(mut self) -> Self {
        self.checker_pages = true;
        self
    }

    pub fn feedback(mut self, enabled: bool) -> Self {
        self.feedback = enabled;
        self
    }

    pub fn page_rgba(self, _page: u32, _rgba: Vec<u8>) -> Self {
        self
    }

    fn config(self, name: String) -> SparseTextureConfig {
        SparseTextureConfig {
            name,
            virtual_width: self.virtual_width,
            virtual_height: self.virtual_height,
            page_size: self.page_size,
            physical_pages: self.physical_pages,
            checker_pages: self.checker_pages,
            feedback: self.feedback,
        }
    }
}

impl MaterialStreamBuilder {
    pub fn per_instance_ids(ids: impl Into<Vec<u32>>) -> Self {
        Self { ids: ids.into() }
    }

    pub fn procedural_grid_tiles(grid: InstanceGrid) -> Self {
        let mut ids = Vec::with_capacity(grid.count().unwrap_or(0) as usize);
        for z in 0..grid.z {
            for y in 0..grid.y {
                for x in 0..grid.x {
                    ids.push(((y / 8) * 16 + (x / 8)) ^ (z / 4));
                }
            }
        }
        Self { ids }
    }

    fn config(self, name: String) -> MaterialStreamConfig {
        MaterialStreamConfig {
            name,
            ids: self.ids,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialSpec {
    pub name: String,
    pub vertex_entrypoint: String,
    pub fragment_entrypoint: String,
    pub abi: MaterialAbi,
    pub source: Option<PathBuf>,
}

impl MaterialSpec {
    pub fn label(&self) -> &str {
        &self.name
    }

    pub fn abi_label(&self) -> &'static str {
        self.abi.label()
    }

    pub fn vertex_entrypoint(&self) -> Option<&str> {
        match self.abi {
            MaterialAbi::CudaTiledInstanceColor => None,
            _ => Some(&self.vertex_entrypoint),
        }
    }

    pub fn fragment_entrypoint(&self) -> Option<&str> {
        match self.abi {
            MaterialAbi::CudaTiledInstanceColor => None,
            _ => Some(&self.fragment_entrypoint),
        }
    }

    pub fn kernel_entrypoint(&self) -> Option<&str> {
        match self.abi {
            MaterialAbi::CudaTiledInstanceColor => Some(&self.vertex_entrypoint),
            _ => None,
        }
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.source.as_deref()
    }

    pub fn is_cuda_tiled(&self) -> bool {
        self.abi.requires_cuda_tiled()
    }

    pub fn execution_kind(&self) -> MaterialExecutionKind {
        self.abi.execution_kind()
    }

    pub fn execution_kind_label(&self) -> &'static str {
        self.execution_kind().label()
    }

    pub fn is_draw_execution(&self) -> bool {
        self.execution_kind() == MaterialExecutionKind::DrawExecution
    }

    pub fn backend(&self) -> DrawBackend {
        self.abi.backend()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterialAbi {
    SimpleColor,
    DirectInstanceColor,
    ComputeCulledInstanceColor,
    CudaTiledInstanceColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialExecutionKind {
    DrawExecution,
    CudaTiled,
}

impl MaterialExecutionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::DrawExecution => "draw-execution",
            Self::CudaTiled => "cuda-tiled",
        }
    }

    pub fn backend(self) -> DrawBackend {
        match self {
            Self::DrawExecution => DrawBackend::HardwareRaster,
            Self::CudaTiled => DrawBackend::CudaTiled,
        }
    }
}

impl fmt::Display for MaterialExecutionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl MaterialAbi {
    pub fn label(&self) -> &'static str {
        match self {
            Self::SimpleColor => "simple-color",
            Self::DirectInstanceColor => "direct-instance-color",
            Self::ComputeCulledInstanceColor => "compute-culled-instance-color",
            Self::CudaTiledInstanceColor => "cuda-tiled-instance-color",
        }
    }

    pub fn backend(&self) -> DrawBackend {
        self.execution_kind().backend()
    }

    pub fn execution_kind(&self) -> MaterialExecutionKind {
        match self {
            Self::SimpleColor | Self::DirectInstanceColor | Self::ComputeCulledInstanceColor => {
                MaterialExecutionKind::DrawExecution
            }
            Self::CudaTiledInstanceColor => MaterialExecutionKind::CudaTiled,
        }
    }

    pub fn execution_kind_label(&self) -> &'static str {
        self.execution_kind().label()
    }

    fn requires_instances(&self) -> bool {
        matches!(
            self,
            Self::DirectInstanceColor
                | Self::ComputeCulledInstanceColor
                | Self::CudaTiledInstanceColor
        )
    }

    fn requires_compute_culling(&self) -> bool {
        matches!(self, Self::ComputeCulledInstanceColor)
    }

    fn requires_cuda_tiled(&self) -> bool {
        matches!(self, Self::CudaTiledInstanceColor)
    }
}

impl fmt::Display for MaterialAbi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeCullSpec {
    pub entrypoint: String,
    pub path: PathBuf,
    pub order: CullOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawPolicy {
    DrawAll,
    ComputeCulled,
    CudaTiled,
}

impl DrawPolicy {
    pub fn backend(self) -> DrawBackend {
        match self {
            Self::DrawAll | Self::ComputeCulled => DrawBackend::HardwareRaster,
            Self::CudaTiled => DrawBackend::CudaTiled,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DrawAll => "draw-all",
            Self::ComputeCulled => "compute-culled",
            Self::CudaTiled => "cuda-tiled",
        }
    }
}

impl fmt::Display for DrawPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullOrder {
    AtomicCompact,
    StableDense,
}

pub type RasterCullOrder = CullOrder;

impl CullOrder {
    fn live_window_order(self) -> neo_live_window::DrawCullOrder {
        match self {
            Self::AtomicCompact => neo_live_window::DrawCullOrder::AtomicCompact,
            Self::StableDense => neo_live_window::DrawCullOrder::StableDense,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AtomicCompact => "atomic-compact",
            Self::StableDense => "stable-dense",
        }
    }

    fn live_window_name(self) -> &'static str {
        self.label()
    }
}

impl fmt::Display for CullOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityMode {
    Frustum,
    ProjectedSize,
}

pub type RasterVisibilityMode = VisibilityMode;

pub const DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS: u32 = 850;
pub const DEFAULT_MIN_PROJECTED_MILLIPIXELS: u32 = DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS;

impl VisibilityMode {
    fn live_window_mode(self) -> neo_live_window::DrawVisibilityMode {
        match self {
            Self::Frustum => neo_live_window::DrawVisibilityMode::Frustum,
            Self::ProjectedSize => neo_live_window::DrawVisibilityMode::ProjectedSize,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Frustum => "frustum",
            Self::ProjectedSize => "projected-size",
        }
    }

    fn live_window_name(self) -> &'static str {
        self.label()
    }
}

impl fmt::Display for VisibilityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    pub fn uses_depth(self, policy: DrawPolicy) -> bool {
        match self {
            Self::Auto => policy != DrawPolicy::DrawAll,
            Self::On => true,
            Self::Off => false,
        }
    }

    fn live_window_mode(self) -> neo_live_window::DrawDepthMode {
        match self {
            Self::Auto => neo_live_window::DrawDepthMode::Auto,
            Self::On => neo_live_window::DrawDepthMode::On,
            Self::Off => neo_live_window::DrawDepthMode::Off,
        }
    }

    #[cfg(windows)]
    fn runtime_mode(self) -> RuntimeDrawDepthMode {
        match self {
            Self::Auto => RuntimeDrawDepthMode::Auto,
            Self::On => RuntimeDrawDepthMode::On,
            Self::Off => RuntimeDrawDepthMode::Off,
        }
    }
}

impl fmt::Display for DrawDepthMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawPolicyConfig {
    pub policy: DrawPolicy,
    pub depth: DrawDepthMode,
    pub cull_order: CullOrder,
    pub visibility: VisibilityMode,
    pub min_projected_millipixels: u32,
}

impl DrawPolicyConfig {
    pub fn draw_all() -> Self {
        Self {
            policy: DrawPolicy::DrawAll,
            depth: DrawDepthMode::Auto,
            cull_order: CullOrder::StableDense,
            visibility: VisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn compute_culled(cull_order: CullOrder) -> Self {
        Self {
            policy: DrawPolicy::ComputeCulled,
            depth: DrawDepthMode::Auto,
            cull_order,
            visibility: VisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn compute_culled_with_visibility(
        cull_order: CullOrder,
        visibility: VisibilityMode,
    ) -> Self {
        Self {
            policy: DrawPolicy::ComputeCulled,
            depth: DrawDepthMode::Auto,
            cull_order,
            visibility,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn cuda_tiled() -> Self {
        Self {
            policy: DrawPolicy::CudaTiled,
            depth: DrawDepthMode::Auto,
            cull_order: CullOrder::StableDense,
            visibility: VisibilityMode::ProjectedSize,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn with_min_projected_millipixels(mut self, min_projected_millipixels: u32) -> Self {
        self.min_projected_millipixels = min_projected_millipixels;
        self
    }

    pub fn with_depth(mut self, depth: DrawDepthMode) -> Self {
        self.depth = depth;
        self
    }

    pub fn backend(self) -> DrawBackend {
        self.policy.backend()
    }

    pub fn policy_label(self) -> &'static str {
        self.policy.label()
    }

    pub fn backend_label(self) -> &'static str {
        self.backend().label()
    }

    pub fn cull_order_label(self) -> &'static str {
        self.cull_order.label()
    }

    pub fn depth_label(self) -> &'static str {
        self.depth.label()
    }

    pub fn uses_depth(self) -> bool {
        self.depth.uses_depth(self.policy)
    }

    pub fn visibility_label(self) -> &'static str {
        self.visibility.label()
    }

    pub fn min_projected_pixels(self) -> f32 {
        self.min_projected_millipixels as f32 / 1000.0
    }
}

impl From<DrawPolicy> for DrawPolicyConfig {
    fn from(policy: DrawPolicy) -> Self {
        match policy {
            DrawPolicy::DrawAll => Self::draw_all(),
            DrawPolicy::ComputeCulled => Self::compute_culled(CullOrder::AtomicCompact),
            DrawPolicy::CudaTiled => Self::cuda_tiled(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetSpec {
    Window,
    Named(String),
}

impl TargetSpec {
    pub fn window() -> Self {
        Self::Window
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetConfig {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawSpec {
    pub name: String,
    pub geometry: GeometryStreamSpec,
    pub instances: Option<InstanceStreamSpec>,
    pub material: String,
    pub target: TargetSpec,
    pub policy: DrawPolicy,
    pub policy_config: DrawPolicyConfig,
}

impl DrawSpec {
    pub fn draw_name(&self) -> &str {
        &self.name
    }

    pub fn geometry_stream(&self) -> &GeometryStreamSpec {
        &self.geometry
    }

    pub fn instance_stream(&self) -> Option<&InstanceStreamSpec> {
        self.instances.as_ref()
    }

    pub fn material_kernel(&self) -> &str {
        &self.material
    }

    pub fn target(&self) -> &TargetSpec {
        &self.target
    }

    pub fn draw_policy(&self) -> DrawPolicy {
        self.policy_config.policy
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        self.policy_config
    }

    pub fn backend(&self) -> DrawBackend {
        self.policy_config().backend()
    }

    pub fn contract(&self) -> DrawContract {
        DrawContract {
            draw: self.name.clone(),
            geometry_stream: self.geometry.mesh.clone(),
            instance_stream: self
                .instances
                .as_ref()
                .map(|instances| instances.name.clone()),
            instance_count: None,
            instance_layout: None,
            material_kernel: self.material.clone(),
            material_abi_label: None,
            target: match &self.target {
                TargetSpec::Window => "window".to_string(),
                TargetSpec::Named(name) => name.clone(),
            },
            target_width: None,
            target_height: None,
            policy: self.policy_config.policy,
            policy_config: self.policy_config,
            backend: self.backend(),
        }
    }
}

pub type DrawIndirectSpec = DrawSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawGraph {
    pub draws: Vec<DrawGraphDraw>,
}

impl DrawGraph {
    pub fn draw(&self, name: &str) -> Result<&DrawGraphDraw> {
        self.draws
            .iter()
            .find(|draw| draw.name == name)
            .ok_or_else(|| anyhow::anyhow!("draw `{name}` was not found in the draw graph"))
    }

    pub fn draws(&self) -> &[DrawGraphDraw] {
        &self.draws
    }

    pub fn contracts(&self) -> impl Iterator<Item = DrawContract> + '_ {
        self.draws.iter().map(DrawGraphDraw::contract)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawGraphDraw {
    pub name: String,
    pub geometry: GeometryStreamSpec,
    pub instances: Option<InstanceStreamConfig>,
    pub material: MaterialSpec,
    pub target: TargetBinding,
    pub policy: DrawPolicy,
    pub depth: DrawDepthMode,
    pub cull_order: CullOrder,
    pub visibility: VisibilityMode,
    pub min_projected_millipixels: u32,
    pub compute_cull: Option<ComputeCullSpec>,
}

impl DrawGraphDraw {
    pub fn draw_name(&self) -> &str {
        &self.name
    }

    pub fn geometry_stream(&self) -> &GeometryStreamSpec {
        &self.geometry
    }

    pub fn instance_stream(&self) -> Option<&InstanceStreamConfig> {
        self.instances.as_ref()
    }

    pub fn material_kernel(&self) -> &MaterialSpec {
        &self.material
    }

    pub fn target(&self) -> &TargetBinding {
        &self.target
    }

    pub fn draw_policy(&self) -> DrawPolicy {
        self.policy
    }

    pub fn compute_cull(&self) -> Option<&ComputeCullSpec> {
        self.compute_cull.as_ref()
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        DrawPolicyConfig {
            policy: self.policy,
            depth: self.depth,
            cull_order: self.cull_order,
            visibility: self.visibility,
            min_projected_millipixels: self.min_projected_millipixels,
        }
    }

    pub fn backend(&self) -> DrawBackend {
        self.policy_config().backend()
    }

    pub fn contract(&self) -> DrawContract {
        DrawContract {
            draw: self.name.clone(),
            geometry_stream: self.geometry.mesh.clone(),
            instance_stream: self
                .instances
                .as_ref()
                .map(|instances| instances.name.clone()),
            instance_count: self
                .instances
                .as_ref()
                .and_then(|instances| instances.grid.count()),
            instance_layout: self.instances.as_ref().map(|instances| instances.layout),
            material_kernel: self.material.name.clone(),
            material_abi_label: Some(self.material.abi_label().to_string()),
            target: self.target.name().to_string(),
            target_width: Some(self.target.dimensions().0),
            target_height: Some(self.target.dimensions().1),
            policy: self.policy,
            policy_config: self.policy_config(),
            backend: self.backend(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawContract {
    pub draw: String,
    pub geometry_stream: String,
    pub instance_stream: Option<String>,
    pub instance_count: Option<u32>,
    pub instance_layout: Option<InstanceLayout>,
    pub material_kernel: String,
    pub material_abi_label: Option<String>,
    pub target: String,
    pub target_width: Option<u32>,
    pub target_height: Option<u32>,
    pub policy: DrawPolicy,
    pub policy_config: DrawPolicyConfig,
    pub backend: DrawBackend,
}

impl DrawContract {
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

    pub fn instance_layout_label(&self) -> Option<&'static str> {
        self.instance_layout.map(InstanceLayout::label)
    }

    pub fn material_label(&self) -> &str {
        &self.material_kernel
    }

    pub fn material_abi_label(&self) -> Option<&str> {
        self.material_abi_label.as_deref()
    }

    pub fn target_dimensions(&self) -> Option<(u32, u32)> {
        Some((self.target_width?, self.target_height?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetBinding {
    Window(WindowConfig),
    Named(TargetConfig),
}

impl TargetBinding {
    pub fn name(&self) -> &str {
        match self {
            Self::Window(_) => "window",
            Self::Named(target) => &target.name,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Window(window) => (window.width, window.height),
            Self::Named(target) => (target.width, target.height),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawExecutionRunPlan {
    pub draw_name: String,
    pub source: PathBuf,
    pub geometry: GeometryStreamSpec,
    pub instances: InstanceStreamConfig,
    pub material: MaterialSpec,
    pub target: TargetBinding,
    pub policy: DrawPolicy,
    pub policy_config: DrawPolicyConfig,
    pub cull_order: CullOrder,
    pub visibility: VisibilityMode,
    pub min_projected_millipixels: u32,
    pub compute_cull: Option<ComputeCullSpec>,
}

pub type HardwareRasterRunPlan = DrawExecutionRunPlan;
pub type RasterRunPlan = DrawExecutionRunPlan;

impl DrawExecutionRunPlan {
    pub fn live_draw_plan(&self) -> neo_live_window::DrawExecutionPlan {
        neo_live_window::DrawExecutionPlan {
            draw_name: self.draw_name.clone(),
            geometry_stream: neo_live_window::GeometryStreamPlan::indexed_u16(
                self.geometry.mesh.clone(),
                self.geometry.vertex_bytes.clone(),
                self.geometry.vertex_stride,
                self.geometry.color_offset,
                self.geometry.indices_u16.clone(),
            ),
            instance_stream: neo_live_window::InstanceStreamPlan {
                name: self.instances.name.clone(),
                grid: neo_live_window::InstanceGrid::new(
                    self.instances.grid.x,
                    self.instances.grid.y,
                    self.instances.grid.z,
                ),
                layout: match self.instances.layout {
                    InstanceLayout::AoSoA32 => neo_live_window::StressInstanceLayout::AoSoA32,
                    InstanceLayout::AoSoA64 => neo_live_window::StressInstanceLayout::AoSoA64,
                },
            },
            target: match &self.target {
                TargetBinding::Window(window) => neo_live_window::TargetPlan {
                    name: "window".to_string(),
                    width: window.width,
                    height: window.height,
                },
                TargetBinding::Named(target) => neo_live_window::TargetPlan {
                    name: target.name.clone(),
                    width: target.width,
                    height: target.height,
                },
            },
            material: match self.material.abi {
                MaterialAbi::DirectInstanceColor => {
                    neo_live_window::MaterialKernelPlan::direct_instance_color(
                        self.material.name.clone(),
                        self.material.vertex_entrypoint.clone(),
                        self.material.fragment_entrypoint.clone(),
                    )
                }
                MaterialAbi::ComputeCulledInstanceColor => {
                    neo_live_window::MaterialKernelPlan::compute_culled_instance_color(
                        self.material.name.clone(),
                        self.material.vertex_entrypoint.clone(),
                        self.material.fragment_entrypoint.clone(),
                    )
                }
                MaterialAbi::SimpleColor => {
                    neo_live_window::MaterialKernelPlan::direct_instance_color(
                        self.material.name.clone(),
                        self.material.vertex_entrypoint.clone(),
                        self.material.fragment_entrypoint.clone(),
                    )
                }
                MaterialAbi::CudaTiledInstanceColor => {
                    unreachable!("CUDA tiled MaterialKernel does not produce DrawExecutionPlan")
                }
            },
            draw_policy: match self.policy {
                DrawPolicy::DrawAll => neo_live_window::DrawPolicyPlan::DrawAll,
                DrawPolicy::ComputeCulled => neo_live_window::DrawPolicyPlan::ComputeCulled,
                DrawPolicy::CudaTiled => neo_live_window::DrawPolicyPlan::ComputeCulled,
            },
            depth: self.policy_config.depth.live_window_mode(),
            cull_order: self.cull_order.live_window_order(),
            visibility: self.visibility.live_window_mode(),
            min_projected_millipixels: self.min_projected_millipixels,
        }
    }

    pub fn live_window_plan(&self) -> neo_live_window::DrawExecutionPlan {
        self.live_draw_plan()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaRunPlan {
    pub draw_name: String,
    pub source: PathBuf,
    pub entrypoint: String,
    pub geometry: GeometryStreamSpec,
    pub instances: InstanceStreamConfig,
    pub material: MaterialSpec,
    pub target: TargetBinding,
    pub policy_config: DrawPolicyConfig,
    pub variant: InstanceStressVariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawRunPlan {
    Cuda(CudaRunPlan),
    DrawExecution(DrawExecutionRunPlan),
    HardwareRaster(HardwareRasterRunPlan),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawBackend {
    CudaTiled,
    HardwareRaster,
}

impl DrawBackend {
    pub fn primary_neo() -> Self {
        Self::CudaTiled
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::CudaTiled => "cuda-tiled",
            Self::HardwareRaster => "hardware-raster",
        }
    }

    pub fn is_primary_neo(self) -> bool {
        self == Self::primary_neo()
    }
}

impl fmt::Display for DrawBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawBackendPreference {
    PrimaryNeo,
    CudaTiled,
    DrawExecution,
    HardwareRaster,
    FirstConfigured,
}

pub type RendererPreference = DrawBackendPreference;

impl DrawBackendPreference {
    pub fn label(self) -> &'static str {
        match self {
            Self::PrimaryNeo => "primary-neo",
            Self::CudaTiled => "cuda-tiled",
            Self::DrawExecution => "draw-execution",
            Self::HardwareRaster => "hardware-raster",
            Self::FirstConfigured => "first-configured",
        }
    }

    pub fn preferred_backend(self) -> Option<DrawBackend> {
        match self {
            Self::PrimaryNeo | Self::CudaTiled => Some(DrawBackend::CudaTiled),
            Self::DrawExecution | Self::HardwareRaster => Some(DrawBackend::HardwareRaster),
            Self::FirstConfigured => None,
        }
    }
}

impl Default for DrawBackendPreference {
    fn default() -> Self {
        Self::PrimaryNeo
    }
}

impl fmt::Display for DrawBackendPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

pub trait DrawPlanRecipe {
    fn backend(&self) -> DrawBackend;
    fn draw_name(&self) -> &str;
    fn source(&self) -> &Path;
    fn geometry(&self) -> &GeometryStreamSpec;
    fn instances(&self) -> &InstanceStreamConfig;
    fn material(&self) -> &MaterialSpec;
    fn target(&self) -> &TargetBinding;
    fn policy_config(&self) -> DrawPolicyConfig;

    fn policy(&self) -> DrawPolicy {
        self.policy_config().policy
    }

    fn contract(&self) -> DrawContract {
        DrawContract {
            draw: self.draw_name().to_string(),
            geometry_stream: self.geometry().mesh.clone(),
            instance_stream: Some(self.instances().name.clone()),
            instance_count: self.instances().grid.count(),
            instance_layout: Some(self.instances().layout),
            material_kernel: self.material().name.clone(),
            material_abi_label: Some(self.material().abi_label().to_string()),
            target: self.target().name().to_string(),
            target_width: Some(self.target().dimensions().0),
            target_height: Some(self.target().dimensions().1),
            policy: self.policy(),
            policy_config: self.policy_config(),
            backend: self.backend(),
        }
    }
}

impl DrawPlanRecipe for CudaRunPlan {
    fn backend(&self) -> DrawBackend {
        DrawBackend::CudaTiled
    }

    fn draw_name(&self) -> &str {
        &self.draw_name
    }

    fn source(&self) -> &Path {
        &self.source
    }

    fn geometry(&self) -> &GeometryStreamSpec {
        &self.geometry
    }

    fn instances(&self) -> &InstanceStreamConfig {
        &self.instances
    }

    fn material(&self) -> &MaterialSpec {
        &self.material
    }

    fn target(&self) -> &TargetBinding {
        &self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy_config
    }
}

impl DrawPlanRecipe for HardwareRasterRunPlan {
    fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    fn draw_name(&self) -> &str {
        &self.draw_name
    }

    fn source(&self) -> &Path {
        &self.source
    }

    fn geometry(&self) -> &GeometryStreamSpec {
        &self.geometry
    }

    fn instances(&self) -> &InstanceStreamConfig {
        &self.instances
    }

    fn material(&self) -> &MaterialSpec {
        &self.material
    }

    fn target(&self) -> &TargetBinding {
        &self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy_config
    }
}

impl DrawPlanRecipe for DrawRunPlan {
    fn backend(&self) -> DrawBackend {
        match self {
            Self::Cuda(plan) => plan.backend(),
            Self::DrawExecution(plan) => plan.backend(),
            Self::HardwareRaster(plan) => plan.backend(),
        }
    }

    fn draw_name(&self) -> &str {
        match self {
            Self::Cuda(plan) => plan.draw_name(),
            Self::DrawExecution(plan) => plan.draw_name(),
            Self::HardwareRaster(plan) => plan.draw_name(),
        }
    }

    fn source(&self) -> &Path {
        match self {
            Self::Cuda(plan) => plan.source(),
            Self::DrawExecution(plan) => plan.source(),
            Self::HardwareRaster(plan) => plan.source(),
        }
    }

    fn geometry(&self) -> &GeometryStreamSpec {
        match self {
            Self::Cuda(plan) => plan.geometry(),
            Self::DrawExecution(plan) => plan.geometry(),
            Self::HardwareRaster(plan) => plan.geometry(),
        }
    }

    fn instances(&self) -> &InstanceStreamConfig {
        match self {
            Self::Cuda(plan) => plan.instances(),
            Self::DrawExecution(plan) => plan.instances(),
            Self::HardwareRaster(plan) => plan.instances(),
        }
    }

    fn material(&self) -> &MaterialSpec {
        match self {
            Self::Cuda(plan) => plan.material(),
            Self::DrawExecution(plan) => plan.material(),
            Self::HardwareRaster(plan) => plan.material(),
        }
    }

    fn target(&self) -> &TargetBinding {
        match self {
            Self::Cuda(plan) => plan.target(),
            Self::DrawExecution(plan) => plan.target(),
            Self::HardwareRaster(plan) => plan.target(),
        }
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        match self {
            Self::Cuda(plan) => plan.policy_config(),
            Self::DrawExecution(plan) => plan.policy_config(),
            Self::HardwareRaster(plan) => plan.policy_config(),
        }
    }
}

impl DrawRunPlan {
    pub fn as_cuda_plan(&self) -> Option<&CudaRunPlan> {
        match self {
            Self::Cuda(plan) => Some(plan),
            Self::DrawExecution(_) | Self::HardwareRaster(_) => None,
        }
    }

    pub fn as_draw_execution_plan(&self) -> Option<&DrawExecutionRunPlan> {
        match self {
            Self::Cuda(_) => None,
            Self::DrawExecution(plan) => Some(plan),
            Self::HardwareRaster(plan) => Some(plan),
        }
    }

    pub fn backend(&self) -> DrawBackend {
        DrawPlanRecipe::backend(self)
    }

    pub fn draw_name(&self) -> &str {
        DrawPlanRecipe::draw_name(self)
    }

    pub fn source(&self) -> &Path {
        DrawPlanRecipe::source(self)
    }

    pub fn target(&self) -> &TargetBinding {
        DrawPlanRecipe::target(self)
    }

    pub fn geometry(&self) -> &GeometryStreamSpec {
        DrawPlanRecipe::geometry(self)
    }

    pub fn instances(&self) -> &InstanceStreamConfig {
        DrawPlanRecipe::instances(self)
    }

    pub fn material(&self) -> &MaterialSpec {
        DrawPlanRecipe::material(self)
    }

    pub fn policy(&self) -> DrawPolicy {
        DrawPlanRecipe::policy(self)
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        DrawPlanRecipe::policy_config(self)
    }

    pub fn contract(&self) -> DrawContract {
        DrawPlanRecipe::contract(self)
    }

    pub fn is_cuda_tiled(&self) -> bool {
        self.backend() == DrawBackend::CudaTiled
    }

    pub fn is_cuda_plan(&self) -> bool {
        self.as_cuda_plan().is_some()
    }

    pub fn is_draw_execution_plan(&self) -> bool {
        self.as_draw_execution_plan().is_some()
    }

    pub fn is_draw_execution(&self) -> bool {
        self.is_draw_execution_plan()
    }

    pub fn is_hardware_raster(&self) -> bool {
        self.is_draw_execution()
    }
}

#[cfg(windows)]
pub struct RuntimeDrawResources {
    pub graph: DrawGraph,
    pub geometry_streams: BTreeMap<String, MeshBuffer>,
    pub instance_streams: BTreeMap<String, InstanceBuffer>,
    pub materials: BTreeMap<String, RuntimeMaterialKernel>,
    pub targets: BTreeMap<String, RuntimeTarget>,
}

#[cfg(windows)]
pub enum RuntimeDraw<'a> {
    DrawExecution(RuntimeDrawExecution<'a>),
    Raster(RuntimeDrawExecution<'a>),
    Cuda(RuntimeCudaDraw<'a>),
}

#[cfg(windows)]
impl<'a> RuntimeDraw<'a> {
    pub fn as_draw_execution(&self) -> Option<&RuntimeDrawExecution<'a>> {
        match self {
            Self::DrawExecution(draw) => Some(draw),
            Self::Raster(draw) => Some(draw),
            Self::Cuda(_) => None,
        }
    }

    pub fn as_cuda_draw(&self) -> Option<&RuntimeCudaDraw<'a>> {
        match self {
            Self::DrawExecution(_) | Self::Raster(_) => None,
            Self::Cuda(draw) => Some(draw),
        }
    }

    pub fn is_draw_execution(&self) -> bool {
        self.as_draw_execution().is_some()
    }

    pub fn is_cuda_draw(&self) -> bool {
        self.as_cuda_draw().is_some()
    }

    pub fn backend(&self) -> RuntimeDrawBackend {
        match self {
            Self::DrawExecution(draw) => draw.backend(),
            Self::Raster(draw) => draw.backend(),
            Self::Cuda(draw) => draw.backend(),
        }
    }

    pub fn policy(&self) -> RuntimeDrawPolicy {
        match self {
            Self::DrawExecution(draw) => draw.policy(),
            Self::Raster(draw) => draw.policy(),
            Self::Cuda(draw) => draw.policy(),
        }
    }

    pub fn policy_config(&self) -> RuntimeDrawPolicyConfig {
        match self {
            Self::DrawExecution(draw) => draw.policy_config(),
            Self::Raster(draw) => draw.policy_config(),
            Self::Cuda(draw) => draw.policy_config(),
        }
    }

    pub fn target(&self) -> RuntimeTarget {
        match self {
            Self::DrawExecution(draw) => draw.target(),
            Self::Raster(draw) => draw.target(),
            Self::Cuda(draw) => draw.target(),
        }
    }

    pub fn material(&self) -> &RuntimeMaterialKernel {
        match self {
            Self::DrawExecution(draw) => draw.material(),
            Self::Raster(draw) => draw.material(),
            Self::Cuda(draw) => draw.material(),
        }
    }

    pub fn geometry(&self) -> RuntimeGeometryStream<'a> {
        match self {
            Self::DrawExecution(draw) => draw.geometry(),
            Self::Raster(draw) => draw.geometry(),
            Self::Cuda(draw) => draw.geometry(),
        }
    }

    pub fn instances(&self) -> Option<RuntimeInstanceStream<'a>> {
        match self {
            Self::DrawExecution(draw) => draw.instances(),
            Self::Raster(draw) => draw.instances(),
            Self::Cuda(draw) => Some(draw.instances()),
        }
    }

    pub fn runtime_contract(&self) -> RuntimeDrawContract {
        match self {
            Self::DrawExecution(draw) => draw.contract(),
            Self::Raster(draw) => draw.contract(),
            Self::Cuda(draw) => draw.contract(),
        }
    }
}

#[cfg(windows)]
impl<'a> RuntimeDrawRecipe<'a> for RuntimeDraw<'a> {
    fn backend(&self) -> RuntimeDrawBackend {
        RuntimeDraw::backend(self)
    }

    fn geometry(&self) -> RuntimeGeometryStream<'a> {
        RuntimeDraw::geometry(self)
    }

    fn instances(&self) -> Option<RuntimeInstanceStream<'a>> {
        RuntimeDraw::instances(self)
    }

    fn material(&self) -> &'a RuntimeMaterialKernel {
        match self {
            Self::DrawExecution(draw) => draw.material(),
            Self::Raster(draw) => draw.material(),
            Self::Cuda(draw) => draw.material(),
        }
    }

    fn target(&self) -> RuntimeTarget {
        RuntimeDraw::target(self)
    }

    fn policy_config(&self) -> RuntimeDrawPolicyConfig {
        RuntimeDraw::policy_config(self)
    }

    fn contract(&self) -> RuntimeDrawContract {
        self.runtime_contract()
    }
}

#[cfg(windows)]
impl RuntimeDrawResources {
    pub fn draw(&self, draw_name: &str) -> Result<RuntimeDraw<'_>> {
        let draw = self.graph.draw(draw_name)?;
        match draw.policy {
            DrawPolicy::CudaTiled => self.cuda_draw(draw_name).map(RuntimeDraw::Cuda),
            DrawPolicy::DrawAll | DrawPolicy::ComputeCulled => self
                .draw_execution(draw_name)
                .map(RuntimeDraw::DrawExecution),
        }
    }

    pub fn draw_execution(&self, draw_name: &str) -> Result<RuntimeDrawExecution<'_>> {
        let draw = self.graph.draw(draw_name)?;
        let mesh = self.geometry_for(draw)?;
        let material = self.material_for(draw)?;
        let target = self.target_for(draw)?;
        let mut builder = RuntimeDrawExecution::execution_builder(
            RuntimeGeometryStream::from_mesh(mesh),
            material,
            target,
        )
        .draw_policy_config(runtime_draw_policy_config(draw));
        if let Some(instances) = self.instance_for(draw)? {
            builder = builder.instance_stream(RuntimeInstanceStream::from_instances(instances));
        }
        Ok(builder.try_build()?)
    }

    pub fn raster_draw(&self, draw_name: &str) -> Result<RuntimeDrawExecution<'_>> {
        self.draw_execution(draw_name)
    }

    pub fn cuda_draw(&self, draw_name: &str) -> Result<RuntimeCudaDraw<'_>> {
        let draw = self.graph.draw(draw_name)?;
        let mesh = self.geometry_for(draw)?;
        let material = self.material_for(draw)?;
        let target = self.target_for(draw)?;
        let instances = self.instance_for(draw)?.ok_or_else(|| {
            anyhow::anyhow!(
                "draw `{}` uses DrawPolicy::CudaTiled but has no InstanceStream",
                draw.name
            )
        })?;
        Ok(
            RuntimeCudaDraw::builder(RuntimeGeometryStream::from_mesh(mesh), material, target)
                .instance_stream(RuntimeInstanceStream::from_instances(instances))
                .draw_policy_config(runtime_draw_policy_config(draw))
                .try_build()?,
        )
    }

    fn geometry_for(&self, draw: &DrawGraphDraw) -> Result<&MeshBuffer> {
        self.geometry_streams
            .get(&draw.geometry.mesh)
            .ok_or_else(|| {
                anyhow::anyhow!("missing runtime GeometryStream `{}`", draw.geometry.mesh)
            })
    }

    fn instance_for(&self, draw: &DrawGraphDraw) -> Result<Option<&InstanceBuffer>> {
        draw.instances
            .as_ref()
            .map(|instances| {
                self.instance_streams.get(&instances.name).ok_or_else(|| {
                    anyhow::anyhow!("missing runtime InstanceStream `{}`", instances.name)
                })
            })
            .transpose()
    }

    fn material_for(&self, draw: &DrawGraphDraw) -> Result<&RuntimeMaterialKernel> {
        self.materials.get(&draw.material.name).ok_or_else(|| {
            anyhow::anyhow!("missing runtime MaterialKernel `{}`", draw.material.name)
        })
    }

    fn target_for(&self, draw: &DrawGraphDraw) -> Result<RuntimeTarget> {
        let name = match &draw.target {
            TargetBinding::Window(_) => "window",
            TargetBinding::Named(target) => target.name.as_str(),
        };
        self.targets
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing runtime Target `{name}`"))
    }
}

#[cfg(windows)]
fn runtime_draw_policy_config(draw: &DrawGraphDraw) -> RuntimeDrawPolicyConfig {
    let policy_config = draw.policy_config();
    RuntimeDrawPolicyConfig {
        policy: match policy_config.policy {
            DrawPolicy::DrawAll => RuntimeDrawPolicy::DrawAll,
            DrawPolicy::ComputeCulled => RuntimeDrawPolicy::ComputeCulled,
            DrawPolicy::CudaTiled => RuntimeDrawPolicy::CudaTiled,
        },
        cull_order: match policy_config.cull_order {
            CullOrder::AtomicCompact => RuntimeCullOrder::AtomicCompact,
            CullOrder::StableDense => RuntimeCullOrder::StableDense,
        },
        depth: policy_config.depth.runtime_mode(),
        visibility: match policy_config.visibility {
            VisibilityMode::Frustum => RuntimeVisibilityMode::Frustum,
            VisibilityMode::ProjectedSize => RuntimeVisibilityMode::ProjectedSize,
        },
        min_projected_millipixels: policy_config.min_projected_millipixels,
    }
}

#[cfg(windows)]
fn runtime_material_kernel(material: &MaterialSpec) -> RuntimeMaterialKernel {
    match material.abi {
        MaterialAbi::SimpleColor => RuntimeMaterialKernel::from_stages(
            material.name.clone(),
            material.vertex_entrypoint.clone(),
            material.fragment_entrypoint.clone(),
        ),
        MaterialAbi::DirectInstanceColor => RuntimeMaterialKernel::from_stages(
            material.name.clone(),
            material.vertex_entrypoint.clone(),
            material.fragment_entrypoint.clone(),
        )
        .with_abi(RuntimeMaterialKernelAbi::direct_instance_color(
            material.vertex_entrypoint.clone(),
            material.fragment_entrypoint.clone(),
        )),
        MaterialAbi::ComputeCulledInstanceColor => RuntimeMaterialKernel::from_stages(
            material.name.clone(),
            material.vertex_entrypoint.clone(),
            material.fragment_entrypoint.clone(),
        )
        .with_abi(RuntimeMaterialKernelAbi::compute_culled_instance_color(
            material.vertex_entrypoint.clone(),
            material.fragment_entrypoint.clone(),
        )),
        MaterialAbi::CudaTiledInstanceColor => RuntimeMaterialKernel::from_cuda_tiled(
            material.name.clone(),
            material
                .kernel_entrypoint()
                .expect("validated CUDA tiled material has a kernel entrypoint"),
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceStressVariant {
    Tiled,
    Macrocell,
}

impl InstanceStressVariant {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::Tiled => "tiled",
            Self::Macrocell => "macrocell",
        }
    }
}

#[cfg(windows)]
fn upload_runtime_instance_stream(
    context: &Context,
    stream: &InstanceStreamConfig,
) -> Result<InstanceBuffer, neo_runtime::RuntimeError> {
    const STRIDE: u32 = 40;
    let count = stream
        .grid
        .count()
        .expect("validated instance grid count fits in u32");
    let byte_len = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(STRIDE as usize))
        .ok_or_else(|| {
            neo_runtime::RuntimeError::Instance(
                "instance stream byte size overflowed usize".to_string(),
            )
        })?;
    let instance_bytes = vec![0u8; byte_len];
    InstanceBuffer::upload_with_layout(
        context,
        InstanceBufferDesc {
            instance_count: count,
            instance_layout: RuntimeInstanceLayout {
                stride: STRIDE,
                attributes: vec![
                    RuntimeInstanceAttribute {
                        semantic: RuntimeInstanceSemantic::Position,
                        format: RuntimeInstanceFormat::F32x3,
                        offset: 0,
                    },
                    RuntimeInstanceAttribute {
                        semantic: RuntimeInstanceSemantic::Rotation,
                        format: RuntimeInstanceFormat::F32x4,
                        offset: 12,
                    },
                    RuntimeInstanceAttribute {
                        semantic: RuntimeInstanceSemantic::Scale,
                        format: RuntimeInstanceFormat::F32x2,
                        offset: 28,
                    },
                    RuntimeInstanceAttribute {
                        semantic: RuntimeInstanceSemantic::Color0,
                        format: RuntimeInstanceFormat::U8x4Unorm,
                        offset: 36,
                    },
                ],
            },
        },
        &instance_bytes,
        match stream.layout {
            InstanceLayout::AoSoA32 => RuntimeDataLayout::AoSoA { group_size: 32 },
            InstanceLayout::AoSoA64 => RuntimeDataLayout::AoSoA { group_size: 64 },
        },
    )
}

#[derive(Debug, Clone)]
pub struct GeometryStreamConfig {
    pub name: String,
    pub source: GeometryStreamSource,
}

pub type MeshSpec = GeometryStreamConfig;

#[derive(Debug, Clone)]
pub enum GeometryStreamSource {
    Builder(MeshBuilder),
}

pub type MeshSource = GeometryStreamSource;

impl GeometryStreamConfig {
    fn vertex_bytes(&self) -> Vec<u8> {
        match &self.source {
            GeometryStreamSource::Builder(builder) => builder.vertex_bytes(),
        }
    }

    fn vertex_stride(&self) -> u32 {
        match &self.source {
            GeometryStreamSource::Builder(builder) => builder.layout.stride,
        }
    }

    fn color_offset(&self) -> u32 {
        match &self.source {
            GeometryStreamSource::Builder(builder) => builder
                .layout
                .attributes
                .iter()
                .find(|attribute| attribute.semantic == VertexSemantic::Color0)
                .map(|attribute| attribute.offset)
                .unwrap_or(12),
        }
    }

    fn indices_u16(&self) -> Vec<u16> {
        match &self.source {
            GeometryStreamSource::Builder(builder) => builder.indices.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeoAppConfig {
    pub window: WindowConfig,
    pub presenter: Presenter,
    pub interop_fallback: InteropFallback,
    pub draw_backend_preference: DrawBackendPreference,
    pub fps: FpsCap,
    pub hot_reload: bool,
    pub max_inflight: u32,
    pub present_ring: usize,
    pub render_policy: RenderPolicy,
}

pub struct NeoApp {
    config: NeoAppConfig,
    kernels: BTreeMap<String, KernelSpec>,
    geometry_streams: BTreeMap<String, GeometryStreamConfig>,
    instance_streams: BTreeMap<String, InstanceStreamConfig>,
    sparse_textures: BTreeMap<String, SparseTextureConfig>,
    material_streams: BTreeMap<String, MaterialStreamConfig>,
    materials: BTreeMap<String, MaterialSpec>,
    targets: BTreeMap<String, TargetConfig>,
    compute_culls: BTreeMap<String, ComputeCullSpec>,
    draws: BTreeMap<String, DrawSpec>,
    context: Option<Context>,
}

pub struct NeoAppParts {
    pub config: NeoAppConfig,
    pub kernels: BTreeMap<String, KernelSpec>,
    pub geometry_streams: BTreeMap<String, GeometryStreamConfig>,
    pub meshes: BTreeMap<String, MeshSpec>,
    pub instance_streams: BTreeMap<String, InstanceStreamConfig>,
    pub sparse_textures: BTreeMap<String, SparseTextureConfig>,
    pub material_streams: BTreeMap<String, MaterialStreamConfig>,
    pub materials: BTreeMap<String, MaterialSpec>,
    pub targets: BTreeMap<String, TargetConfig>,
    pub compute_culls: BTreeMap<String, ComputeCullSpec>,
    pub draws: BTreeMap<String, DrawSpec>,
    pub indirect_draws: BTreeMap<String, DrawIndirectSpec>,
    pub context: Option<Context>,
}

impl Default for NeoApp {
    fn default() -> Self {
        Self::new()
    }
}

impl NeoApp {
    pub fn new() -> Self {
        Self {
            config: NeoAppConfig {
                window: WindowConfig {
                    title: "Neo".to_string(),
                    width: 960,
                    height: 540,
                },
                presenter: Presenter::D3d12Interop,
                interop_fallback: InteropFallback::NoInterop,
                draw_backend_preference: DrawBackendPreference::PrimaryNeo,
                fps: FpsCap {
                    kernel: None,
                    present: None,
                },
                hot_reload: true,
                max_inflight: 8,
                present_ring: 8,
                render_policy: RenderPolicy::Auto,
            },
            kernels: BTreeMap::new(),
            geometry_streams: BTreeMap::new(),
            instance_streams: BTreeMap::new(),
            sparse_textures: BTreeMap::new(),
            material_streams: BTreeMap::new(),
            materials: BTreeMap::new(),
            targets: BTreeMap::new(),
            compute_culls: BTreeMap::new(),
            draws: BTreeMap::new(),
            context: None,
        }
    }

    pub fn window(mut self, title: impl Into<String>, width: u32, height: u32) -> Self {
        self.config.window = WindowConfig {
            title: title.into(),
            width,
            height,
        };
        self
    }

    pub fn presenter(mut self, presenter: Presenter) -> Self {
        self.config.presenter = presenter;
        self
    }

    pub fn interop_fallback(mut self, fallback: InteropFallback) -> Self {
        self.config.interop_fallback = fallback;
        self
    }

    pub fn draw_backend_preference(mut self, preference: DrawBackendPreference) -> Self {
        self.config.draw_backend_preference = preference;
        self
    }

    pub fn renderer_preference(self, preference: RendererPreference) -> Self {
        self.draw_backend_preference(preference)
    }

    pub fn renderer(self, preference: RendererPreference) -> Self {
        self.draw_backend_preference(preference)
    }

    pub fn kernel(mut self, entrypoint: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        let entrypoint = entrypoint.into();
        self.kernels.insert(
            entrypoint.clone(),
            KernelSpec {
                entrypoint,
                path: path.into(),
            },
        );
        self
    }

    pub fn geometry_stream(
        mut self,
        name: impl Into<String>,
        source: impl Into<GeometryStreamSource>,
    ) -> Self {
        let name = name.into();
        self.geometry_streams.insert(
            name.clone(),
            GeometryStreamConfig {
                name,
                source: source.into(),
            },
        );
        self
    }

    pub fn mesh(self, name: impl Into<String>, mesh: impl Into<MeshSource>) -> Self {
        self.geometry_stream(name, mesh)
    }

    pub fn instance_stream(mut self, name: impl Into<String>, grid: InstanceGrid) -> Self {
        self = self.instance_stream_aosoa32(name, grid);
        self
    }

    pub fn instance_stream_aosoa32(self, name: impl Into<String>, grid: InstanceGrid) -> Self {
        self.instance_stream_with_layout(name, grid, InstanceLayout::AoSoA32)
    }

    pub fn instance_stream_aosoa64(self, name: impl Into<String>, grid: InstanceGrid) -> Self {
        self.instance_stream_with_layout(name, grid, InstanceLayout::AoSoA64)
    }

    pub fn try_instance_stream_aosoa(
        self,
        name: impl Into<String>,
        grid: InstanceGrid,
        group_size: u32,
    ) -> Result<Self> {
        let layout = match group_size {
            32 => InstanceLayout::AoSoA32,
            64 => InstanceLayout::AoSoA64,
            _ => {
                bail!("unsupported InstanceStream AoSoA group size {group_size}; expected 32 or 64")
            }
        };
        Ok(self.instance_stream_with_layout(name, grid, layout))
    }

    pub fn instance_stream_config(mut self, config: InstanceStreamConfig) -> Self {
        self.instance_streams.insert(config.name.clone(), config);
        self
    }

    pub fn sparse_texture(
        mut self,
        name: impl Into<String>,
        builder: SparseTextureBuilder,
    ) -> Self {
        let name = name.into();
        self.sparse_textures
            .insert(name.clone(), builder.config(name));
        self
    }

    pub fn material_stream(
        mut self,
        name: impl Into<String>,
        builder: MaterialStreamBuilder,
    ) -> Self {
        let name = name.into();
        self.material_streams
            .insert(name.clone(), builder.config(name));
        self
    }

    pub fn instance_stream_with_layout(
        mut self,
        name: impl Into<String>,
        grid: InstanceGrid,
        layout: InstanceLayout,
    ) -> Self {
        let name = name.into();
        self.instance_streams
            .insert(name.clone(), InstanceStreamConfig { name, grid, layout });
        self
    }

    pub fn instance_stream_layout(
        mut self,
        name: impl Into<String>,
        layout: InstanceLayout,
    ) -> Self {
        let name = name.into();
        if let Some(stream) = self.instance_streams.get_mut(&name) {
            stream.layout = layout;
        } else {
            self.instance_streams.insert(
                name.clone(),
                InstanceStreamConfig {
                    name,
                    grid: InstanceGrid::new(1, 1, 1),
                    layout,
                },
            );
        }
        self
    }

    pub fn material_kernel(
        mut self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        let name = name.into();
        self.materials.insert(
            name.clone(),
            MaterialSpec {
                name,
                vertex_entrypoint: vertex_entrypoint.into(),
                fragment_entrypoint: fragment_entrypoint.into(),
                abi: MaterialAbi::SimpleColor,
                source: None,
            },
        );
        self
    }

    pub fn instance_material_kernel(
        mut self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        let name = name.into();
        self.materials.insert(
            name.clone(),
            MaterialSpec {
                name,
                vertex_entrypoint: vertex_entrypoint.into(),
                fragment_entrypoint: fragment_entrypoint.into(),
                abi: MaterialAbi::ComputeCulledInstanceColor,
                source: None,
            },
        );
        self
    }

    pub fn direct_instance_material_kernel(
        mut self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
        source: impl Into<PathBuf>,
    ) -> Self {
        let name = name.into();
        self.materials.insert(
            name.clone(),
            MaterialSpec {
                name,
                vertex_entrypoint: vertex_entrypoint.into(),
                fragment_entrypoint: fragment_entrypoint.into(),
                abi: MaterialAbi::DirectInstanceColor,
                source: Some(source.into()),
            },
        );
        self
    }

    pub fn cuda_tiled_material_kernel(
        mut self,
        name: impl Into<String>,
        entrypoint: impl Into<String>,
        source: impl Into<PathBuf>,
    ) -> Self {
        let name = name.into();
        self.materials.insert(
            name.clone(),
            MaterialSpec {
                name,
                vertex_entrypoint: entrypoint.into(),
                fragment_entrypoint: String::new(),
                abi: MaterialAbi::CudaTiledInstanceColor,
                source: Some(source.into()),
            },
        );
        self
    }

    pub fn draw_execution_material_kernel(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        self.material_kernel(name, vertex_entrypoint, fragment_entrypoint)
    }

    pub fn draw_execution_instance_material_kernel(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        self.instance_material_kernel(name, vertex_entrypoint, fragment_entrypoint)
    }

    pub fn direct_draw_execution_instance_material_kernel(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
        source: impl Into<PathBuf>,
    ) -> Self {
        self.direct_instance_material_kernel(name, vertex_entrypoint, fragment_entrypoint, source)
    }

    pub fn raster_pipeline(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        self.material_kernel(name, vertex_entrypoint, fragment_entrypoint)
    }

    pub fn raster_instance_material(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        self.instance_material_kernel(name, vertex_entrypoint, fragment_entrypoint)
    }

    pub fn raster_direct_instance_material(
        self,
        name: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
        source: impl Into<PathBuf>,
    ) -> Self {
        self.direct_instance_material_kernel(name, vertex_entrypoint, fragment_entrypoint, source)
    }

    pub fn target(mut self, name: impl Into<String>, width: u32, height: u32) -> Self {
        let name = name.into();
        self.targets.insert(
            name.clone(),
            TargetConfig {
                name,
                width,
                height,
            },
        );
        self
    }

    pub fn compute_cull(self, entrypoint: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.compute_cull_with_order(entrypoint, path, CullOrder::AtomicCompact)
    }

    pub fn compute_cull_with_order(
        mut self,
        entrypoint: impl Into<String>,
        path: impl Into<PathBuf>,
        order: CullOrder,
    ) -> Self {
        let entrypoint = entrypoint.into();
        self.compute_culls.insert(
            entrypoint.clone(),
            ComputeCullSpec {
                entrypoint,
                path: path.into(),
                order,
            },
        );
        self
    }

    pub fn draw(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
    ) -> Self {
        self.draw_with_policy(
            name,
            geometry_mesh,
            instances,
            material,
            TargetSpec::Window,
            DrawPolicy::ComputeCulled,
        )
    }

    pub fn draw_with_policy(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
        target: TargetSpec,
        policy: DrawPolicy,
    ) -> Self {
        let policy_config = match policy {
            DrawPolicy::DrawAll => DrawPolicyConfig::draw_all(),
            DrawPolicy::ComputeCulled => DrawPolicyConfig::compute_culled(
                self.compute_culls
                    .values()
                    .next()
                    .map(|cull| cull.order)
                    .unwrap_or(CullOrder::AtomicCompact),
            ),
            DrawPolicy::CudaTiled => DrawPolicyConfig::cuda_tiled(),
        };
        self.draw_with_policy_config(
            name,
            geometry_mesh,
            instances,
            material,
            target,
            policy_config,
        )
    }

    pub fn draw_with_policy_config(
        mut self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
        target: TargetSpec,
        policy_config: DrawPolicyConfig,
    ) -> Self {
        let name = name.into();
        self.draws.insert(
            name.clone(),
            DrawSpec {
                name,
                geometry: GeometryStreamSpec {
                    mesh: geometry_mesh.into(),
                    vertex_bytes: Vec::new(),
                    vertex_stride: 0,
                    color_offset: 0,
                    indices_u16: Vec::new(),
                },
                instances: instances.map(|name| InstanceStreamSpec { name: name.into() }),
                material: material.into(),
                target,
                policy: policy_config.policy,
                policy_config,
            },
        );
        self
    }

    pub fn draw_all(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
        target: TargetSpec,
    ) -> Self {
        self.draw_with_policy(
            name,
            geometry_mesh,
            instances,
            material,
            target,
            DrawPolicy::DrawAll,
        )
    }

    pub fn draw_compute_culled(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: impl Into<String>,
        material: impl Into<String>,
        target: TargetSpec,
    ) -> Self {
        self.draw_with_policy(
            name,
            geometry_mesh,
            Some(instances),
            material,
            target,
            DrawPolicy::ComputeCulled,
        )
    }

    pub fn draw_compute_culled_projected(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: impl Into<String>,
        material: impl Into<String>,
        target: TargetSpec,
        min_projected_millipixels: u32,
    ) -> Self {
        self.draw_with_policy_config(
            name,
            geometry_mesh,
            Some(instances),
            material,
            target,
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize,
            )
            .with_min_projected_millipixels(min_projected_millipixels),
        )
    }

    pub fn draw_cuda_tiled(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: impl Into<String>,
        material: impl Into<String>,
        target: TargetSpec,
    ) -> Self {
        self.draw_with_policy_config(
            name,
            geometry_mesh,
            Some(instances),
            material,
            target,
            DrawPolicyConfig::cuda_tiled(),
        )
    }

    pub fn draw_indirect(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
    ) -> Self {
        self.draw(name, geometry_mesh, instances, material)
    }

    pub fn draw_indirect_with_policy(
        self,
        name: impl Into<String>,
        geometry_mesh: impl Into<String>,
        instances: Option<impl Into<String>>,
        material: impl Into<String>,
        target: TargetSpec,
        policy: DrawPolicy,
    ) -> Self {
        self.draw_with_policy(name, geometry_mesh, instances, material, target, policy)
    }

    pub fn target_fps(mut self, fps: f32) -> Self {
        self.config.fps.kernel = Some(fps);
        self.config.fps.present = Some(fps);
        self
    }

    pub fn kernel_target_fps(mut self, fps: f32) -> Self {
        self.config.fps.kernel = Some(fps);
        self
    }

    pub fn present_target_fps(mut self, fps: f32) -> Self {
        self.config.fps.present = Some(fps);
        self
    }

    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload = enabled;
        self
    }

    pub fn max_inflight(mut self, max_inflight: u32) -> Self {
        self.config.max_inflight = max_inflight;
        self
    }

    pub fn present_ring(mut self, present_ring: usize) -> Self {
        self.config.present_ring = present_ring;
        self
    }

    pub fn render_policy(mut self, policy: RenderPolicy) -> Self {
        self.config.render_policy = policy;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.config.window.width == 0 || self.config.window.height == 0 {
            bail!("window width and height must be greater than zero");
        }
        if self.config.max_inflight == 0 {
            bail!("max_inflight must be greater than zero");
        }
        if self.config.present_ring == 0 {
            bail!("present_ring must be greater than zero");
        }
        for fps in [self.config.fps.kernel, self.config.fps.present]
            .into_iter()
            .flatten()
        {
            if !fps.is_finite() || fps <= 0.0 {
                bail!("FPS caps must be finite and greater than zero");
            }
        }
        for target in self.targets.values() {
            if target.width == 0 || target.height == 0 {
                bail!(
                    "target `{}` width and height must be greater than zero",
                    target.name
                );
            }
        }
        for stream in self.instance_streams.values() {
            stream
                .grid
                .validate()
                .map_err(|err| anyhow::anyhow!("instance stream `{}`: {err}", stream.name))?;
        }
        for draw in self.draws.values() {
            if !self.geometry_streams.contains_key(&draw.geometry.mesh) {
                bail!(
                    "draw `{}` references missing geometry stream `{}`",
                    draw.name,
                    draw.geometry.mesh
                );
            }
            if !self.materials.contains_key(&draw.material) {
                bail!(
                    "draw `{}` references missing material `{}`",
                    draw.name,
                    draw.material
                );
            }
            let material = self
                .materials
                .get(&draw.material)
                .expect("material existence was just checked");
            if material.abi.requires_instances() && draw.instances.is_none() {
                bail!(
                    "draw `{}` uses material `{}` which requires an InstanceStream",
                    draw.name,
                    material.name
                );
            }
            if material.abi.requires_compute_culling()
                && !matches!(
                    draw.policy,
                    DrawPolicy::ComputeCulled | DrawPolicy::CudaTiled
                )
            {
                bail!(
                    "draw `{}` uses material `{}` which requires DrawPolicy::ComputeCulled or DrawPolicy::CudaTiled",
                    draw.name,
                    material.name
                );
            }
            if draw.policy == DrawPolicy::CudaTiled && !material.abi.requires_cuda_tiled() {
                bail!(
                    "draw `{}` uses DrawPolicy::CudaTiled but material `{}` is not a CUDA tiled MaterialKernel",
                    draw.name,
                    material.name
                );
            }
            if material.abi.requires_cuda_tiled() && draw.policy != DrawPolicy::CudaTiled {
                bail!(
                    "draw `{}` uses CUDA tiled MaterialKernel `{}` but its DrawPolicy is not CudaTiled",
                    draw.name,
                    material.name
                );
            }
            if material.abi.requires_cuda_tiled() && material.source.is_none() {
                bail!(
                    "CUDA tiled MaterialKernel `{}` requires an explicit source path",
                    material.name
                );
            }
            if draw.policy == DrawPolicy::CudaTiled && draw.instances.is_none() {
                bail!(
                    "draw `{}` uses DrawPolicy::CudaTiled but has no InstanceStream",
                    draw.name
                );
            }
            if let Some(instances) = &draw.instances {
                if !self.instance_streams.contains_key(&instances.name) {
                    bail!(
                        "draw `{}` references missing instance stream `{}`",
                        draw.name,
                        instances.name
                    );
                }
            }
            if let TargetSpec::Named(target) = &draw.target {
                if !self.targets.contains_key(target) {
                    bail!("draw `{}` references missing target `{target}`", draw.name);
                }
            }
            if draw.policy == DrawPolicy::ComputeCulled && self.compute_culls.is_empty() {
                bail!(
                    "draw `{}` uses compute culling but no compute_cull kernel is configured",
                    draw.name
                );
            }
        }
        Ok(())
    }

    pub fn context(&mut self) -> Result<&Context> {
        if self.context.is_none() {
            self.context = Some(Context::new_default_device()?);
        }
        Ok(self.context.as_ref().expect("context was just initialized"))
    }

    pub fn context_mut(&mut self) -> Result<&mut Context> {
        if self.context.is_none() {
            self.context = Some(Context::new_default_device()?);
        }
        Ok(self.context.as_mut().expect("context was just initialized"))
    }

    pub fn kernel_spec(&self, entrypoint: &str) -> Option<&KernelSpec> {
        self.kernels.get(entrypoint)
    }

    pub fn kernel_specs(&self) -> impl Iterator<Item = &KernelSpec> {
        self.kernels.values()
    }

    pub fn mesh_spec(&self, name: &str) -> Option<&MeshSpec> {
        self.geometry_stream_spec(name)
    }

    pub fn geometry_stream_spec(&self, name: &str) -> Option<&GeometryStreamConfig> {
        self.geometry_streams.get(name)
    }

    pub fn geometry_stream_specs(&self) -> impl Iterator<Item = &GeometryStreamConfig> {
        self.geometry_streams.values()
    }

    pub fn instance_stream_spec(&self, name: &str) -> Option<&InstanceStreamConfig> {
        self.instance_streams.get(name)
    }

    pub fn instance_stream_specs(&self) -> impl Iterator<Item = &InstanceStreamConfig> {
        self.instance_streams.values()
    }

    pub fn material_spec(&self, name: &str) -> Option<&MaterialSpec> {
        self.materials.get(name)
    }

    pub fn material_specs(&self) -> impl Iterator<Item = &MaterialSpec> {
        self.materials.values()
    }

    pub fn draw_spec(&self, name: &str) -> Option<&DrawSpec> {
        self.draws.get(name)
    }

    pub fn draw_specs(&self) -> impl Iterator<Item = &DrawSpec> {
        self.draws.values()
    }

    pub fn draw_contract(&self, name: &str) -> Option<DrawContract> {
        self.draw_spec(name).map(DrawSpec::contract)
    }

    pub fn draw_contracts(&self) -> impl Iterator<Item = DrawContract> + '_ {
        self.draw_specs().map(DrawSpec::contract)
    }

    pub fn indirect_draw_spec(&self, name: &str) -> Option<&DrawIndirectSpec> {
        self.draw_spec(name)
    }

    pub fn target_spec(&self, name: &str) -> Option<&TargetConfig> {
        self.targets.get(name)
    }

    pub fn target_specs(&self) -> impl Iterator<Item = &TargetConfig> {
        self.targets.values()
    }

    pub fn draw_graph(&self) -> Result<DrawGraph> {
        self.validate()?;
        let compute_cull = self.compute_culls.values().next().cloned();
        let draws = self
            .draws
            .values()
            .map(|draw| {
                let mesh = self
                    .geometry_streams
                    .get(&draw.geometry.mesh)
                    .expect("draw graph validation checked geometry existence");
                let geometry = GeometryStreamSpec {
                    mesh: draw.geometry.mesh.clone(),
                    vertex_bytes: mesh.vertex_bytes(),
                    vertex_stride: mesh.vertex_stride(),
                    color_offset: mesh.color_offset(),
                    indices_u16: mesh.indices_u16(),
                };
                let instances = draw
                    .instances
                    .as_ref()
                    .and_then(|instances| self.instance_streams.get(&instances.name))
                    .cloned();
                let material = self
                    .materials
                    .get(&draw.material)
                    .expect("draw graph validation checked material existence")
                    .clone();
                let target = match &draw.target {
                    TargetSpec::Window => TargetBinding::Window(self.config.window.clone()),
                    TargetSpec::Named(name) => TargetBinding::Named(
                        self.targets
                            .get(name)
                            .expect("draw graph validation checked target existence")
                            .clone(),
                    ),
                };
                DrawGraphDraw {
                    name: draw.name.clone(),
                    geometry,
                    instances,
                    material,
                    target,
                    policy: draw.policy,
                    depth: draw.policy_config.depth,
                    cull_order: if matches!(
                        draw.policy_config.policy,
                        DrawPolicy::ComputeCulled | DrawPolicy::CudaTiled
                    ) {
                        draw.policy_config.cull_order
                    } else {
                        CullOrder::StableDense
                    },
                    visibility: if matches!(
                        draw.policy_config.policy,
                        DrawPolicy::ComputeCulled | DrawPolicy::CudaTiled
                    ) {
                        draw.policy_config.visibility
                    } else {
                        VisibilityMode::Frustum
                    },
                    min_projected_millipixels: if matches!(
                        draw.policy_config.policy,
                        DrawPolicy::ComputeCulled | DrawPolicy::CudaTiled
                    ) {
                        draw.policy_config.min_projected_millipixels
                    } else {
                        DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
                    },
                    compute_cull: if draw.policy_config.policy == DrawPolicy::ComputeCulled {
                        compute_cull.clone()
                    } else {
                        None
                    },
                }
            })
            .collect();
        Ok(DrawGraph { draws })
    }

    fn draw_execution_run_plan_from_draw(
        &self,
        draw: DrawGraphDraw,
    ) -> Result<DrawExecutionRunPlan> {
        let policy_config = draw.policy_config();
        let instances = draw
            .instances
            .ok_or_else(|| anyhow::anyhow!("draw `{}` is missing its InstanceStream", draw.name))?;
        if draw.material.abi == MaterialAbi::SimpleColor {
            bail!(
                "live-window raster execution requires material `{}` to be a DirectInstanceColor or ComputeCulledInstanceColor MaterialKernel",
                draw.material.name
            );
        }
        let compute_cull = if draw.policy == DrawPolicy::ComputeCulled {
            Some(draw.compute_cull.ok_or_else(|| {
                anyhow::anyhow!("draw `{}` is missing its compute cull stage", draw.name)
            })?)
        } else {
            draw.compute_cull
        };
        let source = draw
            .material
            .source
            .clone()
            .or_else(|| compute_cull.as_ref().map(|cull| cull.path.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "draw `{}` is missing a MaterialKernel source path for raster execution",
                    draw.name
                )
            })?;
        Ok(HardwareRasterRunPlan {
            draw_name: draw.name,
            source,
            geometry: draw.geometry,
            instances,
            material: draw.material,
            target: draw.target,
            policy: draw.policy,
            policy_config,
            cull_order: draw.cull_order,
            visibility: draw.visibility,
            min_projected_millipixels: draw.min_projected_millipixels,
            compute_cull,
        })
    }

    fn cuda_run_plan_from_draw(&self, draw: DrawGraphDraw) -> Result<CudaRunPlan> {
        let instances = draw
            .instances
            .ok_or_else(|| anyhow::anyhow!("draw `{}` is missing its InstanceStream", draw.name))?;
        let source = draw
            .material
            .source
            .clone()
            .or_else(|| {
                self.kernels
                    .get("instance_raster")
                    .map(|kernel| kernel.path.clone())
            })
            .or_else(|| self.kernels.get("image").map(|kernel| kernel.path.clone()))
            .unwrap_or_else(|| {
                Path::new("examples/stress-quads/three_d_instances.neo").to_path_buf()
            });
        let entrypoint = draw
            .material
            .kernel_entrypoint()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "draw `{}` CUDA MaterialKernel `{}` is missing a kernel entrypoint",
                    draw.name,
                    draw.material.name
                )
            })?
            .to_string();
        Ok(CudaRunPlan {
            draw_name: draw.name,
            source,
            entrypoint,
            geometry: draw.geometry,
            instances,
            material: draw.material,
            target: draw.target,
            policy_config: DrawPolicyConfig {
                policy: draw.policy,
                depth: draw.depth,
                cull_order: draw.cull_order,
                visibility: draw.visibility,
                min_projected_millipixels: draw.min_projected_millipixels,
            },
            variant: if self.sparse_textures.is_empty() && self.material_streams.is_empty() {
                InstanceStressVariant::Tiled
            } else {
                InstanceStressVariant::Macrocell
            },
        })
    }

    pub fn draw_run_plan(&self) -> Result<Option<DrawRunPlan>> {
        let graph = self.draw_graph()?;
        let Some(draw) = self.select_draw_for_run_plan(graph) else {
            return Ok(None);
        };
        match draw.policy {
            DrawPolicy::CudaTiled => self
                .cuda_run_plan_from_draw(draw)
                .map(DrawRunPlan::Cuda)
                .map(Some),
            DrawPolicy::DrawAll | DrawPolicy::ComputeCulled => self
                .draw_execution_run_plan_from_draw(draw)
                .map(DrawRunPlan::DrawExecution)
                .map(Some),
        }
    }

    fn select_draw_for_run_plan(&self, graph: DrawGraph) -> Option<DrawGraphDraw> {
        let mut draws = graph.draws;
        if let Some(preferred_backend) = self.config.draw_backend_preference.preferred_backend()
            && let Some(index) = draws
                .iter()
                .position(|draw| draw.backend() == preferred_backend)
        {
            return Some(draws.remove(index));
        }
        draws.into_iter().next()
    }

    pub fn draw_execution_run_plan(&self) -> Result<Option<DrawExecutionRunPlan>> {
        let graph = self.draw_graph()?;
        let Some(draw) = graph
            .draws
            .into_iter()
            .find(|draw| draw.policy != DrawPolicy::CudaTiled)
        else {
            return Ok(None);
        };
        self.draw_execution_run_plan_from_draw(draw).map(Some)
    }

    pub fn hardware_raster_run_plan(&self) -> Result<Option<HardwareRasterRunPlan>> {
        self.draw_execution_run_plan()
    }

    pub fn raster_run_plan(&self) -> Result<Option<RasterRunPlan>> {
        self.draw_execution_run_plan()
    }

    pub fn cuda_run_plan(&self) -> Result<Option<CudaRunPlan>> {
        let graph = self.draw_graph()?;
        let Some(draw) = graph
            .draws
            .into_iter()
            .find(|draw| draw.policy == DrawPolicy::CudaTiled)
        else {
            return Ok(None);
        };
        self.cuda_run_plan_from_draw(draw).map(Some)
    }

    pub fn presenter_kind(&self) -> Presenter {
        self.config.presenter
    }

    pub fn config(&self) -> &NeoAppConfig {
        &self.config
    }

    pub fn into_parts(self) -> NeoAppParts {
        NeoAppParts {
            config: self.config,
            kernels: self.kernels,
            geometry_streams: self.geometry_streams.clone(),
            meshes: self.geometry_streams,
            instance_streams: self.instance_streams,
            sparse_textures: self.sparse_textures,
            material_streams: self.material_streams,
            materials: self.materials,
            targets: self.targets,
            compute_culls: self.compute_culls,
            draws: self.draws.clone(),
            indirect_draws: self.draws,
            context: self.context,
        }
    }

    pub fn run(self) -> Result<()> {
        let mut app = self;
        app.validate()?;
        let _friendly_geometry_streams = app.build_geometry_streams()?;
        let draw_plan = app.draw_run_plan()?;
        let args = app.try_live_window_args()?;
        match draw_plan {
            Some(DrawRunPlan::DrawExecution(plan)) | Some(DrawRunPlan::HardwareRaster(plan)) => {
                neo_live_window::run_from_args_with_draw_execution_plan(args, plan.live_draw_plan())
            }
            Some(DrawRunPlan::Cuda(_)) | None => neo_live_window::run_from_args(args),
        }
    }

    pub fn build_geometry_streams(&mut self) -> Result<BTreeMap<String, MeshBuffer>> {
        let builders: Vec<(String, MeshBuilder)> = self
            .geometry_streams
            .iter()
            .map(|(name, spec)| match &spec.source {
                GeometryStreamSource::Builder(builder) => (name.clone(), builder.clone()),
            })
            .collect();
        if builders.is_empty() {
            return Ok(BTreeMap::new());
        }
        let context = self.context()?;
        builders
            .into_iter()
            .map(|(name, builder)| builder.build(context).map(|mesh| (name, mesh)))
            .collect()
    }

    pub fn build_meshes(&mut self) -> Result<BTreeMap<String, MeshBuffer>> {
        self.build_geometry_streams()
    }

    #[cfg(windows)]
    pub fn build_instance_streams(&mut self) -> Result<BTreeMap<String, InstanceBuffer>> {
        let streams: Vec<InstanceStreamConfig> = self.instance_streams.values().cloned().collect();
        if streams.is_empty() {
            return Ok(BTreeMap::new());
        }
        let context = self.context()?;
        streams
            .into_iter()
            .map(|stream| {
                upload_runtime_instance_stream(context, &stream)
                    .map(|instances| (stream.name.clone(), instances))
                    .map_err(Into::into)
            })
            .collect()
    }

    #[cfg(windows)]
    pub fn build_runtime_draw_resources(&mut self) -> Result<RuntimeDrawResources> {
        let graph = self.draw_graph()?;
        let geometry_streams = self.build_geometry_streams()?;
        let instance_streams = self.build_instance_streams()?;
        let materials = self
            .materials
            .iter()
            .map(|(name, material)| (name.clone(), runtime_material_kernel(material)))
            .collect();
        let mut targets = BTreeMap::new();
        targets.insert(
            "window".to_string(),
            RuntimeTarget::new(self.config.window.width, self.config.window.height)?,
        );
        for target in self.targets.values() {
            targets.insert(
                target.name.clone(),
                RuntimeTarget::new(target.width, target.height)?,
            );
        }
        Ok(RuntimeDrawResources {
            graph,
            geometry_streams,
            instance_streams,
            materials,
            targets,
        })
    }

    pub fn live_window_args(&self) -> Vec<String> {
        self.try_live_window_args()
            .expect("NeoApp::live_window_args requires a valid app configuration")
    }

    pub fn try_live_window_args(&self) -> Result<Vec<String>> {
        let draw_plan = self.draw_run_plan()?;
        let draw_execution_mode = matches!(
            &draw_plan,
            Some(DrawRunPlan::DrawExecution(_) | DrawRunPlan::HardwareRaster(_))
        );
        let cuda_draw_mode = matches!(&draw_plan, Some(DrawRunPlan::Cuda(_)));
        let mesh_mode =
            self.geometry_streams.contains_key("quad") && self.kernels.contains_key("raster");
        let source = if let Some(plan) = &draw_plan {
            Some(plan.source().to_path_buf())
        } else if mesh_mode {
            self.kernels.get("raster").map(|kernel| kernel.path.clone())
        } else {
            self.kernels
                .get("image")
                .or_else(|| self.kernels.values().next())
                .map(|kernel| kernel.path.clone())
        }
        .unwrap_or_else(|| Path::new("examples/live-window/live.neo").to_path_buf());

        let (output_width, output_height) = if let Some(plan) = &draw_plan {
            plan.target().dimensions()
        } else {
            (self.config.window.width, self.config.window.height)
        };

        let mut args = vec![
            source.display().to_string(),
            "--title".to_string(),
            self.config.window.title.clone(),
            "--width".to_string(),
            output_width.to_string(),
            "--height".to_string(),
            output_height.to_string(),
            "--presenter".to_string(),
            self.config.presenter.live_window_name().to_string(),
            "--interop-fallback".to_string(),
            self.config.interop_fallback.live_window_name().to_string(),
            "--max-inflight".to_string(),
            self.config.max_inflight.to_string(),
            "--present-ring".to_string(),
            self.config.present_ring.to_string(),
            "--render-policy".to_string(),
            self.config.render_policy.live_window_name().to_string(),
        ];
        args.push("--mode".to_string());
        args.push(if draw_execution_mode {
            "draw-stress".to_string()
        } else if cuda_draw_mode {
            "instance-stress".to_string()
        } else if mesh_mode {
            "mesh-demo".to_string()
        } else {
            "kernel-throughput".to_string()
        });
        if let Some(fps) = self.config.fps.kernel {
            args.push("--kernel-target-fps".to_string());
            args.push(fps.to_string());
        }
        if let Some(fps) = self.config.fps.present {
            args.push("--present-target-fps".to_string());
            args.push(fps.to_string());
        }
        if draw_execution_mode {
            if let Some(DrawRunPlan::DrawExecution(plan) | DrawRunPlan::HardwareRaster(plan)) =
                &draw_plan
            {
                args.push("--instance-grid".to_string());
                args.push(plan.instances.grid.live_window_name());
                args.push("--instance-layout".to_string());
                args.push(plan.instances.layout.live_window_name().to_string());
                args.push("--draw-policy".to_string());
                args.push(plan.policy.label().to_string());
                args.push("--cull-order".to_string());
                args.push(plan.cull_order.live_window_name().to_string());
                args.push("--visibility".to_string());
                args.push(plan.visibility.live_window_name().to_string());
                args.push("--min-projected-pixels".to_string());
                args.push(format!(
                    "{:.3}",
                    plan.min_projected_millipixels as f32 / 1000.0
                ));
            }
        } else if let Some(DrawRunPlan::Cuda(plan)) = &draw_plan {
            args.push("--instance-grid".to_string());
            args.push(plan.instances.grid.live_window_name());
            args.push("--instance-layout".to_string());
            args.push(plan.instances.layout.live_window_name().to_string());
            args.push("--instance-stress-variant".to_string());
            args.push(plan.variant.live_window_name().to_string());
            args.push("--instance-debug-view".to_string());
            args.push("off".to_string());
            if !self.sparse_textures.is_empty() || !self.material_streams.is_empty() {
                args.push("--instance-materials".to_string());
                args.push("sparse-texture".to_string());
            }
            if self
                .sparse_textures
                .values()
                .any(|texture| texture.feedback)
            {
                args.push("--sparse-feedback".to_string());
                args.push("sampled".to_string());
            }
        }
        args.push(if self.config.hot_reload {
            "--hot-reload".to_string()
        } else {
            "--no-hot-reload".to_string()
        });
        Ok(args)
    }
}

#[derive(Debug, Clone)]
pub struct MeshBuilder {
    vertices: Vec<FriendlyVertex>,
    indices: Vec<u16>,
    layout: VertexLayout,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct FriendlyVertex {
    position: [f32; 3],
    color_bgra: u32,
}

pub type GeometryBuilder = MeshBuilder;

impl MeshBuilder {
    pub fn quad() -> Self {
        Self {
            vertices: vec![
                FriendlyVertex {
                    position: [-0.65, -0.65, 0.0],
                    color_bgra: 0xffff_4040,
                },
                FriendlyVertex {
                    position: [0.65, -0.65, 0.0],
                    color_bgra: 0xff40_ff40,
                },
                FriendlyVertex {
                    position: [0.65, 0.65, 0.0],
                    color_bgra: 0xff40_40ff,
                },
                FriendlyVertex {
                    position: [-0.65, 0.65, 0.0],
                    color_bgra: 0xffff_ff40,
                },
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            layout: friendly_vertex_layout(),
        }
    }

    pub fn cube() -> Self {
        let mut mesh = Self::quad();
        mesh.vertices = vec![
            FriendlyVertex {
                position: [-0.5, -0.5, -0.5],
                color_bgra: 0xffff_4040,
            },
            FriendlyVertex {
                position: [0.5, -0.5, -0.5],
                color_bgra: 0xff40_ff40,
            },
            FriendlyVertex {
                position: [0.5, 0.5, -0.5],
                color_bgra: 0xff40_40ff,
            },
            FriendlyVertex {
                position: [-0.5, 0.5, -0.5],
                color_bgra: 0xffff_ff40,
            },
            FriendlyVertex {
                position: [-0.5, -0.5, 0.5],
                color_bgra: 0xffff_40ff,
            },
            FriendlyVertex {
                position: [0.5, -0.5, 0.5],
                color_bgra: 0xff40_ffff,
            },
            FriendlyVertex {
                position: [0.5, 0.5, 0.5],
                color_bgra: 0xffffffff,
            },
            FriendlyVertex {
                position: [-0.5, 0.5, 0.5],
                color_bgra: 0xff80_80ff,
            },
        ];
        mesh.indices = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];
        mesh
    }

    pub fn colored(self) -> Self {
        self
    }

    pub fn vertices(mut self, vertices: &[[f32; 3]]) -> Self {
        self.vertices = vertices
            .iter()
            .enumerate()
            .map(|(idx, position)| FriendlyVertex {
                position: *position,
                color_bgra: default_color(idx),
            })
            .collect();
        self
    }

    pub fn indices_u16(mut self, indices: &[u16]) -> Self {
        self.indices = indices.to_vec();
        self
    }

    pub fn layout(mut self, layout: VertexLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn desc(&self) -> MeshBufferDesc {
        MeshBufferDesc {
            vertex_count: self.vertices.len() as u32,
            vertex_layout: self.layout.clone(),
            index_format: if self.indices.is_empty() {
                IndexFormat::None
            } else {
                IndexFormat::U16
            },
            index_count: self.indices.len() as u32,
            topology: PrimitiveTopology::TriangleList,
        }
    }

    pub fn build(&self, context: &Context) -> Result<MeshBuffer> {
        MeshBuffer::upload_typed(context, self.desc(), &self.vertices, &self.indices)
            .map_err(Into::into)
    }

    fn vertex_bytes(&self) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                self.vertices.as_ptr().cast::<u8>(),
                std::mem::size_of_val(self.vertices.as_slice()),
            )
            .to_vec()
        }
    }
}

impl From<MeshBuilder> for GeometryStreamSource {
    fn from(value: MeshBuilder) -> Self {
        Self::Builder(value)
    }
}

fn friendly_vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<FriendlyVertex>() as u32,
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
    }
}

fn default_color(idx: usize) -> u32 {
    const COLORS: [u32; 6] = [
        0xffff_4040,
        0xff40_ff40,
        0xff40_40ff,
        0xffff_ff40,
        0xffff_40ff,
        0xff40_ffff,
    ];
    COLORS[idx % COLORS.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_draw_plan_recipe<D: DrawPlanRecipe>(
        plan: &D,
        backend: DrawBackend,
        policy: DrawPolicy,
        source: &str,
    ) {
        assert_eq!(plan.backend(), backend);
        assert_eq!(plan.policy(), policy);
        assert_eq!(plan.policy_config().policy, policy);
        assert_eq!(plan.source(), Path::new(source));
        assert_eq!(plan.geometry().mesh, "quad");
        assert_eq!(plan.instances().name, "instances");
        assert_eq!(plan.material().name, "material");
        assert!(plan.target().dimensions().0 > 0);
        assert!(plan.target().dimensions().1 > 0);
        let contract = plan.contract();
        assert_eq!(contract.backend, backend);
        assert_eq!(contract.policy, policy);
        assert_eq!(contract.policy_config, plan.policy_config());
        assert_eq!(contract.material_label(), plan.material().name);
        assert_eq!(
            contract.material_abi_label(),
            Some(plan.material().abi_label())
        );
        assert_eq!(contract.instance_count, plan.instances().grid.count());
        assert_eq!(contract.instance_layout, Some(plan.instances().layout));
        assert_eq!(
            contract.instance_layout_label(),
            Some(plan.instances().layout.label())
        );
        assert_eq!(contract.target_width, Some(plan.target().dimensions().0));
        assert_eq!(contract.target_height, Some(plan.target().dimensions().1));
        assert_eq!(
            contract.target_dimensions(),
            Some(plan.target().dimensions())
        );
    }

    #[cfg(windows)]
    fn assert_runtime_draw_recipe<'a, D: RuntimeDrawRecipe<'a>>(
        draw: &D,
        backend: neo_runtime::DrawBackend,
        policy: neo_runtime::DrawPolicy,
    ) {
        assert_eq!(draw.backend(), backend);
        assert_eq!(draw.policy(), policy);
        assert_eq!(draw.policy_config().policy, policy);
        assert_eq!(draw.geometry().mesh().desc().vertex_count, 4);
        assert_eq!(draw.material().label(), "material");
        assert!(draw.instances().is_some());
        assert!(draw.target().width > 0);
        assert!(draw.target().height > 0);
        let contract = draw.contract();
        assert_eq!(contract.backend, backend);
        assert_eq!(contract.policy, policy);
        assert_eq!(contract.policy_config, draw.policy_config());
        assert_eq!(contract.backend_label(), backend.label());
        assert_eq!(contract.policy_label(), policy.label());
        assert_eq!(contract.geometry_vertex_count, 4);
        assert_eq!(contract.geometry_index_count, 6);
        assert_eq!(contract.material_kernel, "material");
        assert!(contract.instance_count.is_some());
    }

    #[test]
    fn builder_defaults_are_stable() {
        let app = NeoApp::new();
        assert_eq!(app.config().window.width, 960);
        assert_eq!(app.config().window.height, 540);
        assert_eq!(app.presenter_kind(), Presenter::D3d12Interop);
        assert!(app.config().hot_reload);
        assert_eq!(app.config().max_inflight, 8);
        assert_eq!(app.config().present_ring, 8);
        assert_eq!(app.config().render_policy, RenderPolicy::Auto);
    }

    #[test]
    fn invalid_sizes_and_fps_are_rejected() {
        let err = NeoApp::new().window("Neo", 0, 720).validate().unwrap_err();
        assert!(err.to_string().contains("width and height"));
        let err = NeoApp::new().target_fps(f32::NAN).validate().unwrap_err();
        assert!(err.to_string().contains("FPS caps"));
        let err = NeoApp::new().present_ring(0).validate().unwrap_err();
        assert!(err.to_string().contains("present_ring"));
    }

    #[test]
    fn presenter_fallback_policy_maps_to_live_window_args() {
        let args = NeoApp::new()
            .presenter(Presenter::D3d12Upload)
            .interop_fallback(InteropFallback::Fail)
            .render_policy(RenderPolicy::ForceRender)
            .kernel("image", "examples/live-window/live.neo")
            .live_window_args();
        assert!(args.windows(2).any(|pair| pair == ["--presenter", "d3d12"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--interop-fallback", "fail"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--render-policy", "force-render"])
        );
    }

    #[test]
    fn mesh_builder_quad_matches_manual_runtime_layout() {
        let desc = MeshBuilder::quad().colored().desc();
        assert_eq!(desc.vertex_count, 4);
        assert_eq!(desc.index_count, 6);
        assert_eq!(desc.index_format, IndexFormat::U16);
        assert_eq!(desc.vertex_layout, friendly_vertex_layout());
    }

    #[test]
    fn sparse_texture_and_material_builders_export_friendly_configs() {
        let grid = InstanceGrid::new(16, 16, 8);
        let app = NeoApp::new()
            .sparse_texture(
                "atlas",
                SparseTextureBuilder::new()
                    .virtual_size(4096, 2048)
                    .page_size(128)
                    .checker_pages()
                    .feedback(true),
            )
            .material_stream(
                "materials",
                MaterialStreamBuilder::procedural_grid_tiles(grid),
            )
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", grid)
            .cuda_tiled_material_kernel(
                "material",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .draw_cuda_tiled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            );
        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-stress-variant", "macrocell"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-materials", "sparse-texture"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--sparse-feedback", "sampled"])
        );
        let parts = app.into_parts();
        let atlas = parts.sparse_textures.get("atlas").unwrap();
        assert_eq!(atlas.virtual_width, 4096);
        assert_eq!(atlas.virtual_height, 2048);
        assert_eq!(atlas.page_size, 128);
        assert_eq!(atlas.physical_pages, 256);
        assert!(atlas.checker_pages);
        assert!(atlas.feedback);
        let materials = parts.material_streams.get("materials").unwrap();
        assert_eq!(materials.ids.len(), grid.count().unwrap() as usize);
        assert_eq!(materials.ids[0], 0);
    }

    #[test]
    fn fluent_builder_selects_mesh_demo_mode() {
        let args = NeoApp::new()
            .kernel("image", "examples/live-window/live.neo")
            .kernel("raster", "examples/mesh-buffer/raster.neo")
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .target_fps(240.0)
            .hot_reload(false)
            .live_window_args();
        assert!(args.windows(2).any(|pair| pair == ["--mode", "mesh-demo"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--kernel-target-fps", "240"])
        );
        assert!(args.iter().any(|arg| arg == "--no-hot-reload"));
    }

    #[test]
    fn fluent_builder_selects_image_throughput_for_quad_stress() {
        let args = NeoApp::new()
            .kernel("image", "examples/stress-quads/million_quads.neo")
            .target_fps(240.0)
            .hot_reload(false)
            .live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "kernel-throughput"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--title", "Neo"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--present-target-fps", "240"])
        );
    }

    #[test]
    fn fluent_builder_selects_draw_stress_for_compute_culled_draw() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_with_layout(
                "instances",
                InstanceGrid::new(32, 32, 8),
                InstanceLayout::AoSoA64,
            )
            .target("window-copy", 1280, 720)
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("window-copy"),
            )
            .target_fps(240.0)
            .hot_reload(false);
        app.validate().unwrap();
        let draw = app.draw_spec("main").unwrap();
        assert_eq!(draw.geometry.mesh, "quad");
        assert_eq!(draw.instances.as_ref().unwrap().name, "instances");
        assert_eq!(draw.target, TargetSpec::named("window-copy"));
        assert_eq!(draw.policy, DrawPolicy::ComputeCulled);
        assert_eq!(app.target_spec("window-copy").unwrap().width, 1280);
        let instances = app.instance_stream_spec("instances").unwrap();
        assert_eq!(instances.grid, InstanceGrid::new(32, 32, 8));
        assert_eq!(instances.layout, InstanceLayout::AoSoA64);
        let material = app.material_spec("material").unwrap();
        assert_eq!(material.vertex_entrypoint, "quad_vs");
        assert_eq!(material.fragment_entrypoint, "quad_fs");
        assert_eq!(material.abi, MaterialAbi::ComputeCulledInstanceColor);
        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "draw-stress"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--width", "1280"]));
        assert!(args.windows(2).any(|pair| pair == ["--height", "720"]));
        assert!(
            args.iter()
                .any(|arg| arg == "examples/stress-quads/hardware_raster.neo")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-grid", "32x32x8"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-layout", "aosoa64"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--draw-policy", "compute-culled"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--cull-order", "atomic-compact"])
        );
    }

    #[test]
    fn modern_draw_api_records_stream_material_target_and_policy() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(8, 8, 2))
            .target("main-target", 1280, 720)
            .direct_instance_material_kernel(
                "material",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .draw_all(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::named("main-target"),
            );

        let draw = app.draw_spec("main").unwrap();
        assert_eq!(draw.draw_name(), "main");
        assert_eq!(draw.geometry.mesh, "quad");
        assert_eq!(draw.geometry_stream().mesh, "quad");
        assert_eq!(draw.instances.as_ref().unwrap().name, "instances");
        assert_eq!(draw.instance_stream().unwrap().name, "instances");
        assert_eq!(draw.material, "material");
        assert_eq!(draw.material_kernel(), "material");
        assert_eq!(draw.target, TargetSpec::named("main-target"));
        assert_eq!(draw.target(), &TargetSpec::named("main-target"));
        assert_eq!(draw.policy, DrawPolicy::DrawAll);
        assert_eq!(draw.draw_policy(), DrawPolicy::DrawAll);
        assert_eq!(draw.policy_config().policy, DrawPolicy::DrawAll);
        assert_eq!(draw.backend(), DrawBackend::HardwareRaster);
        let contract = draw.contract();
        assert_eq!(contract.draw, "main");
        assert_eq!(contract.geometry_stream, "quad");
        assert_eq!(contract.instance_stream.as_deref(), Some("instances"));
        assert_eq!(contract.instance_count, None);
        assert_eq!(contract.instance_layout, None);
        assert_eq!(contract.instance_layout_label(), None);
        assert_eq!(contract.material_kernel, "material");
        assert_eq!(contract.material_label(), "material");
        assert_eq!(contract.material_abi_label(), None);
        assert_eq!(contract.target, "main-target");
        assert_eq!(contract.target_width, None);
        assert_eq!(contract.target_height, None);
        assert_eq!(contract.target_dimensions(), None);
        assert_eq!(contract.policy_label(), "draw-all");
        assert_eq!(contract.backend_label(), "hardware-raster");
        assert_eq!(app.draw_contract("main").unwrap(), contract);
        assert_eq!(
            app.geometry_stream_specs()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["quad"]
        );
        assert_eq!(
            app.instance_stream_specs()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["instances"]
        );
        assert_eq!(
            app.material_specs()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["material"]
        );
        assert_eq!(
            app.target_specs()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["main-target"]
        );
        assert_eq!(
            app.draw_specs()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["main"]
        );
        assert_eq!(
            app.draw_contracts()
                .map(|contract| contract.draw)
                .collect::<Vec<_>>(),
            vec!["main"]
        );
    }

    #[test]
    fn target_spec_window_helper_exports_window_target() {
        let plan = NeoApp::new()
            .window("Neo Target", 1234, 567)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_with_policy(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::window(),
                DrawPolicy::ComputeCulled,
            )
            .draw_execution_run_plan()
            .unwrap()
            .unwrap();

        assert_eq!(
            plan.target,
            TargetBinding::Window(WindowConfig {
                title: "Neo Target".to_string(),
                width: 1234,
                height: 567,
            })
        );
        assert_eq!(plan.target.dimensions(), (1234, 567));
        assert_eq!(plan.cull_order, CullOrder::AtomicCompact);
    }

    #[test]
    fn draw_policy_helpers_store_first_class_policy() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(4, 4, 4))
            .direct_instance_material_kernel(
                "direct",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .instance_material_kernel("culled", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_all(
                "draw-all",
                "quad",
                Some("instances"),
                "direct",
                TargetSpec::window(),
            )
            .draw_compute_culled(
                "draw-culled",
                "quad",
                "instances",
                "culled",
                TargetSpec::window(),
            );

        assert_eq!(
            app.draw_spec("draw-all").unwrap().policy,
            DrawPolicy::DrawAll
        );
        assert_eq!(DrawPolicy::DrawAll.backend(), DrawBackend::HardwareRaster);
        assert_eq!(DrawPolicy::DrawAll.label(), "draw-all");
        assert_eq!(DrawPolicy::DrawAll.to_string(), "draw-all");
        assert_eq!(
            DrawPolicy::ComputeCulled.backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(DrawPolicy::ComputeCulled.label(), "compute-culled");
        assert_eq!(DrawPolicy::CudaTiled.backend(), DrawBackend::CudaTiled);
        assert_eq!(DrawPolicy::CudaTiled.label(), "cuda-tiled");
        assert_eq!(DrawBackend::primary_neo(), DrawBackend::CudaTiled);
        assert!(DrawBackend::CudaTiled.is_primary_neo());
        assert!(!DrawBackend::HardwareRaster.is_primary_neo());
        assert_eq!(
            DrawBackendPreference::default(),
            DrawBackendPreference::PrimaryNeo
        );
        assert_eq!(DrawBackendPreference::PrimaryNeo.label(), "primary-neo");
        assert_eq!(DrawBackendPreference::PrimaryNeo.to_string(), "primary-neo");
        assert_eq!(
            DrawBackendPreference::PrimaryNeo.preferred_backend(),
            Some(DrawBackend::CudaTiled)
        );
        assert_eq!(
            DrawBackendPreference::DrawExecution.label(),
            "draw-execution"
        );
        assert_eq!(
            DrawBackendPreference::DrawExecution.to_string(),
            "draw-execution"
        );
        assert_eq!(
            DrawBackendPreference::DrawExecution.preferred_backend(),
            Some(DrawBackend::HardwareRaster)
        );
        assert_eq!(
            DrawBackendPreference::HardwareRaster.preferred_backend(),
            Some(DrawBackend::HardwareRaster)
        );
        assert_eq!(
            DrawBackendPreference::FirstConfigured.preferred_backend(),
            None
        );
        let legacy_preference: RendererPreference = DrawBackendPreference::HardwareRaster;
        assert_eq!(
            legacy_preference.preferred_backend(),
            Some(DrawBackend::HardwareRaster)
        );
        assert_eq!(DrawBackend::HardwareRaster.label(), "hardware-raster");
        assert_eq!(DrawBackend::HardwareRaster.to_string(), "hardware-raster");
        assert_eq!(DrawBackend::CudaTiled.label(), "cuda-tiled");
        assert_eq!(CullOrder::AtomicCompact.label(), "atomic-compact");
        assert_eq!(CullOrder::StableDense.to_string(), "stable-dense");
        let neutral_cull_order: CullOrder = CullOrder::StableDense;
        assert_eq!(neutral_cull_order.label(), "stable-dense");
        let legacy_cull_order: RasterCullOrder = RasterCullOrder::StableDense;
        assert_eq!(legacy_cull_order, neutral_cull_order);
        assert_eq!(VisibilityMode::Frustum.label(), "frustum");
        assert_eq!(VisibilityMode::ProjectedSize.to_string(), "projected-size");
        let neutral_visibility: VisibilityMode = VisibilityMode::ProjectedSize;
        assert_eq!(neutral_visibility.label(), "projected-size");
        let legacy_visibility: RasterVisibilityMode = RasterVisibilityMode::ProjectedSize;
        assert_eq!(legacy_visibility, neutral_visibility);
        assert_eq!(DrawDepthMode::Auto.label(), "auto");
        assert_eq!(DrawDepthMode::Auto.to_string(), "auto");
        assert!(!DrawDepthMode::Auto.uses_depth(DrawPolicy::DrawAll));
        assert!(DrawDepthMode::Auto.uses_depth(DrawPolicy::ComputeCulled));
        assert!(DrawDepthMode::On.uses_depth(DrawPolicy::DrawAll));
        assert!(!DrawDepthMode::Off.uses_depth(DrawPolicy::ComputeCulled));
        assert_eq!(
            DEFAULT_MIN_PROJECTED_MILLIPIXELS,
            DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
        );
        assert_eq!(
            app.draw_spec("draw-all").unwrap().policy_config,
            DrawPolicyConfig::draw_all()
        );
        assert_eq!(
            DrawPolicyConfig::draw_all().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(DrawPolicyConfig::draw_all().policy_label(), "draw-all");
        assert_eq!(
            DrawPolicyConfig::draw_all().backend_label(),
            "hardware-raster"
        );
        assert_eq!(DrawPolicyConfig::draw_all().depth_label(), "auto");
        assert!(!DrawPolicyConfig::draw_all().uses_depth());
        assert!(DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).uses_depth());
        assert!(
            DrawPolicyConfig::draw_all()
                .with_depth(DrawDepthMode::On)
                .uses_depth()
        );
        assert_eq!(
            DrawPolicyConfig::draw_all().cull_order_label(),
            "stable-dense"
        );
        assert_eq!(DrawPolicyConfig::draw_all().visibility_label(), "frustum");
        assert_eq!(
            app.draw_spec("draw-all").unwrap().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            app.draw_spec("draw-culled").unwrap().policy,
            DrawPolicy::ComputeCulled
        );
        assert_eq!(
            app.draw_spec("draw-culled").unwrap().policy_config,
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact)
        );
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).policy_label(),
            "compute-culled"
        );
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).cull_order_label(),
            "atomic-compact"
        );
        assert_eq!(
            DrawPolicyConfig::cuda_tiled().backend(),
            DrawBackend::CudaTiled
        );
        assert_eq!(DrawPolicyConfig::cuda_tiled().policy_label(), "cuda-tiled");
        assert_eq!(
            DrawPolicyConfig::cuda_tiled().visibility_label(),
            "projected-size"
        );
        assert_eq!(DrawPolicyConfig::cuda_tiled().min_projected_pixels(), 0.85);
        assert_eq!(
            app.draw_spec("draw-culled").unwrap().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            app.draw_spec("draw-culled")
                .unwrap()
                .policy_config
                .visibility,
            VisibilityMode::Frustum
        );
        assert_eq!(
            app.draw_spec("draw-culled")
                .unwrap()
                .policy_config
                .min_projected_millipixels,
            DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
        );
        let contract = app.draw_spec("draw-culled").unwrap().contract();
        assert_eq!(contract.cull_order_label(), "atomic-compact");
        assert_eq!(contract.visibility_label(), "frustum");
        assert_eq!(contract.min_projected_pixels(), 0.85);
    }

    #[test]
    fn compute_cull_order_is_an_explicit_draw_policy_knob() {
        let plan = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull_with_order(
                "raster_cull",
                "examples/stress-quads/hardware_raster.neo",
                CullOrder::StableDense,
            )
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .draw_execution_run_plan()
            .unwrap()
            .unwrap();

        let _neutral_plan: DrawExecutionRunPlan = plan.clone();
        assert_eq!(plan.cull_order, CullOrder::StableDense);
        assert_eq!(plan.visibility, VisibilityMode::Frustum);
        assert_eq!(
            plan.min_projected_millipixels,
            DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
        );
        let live_plan: neo_live_window::DrawExecutionPlan = plan.live_draw_plan();
        assert_eq!(
            live_plan.cull_order,
            neo_live_window::DrawCullOrder::StableDense
        );
        assert_eq!(plan.live_draw_plan(), plan.live_window_plan());
    }

    #[test]
    fn draw_policy_config_is_visible_from_draw_to_raster_plan() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_with_policy_config(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::window(),
                DrawPolicyConfig::compute_culled_with_visibility(
                    CullOrder::StableDense,
                    VisibilityMode::ProjectedSize,
                )
                .with_min_projected_millipixels(500)
                .with_depth(DrawDepthMode::Off),
            );

        let spec = app.draw_spec("main").unwrap();
        assert_eq!(spec.policy, DrawPolicy::ComputeCulled);
        assert_eq!(spec.backend(), DrawBackend::HardwareRaster);
        assert_eq!(
            spec.policy_config,
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize
            )
            .with_min_projected_millipixels(500)
            .with_depth(DrawDepthMode::Off)
        );

        let graph = app.draw_graph().unwrap();
        assert_eq!(
            graph.draws[0].policy_config(),
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize
            )
            .with_min_projected_millipixels(500)
            .with_depth(DrawDepthMode::Off)
        );
        assert_eq!(graph.draws[0].policy, DrawPolicy::ComputeCulled);
        assert_eq!(graph.draws[0].backend(), DrawBackend::HardwareRaster);
        assert_eq!(graph.draws[0].depth, DrawDepthMode::Off);
        assert_eq!(graph.draws[0].cull_order, CullOrder::StableDense);
        assert_eq!(graph.draws[0].visibility, VisibilityMode::ProjectedSize);

        let plan = app.draw_execution_run_plan().unwrap().unwrap();
        assert_eq!(plan.policy, DrawPolicy::ComputeCulled);
        assert_eq!(plan.backend(), DrawBackend::HardwareRaster);
        assert_eq!(plan.policy_config.depth, DrawDepthMode::Off);
        assert_eq!(plan.cull_order, CullOrder::StableDense);
        assert_eq!(plan.visibility, VisibilityMode::ProjectedSize);
        assert_eq!(plan.min_projected_millipixels, 500);

        let contract = spec.contract();
        assert_eq!(contract.policy_config, spec.policy_config);
        assert_eq!(contract.depth_label(), "off");
        assert!(!contract.uses_depth());

        assert_eq!(
            plan.live_draw_plan().visibility,
            neo_live_window::HardwareRasterVisibilityMode::ProjectedSize
        );
        assert_eq!(
            plan.live_draw_plan().depth,
            neo_live_window::DrawDepthMode::Off
        );
        assert_eq!(plan.live_draw_plan().min_projected_millipixels, 500);
    }

    #[test]
    fn projected_cull_draw_helper_exports_optimized_draw_policy() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull_with_order(
                "raster_cull",
                "examples/stress-quads/hardware_raster.neo",
                CullOrder::StableDense,
            )
            .draw_compute_culled_projected(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
                850,
            );

        let spec = app.draw_spec("main").unwrap();
        assert_eq!(spec.backend(), DrawBackend::HardwareRaster);
        assert_eq!(
            spec.policy_config,
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize
            )
            .with_min_projected_millipixels(850)
        );

        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--draw-policy", "compute-culled"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--cull-order", "stable-dense"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--visibility", "projected-size"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--min-projected-pixels", "0.850"])
        );
    }

    #[test]
    fn cuda_tiled_draw_policy_uses_primary_cuda_instance_path() {
        let app = NeoApp::new()
            .window("Neo CUDA Draw", 960, 540)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(8, 8, 4))
            .cuda_tiled_material_kernel(
                "material",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .draw_cuda_tiled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            );

        let spec = app.draw_spec("main").unwrap();
        assert_eq!(spec.policy, DrawPolicy::CudaTiled);
        assert_eq!(spec.policy_config, DrawPolicyConfig::cuda_tiled());
        assert_eq!(spec.backend(), DrawBackend::CudaTiled);

        let cuda_plan = app.cuda_run_plan().unwrap().unwrap();
        assert_eq!(cuda_plan.draw_name, "main");
        assert_eq!(
            cuda_plan.source,
            PathBuf::from("examples/stress-quads/three_d_instances.neo")
        );
        assert_eq!(cuda_plan.entrypoint, "instance_raster");
        assert_eq!(cuda_plan.instances.grid, InstanceGrid::new(8, 8, 4));
        assert_eq!(cuda_plan.target.dimensions(), (960, 540));
        assert_draw_plan_recipe(
            &cuda_plan,
            DrawBackend::CudaTiled,
            DrawPolicy::CudaTiled,
            "examples/stress-quads/three_d_instances.neo",
        );
        assert!(app.draw_execution_run_plan().unwrap().is_none());

        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "instance-stress"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-stress-variant", "tiled"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-grid", "8x8x4"])
        );
        assert!(
            !args
                .windows(2)
                .any(|pair| pair == ["--mode", "draw-stress"])
        );
        assert!(!args.iter().any(|arg| arg == "--draw-policy"));
    }

    #[test]
    fn draw_run_plan_selects_policy_backend_from_primary_draw() {
        let app = NeoApp::new()
            .window("Neo CUDA Draw", 960, 540)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(8, 8, 4))
            .cuda_tiled_material_kernel(
                "material",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .draw_cuda_tiled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            );

        let plan = app.draw_run_plan().unwrap().unwrap();
        assert_eq!(plan.draw_name(), "main");
        assert_eq!(
            plan.source(),
            Path::new("examples/stress-quads/three_d_instances.neo")
        );
        assert_eq!(plan.target().dimensions(), (960, 540));
        assert_eq!(plan.policy(), DrawPolicy::CudaTiled);
        assert_eq!(plan.policy_config(), DrawPolicyConfig::cuda_tiled());
        assert_eq!(plan.geometry().mesh, "quad");
        assert_eq!(plan.instances().name, "instances");
        assert_eq!(plan.instances().layout, InstanceLayout::AoSoA32);
        assert_eq!(plan.material().name, "material");
        assert_eq!(plan.material().kernel_entrypoint(), Some("instance_raster"));
        assert_eq!(plan.backend(), DrawBackend::CudaTiled);
        let contract = plan.contract();
        assert_eq!(contract.draw, "main");
        assert_eq!(contract.geometry_stream, "quad");
        assert_eq!(contract.instance_stream.as_deref(), Some("instances"));
        assert_eq!(contract.material_kernel, "material");
        assert_eq!(
            contract.material_abi_label(),
            Some("cuda-tiled-instance-color")
        );
        assert_eq!(contract.target, "window");
        assert_eq!(contract.policy_config, DrawPolicyConfig::cuda_tiled());
        assert_eq!(contract.policy_label(), "cuda-tiled");
        assert_eq!(contract.backend_label(), "cuda-tiled");
        assert_draw_plan_recipe(
            &plan,
            DrawBackend::CudaTiled,
            DrawPolicy::CudaTiled,
            "examples/stress-quads/three_d_instances.neo",
        );
        assert!(plan.is_cuda_tiled());
        assert!(plan.is_cuda_plan());
        assert!(!plan.is_draw_execution_plan());
        assert!(!plan.is_draw_execution());
        assert_eq!(plan.as_cuda_plan().unwrap().entrypoint, "instance_raster");
        assert!(plan.as_draw_execution_plan().is_none());
        assert!(!plan.is_hardware_raster());
    }

    #[test]
    fn draw_run_plan_exposes_complete_draw_vocabulary_for_hardware_raster() {
        let app = NeoApp::new()
            .window("Neo Raster Draw", 1024, 768)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa64("instances", InstanceGrid::new(16, 16, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull_with_order(
                "raster_cull",
                "examples/stress-quads/hardware_raster.neo",
                CullOrder::StableDense,
            )
            .draw_compute_culled_projected(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
                850,
            );

        let plan = app.draw_run_plan().unwrap().unwrap();
        assert_eq!(plan.draw_name(), "main");
        assert_eq!(plan.geometry().mesh, "quad");
        assert_eq!(plan.instances().name, "instances");
        assert_eq!(plan.instances().layout, InstanceLayout::AoSoA64);
        assert_eq!(plan.material().name, "material");
        assert_eq!(plan.target().dimensions(), (1024, 768));
        assert_eq!(plan.policy(), DrawPolicy::ComputeCulled);
        assert_eq!(
            plan.policy_config(),
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize,
            )
            .with_min_projected_millipixels(850)
        );
        assert_eq!(plan.backend(), DrawBackend::HardwareRaster);
        assert!(matches!(plan, DrawRunPlan::DrawExecution(_)));
        let contract = plan.contract();
        assert_eq!(contract.draw, "main");
        assert_eq!(contract.geometry_stream, "quad");
        assert_eq!(contract.instance_stream.as_deref(), Some("instances"));
        assert_eq!(contract.material_kernel, "material");
        assert_eq!(
            contract.material_abi_label(),
            Some("compute-culled-instance-color")
        );
        assert_eq!(contract.target, "window");
        assert_eq!(contract.policy_config, plan.policy_config());
        assert_eq!(contract.policy_label(), "compute-culled");
        assert_eq!(contract.backend_label(), "hardware-raster");
        assert_draw_plan_recipe(
            &plan,
            DrawBackend::HardwareRaster,
            DrawPolicy::ComputeCulled,
            "examples/stress-quads/hardware_raster.neo",
        );
        assert!(!plan.is_cuda_tiled());
        assert!(!plan.is_cuda_plan());
        assert!(plan.is_draw_execution_plan());
        assert!(plan.is_draw_execution());
        assert_eq!(
            plan.as_draw_execution_plan()
                .unwrap()
                .compute_cull
                .as_ref()
                .unwrap()
                .entrypoint,
            "raster_cull"
        );
        assert!(plan.as_cuda_plan().is_none());
        assert!(plan.is_hardware_raster());
        let legacy_plan =
            DrawRunPlan::HardwareRaster(plan.as_draw_execution_plan().unwrap().clone());
        assert!(legacy_plan.is_draw_execution());
        assert_eq!(legacy_plan.backend(), plan.backend());
    }

    #[test]
    fn live_window_args_follow_draw_policy_not_backend_probe_order() {
        let app = NeoApp::new()
            .window("Neo Mixed Draw", 960, 540)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(8, 8, 4))
            .cuda_tiled_material_kernel(
                "cuda-material",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .instance_material_kernel("raster-material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_cuda_tiled(
                "a-primary-cuda",
                "quad",
                "instances",
                "cuda-material",
                TargetSpec::window(),
            )
            .draw_compute_culled(
                "z-comparison-raster",
                "quad",
                "instances",
                "raster-material",
                TargetSpec::window(),
            );

        let plan = app.draw_run_plan().unwrap().unwrap();
        assert_eq!(plan.backend(), DrawBackend::CudaTiled);

        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "instance-stress"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-stress-variant", "tiled"])
        );
        assert!(
            args.iter()
                .any(|arg| arg == "examples/stress-quads/three_d_instances.neo")
        );
        assert!(!args.iter().any(|arg| arg == "--draw-policy"));
    }

    #[test]
    fn draw_run_plan_prefers_primary_cuda_over_configured_order() {
        fn mixed_app() -> NeoApp {
            NeoApp::new()
                .window("Neo Mixed Draw", 960, 540)
                .geometry_stream("quad", MeshBuilder::quad().colored())
                .instance_stream_aosoa32("instances", InstanceGrid::new(8, 8, 4))
                .cuda_tiled_material_kernel(
                    "cuda-material",
                    "instance_raster",
                    "examples/stress-quads/three_d_instances.neo",
                )
                .instance_material_kernel("raster-material", "quad_vs", "quad_fs")
                .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
                .draw_compute_culled(
                    "a-comparison-raster",
                    "quad",
                    "instances",
                    "raster-material",
                    TargetSpec::window(),
                )
                .draw_cuda_tiled(
                    "z-primary-cuda",
                    "quad",
                    "instances",
                    "cuda-material",
                    TargetSpec::window(),
                )
        }

        let app = mixed_app();

        let plan = app.draw_run_plan().unwrap().unwrap();
        assert_eq!(plan.draw_name(), "z-primary-cuda");
        assert_eq!(plan.backend(), DrawBackend::CudaTiled);
        assert!(plan.backend().is_primary_neo());

        let hardware_plan = mixed_app()
            .draw_backend_preference(DrawBackendPreference::DrawExecution)
            .draw_run_plan()
            .unwrap()
            .unwrap();
        assert_eq!(hardware_plan.draw_name(), "a-comparison-raster");
        assert_eq!(hardware_plan.backend(), DrawBackend::HardwareRaster);

        let legacy_hardware_plan = mixed_app()
            .renderer_preference(RendererPreference::HardwareRaster)
            .draw_run_plan()
            .unwrap()
            .unwrap();
        assert_eq!(legacy_hardware_plan.draw_name(), "a-comparison-raster");
        assert_eq!(legacy_hardware_plan.backend(), DrawBackend::HardwareRaster);

        let ordered_plan = mixed_app()
            .draw_backend_preference(DrawBackendPreference::FirstConfigured)
            .draw_run_plan()
            .unwrap()
            .unwrap();
        assert_eq!(ordered_plan.draw_name(), "a-comparison-raster");
        assert_eq!(ordered_plan.backend(), DrawBackend::HardwareRaster);
    }

    #[test]
    fn cuda_tiled_draw_policy_rejects_hardware_raster_material() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(8, 8, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .draw_cuda_tiled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .draw_graph()
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("is not a CUDA tiled MaterialKernel")
        );
    }

    #[test]
    fn material_kernel_builders_are_primary_draw_vocabulary() {
        let app = NeoApp::new()
            .material_kernel("plain", "plain_vs", "plain_fs")
            .instance_material_kernel("culled", "quad_vs", "quad_fs")
            .direct_instance_material_kernel(
                "direct",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .draw_execution_material_kernel("draw-plain", "plain_vs", "plain_fs")
            .draw_execution_instance_material_kernel("draw-culled", "quad_vs", "quad_fs")
            .direct_draw_execution_instance_material_kernel(
                "draw-direct",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .cuda_tiled_material_kernel(
                "cuda",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .raster_pipeline("legacy-plain", "plain_vs", "plain_fs")
            .raster_instance_material("legacy-culled", "quad_vs", "quad_fs")
            .raster_direct_instance_material(
                "legacy-direct",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            );

        assert_eq!(
            app.material_spec("plain").unwrap().abi,
            MaterialAbi::SimpleColor
        );
        assert_eq!(app.material_spec("plain").unwrap().label(), "plain");
        assert_eq!(
            app.material_spec("plain").unwrap().abi_label(),
            "simple-color"
        );
        assert_eq!(MaterialAbi::SimpleColor.to_string(), "simple-color");
        assert_eq!(
            MaterialAbi::SimpleColor.execution_kind(),
            MaterialExecutionKind::DrawExecution
        );
        assert_eq!(
            MaterialAbi::SimpleColor.execution_kind_label(),
            "draw-execution"
        );
        assert_eq!(
            MaterialExecutionKind::DrawExecution.backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            MaterialExecutionKind::DrawExecution.to_string(),
            "draw-execution"
        );
        assert_eq!(
            app.material_spec("plain").unwrap().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            app.material_spec("plain").unwrap().execution_kind(),
            MaterialExecutionKind::DrawExecution
        );
        assert_eq!(
            app.material_spec("plain").unwrap().execution_kind_label(),
            "draw-execution"
        );
        assert_eq!(
            app.material_spec("culled").unwrap().abi,
            MaterialAbi::ComputeCulledInstanceColor
        );
        assert_eq!(
            app.material_spec("culled").unwrap().abi_label(),
            "compute-culled-instance-color"
        );
        assert_eq!(
            app.material_spec("culled").unwrap().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            app.material_spec("direct").unwrap().abi,
            MaterialAbi::DirectInstanceColor
        );
        assert_eq!(
            app.material_spec("direct").unwrap().abi_label(),
            "direct-instance-color"
        );
        assert_eq!(
            app.material_spec("direct").unwrap().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().abi,
            MaterialAbi::CudaTiledInstanceColor
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().abi_label(),
            "cuda-tiled-instance-color"
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().backend(),
            DrawBackend::CudaTiled
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().execution_kind(),
            MaterialExecutionKind::CudaTiled
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().execution_kind_label(),
            "cuda-tiled"
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().kernel_entrypoint(),
            Some("instance_raster")
        );
        assert_eq!(app.material_spec("cuda").unwrap().vertex_entrypoint(), None);
        assert_eq!(
            app.material_spec("cuda").unwrap().fragment_entrypoint(),
            None
        );
        assert_eq!(
            app.material_spec("cuda").unwrap().source_path(),
            Some(Path::new("examples/stress-quads/three_d_instances.neo"))
        );
        assert!(app.material_spec("cuda").unwrap().is_cuda_tiled());
        assert!(!app.material_spec("cuda").unwrap().is_draw_execution());
        assert!(app.material_spec("culled").unwrap().is_draw_execution());
        assert_eq!(
            app.material_spec("draw-plain").unwrap().abi,
            app.material_spec("plain").unwrap().abi
        );
        assert_eq!(
            app.material_spec("draw-culled").unwrap().abi,
            app.material_spec("culled").unwrap().abi
        );
        assert_eq!(
            app.material_spec("draw-direct").unwrap().source,
            app.material_spec("direct").unwrap().source
        );
        assert_eq!(
            app.material_spec("draw-direct").unwrap().execution_kind(),
            MaterialExecutionKind::DrawExecution
        );
        assert_eq!(
            app.material_spec("legacy-culled").unwrap().abi,
            app.material_spec("culled").unwrap().abi
        );
        assert_eq!(
            app.material_spec("legacy-direct").unwrap().source,
            app.material_spec("direct").unwrap().source
        );
    }

    #[test]
    fn geometry_stream_config_is_primary_mesh_compatibility_vocabulary() {
        let app = NeoApp::new()
            .geometry_stream("quad", GeometryBuilder::quad().colored())
            .mesh("legacy-quad", MeshBuilder::quad().colored());

        assert!(app.geometry_stream_spec("quad").is_some());
        assert!(app.geometry_stream_spec("legacy-quad").is_some());
        assert!(app.mesh_spec("quad").is_some());
        assert!(app.mesh_spec("legacy-quad").is_some());
        let stream_config: &GeometryStreamConfig = app.geometry_stream_spec("quad").unwrap();
        let legacy_mesh_spec: &MeshSpec = app.mesh_spec("legacy-quad").unwrap();
        let geometry_builder_desc = GeometryBuilder::quad().colored().desc();
        let mesh_builder_desc = MeshBuilder::quad().colored().desc();
        let _stream_source: &GeometryStreamSource = &stream_config.source;
        let _legacy_mesh_source: &MeshSource = &legacy_mesh_spec.source;
        assert_eq!(
            geometry_builder_desc.vertex_count,
            mesh_builder_desc.vertex_count
        );
        assert_eq!(
            stream_config.vertex_bytes(),
            legacy_mesh_spec.vertex_bytes()
        );
    }

    #[test]
    fn draw_indirect_helpers_are_compatibility_spelling() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_indirect_with_policy(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::window(),
                DrawPolicy::ComputeCulled,
            );

        let draw = app.draw_spec("main").unwrap();
        let indirect = app.indirect_draw_spec("main").unwrap();
        assert_eq!(indirect, draw);
        assert_eq!(indirect.policy, DrawPolicy::ComputeCulled);
        assert_eq!(indirect.backend(), DrawBackend::HardwareRaster);
    }

    #[test]
    fn instance_stream_aosoa_helpers_are_explicit_layout_vocabulary() {
        let app = NeoApp::new()
            .instance_stream("default", InstanceGrid::new(1, 2, 3))
            .instance_stream_aosoa32("aosoa32", InstanceGrid::new(4, 5, 6))
            .instance_stream_aosoa64("aosoa64", InstanceGrid::new(7, 8, 9))
            .try_instance_stream_aosoa("try64", InstanceGrid::new(2, 2, 2), 64)
            .unwrap();

        assert_eq!(
            app.instance_stream_spec("default").unwrap().layout,
            InstanceLayout::AoSoA32
        );
        assert_eq!(
            app.instance_stream_spec("aosoa32").unwrap().layout,
            InstanceLayout::AoSoA32
        );
        assert_eq!(
            app.instance_stream_spec("aosoa64").unwrap().layout,
            InstanceLayout::AoSoA64
        );
        assert_eq!(
            app.instance_stream_spec("try64").unwrap().layout,
            InstanceLayout::AoSoA64
        );
        assert_eq!(InstanceLayout::AoSoA32.label(), "aosoa32");
        assert_eq!(InstanceLayout::AoSoA32.to_string(), "aosoa32");
        assert_eq!(InstanceLayout::AoSoA64.label(), "aosoa64");
        assert_eq!(InstanceLayout::AoSoA64.to_string(), "aosoa64");

        let err =
            match NeoApp::new().try_instance_stream_aosoa("bad", InstanceGrid::new(1, 1, 1), 48) {
                Ok(_) => panic!("expected unsupported AoSoA group size error"),
                Err(err) => err,
            };
        assert!(err.to_string().contains("expected 32 or 64"));
    }

    #[test]
    fn fluent_builder_exports_draw_all_raster_policy() {
        let args = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .direct_instance_material_kernel(
                "material",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .draw_all(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::Window,
            )
            .live_window_args();

        assert!(
            args.windows(2)
                .any(|pair| pair == ["--draw-policy", "draw-all"])
        );
        assert!(
            args.iter()
                .any(|arg| arg == "examples/stress-quads/hardware_raster.neo")
        );
    }

    #[test]
    fn draw_graph_exports_resolved_draw_vocabulary() {
        let graph = NeoApp::new()
            .window("graph-window", 800, 600)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_with_layout(
                "instances",
                InstanceGrid::new(32, 32, 8),
                InstanceLayout::AoSoA64,
            )
            .target("hdr-target", 1280, 720)
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("hdr-target"),
            )
            .draw_graph()
            .unwrap();

        assert_eq!(graph.draws.len(), 1);
        assert_eq!(graph.draws().len(), 1);
        let draw = graph.draw("main").unwrap();
        assert_eq!(draw.draw_name(), "main");
        assert_eq!(draw.name, "main");
        assert_eq!(draw.geometry.mesh, "quad");
        assert_eq!(draw.geometry_stream().mesh, "quad");
        assert_eq!(draw.geometry.indices_u16, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            draw.instances.as_ref().unwrap().grid,
            InstanceGrid::new(32, 32, 8)
        );
        assert_eq!(
            draw.instance_stream().unwrap().grid,
            InstanceGrid::new(32, 32, 8)
        );
        assert_eq!(
            draw.instances.as_ref().unwrap().layout,
            InstanceLayout::AoSoA64
        );
        assert_eq!(draw.material.vertex_entrypoint, "quad_vs");
        assert_eq!(draw.material_kernel().name, "material");
        assert_eq!(draw.material.fragment_entrypoint, "quad_fs");
        assert_eq!(draw.material.abi, MaterialAbi::ComputeCulledInstanceColor);
        assert_eq!(
            draw.target,
            TargetBinding::Named(TargetConfig {
                name: "hdr-target".to_string(),
                width: 1280,
                height: 720,
            })
        );
        assert_eq!(draw.target().name(), "hdr-target");
        assert_eq!(draw.target().dimensions(), (1280, 720));
        assert_eq!(draw.policy, DrawPolicy::ComputeCulled);
        assert_eq!(draw.draw_policy(), DrawPolicy::ComputeCulled);
        assert_eq!(draw.policy_config().policy, DrawPolicy::ComputeCulled);
        let contract = draw.contract();
        assert_eq!(
            graph
                .contracts()
                .map(|contract| contract.draw)
                .collect::<Vec<_>>(),
            vec!["main"]
        );
        assert_eq!(contract.draw, "main");
        assert_eq!(contract.geometry_stream, "quad");
        assert_eq!(contract.instance_stream.as_deref(), Some("instances"));
        assert_eq!(contract.instance_count, Some(32 * 32 * 8));
        assert_eq!(contract.instance_layout, Some(InstanceLayout::AoSoA64));
        assert_eq!(contract.instance_layout_label(), Some("aosoa64"));
        assert_eq!(contract.material_kernel, "material");
        assert_eq!(
            contract.material_abi_label(),
            Some("compute-culled-instance-color")
        );
        assert_eq!(contract.target, "hdr-target");
        assert_eq!(contract.target_width, Some(1280));
        assert_eq!(contract.target_height, Some(720));
        assert_eq!(contract.target_dimensions(), Some((1280, 720)));
        assert_eq!(contract.policy_label(), "compute-culled");
        assert_eq!(contract.backend_label(), "hardware-raster");
        assert_eq!(
            draw.compute_cull.as_ref().unwrap().path,
            PathBuf::from("examples/stress-quads/hardware_raster.neo")
        );
        assert_eq!(
            draw.compute_cull().unwrap().path,
            PathBuf::from("examples/stress-quads/hardware_raster.neo")
        );

        let err = graph.draw("missing").unwrap_err();
        assert!(err.to_string().contains("draw `missing` was not found"));
    }

    #[test]
    fn draw_graph_exports_window_target() {
        let graph = NeoApp::new()
            .window("main-window", 1024, 768)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .draw_graph()
            .unwrap();

        assert_eq!(
            graph.draws[0].target,
            TargetBinding::Window(WindowConfig {
                title: "main-window".to_string(),
                width: 1024,
                height: 768,
            })
        );
    }

    #[test]
    fn draw_execution_run_plan_resolves_from_draw_graph() {
        let app = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_with_layout(
                "instances",
                InstanceGrid::new(8, 8, 4),
                InstanceLayout::AoSoA64,
            )
            .target("swap-target", 1600, 900)
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("swap-target"),
            );
        let plan = app.draw_execution_run_plan().unwrap().unwrap();
        let legacy_plan = app.hardware_raster_run_plan().unwrap().unwrap();

        assert_eq!(plan.draw_name, "main");
        assert_eq!(legacy_plan.source, plan.source);
        assert_eq!(legacy_plan.policy_config, plan.policy_config);
        assert_eq!(
            plan.source,
            PathBuf::from("examples/stress-quads/hardware_raster.neo")
        );
        assert_eq!(plan.geometry.mesh, "quad");
        assert_eq!(plan.geometry.indices_u16, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(plan.instances.grid, InstanceGrid::new(8, 8, 4));
        assert_eq!(plan.instances.layout, InstanceLayout::AoSoA64);
        assert_eq!(plan.material.name, "material");
        assert_eq!(plan.target.dimensions(), (1600, 900));
        assert_eq!(plan.policy, DrawPolicy::ComputeCulled);
        assert_draw_plan_recipe(
            &plan,
            DrawBackend::HardwareRaster,
            DrawPolicy::ComputeCulled,
            "examples/stress-quads/hardware_raster.neo",
        );
        assert_eq!(
            plan.compute_cull.as_ref().unwrap().entrypoint,
            "raster_cull"
        );
    }

    #[test]
    fn draw_execution_run_plan_exports_live_window_hardware_plan() {
        let plan = NeoApp::new()
            .window("Neo Raster", 1024, 768)
            .geometry_stream("quad-mesh", MeshBuilder::quad().colored())
            .instance_stream_with_layout(
                "quad-instances",
                InstanceGrid::new(16, 16, 4),
                InstanceLayout::AoSoA32,
            )
            .target("swap-target", 1600, 900)
            .instance_material_kernel("lit-quads", "lit_quad_vs", "lit_quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main-draw",
                "quad-mesh",
                "quad-instances",
                "lit-quads",
                TargetSpec::named("swap-target"),
            )
            .raster_run_plan()
            .unwrap()
            .unwrap()
            .live_draw_plan();

        assert_eq!(plan.draw_name, "main-draw");
        assert_eq!(plan.geometry_stream.name, "quad-mesh");
        assert_eq!(plan.geometry_stream.indices_u16, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(plan.instance_stream.name, "quad-instances");
        assert_eq!(plan.instance_stream.grid.x(), 16);
        assert_eq!(plan.instance_stream.grid.y(), 16);
        assert_eq!(plan.instance_stream.grid.z(), 4);
        assert_eq!(
            plan.instance_stream.layout,
            neo_live_window::StressInstanceLayout::AoSoA32
        );
        assert_eq!(plan.target.name, "swap-target");
        assert_eq!(plan.target.width, 1600);
        assert_eq!(plan.target.height, 900);
        assert_eq!(plan.material.name, "lit-quads");
        assert_eq!(plan.material.vertex_entrypoint, "lit_quad_vs");
        assert_eq!(plan.material.fragment_entrypoint, "lit_quad_fs");
        assert_eq!(
            plan.draw_policy,
            neo_live_window::HardwareRasterDrawPolicy::ComputeCulled
        );
    }

    #[test]
    fn raster_window_target_exports_stable_live_window_target_name() {
        let plan = NeoApp::new()
            .window("Neo Raster", 1024, 768)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .raster_run_plan()
            .unwrap()
            .unwrap()
            .live_draw_plan();

        assert_eq!(plan.target.name, "window");
        assert_eq!(plan.target.width, 1024);
        assert_eq!(plan.target.height, 768);
    }

    #[test]
    fn draw_execution_run_plan_exports_draw_all_direct_material() {
        let plan = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .direct_instance_material_kernel(
                "material",
                "quad_vs_direct",
                "quad_fs",
                "examples/stress-quads/hardware_raster.neo",
            )
            .draw_all(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::Window,
            )
            .raster_run_plan()
            .unwrap()
            .unwrap();
        assert_eq!(plan.policy, DrawPolicy::DrawAll);
        assert!(plan.compute_cull.is_none());
        assert_eq!(
            plan.source,
            PathBuf::from("examples/stress-quads/hardware_raster.neo")
        );
        let live_plan = plan.live_draw_plan();
        assert_eq!(
            live_plan.draw_policy,
            neo_live_window::DrawPolicyPlan::DrawAll
        );
        assert_eq!(
            live_plan.material.kind,
            neo_live_window::MaterialKernelPlanKind::DirectInstanceColor
        );
    }

    #[test]
    fn named_target_dimensions_drive_raster_live_window_args() {
        let args = NeoApp::new()
            .window("small-control-window", 640, 360)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream("instances", InstanceGrid::new(4, 4, 4))
            .target("render-target", 1920, 1080)
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("render-target"),
            )
            .live_window_args();

        assert!(args.windows(2).any(|pair| pair == ["--width", "1920"]));
        assert!(args.windows(2).any(|pair| pair == ["--height", "1080"]));
    }

    #[test]
    fn draw_graph_validation_rejects_missing_material() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled("main", "quad", "instances", "missing", TargetSpec::window())
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("missing material"));
    }

    #[test]
    fn draw_graph_validation_rejects_missing_instance_stream() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("missing instance stream"));
    }

    #[test]
    fn draw_graph_validation_rejects_invalid_instance_grid() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .instance_stream("instances", InstanceGrid::new(32, 0, 8))
            .material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            )
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("grid dimensions"));
    }

    #[test]
    fn draw_graph_validation_rejects_missing_target() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .instance_stream("instances", InstanceGrid::new(32, 32, 8))
            .material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("missing"),
            )
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("missing target"));
    }

    #[test]
    fn draw_graph_validation_rejects_invalid_target_size() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .instance_stream("instances", InstanceGrid::new(32, 32, 8))
            .target("bad", 0, 720)
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_compute_culled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::named("bad"),
            )
            .validate()
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("target `bad` width and height must be greater than zero")
        );
    }

    #[test]
    fn draw_graph_validation_rejects_instance_material_without_instance_stream() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw("main", "quad", None::<String>, "material")
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("requires an InstanceStream"));
    }

    #[test]
    fn draw_graph_validation_rejects_instance_material_without_compute_culling() {
        let err = NeoApp::new()
            .geometry_stream("quad", MeshBuilder::quad())
            .instance_stream("instances", InstanceGrid::new(32, 32, 8))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull("raster_cull", "examples/stress-quads/hardware_raster.neo")
            .draw_all(
                "main",
                "quad",
                Some("instances"),
                "material",
                TargetSpec::Window,
            )
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("DrawPolicy::ComputeCulled"));
    }

    #[test]
    fn escape_hatch_into_parts_exposes_configuration() {
        let app = NeoApp::new()
            .kernel("image", "examples/live-window/live.neo")
            .mesh("quad", MeshBuilder::quad());
        let parts = app.into_parts();
        assert_eq!(parts.config.presenter, Presenter::D3d12Interop);
        assert!(parts.kernels.contains_key("image"));
        assert!(parts.geometry_streams.contains_key("quad"));
        assert!(parts.meshes.contains_key("quad"));
    }

    #[test]
    fn context_escape_hatch_skips_without_cuda() {
        let mut app = NeoApp::new();
        match app.context() {
            Ok(ctx) => ctx.synchronize().unwrap(),
            Err(err) => eprintln!("skipping context escape hatch test without CUDA: {err}"),
        }
    }

    #[test]
    fn build_geometry_streams_uses_runtime_meshbuffer_when_cuda_exists() {
        let mut app = NeoApp::new().geometry_stream("quad", MeshBuilder::quad());
        match app.build_geometry_streams() {
            Ok(geometry_streams) => {
                let mesh = geometry_streams.get("quad").unwrap();
                assert_eq!(mesh.desc().vertex_count, 4);
            }
            Err(err) => {
                eprintln!("skipping runtime GeometryStream build test without CUDA: {err}");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn runtime_resources_materialize_cuda_draw_from_friendly_graph() {
        let mut app = NeoApp::new()
            .window("Neo CUDA Draw", 64, 32)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa32("instances", InstanceGrid::new(1, 1, 1))
            .cuda_tiled_material_kernel(
                "material",
                "instance_raster",
                "examples/stress-quads/three_d_instances.neo",
            )
            .draw_cuda_tiled(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
            );

        match app.build_runtime_draw_resources() {
            Ok(resources) => {
                let draw = resources.cuda_draw("main").unwrap();
                assert_eq!(draw.policy(), neo_runtime::DrawPolicy::CudaTiled);
                assert_eq!(draw.target(), neo_runtime::Target::new(64, 32).unwrap());
                assert_eq!(draw.material().kernel_entrypoint(), Some("instance_raster"));
                assert_eq!(draw.geometry().mesh().desc().vertex_count, 4);
                assert_eq!(draw.instances().instances().desc().instance_count, 1);

                let draw = resources.draw("main").unwrap();
                assert!(matches!(draw, RuntimeDraw::Cuda(_)));
                assert!(draw.as_cuda_draw().is_some());
                assert!(draw.as_draw_execution().is_none());
                assert!(draw.is_cuda_draw());
                assert!(!draw.is_draw_execution());
                assert_eq!(draw.backend(), neo_runtime::DrawBackend::CudaTiled);
                assert_runtime_draw_recipe(
                    &draw,
                    neo_runtime::DrawBackend::CudaTiled,
                    neo_runtime::DrawPolicy::CudaTiled,
                );
                assert_eq!(draw.policy(), neo_runtime::DrawPolicy::CudaTiled);
                assert_eq!(
                    draw.policy_config(),
                    neo_runtime::DrawPolicyConfig::cuda_tiled()
                );
                assert_eq!(draw.target(), neo_runtime::Target::new(64, 32).unwrap());
                assert_eq!(draw.geometry().mesh().desc().vertex_count, 4);
                assert_eq!(
                    draw.instances().unwrap().instances().desc().instance_count,
                    1
                );
                let contract = draw.runtime_contract();
                assert_eq!(contract.backend_label(), "cuda-tiled");
                assert_eq!(contract.policy_label(), "cuda-tiled");
                assert_eq!(contract.target_width, 64);
                assert_eq!(contract.target_height, 32);
                assert_eq!(contract.instance_count, Some(1));
            }
            Err(err) => eprintln!("skipping runtime CudaDraw materialization without CUDA: {err}"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn runtime_resources_materialize_draw_execution_from_friendly_graph() {
        let mut app = NeoApp::new()
            .window("Neo Raster Draw", 64, 32)
            .geometry_stream("quad", MeshBuilder::quad().colored())
            .instance_stream_aosoa64("instances", InstanceGrid::new(1, 1, 1))
            .instance_material_kernel("material", "quad_vs", "quad_fs")
            .compute_cull_with_order(
                "raster_cull",
                "examples/stress-quads/hardware_raster.neo",
                CullOrder::StableDense,
            )
            .draw_compute_culled_projected(
                "main",
                "quad",
                "instances",
                "material",
                TargetSpec::window(),
                850,
            );

        match app.build_runtime_draw_resources() {
            Ok(resources) => {
                let draw = resources.draw_execution("main").unwrap();
                assert_eq!(draw.policy(), neo_runtime::DrawPolicy::ComputeCulled);
                assert_eq!(draw.target(), neo_runtime::Target::new(64, 32).unwrap());
                assert_eq!(draw.material().vertex_entrypoint(), "quad_vs");
                assert_eq!(draw.geometry().mesh().desc().vertex_count, 4);
                assert_eq!(
                    draw.instances().unwrap().instances().data_layout(),
                    neo_runtime::DataLayout::AoSoA { group_size: 64 }
                );
                let legacy_draw = resources.raster_draw("main").unwrap();
                assert_eq!(legacy_draw.backend(), draw.backend());
                assert_eq!(legacy_draw.policy_config(), draw.policy_config());

                let draw = resources.draw("main").unwrap();
                assert!(matches!(draw, RuntimeDraw::DrawExecution(_)));
                assert!(draw.as_draw_execution().is_some());
                assert!(draw.as_cuda_draw().is_none());
                assert!(draw.is_draw_execution());
                assert!(!draw.is_cuda_draw());
                assert_eq!(draw.backend(), neo_runtime::DrawBackend::HardwareRaster);
                assert_runtime_draw_recipe(
                    &draw,
                    neo_runtime::DrawBackend::HardwareRaster,
                    neo_runtime::DrawPolicy::ComputeCulled,
                );
                assert_eq!(draw.policy(), neo_runtime::DrawPolicy::ComputeCulled);
                assert_eq!(
                    draw.policy_config(),
                    neo_runtime::DrawPolicyConfig::compute_culled_with_visibility(
                        neo_runtime::CullOrder::StableDense,
                        neo_runtime::VisibilityMode::ProjectedSize
                    )
                    .with_min_projected_millipixels(850)
                );
                assert_eq!(draw.target(), neo_runtime::Target::new(64, 32).unwrap());
                assert_eq!(draw.geometry().mesh().desc().vertex_count, 4);
                assert_eq!(
                    draw.instances().unwrap().instances().data_layout(),
                    neo_runtime::DataLayout::AoSoA { group_size: 64 }
                );
                let contract = draw.runtime_contract();
                assert_eq!(contract.backend_label(), "hardware-raster");
                assert_eq!(contract.policy_label(), "compute-culled");
                assert_eq!(contract.target_width, 64);
                assert_eq!(contract.target_height, 32);
                assert_eq!(contract.instance_count, Some(1));
                assert_eq!(contract.instance_layout.as_deref(), Some("aosoa64"));

                let legacy_wrapper = RuntimeDraw::Raster(resources.raster_draw("main").unwrap());
                assert!(legacy_wrapper.as_draw_execution().is_some());
                assert_eq!(legacy_wrapper.backend(), draw.backend());
            }
            Err(err) => {
                eprintln!("skipping runtime DrawExecution materialization without CUDA: {err}")
            }
        }
    }
}
