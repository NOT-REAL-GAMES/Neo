#[cfg(windows)]
use std::{fmt, sync::Arc};

#[cfg(windows)]
use crate::{
    Context, DataLayout, InstanceBuffer, InstanceBufferDesc, MeshBuffer, NeoD3d12InteropDevice,
    RuntimeError, SharedGpuBuffer,
};

#[cfg(windows)]
#[derive(Clone)]
pub struct DrawDevice {
    interop: Arc<NeoD3d12InteropDevice>,
}

#[cfg(windows)]
pub type RasterDevice = DrawDevice;

#[cfg(windows)]
impl DrawDevice {
    pub fn new(ctx: &Context) -> Result<Self, RuntimeError> {
        Ok(Self {
            interop: Arc::new(NeoD3d12InteropDevice::new(ctx)?),
        })
    }

    pub fn from_interop(interop: NeoD3d12InteropDevice) -> Self {
        Self {
            interop: Arc::new(interop),
        }
    }

    pub fn interop(&self) -> &NeoD3d12InteropDevice {
        &self.interop
    }

    pub fn create_shared_gpu_buffer(&self, byte_len: u64) -> Result<SharedGpuBuffer, RuntimeError> {
        self.interop.create_shared_gpu_buffer(byte_len)
    }
}

#[cfg(windows)]
pub struct DrawPipeline {
    label: String,
}

#[cfg(windows)]
pub type RasterPipeline = DrawPipeline;

#[cfg(windows)]
impl DrawPipeline {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct GeometryStream<'a> {
    mesh: &'a MeshBuffer,
}

#[cfg(windows)]
impl<'a> GeometryStream<'a> {
    pub fn from_mesh(mesh: &'a MeshBuffer) -> Self {
        Self { mesh }
    }

    pub fn mesh(&self) -> &'a MeshBuffer {
        self.mesh
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct InstanceStream<'a> {
    instances: &'a InstanceBuffer,
}

#[cfg(windows)]
impl<'a> InstanceStream<'a> {
    pub fn from_instances(instances: &'a InstanceBuffer) -> Self {
        Self { instances }
    }

    pub fn instances(&self) -> &'a InstanceBuffer {
        self.instances
    }
}

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialKernelAbi {
    pub kind: MaterialKernelKind,
    pub vertex_entrypoint: String,
    pub fragment_entrypoint: String,
    pub kernel_entrypoint: String,
    pub vertex_requirements: Vec<MaterialVertexRequirement>,
    pub fragment_requirements: Vec<MaterialFragmentRequirement>,
    pub bindings: Vec<MaterialBinding>,
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialKernelKind {
    DrawExecution,
    HardwareRaster,
    CudaTiled,
}

#[cfg(windows)]
impl MaterialKernelKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::DrawExecution => "draw-execution",
            Self::HardwareRaster => "hardware-raster",
            Self::CudaTiled => "cuda-tiled",
        }
    }

    pub fn backend(self) -> DrawBackend {
        match self {
            Self::DrawExecution | Self::HardwareRaster => DrawBackend::HardwareRaster,
            Self::CudaTiled => DrawBackend::CudaTiled,
        }
    }

    pub fn is_draw_execution(self) -> bool {
        matches!(self, Self::DrawExecution | Self::HardwareRaster)
    }
}

#[cfg(windows)]
impl fmt::Display for MaterialKernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
impl MaterialKernelAbi {
    pub fn kind_label(&self) -> &'static str {
        self.kind.label()
    }

    pub fn simple_color(
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        Self {
            kind: MaterialKernelKind::DrawExecution,
            vertex_entrypoint: vertex_entrypoint.into(),
            fragment_entrypoint: fragment_entrypoint.into(),
            kernel_entrypoint: String::new(),
            vertex_requirements: vec![
                MaterialVertexRequirement::ClipPositionOutput,
                MaterialVertexRequirement::VertexColorOutput,
            ],
            fragment_requirements: vec![MaterialFragmentRequirement::InterpolatedColorInput],
            bindings: vec![MaterialBinding::draw_params(0, 0)],
        }
    }

    pub fn compute_culled_instance_color(
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        Self {
            kind: MaterialKernelKind::DrawExecution,
            vertex_entrypoint: vertex_entrypoint.into(),
            fragment_entrypoint: fragment_entrypoint.into(),
            kernel_entrypoint: String::new(),
            vertex_requirements: vec![
                MaterialVertexRequirement::VisibleInstanceStream,
                MaterialVertexRequirement::InstancePosition,
                MaterialVertexRequirement::GeometryPosition,
                MaterialVertexRequirement::ClipPositionOutput,
                MaterialVertexRequirement::VertexColorOutput,
            ],
            fragment_requirements: vec![MaterialFragmentRequirement::InterpolatedColorInput],
            bindings: vec![
                MaterialBinding::draw_params(0, 0),
                MaterialBinding::visible_instance_stream(1, 0),
                MaterialBinding::instance_stream(2, 1),
                MaterialBinding::geometry_stream(3, 2),
            ],
        }
    }

    pub fn direct_instance_color(
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        Self {
            kind: MaterialKernelKind::DrawExecution,
            vertex_entrypoint: vertex_entrypoint.into(),
            fragment_entrypoint: fragment_entrypoint.into(),
            kernel_entrypoint: String::new(),
            vertex_requirements: vec![
                MaterialVertexRequirement::DirectInstanceId,
                MaterialVertexRequirement::InstancePosition,
                MaterialVertexRequirement::GeometryPosition,
                MaterialVertexRequirement::ClipPositionOutput,
                MaterialVertexRequirement::VertexColorOutput,
            ],
            fragment_requirements: vec![MaterialFragmentRequirement::InterpolatedColorInput],
            bindings: vec![
                MaterialBinding::draw_params(0, 0),
                MaterialBinding::instance_stream(1, 1),
                MaterialBinding::geometry_stream(2, 2),
            ],
        }
    }

    pub fn cuda_tiled_instance_color(kernel_entrypoint: impl Into<String>) -> Self {
        Self {
            kind: MaterialKernelKind::CudaTiled,
            vertex_entrypoint: String::new(),
            fragment_entrypoint: String::new(),
            kernel_entrypoint: kernel_entrypoint.into(),
            vertex_requirements: Vec::new(),
            fragment_requirements: Vec::new(),
            bindings: vec![
                MaterialBinding::draw_params(0, 0),
                MaterialBinding::instance_stream(1, 1),
                MaterialBinding::geometry_stream(2, 2),
            ],
        }
    }

    pub fn is_draw_execution(&self) -> bool {
        self.kind.is_draw_execution()
    }

    pub fn is_hardware_raster(&self) -> bool {
        self.is_draw_execution()
    }

    pub fn is_cuda_tiled(&self) -> bool {
        self.kind == MaterialKernelKind::CudaTiled
    }

    pub fn backend(&self) -> DrawBackend {
        self.kind.backend()
    }

    pub fn vertex_entrypoint(&self) -> Option<&str> {
        self.is_draw_execution()
            .then_some(self.vertex_entrypoint.as_str())
    }

    pub fn fragment_entrypoint(&self) -> Option<&str> {
        self.is_draw_execution()
            .then_some(self.fragment_entrypoint.as_str())
    }

    pub fn kernel_entrypoint(&self) -> Option<&str> {
        self.is_cuda_tiled()
            .then_some(self.kernel_entrypoint.as_str())
    }

    pub fn requires_instance_stream(&self) -> bool {
        if self.is_cuda_tiled() {
            return true;
        }
        self.vertex_requirements.iter().any(|requirement| {
            matches!(
                requirement,
                MaterialVertexRequirement::VisibleInstanceStream
                    | MaterialVertexRequirement::DirectInstanceId
                    | MaterialVertexRequirement::InstancePosition
            )
        })
    }

    pub fn requires_compute_culling(&self) -> bool {
        if self.is_cuda_tiled() {
            return false;
        }
        self.vertex_requirements
            .contains(&MaterialVertexRequirement::VisibleInstanceStream)
    }

    pub fn binding(&self, kind: MaterialBindingKind) -> Option<&MaterialBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.kind.matches(kind))
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialBinding {
    pub kind: MaterialBindingKind,
    pub root_parameter_index: u32,
    pub shader_register: u32,
    pub register_space: u32,
}

#[cfg(windows)]
impl MaterialBinding {
    pub fn draw_params(root_parameter_index: u32, shader_register: u32) -> Self {
        Self {
            kind: MaterialBindingKind::DrawParams,
            root_parameter_index,
            shader_register,
            register_space: 0,
        }
    }

    pub fn raster_params(root_parameter_index: u32, shader_register: u32) -> Self {
        Self {
            kind: MaterialBindingKind::RasterParams,
            root_parameter_index,
            shader_register,
            register_space: 0,
        }
    }

    pub fn visible_instance_stream(root_parameter_index: u32, shader_register: u32) -> Self {
        Self {
            kind: MaterialBindingKind::VisibleInstanceStream,
            root_parameter_index,
            shader_register,
            register_space: 0,
        }
    }

    pub fn instance_stream(root_parameter_index: u32, shader_register: u32) -> Self {
        Self {
            kind: MaterialBindingKind::InstanceStream,
            root_parameter_index,
            shader_register,
            register_space: 0,
        }
    }

    pub fn geometry_stream(root_parameter_index: u32, shader_register: u32) -> Self {
        Self {
            kind: MaterialBindingKind::GeometryStream,
            root_parameter_index,
            shader_register,
            register_space: 0,
        }
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialBindingKind {
    DrawParams,
    RasterParams,
    VisibleInstanceStream,
    InstanceStream,
    GeometryStream,
}

#[cfg(windows)]
impl MaterialBindingKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::DrawParams => "draw params",
            Self::RasterParams => "raster params",
            Self::VisibleInstanceStream => "visible InstanceStream",
            Self::InstanceStream => "InstanceStream",
            Self::GeometryStream => "GeometryStream",
        }
    }

    pub fn matches(self, requested: Self) -> bool {
        self == requested
            || matches!(
                (self, requested),
                (Self::DrawParams, Self::RasterParams) | (Self::RasterParams, Self::DrawParams)
            )
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialVertexRequirement {
    VisibleInstanceStream,
    DirectInstanceId,
    InstancePosition,
    GeometryPosition,
    ClipPositionOutput,
    VertexColorOutput,
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialFragmentRequirement {
    InterpolatedColorInput,
}

#[cfg(windows)]
pub struct MaterialKernel {
    label: String,
    vertex_entrypoint: String,
    fragment_entrypoint: String,
    abi: MaterialKernelAbi,
}

#[cfg(windows)]
impl MaterialKernel {
    pub fn new(label: impl Into<String>) -> Self {
        Self::from_stages(label, "quad_vs", "quad_fs")
    }

    pub fn from_stages(
        label: impl Into<String>,
        vertex_entrypoint: impl Into<String>,
        fragment_entrypoint: impl Into<String>,
    ) -> Self {
        let vertex_entrypoint = vertex_entrypoint.into();
        let fragment_entrypoint = fragment_entrypoint.into();
        Self {
            label: label.into(),
            abi: MaterialKernelAbi::simple_color(
                vertex_entrypoint.clone(),
                fragment_entrypoint.clone(),
            ),
            vertex_entrypoint,
            fragment_entrypoint,
        }
    }

    pub fn from_cuda_tiled(label: impl Into<String>, kernel_entrypoint: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            abi: MaterialKernelAbi::cuda_tiled_instance_color(kernel_entrypoint),
            vertex_entrypoint: String::new(),
            fragment_entrypoint: String::new(),
        }
    }

    pub fn with_abi(mut self, abi: MaterialKernelAbi) -> Self {
        self.vertex_entrypoint = abi.vertex_entrypoint.clone();
        self.fragment_entrypoint = abi.fragment_entrypoint.clone();
        self.abi = abi;
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn kind_label(&self) -> &'static str {
        self.abi.kind_label()
    }

    pub fn vertex_entrypoint(&self) -> &str {
        &self.vertex_entrypoint
    }

    pub fn fragment_entrypoint(&self) -> &str {
        &self.fragment_entrypoint
    }

    pub fn kernel_entrypoint(&self) -> Option<&str> {
        self.abi.kernel_entrypoint()
    }

    pub fn abi(&self) -> &MaterialKernelAbi {
        &self.abi
    }

    pub fn backend(&self) -> DrawBackend {
        self.abi.backend()
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawPolicy {
    DrawAll,
    ComputeCulled,
    CudaTiled,
}

#[cfg(windows)]
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

#[cfg(windows)]
impl fmt::Display for DrawPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawBackend {
    HardwareRaster,
    CudaTiled,
}

#[cfg(windows)]
impl DrawBackend {
    pub fn primary_neo() -> Self {
        Self::CudaTiled
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::HardwareRaster => "hardware-raster",
            Self::CudaTiled => "cuda-tiled",
        }
    }

    pub fn is_primary_neo(self) -> bool {
        self == Self::primary_neo()
    }
}

#[cfg(windows)]
impl fmt::Display for DrawBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullOrder {
    AtomicCompact,
    StableDense,
}

#[cfg(windows)]
pub type RasterCullOrder = CullOrder;

#[cfg(windows)]
impl CullOrder {
    pub fn label(self) -> &'static str {
        match self {
            Self::AtomicCompact => "atomic-compact",
            Self::StableDense => "stable-dense",
        }
    }
}

#[cfg(windows)]
impl fmt::Display for CullOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityMode {
    Frustum,
    ProjectedSize,
}

#[cfg(windows)]
pub type RasterVisibilityMode = VisibilityMode;

#[cfg(windows)]
impl VisibilityMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Frustum => "frustum",
            Self::ProjectedSize => "projected-size",
        }
    }
}

#[cfg(windows)]
impl fmt::Display for VisibilityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawDepthMode {
    Auto,
    On,
    Off,
}

#[cfg(windows)]
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
}

#[cfg(windows)]
impl fmt::Display for DrawDepthMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(windows)]
pub const DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS: u32 = 850;

#[cfg(windows)]
pub const DEFAULT_MIN_PROJECTED_MILLIPIXELS: u32 = DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS;

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawPolicyConfig {
    pub policy: DrawPolicy,
    pub depth: DrawDepthMode,
    pub cull_order: CullOrder,
    pub visibility: VisibilityMode,
    pub min_projected_millipixels: u32,
}

#[cfg(windows)]
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

#[cfg(windows)]
impl From<DrawPolicy> for DrawPolicyConfig {
    fn from(policy: DrawPolicy) -> Self {
        match policy {
            DrawPolicy::DrawAll => Self::draw_all(),
            DrawPolicy::ComputeCulled => Self::compute_culled(CullOrder::AtomicCompact),
            DrawPolicy::CudaTiled => Self::cuda_tiled(),
        }
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub width: u32,
    pub height: u32,
}

#[cfg(windows)]
impl Target {
    pub fn new(width: u32, height: u32) -> Result<Self, RuntimeError> {
        if width == 0 || height == 0 {
            return Err(RuntimeError::Raster(
                "target width and height must be greater than zero".to_string(),
            ));
        }
        Ok(Self { width, height })
    }
}

#[cfg(windows)]
pub type RasterTarget = Target;

#[cfg(windows)]
pub type RenderTarget = Target;

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawPass {
    pub target: Target,
}

#[cfg(windows)]
pub type RasterPass = DrawPass;

#[cfg(windows)]
pub trait DrawRecipe<'a> {
    fn backend(&self) -> DrawBackend;
    fn geometry(&self) -> GeometryStream<'a>;
    fn instances(&self) -> Option<InstanceStream<'a>>;
    fn material(&self) -> &'a MaterialKernel;
    fn target(&self) -> Target;
    fn policy_config(&self) -> DrawPolicyConfig;

    fn policy(&self) -> DrawPolicy {
        self.policy_config().policy
    }

    fn contract(&self) -> DrawContract {
        let geometry = self.geometry();
        let instances = self.instances();
        let material = self.material();
        let target = self.target();
        let policy_config = self.policy_config();
        let policy = policy_config.policy;
        let backend = self.backend();
        DrawContract {
            geometry_vertex_count: geometry.mesh().desc().vertex_count,
            geometry_index_count: geometry.mesh().desc().index_count,
            instance_count: instances.map(|instances| instances.instances().desc().instance_count),
            instance_layout: instances.map(|instances| instances.instances().layout_label()),
            material_kernel: material.label().to_string(),
            material_kind_label: material.kind_label().to_string(),
            target_width: target.width,
            target_height: target.height,
            policy,
            policy_config,
            backend,
        }
    }
}

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawContract {
    pub geometry_vertex_count: u32,
    pub geometry_index_count: u32,
    pub instance_count: Option<u32>,
    pub instance_layout: Option<String>,
    pub material_kernel: String,
    pub material_kind_label: String,
    pub target_width: u32,
    pub target_height: u32,
    pub policy: DrawPolicy,
    pub policy_config: DrawPolicyConfig,
    pub backend: DrawBackend,
}

#[cfg(windows)]
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

    pub fn material_label(&self) -> &str {
        &self.material_kernel
    }

    pub fn material_kind_label(&self) -> &str {
        &self.material_kind_label
    }
}

#[cfg(windows)]
pub struct DrawExecution<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
pub type RasterDraw<'a> = DrawExecution<'a>;

#[cfg(windows)]
pub type RasterDrawBuilder<'a> = DrawExecutionBuilder<'a>;

#[cfg(windows)]
impl<'a> DrawExecution<'a> {
    pub fn execution_builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> DrawExecutionBuilder<'a> {
        Self::builder(geometry, material, target)
    }

    pub fn builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> DrawExecutionBuilder<'a> {
        DrawExecutionBuilder {
            geometry,
            instances: None,
            material,
            target,
            policy: DrawPolicyConfig::draw_all(),
        }
    }

    pub fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    pub fn instances(&self) -> Option<InstanceStream<'a>> {
        self.instances
    }

    pub fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn policy(&self) -> DrawPolicy {
        self.policy.policy
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }

    pub fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    pub fn contract(&self) -> DrawContract {
        DrawRecipe::contract(self)
    }
}

#[cfg(windows)]
impl<'a> DrawRecipe<'a> for DrawExecution<'a> {
    fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    fn instances(&self) -> Option<InstanceStream<'a>> {
        self.instances
    }

    fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    fn target(&self) -> Target {
        self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }
}

#[cfg(windows)]
pub struct DrawExecutionBuilder<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> DrawExecutionBuilder<'a> {
    pub fn instance_stream(mut self, instances: InstanceStream<'a>) -> Self {
        self.instances = Some(instances);
        self
    }

    pub fn draw_policy(mut self, policy: DrawPolicy) -> Self {
        self.policy = policy.into();
        self
    }

    pub fn draw_policy_config(mut self, policy: DrawPolicyConfig) -> Self {
        self.policy = policy;
        self
    }

    pub fn compute_culled(mut self, cull_order: CullOrder) -> Self {
        self.policy = DrawPolicyConfig::compute_culled(cull_order);
        self
    }

    pub fn compute_culled_with_visibility(
        mut self,
        cull_order: CullOrder,
        visibility: VisibilityMode,
    ) -> Self {
        self.policy = DrawPolicyConfig::compute_culled_with_visibility(cull_order, visibility);
        self
    }

    pub fn compute_culled_projected(
        mut self,
        cull_order: CullOrder,
        min_projected_millipixels: u32,
    ) -> Self {
        self.policy = DrawPolicyConfig::compute_culled_with_visibility(
            cull_order,
            VisibilityMode::ProjectedSize,
        )
        .with_min_projected_millipixels(min_projected_millipixels);
        self
    }

    pub fn try_build(self) -> Result<DrawExecution<'a>, RuntimeError> {
        let abi = self.material.abi();
        if abi.is_cuda_tiled() {
            return Err(RuntimeError::Raster(format!(
                "DrawExecution requires a draw-execution MaterialKernel, got CUDA tiled material `{}`",
                self.material.label()
            )));
        }
        let policy = self.policy.policy;
        if policy == DrawPolicy::CudaTiled {
            return Err(RuntimeError::Raster(format!(
                "DrawExecution material `{}` cannot use DrawPolicy::CudaTiled",
                self.material.label()
            )));
        }
        if abi.requires_instance_stream() && self.instances.is_none() {
            return Err(RuntimeError::Raster(format!(
                "raster material `{}` requires an explicit InstanceStream",
                self.material.label()
            )));
        }
        if abi.requires_compute_culling() && policy != DrawPolicy::ComputeCulled {
            return Err(RuntimeError::Raster(format!(
                "raster material `{}` requires DrawPolicy::ComputeCulled",
                self.material.label()
            )));
        }
        if policy == DrawPolicy::ComputeCulled && !abi.requires_compute_culling() {
            return Err(RuntimeError::Raster(format!(
                "DrawPolicy::ComputeCulled requires material `{}` to read the visible InstanceStream",
                self.material.label()
            )));
        }
        if policy == DrawPolicy::ComputeCulled && self.instances.is_none() {
            return Err(RuntimeError::Raster(
                "DrawPolicy::ComputeCulled requires an explicit InstanceStream".to_string(),
            ));
        }
        Ok(DrawExecution {
            geometry: self.geometry,
            instances: self.instances,
            material: self.material,
            target: self.target,
            policy: self.policy,
        })
    }

    pub fn build(self) -> DrawExecution<'a> {
        self.try_build()
            .expect("invalid draw execution recipe; use try_build for recoverable validation")
    }
}

#[cfg(windows)]
pub struct CudaDraw<'a> {
    geometry: GeometryStream<'a>,
    instances: InstanceStream<'a>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> CudaDraw<'a> {
    pub fn builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> CudaDrawBuilder<'a> {
        CudaDrawBuilder {
            geometry,
            instances: None,
            material,
            target,
            policy: DrawPolicyConfig::cuda_tiled(),
        }
    }

    pub fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    pub fn instances(&self) -> InstanceStream<'a> {
        self.instances
    }

    pub fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn policy(&self) -> DrawPolicy {
        self.policy.policy
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }

    pub fn backend(&self) -> DrawBackend {
        DrawBackend::CudaTiled
    }

    pub fn contract(&self) -> DrawContract {
        DrawRecipe::contract(self)
    }
}

#[cfg(windows)]
impl<'a> DrawRecipe<'a> for CudaDraw<'a> {
    fn backend(&self) -> DrawBackend {
        DrawBackend::CudaTiled
    }

    fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    fn instances(&self) -> Option<InstanceStream<'a>> {
        Some(self.instances)
    }

    fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    fn target(&self) -> Target {
        self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }
}

#[cfg(windows)]
pub struct CudaDrawBuilder<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> CudaDrawBuilder<'a> {
    pub fn instance_stream(mut self, instances: InstanceStream<'a>) -> Self {
        self.instances = Some(instances);
        self
    }

    pub fn draw_policy_config(mut self, policy: DrawPolicyConfig) -> Self {
        self.policy = policy;
        self
    }

    pub fn try_build(self) -> Result<CudaDraw<'a>, RuntimeError> {
        let abi = self.material.abi();
        if !abi.is_cuda_tiled() {
            return Err(RuntimeError::Raster(format!(
                "CudaDraw requires a CUDA tiled MaterialKernel, got hardware raster material `{}`",
                self.material.label()
            )));
        }
        if self.policy.policy != DrawPolicy::CudaTiled {
            return Err(RuntimeError::Raster(format!(
                "CudaDraw material `{}` requires DrawPolicy::CudaTiled",
                self.material.label()
            )));
        }
        let instances = self.instances.ok_or_else(|| {
            RuntimeError::Raster(format!(
                "CudaDraw material `{}` requires an explicit InstanceStream",
                self.material.label()
            ))
        })?;
        Ok(CudaDraw {
            geometry: self.geometry,
            instances,
            material: self.material,
            target: self.target,
            policy: self.policy,
        })
    }

    pub fn build(self) -> CudaDraw<'a> {
        self.try_build()
            .expect("invalid CUDA draw recipe; use try_build for recoverable validation")
    }
}

#[cfg(windows)]
pub struct IndirectDrawBuffer {
    buffer: SharedGpuBuffer,
    command_capacity: u32,
}

#[cfg(windows)]
impl IndirectDrawBuffer {
    pub fn new(
        device: &NeoD3d12InteropDevice,
        command_capacity: u32,
    ) -> Result<Self, RuntimeError> {
        if command_capacity == 0 {
            return Err(RuntimeError::Raster(
                "indirect draw command capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(command_capacity)
            .checked_mul(std::mem::size_of::<DrawIndexedIndirectCommand>() as u64)
            .ok_or_else(|| {
                RuntimeError::Raster("indirect draw buffer size overflow".to_string())
            })?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            command_capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn command_capacity(&self) -> u32 {
        self.command_capacity
    }
}

#[cfg(windows)]
pub struct VisibleInstanceStream {
    buffer: SharedGpuBuffer,
    capacity: u32,
}

#[cfg(windows)]
impl VisibleInstanceStream {
    pub fn new(device: &NeoD3d12InteropDevice, capacity: u32) -> Result<Self, RuntimeError> {
        if capacity == 0 {
            return Err(RuntimeError::Raster(
                "visible instance stream capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(capacity)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(windows)]
pub struct SharedInstanceStream {
    buffer: SharedGpuBuffer,
    desc: InstanceBufferDesc,
    data_layout: DataLayout,
    byte_len: usize,
}

#[cfg(windows)]
impl SharedInstanceStream {
    pub fn upload_typed<I>(
        ctx: &Context,
        device: &NeoD3d12InteropDevice,
        desc: InstanceBufferDesc,
        instances: &[I],
        data_layout: DataLayout,
    ) -> Result<Self, RuntimeError>
    where
        I: Copy,
    {
        let packed = InstanceBuffer::pack_typed_with_layout(&desc, instances, data_layout)?;
        let byte_len = packed.len();
        let mut buffer = device.create_shared_gpu_buffer(byte_len as u64)?;
        let stream = ctx.default_stream();
        buffer.upload_bytes_on_stream(&stream, &packed)?;
        ctx.synchronize()?;
        Ok(Self {
            buffer,
            desc,
            data_layout,
            byte_len,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn desc(&self) -> &InstanceBufferDesc {
        &self.desc
    }

    pub fn data_layout(&self) -> DataLayout {
        self.data_layout
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count_per_instance: u32,
    pub instance_count: u32,
    pub start_index_location: u32,
    pub base_vertex_location: i32,
    pub start_instance_location: u32,
}

impl DrawIndexedIndirectCommand {
    pub const BYTE_LEN: usize = std::mem::size_of::<Self>();

    pub fn indexed_quad(instance_count: u32) -> Self {
        Self {
            index_count_per_instance: 6,
            instance_count,
            start_index_location: 0,
            base_vertex_location: 0,
            start_instance_location: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}
