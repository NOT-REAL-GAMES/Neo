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
