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
