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

