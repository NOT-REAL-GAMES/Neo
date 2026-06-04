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
