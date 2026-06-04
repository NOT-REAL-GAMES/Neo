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
