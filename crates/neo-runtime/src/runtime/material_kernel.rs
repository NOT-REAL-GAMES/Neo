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
