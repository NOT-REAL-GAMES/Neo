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
