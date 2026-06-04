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
