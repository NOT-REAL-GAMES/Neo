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
