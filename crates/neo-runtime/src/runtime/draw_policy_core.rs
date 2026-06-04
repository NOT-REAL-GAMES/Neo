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
