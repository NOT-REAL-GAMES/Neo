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
