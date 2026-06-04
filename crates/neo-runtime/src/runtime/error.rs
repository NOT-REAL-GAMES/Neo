use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("Neo compile error: {0}")]
    Neo(#[from] neo_lang::LowerError),
    #[error("Neo parse error: {0}")]
    Parse(#[from] neo_lang::ParseError),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] DriverError),
    #[error("NVRTC compile error: {0}")]
    Nvrtc(String),
    #[error("kernel entrypoint `{0}` was not found in Neo source")]
    MissingEntrypoint(String),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("expected {expected} bytes for {width}x{height} RGBA image, got {actual}")]
    InvalidImageBuffer {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("host buffer size overflow")]
    HostBufferTooLarge,
    #[error("mesh buffer error: {0}")]
    Mesh(String),
    #[error("instance buffer error: {0}")]
    Instance(String),
    #[error("visibility grid error: {0}")]
    VisibilityGrid(String),
    #[error("sparse texture error: {0}")]
    SparseTexture(String),
    #[error("material stream error: {0}")]
    MaterialStream(String),
    #[cfg(windows)]
    #[error("D3D12 interop error: {0}")]
    D3d12Interop(String),
    #[cfg(windows)]
    #[error("raster error: {0}")]
    Raster(String),
    #[cfg(windows)]
    #[error("Windows graphics error: {0}")]
    Windows(#[from] windows::core::Error),
}
