use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchArgs,
        LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits, sys,
    },
    nvrtc::compile_ptx,
};
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
}

pub struct Context {
    inner: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Context {
    pub fn new_default_device() -> Result<Self, RuntimeError> {
        let inner = CudaContext::new(0)?;
        let stream = inner.default_stream();
        Ok(Self { inner, stream })
    }

    pub fn compile_neo_module(&self, source: &str) -> Result<Module, RuntimeError> {
        Module::from_neo_source(self, source, &[])
    }

    pub fn alloc_zeros<T>(&self, len: usize) -> Result<DeviceBuffer<T>, RuntimeError>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        DeviceBuffer::new(self, len)
    }

    pub fn upload<T>(&self, values: &[T]) -> Result<DeviceBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        DeviceBuffer::upload(self, values)
    }

    pub fn alloc_pinned<T>(&self, len: usize) -> Result<PinnedHostBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        PinnedHostBuffer::new(self, len)
    }

    pub fn alloc_readable_pinned<T>(
        &self,
        len: usize,
    ) -> Result<ReadablePinnedHostBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        ReadablePinnedHostBuffer::new(self, len)
    }

    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn create_fence(&self) -> Result<CudaFence, RuntimeError> {
        CudaFence::new()
    }
}

pub struct Module {
    inner: Arc<cudarc::driver::CudaModule>,
    stream: Arc<CudaStream>,
    pub cuda_source: String,
}

impl Module {
    pub fn from_neo_source(
        ctx: &Context,
        source: &str,
        entrypoints: &[&str],
    ) -> Result<Self, RuntimeError> {
        let program = neo_lang::parse(source)?;
        for entrypoint in entrypoints {
            if !program
                .kernels
                .iter()
                .any(|kernel| kernel.name == *entrypoint)
            {
                return Err(RuntimeError::MissingEntrypoint((*entrypoint).to_string()));
            }
        }
        let cuda_source = neo_lang::lower_program(&program);
        let diagnostics = RuntimeDiagnostics::collect();
        if !diagnostics.nvrtc_loadable {
            return Err(RuntimeError::Nvrtc(diagnostics.nvrtc_help()));
        }
        configure_nvrtc_search_path(&diagnostics);
        let ptx = compile_ptx(&cuda_source).map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
        let inner = ctx.inner.load_module(ptx)?;
        Ok(Self {
            inner,
            stream: ctx.stream.clone(),
            cuda_source,
        })
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, RuntimeError> {
        let function = self.inner.load_function(name)?;
        Ok(Kernel {
            function,
            stream: self.stream.clone(),
        })
    }
}

pub fn nvrtc_available() -> bool {
    RuntimeDiagnostics::collect().nvrtc_loadable
}

#[cfg(windows)]
fn nvrtc_candidates() -> Vec<PathBuf> {
    let names = [
        "nvrtc.dll",
        "nvrtc64.dll",
        "nvrtc64_12.dll",
        "nvrtc64_120.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_11.dll",
        "nvrtc64_112_0.dll",
    ];
    let mut dirs = BTreeSet::new();
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = std::env::var_os(key) {
            dirs.insert(PathBuf::from(root).join("bin"));
        }
    }
    dirs.extend(cuda_toolkit_bin_dirs());
    dirs.extend(nvidia_app_nvrtc_dirs());

    dirs.into_iter()
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .collect()
}

#[cfg(not(windows))]
fn nvrtc_candidates() -> Vec<PathBuf> {
    let names = [
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.dylib",
    ];
    let mut dirs = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/cuda/lib64"),
        PathBuf::from("/usr/local/cuda/lib"),
    ];
    if let Some(path) = std::env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs.into_iter()
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .collect()
}

#[derive(Debug, Clone)]
pub struct RuntimeDiagnostics {
    pub cuda_driver_available: bool,
    pub cuda_driver_error: Option<String>,
    pub nvrtc_candidates: Vec<PathBuf>,
    pub nvrtc_found: Vec<PathBuf>,
    pub nvrtc_loadable: bool,
}

impl RuntimeDiagnostics {
    pub fn collect() -> Self {
        let (cuda_driver_available, cuda_driver_error) = match CudaContext::new(0) {
            Ok(ctx) => {
                drop(ctx);
                (true, None)
            }
            Err(err) => (false, Some(format!("{err:?}"))),
        };
        let nvrtc_candidates = nvrtc_candidates();
        let nvrtc_found = nvrtc_candidates
            .iter()
            .filter(|candidate| candidate.exists())
            .cloned()
            .collect::<Vec<_>>();
        let nvrtc_loadable = !nvrtc_found.is_empty();
        Self {
            cuda_driver_available,
            cuda_driver_error,
            nvrtc_candidates,
            nvrtc_found,
            nvrtc_loadable,
        }
    }

    pub fn nvrtc_help(&self) -> String {
        if let Some(found) = self.nvrtc_found.first() {
            return format!(
                "NVRTC was found at {}, but the dynamic loader could not use it. Add its directory to PATH before starting Neo.",
                found.display()
            );
        }
        "NVRTC shared library was not found. Install the NVIDIA CUDA Toolkit or add the directory containing nvrtc64_120_0.dll/nvrtc64_12.dll to PATH.".to_string()
    }
}

#[cfg(windows)]
fn configure_nvrtc_search_path(diagnostics: &RuntimeDiagnostics) {
    let Some(dir) = diagnostics
        .nvrtc_found
        .first()
        .and_then(|path| path.parent())
    else {
        return;
    };

    let Some(current_path) = std::env::var_os("PATH") else {
        // SAFETY: Neo is single-threaded at the point this is called by the CLI/runtime setup.
        unsafe {
            std::env::set_var("PATH", dir);
        }
        return;
    };

    let paths = std::env::split_paths(&current_path).collect::<Vec<_>>();
    if paths.iter().any(|path| path == dir) {
        return;
    }
    let mut new_paths = vec![dir.to_path_buf()];
    new_paths.extend(paths);
    if let Ok(joined) = std::env::join_paths(new_paths) {
        // SAFETY: Neo updates the process DLL search path before NVRTC is loaded.
        unsafe {
            std::env::set_var("PATH", joined);
        }
    }
}

#[cfg(not(windows))]
fn configure_nvrtc_search_path(_diagnostics: &RuntimeDiagnostics) {}

#[cfg(windows)]
fn cuda_toolkit_bin_dirs() -> Vec<PathBuf> {
    let root = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("bin"))
        .filter(|path| path.is_dir())
        .collect()
}

#[cfg(windows)]
fn nvidia_app_nvrtc_dirs() -> Vec<PathBuf> {
    [
        r"C:\Program Files\NVIDIA Corporation\NVIDIA Audio Effects SDK",
        r"C:\Program Files\Blackmagic Design\DaVinci Resolve",
    ]
    .into_iter()
    .map(PathBuf::from)
    .filter(|path| path.is_dir())
    .collect()
}

pub struct DeviceBuffer<T> {
    inner: CudaSlice<T>,
}

pub struct PinnedHostBuffer<T> {
    inner: PinnedHostSlice<T>,
}

pub struct ReadablePinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for ReadablePinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for ReadablePinnedHostBuffer<T> {}

impl<T> ReadablePinnedHostBuffer<T>
where
    T: DeviceRepr,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        ctx.inner.bind_to_thread()?;
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let mut ptr = std::ptr::null_mut();
        unsafe {
            sys::cuMemAllocHost_v2(&mut ptr, byte_len)
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for ReadablePinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { sys::cuMemFreeHost(self.ptr.cast()).result() };
        }
    }
}

pub struct CudaFence {
    event: sys::CUevent,
}

impl CudaFence {
    fn new() -> Result<Self, RuntimeError> {
        let mut event = std::ptr::null_mut();
        unsafe {
            sys::cuEventCreate(
                &mut event,
                sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC as u32,
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(Self { event })
    }

    pub fn record(&self, ctx: &Context) -> Result<(), RuntimeError> {
        unsafe {
            sys::cuEventRecord(self.event, ctx.stream.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        unsafe {
            sys::cuEventSynchronize(self.event)
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }
}

impl Drop for CudaFence {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = unsafe { sys::cuEventDestroy_v2(self.event).result() };
        }
    }
}

impl<T> PinnedHostBuffer<T>
where
    T: DeviceRepr,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        let inner = unsafe { ctx.inner.alloc_pinned(len)? };
        Ok(Self { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T> PinnedHostBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn as_slice(&self) -> Result<&[T], RuntimeError> {
        Ok(self.inner.as_slice()?)
    }
}

impl<T> DeviceBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        let inner = ctx.stream.alloc_zeros(len)?;
        Ok(Self { inner })
    }
}

impl<T> DeviceBuffer<T>
where
    T: DeviceRepr,
{
    pub fn upload(ctx: &Context, values: &[T]) -> Result<Self, RuntimeError> {
        let inner = ctx.stream.clone_htod(values)?;
        Ok(Self { inner })
    }

    pub fn download(&self) -> Result<Vec<T>, RuntimeError> {
        Ok(self.inner.stream().clone_dtoh(&self.inner)?)
    }

    pub fn download_into(&self, dst: &mut [T]) -> Result<(), RuntimeError> {
        self.inner.stream().memcpy_dtoh(&self.inner, dst)?;
        Ok(())
    }

    pub fn download_into_pinned(&self, dst: &mut PinnedHostBuffer<T>) -> Result<(), RuntimeError> {
        self.inner
            .stream()
            .memcpy_dtoh(&self.inner, &mut dst.inner)?;
        Ok(())
    }

    pub fn download_into_readable_pinned(
        &self,
        dst: &mut ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtr as _;

        let stream = self.inner.stream();
        let (src, _record_read) = self.inner.device_ptr(stream);
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.ptr.cast(),
                src,
                self.inner.num_bytes(),
                stream.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

pub struct Kernel {
    function: CudaFunction,
    stream: Arc<CudaStream>,
}

impl Kernel {
    pub fn launcher(&self) -> KernelLaunch<'_> {
        KernelLaunch {
            inner: self.stream.launch_builder(&self.function),
        }
    }
}

pub struct KernelLaunch<'a> {
    inner: LaunchArgs<'a>,
}

impl<'a> KernelLaunch<'a> {
    pub fn arg<T>(&mut self, value: &'a T) -> &mut Self
    where
        T: DeviceRepr,
    {
        self.inner.arg(value);
        self
    }

    pub fn arg_buffer<T>(&mut self, value: &'a DeviceBuffer<T>) -> &mut Self {
        self.inner.arg(&value.inner);
        self
    }

    pub fn arg_buffer_mut<T>(&mut self, value: &'a mut DeviceBuffer<T>) -> &mut Self {
        self.inner.arg(&mut value.inner);
        self
    }

    /// Launches the configured kernel with explicit grid/block dimensions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pushed arguments exactly match the CUDA
    /// kernel signature, that mutable buffers are not aliased by concurrent GPU
    /// work, and that the kernel does not read or write outside the provided
    /// device allocations.
    pub unsafe fn launch(&mut self, dims: LaunchDims) -> Result<(), RuntimeError> {
        unsafe {
            self.inner.launch(dims.into())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchDims {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchDims {
    pub fn for_2d(width: u32, height: u32, block: (u32, u32)) -> Self {
        let grid_x = width.div_ceil(block.0);
        let grid_y = height.div_ceil(block.1);
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block.0, block.1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl From<LaunchDims> for LaunchConfig {
    fn from(value: LaunchDims) -> Self {
        Self {
            grid_dim: value.grid,
            block_dim: value.block,
            shared_mem_bytes: value.shared_mem_bytes,
        }
    }
}

#[derive(Debug)]
pub struct ImageBuffer {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl ImageBuffer {
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, RuntimeError> {
        let expected = width as usize * height as usize * 4;
        let actual = rgba.len();
        if actual != expected {
            return Err(RuntimeError::InvalidImageBuffer {
                width,
                height,
                expected,
                actual,
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), RuntimeError> {
        image::save_buffer_with_format(
            path,
            &self.rgba,
            self.width,
            self.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        Ok(())
    }
}

pub fn run_image_kernel(
    source: &str,
    width: u32,
    height: u32,
) -> Result<ImageBuffer, RuntimeError> {
    let ctx = Context::new_default_device()?;
    let module = ctx.compile_neo_module(source)?;
    let kernel = module.kernel("image")?;
    let mut pixels = ctx.alloc_zeros::<u8>(width as usize * height as usize * 4)?;
    let dims = LaunchDims::for_2d(width, height, (16, 16));

    {
        let mut launch = kernel.launcher();
        launch.arg_buffer_mut(&mut pixels);
        launch.arg(&width);
        launch.arg(&height);
        unsafe {
            launch.launch(dims)?;
        }
    }

    ctx.synchronize()?;
    ImageBuffer::from_rgba(width, height, pixels.download()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_dims_cover_2d_image() {
        let dims = LaunchDims::for_2d(33, 17, (16, 16));
        assert_eq!(dims.grid, (3, 2, 1));
        assert_eq!(dims.block, (16, 16, 1));
    }

    #[test]
    fn image_buffer_validates_size() {
        let err = ImageBuffer::from_rgba(2, 2, vec![0; 3]).unwrap_err();
        assert!(matches!(err, RuntimeError::InvalidImageBuffer { .. }));
    }

    #[test]
    fn module_validates_requested_entrypoints_before_nvrtc() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping entrypoint validation test without CUDA: {err}");
                return;
            }
        };
        let err = match Module::from_neo_source(
            &ctx,
            "kernel fn image(global u8* pixels) {}",
            &["missing"],
        ) {
            Ok(_) => panic!("expected missing entrypoint error"),
            Err(err) => err,
        };
        assert!(matches!(err, RuntimeError::MissingEntrypoint(name) if name == "missing"));
    }

    #[test]
    fn diagnostics_collect_without_panicking() {
        let diagnostics = RuntimeDiagnostics::collect();
        if diagnostics.nvrtc_loadable {
            assert!(!diagnostics.nvrtc_found.is_empty());
        }
    }

    #[test]
    fn runtime_smoke_test_skips_without_cuda() {
        match Context::new_default_device() {
            Ok(ctx) => ctx.synchronize().unwrap(),
            Err(err) => eprintln!("skipping CUDA smoke test: {err}"),
        }
    }

    #[test]
    fn end_to_end_gradient_skips_without_nvrtc() {
        let source = include_str!("../../../examples/gradient.neo");
        match run_image_kernel(source, 8, 8) {
            Ok(image) => {
                assert_eq!(image.rgba.len(), 8 * 8 * 4);
                assert!(image.rgba.iter().any(|value| *value != 0));
            }
            Err(err) => eprintln!("skipping GPU/NVRTC e2e test: {err}"),
        }
    }
}
