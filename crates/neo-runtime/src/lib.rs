use std::{
    any::Any,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::Arc,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaGraph as CudarcCudaGraph, CudaSlice, CudaStream, DeviceRepr,
        DriverError, LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits,
        sys,
    },
    nvrtc::{Ptx, compile_ptx, result as nvrtc_result},
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

    pub fn create_stream(&self) -> Result<Stream, RuntimeError> {
        Ok(Stream {
            inner: self.inner.new_stream()?,
        })
    }

    pub fn default_stream(&self) -> Stream {
        Stream {
            inner: self.stream.clone(),
        }
    }

    /// Disables cudarc's automatic multi-stream event tracking for future allocations.
    ///
    /// Callers that use this must provide their own stream/fence lifetime ordering.
    ///
    /// # Safety
    ///
    /// The caller must ensure all buffers allocated after this call are not used
    /// concurrently across streams unless explicit CUDA stream waits, fences, or
    /// other ordering guarantees protect those accesses.
    pub unsafe fn disable_automatic_event_tracking(&self) {
        unsafe {
            self.inner.disable_event_tracking();
        }
    }
}

#[derive(Clone)]
pub struct Stream {
    inner: Arc<CudaStream>,
}

impl Stream {
    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.inner.synchronize()?;
        Ok(())
    }

    pub fn create_fence(&self) -> Result<CudaFence, RuntimeError> {
        CudaFence::new()
    }

    pub fn begin_graph_capture(&self) -> Result<(), RuntimeError> {
        self.inner
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        Ok(())
    }

    pub fn end_graph_capture(&self) -> Result<Option<CudaGraph>, RuntimeError> {
        let no_flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0) };
        let graph = self.inner.end_capture(no_flags)?;
        Ok(graph.map(|inner| CudaGraph { inner }))
    }
}

pub struct CudaGraph {
    inner: CudarcCudaGraph,
}

impl CudaGraph {
    pub fn launch(&self) -> Result<(), RuntimeError> {
        self.inner.launch()?;
        Ok(())
    }

    pub fn upload(&self) -> Result<(), RuntimeError> {
        self.inner.upload()?;
        Ok(())
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
            if !program.kernels.iter().any(|kernel| {
                kernel.kind == neo_lang::EntryPointKind::Kernel && kernel.name == *entrypoint
            }) {
                return Err(RuntimeError::MissingEntrypoint((*entrypoint).to_string()));
            }
        }
        let cuda_source = format!(
            "{}\n{}",
            runtime_cuda_prelude(),
            neo_lang::lower_program(&program)
        );
        let diagnostics = RuntimeDiagnostics::collect();
        if !diagnostics.nvrtc_loadable {
            return Err(RuntimeError::Nvrtc(diagnostics.nvrtc_help()));
        }
        configure_nvrtc_search_path(&diagnostics);
        let ptx = compile_cuda_image_checked(ctx, &cuda_source, &diagnostics)?;
        let inner = load_cuda_module_checked(ctx, ptx)?;
        Ok(Self {
            inner,
            stream: ctx.stream.clone(),
            cuda_source,
        })
    }

    pub fn from_cuda_source(ctx: &Context, cuda_source: String) -> Result<Self, RuntimeError> {
        let diagnostics = RuntimeDiagnostics::collect();
        if !diagnostics.nvrtc_loadable {
            return Err(RuntimeError::Nvrtc(diagnostics.nvrtc_help()));
        }
        configure_nvrtc_search_path(&diagnostics);
        let ptx = compile_cuda_image_checked(ctx, &cuda_source, &diagnostics)?;
        let inner = load_cuda_module_checked(ctx, ptx)?;
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

    pub fn kernel_on_stream(&self, name: &str, stream: &Stream) -> Result<Kernel, RuntimeError> {
        let function = self.inner.load_function(name)?;
        Ok(Kernel {
            function,
            stream: stream.inner.clone(),
        })
    }
}

const MESH_MAGIC: u32 = 0x4d48_454e;
const MESH_VERSION: u32 = 1;
const MESH_HEADER_BYTES: usize = 48;
const MESH_ATTRIBUTE_BYTES: usize = 16;

const MESH_SEMANTIC_POSITION: u32 = 1;
const MESH_SEMANTIC_NORMAL: u32 = 2;
const MESH_SEMANTIC_UV0: u32 = 3;
const MESH_SEMANTIC_COLOR0: u32 = 4;

const MESH_FORMAT_F32X2: u32 = 1;
const MESH_FORMAT_F32X3: u32 = 2;
const MESH_FORMAT_F32X4: u32 = 3;
const MESH_FORMAT_U8X4_UNORM: u32 = 4;

const MESH_INDEX_NONE: u32 = 0;
const MESH_INDEX_U16: u32 = 1;
const MESH_INDEX_U32: u32 = 2;
const MESH_TOPOLOGY_TRIANGLE_LIST: u32 = 1;

const INSTANCE_MAGIC: u32 = 0x4948_454e;
const INSTANCE_VERSION: u32 = 2;
const INSTANCE_HEADER_BYTES: usize = 40;
const INSTANCE_ATTRIBUTE_BYTES: usize = 16;

const INSTANCE_SEMANTIC_POSITION: u32 = 1;
const INSTANCE_SEMANTIC_ROTATION: u32 = 2;
const INSTANCE_SEMANTIC_SCALE: u32 = 3;
const INSTANCE_SEMANTIC_COLOR0: u32 = 4;

const INSTANCE_FORMAT_F32X2: u32 = 1;
const INSTANCE_FORMAT_F32X3: u32 = 2;
const INSTANCE_FORMAT_F32X4: u32 = 3;
const INSTANCE_FORMAT_U8X4_UNORM: u32 = 4;

pub const VISIBILITY_GRID_MAGIC: u32 = 0x4e45_4f4d;
pub const VISIBILITY_GRID_HEADER_U32S: usize = 8;
pub const VISIBILITY_GRID_RECORD_U32S: usize = 6;
pub const DEFAULT_MACROCELL_SIZE: u32 = 8;

pub const SPARSE_TEXTURE_MAGIC: u32 = 0x5354_584e;
pub const SPARSE_TEXTURE_VERSION: u32 = 1;
pub const SPARSE_TEXTURE_HEADER_U32S: usize = 20;
pub const SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S: usize = 1;
pub const DEFAULT_SPARSE_TEXTURE_PAGE_SIZE: u32 = 128;
pub const DEFAULT_SPARSE_TEXTURE_GUTTER: u32 = 1;
const SPARSE_TEXTURE_FORMAT_RGBA8_UNORM: u32 = 1;
const SPARSE_TEXTURE_ENTRY_RESIDENT: u32 = 1 << 31;
const SPARSE_TEXTURE_ENTRY_PHYSICAL_MASK: u32 = 0x00ff_ffff;
const SPARSE_TEXTURE_HEADER_FEEDBACK_FLAGS_U32: usize = 18;
const SPARSE_TEXTURE_FEEDBACK_ENABLED: u32 = 1;

pub const MATERIAL_STREAM_MAGIC: u32 = 0x4d53_584e;
pub const MATERIAL_STREAM_VERSION: u32 = 1;
pub const MATERIAL_STREAM_HEADER_U32S: usize = 8;

pub const DEFAULT_AOSOA_GROUP_SIZE: u32 = 32;
const DATA_LAYOUT_AOS: u32 = 0;
const DATA_LAYOUT_SOA: u32 = 1;
const DATA_LAYOUT_AOSOA: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLayout {
    AoS,
    SoA,
    AoSoA { group_size: u32 },
}

impl DataLayout {
    pub fn aosoa32() -> Self {
        Self::AoSoA {
            group_size: DEFAULT_AOSOA_GROUP_SIZE,
        }
    }

    pub fn aosoa64() -> Self {
        Self::AoSoA { group_size: 64 }
    }

    fn code(self) -> u32 {
        match self {
            Self::AoS => DATA_LAYOUT_AOS,
            Self::SoA => DATA_LAYOUT_SOA,
            Self::AoSoA { .. } => DATA_LAYOUT_AOSOA,
        }
    }

    fn group_size(self) -> u32 {
        match self {
            Self::AoS | Self::SoA => 1,
            Self::AoSoA { group_size } => group_size,
        }
    }

    fn label(self) -> String {
        match self {
            Self::AoS => "aos".to_string(),
            Self::SoA => "soa".to_string(),
            Self::AoSoA { group_size } => format!("aosoa{group_size}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferFormat {
    F32x2,
    F32x3,
    F32x4,
    U8x4Unorm,
}

impl BufferFormat {
    fn code(self) -> u32 {
        match self {
            Self::F32x2 => INSTANCE_FORMAT_F32X2,
            Self::F32x3 => INSTANCE_FORMAT_F32X3,
            Self::F32x4 => INSTANCE_FORMAT_F32X4,
            Self::U8x4Unorm => INSTANCE_FORMAT_U8X4_UNORM,
        }
    }

    fn byte_len(self) -> u32 {
        match self {
            Self::F32x2 => 8,
            Self::F32x3 => 12,
            Self::F32x4 => 16,
            Self::U8x4Unorm => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferField {
    pub semantic: u32,
    pub format: BufferFormat,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredBufferDesc {
    pub element_count: u32,
    pub source_stride: u32,
    pub layout: DataLayout,
    pub fields: Vec<BufferField>,
}

pub struct StructuredBuffer {
    buffer: DeviceBuffer<u8>,
    desc: StructuredBufferDesc,
    byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexSemantic {
    Position,
    Normal,
    Uv0,
    Color0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    F32x2,
    F32x3,
    F32x4,
    U8x4Unorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    None,
    U16,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    TriangleList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub semantic: VertexSemantic,
    pub format: VertexFormat,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshBufferDesc {
    pub vertex_count: u32,
    pub vertex_layout: VertexLayout,
    pub index_format: IndexFormat,
    pub index_count: u32,
    pub topology: PrimitiveTopology,
}

pub struct MeshBuffer {
    buffer: DeviceBuffer<u8>,
    desc: MeshBufferDesc,
    byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceSemantic {
    Position,
    Rotation,
    Scale,
    Color0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceFormat {
    F32x2,
    F32x3,
    F32x4,
    U8x4Unorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceAttribute {
    pub semantic: InstanceSemantic,
    pub format: InstanceFormat,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceLayout {
    pub stride: u32,
    pub attributes: Vec<InstanceAttribute>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceBufferDesc {
    pub instance_count: u32,
    pub instance_layout: InstanceLayout,
}

pub struct InstanceBuffer {
    buffer: DeviceBuffer<u8>,
    desc: InstanceBufferDesc,
    byte_len: usize,
    data_layout: DataLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibilityGridDesc {
    pub cells: [u32; 3],
    pub macrocell_size: u32,
}

impl VisibilityGridDesc {
    pub fn macrocell_lattice(cells: [u32; 3]) -> Self {
        Self {
            cells,
            macrocell_size: DEFAULT_MACROCELL_SIZE,
        }
    }
}

pub struct VisibilityGrid {
    buffer: DeviceBuffer<u8>,
    desc: VisibilityGridDesc,
    macrocell_dims: [u32; 3],
    macrocell_count: u32,
    byte_len: usize,
}

pub type AccelerationGrid = VisibilityGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseTextureFormat {
    Rgba8Unorm,
}

impl SparseTextureFormat {
    fn code(self) -> u32 {
        match self {
            Self::Rgba8Unorm => SPARSE_TEXTURE_FORMAT_RGBA8_UNORM,
        }
    }

    fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8Unorm => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseTextureDesc {
    pub virtual_width: u32,
    pub virtual_height: u32,
    pub page_size: u32,
    pub mip_count: u32,
    pub format: SparseTextureFormat,
    pub physical_pages: u32,
    pub gutter: u32,
}

impl SparseTextureDesc {
    pub fn rgba8(virtual_width: u32, virtual_height: u32, physical_pages: u32) -> Self {
        Self {
            virtual_width,
            virtual_height,
            page_size: DEFAULT_SPARSE_TEXTURE_PAGE_SIZE,
            mip_count: 1,
            format: SparseTextureFormat::Rgba8Unorm,
            physical_pages,
            gutter: DEFAULT_SPARSE_TEXTURE_GUTTER,
        }
    }
}

pub struct SparseTextureAtlas {
    buffer: DeviceBuffer<u8>,
    desc: SparseTextureDesc,
    page_dims: [u32; 2],
    byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseTextureFeedbackSummary {
    pub active_pages: u32,
    pub total_requests: u64,
    pub hottest_page: Option<u32>,
    pub hottest_requests: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialStreamDesc {
    pub material_count: u32,
}

pub struct MaterialStream {
    buffer: DeviceBuffer<u8>,
    desc: MaterialStreamDesc,
    byte_len: usize,
}

impl StructuredBuffer {
    pub fn upload_aos(
        ctx: &Context,
        mut desc: StructuredBufferDesc,
        source_bytes: &[u8],
    ) -> Result<Self, RuntimeError> {
        desc.layout = DataLayout::AoS;
        Self::upload(ctx, desc, source_bytes)
    }

    pub fn upload_soa(
        ctx: &Context,
        mut desc: StructuredBufferDesc,
        source_bytes: &[u8],
    ) -> Result<Self, RuntimeError> {
        desc.layout = DataLayout::SoA;
        Self::upload(ctx, desc, source_bytes)
    }

    pub fn upload_aosoa(
        ctx: &Context,
        mut desc: StructuredBufferDesc,
        source_bytes: &[u8],
        group_size: u32,
    ) -> Result<Self, RuntimeError> {
        desc.layout = DataLayout::AoSoA { group_size };
        Self::upload(ctx, desc, source_bytes)
    }

    fn upload(
        ctx: &Context,
        desc: StructuredBufferDesc,
        source_bytes: &[u8],
    ) -> Result<Self, RuntimeError> {
        let blob = pack_structured_buffer(&desc, source_bytes)?;
        let byte_len = blob.len();
        let buffer = DeviceBuffer::upload(ctx, &blob)?;
        Ok(Self {
            buffer,
            desc,
            byte_len,
        })
    }

    pub fn desc(&self) -> &StructuredBufferDesc {
        &self.desc
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl InstanceBuffer {
    pub fn upload(
        ctx: &Context,
        desc: InstanceBufferDesc,
        instance_bytes: &[u8],
    ) -> Result<Self, RuntimeError> {
        Self::upload_with_layout(ctx, desc, instance_bytes, DataLayout::AoS)
    }

    pub fn upload_with_layout(
        ctx: &Context,
        desc: InstanceBufferDesc,
        instance_bytes: &[u8],
        data_layout: DataLayout,
    ) -> Result<Self, RuntimeError> {
        let blob = pack_instance_buffer_with_layout(&desc, instance_bytes, data_layout)?;
        let byte_len = blob.len();
        let buffer = DeviceBuffer::upload(ctx, &blob)?;
        Ok(Self {
            buffer,
            desc,
            byte_len,
            data_layout,
        })
    }

    pub fn upload_typed<I>(
        ctx: &Context,
        desc: InstanceBufferDesc,
        instances: &[I],
    ) -> Result<Self, RuntimeError>
    where
        I: Copy,
    {
        Self::upload(ctx, desc, slice_as_bytes(instances))
    }

    pub fn upload_typed_with_layout<I>(
        ctx: &Context,
        desc: InstanceBufferDesc,
        instances: &[I],
        data_layout: DataLayout,
    ) -> Result<Self, RuntimeError>
    where
        I: Copy,
    {
        Self::upload_with_layout(ctx, desc, slice_as_bytes(instances), data_layout)
    }

    pub fn pack_typed_with_layout<I>(
        desc: &InstanceBufferDesc,
        instances: &[I],
        data_layout: DataLayout,
    ) -> Result<Vec<u8>, RuntimeError>
    where
        I: Copy,
    {
        pack_instance_buffer_with_layout(desc, slice_as_bytes(instances), data_layout)
    }

    pub fn desc(&self) -> &InstanceBufferDesc {
        &self.desc
    }

    pub fn data_layout(&self) -> DataLayout {
        self.data_layout
    }

    pub fn layout_label(&self) -> String {
        self.data_layout.label()
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        self.buffer.device_ptr_arg()
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

impl VisibilityGrid {
    pub fn upload(ctx: &Context, desc: VisibilityGridDesc) -> Result<Self, RuntimeError> {
        let packed = pack_visibility_grid(&desc)?;
        let byte_len = packed.len();
        let macrocell_dims = visibility_macrocell_dims(&desc)?;
        let macrocell_count = visibility_macrocell_count(macrocell_dims)?;
        let buffer = DeviceBuffer::upload(ctx, &packed)?;
        Ok(Self {
            buffer,
            desc,
            macrocell_dims,
            macrocell_count,
            byte_len,
        })
    }

    pub fn pack(desc: &VisibilityGridDesc) -> Result<Vec<u8>, RuntimeError> {
        pack_visibility_grid(desc)
    }

    pub fn desc(&self) -> VisibilityGridDesc {
        self.desc
    }

    pub fn macrocell_dims(&self) -> [u32; 3] {
        self.macrocell_dims
    }

    pub fn macrocell_count(&self) -> u32 {
        self.macrocell_count
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

impl SparseTextureAtlas {
    pub fn new(ctx: &Context, desc: SparseTextureDesc) -> Result<Self, RuntimeError> {
        let blob = pack_sparse_texture(&desc)?;
        let page_dims = sparse_texture_page_dims(&desc)?;
        let byte_len = blob.len();
        let buffer = DeviceBuffer::upload(ctx, &blob)?;
        Ok(Self {
            buffer,
            desc,
            page_dims,
            byte_len,
        })
    }

    pub fn pack(desc: &SparseTextureDesc) -> Result<Vec<u8>, RuntimeError> {
        pack_sparse_texture(desc)
    }

    pub fn upload_page(&mut self, page_index: u32, rgba: &[u8]) -> Result<(), RuntimeError> {
        let offset = sparse_texture_physical_page_offset(&self.desc, page_index)?;
        self.validate_page_bytes(rgba)?;
        self.buffer.upload_range(offset, rgba)
    }

    pub fn upload_checker_pages(&mut self) -> Result<(), RuntimeError> {
        let page_bytes = sparse_texture_page_bytes(&self.desc)?;
        for page in 0..self.desc.physical_pages {
            let mut rgba = vec![0u8; page_bytes];
            fill_sparse_checker_page(&self.desc, page, &mut rgba)?;
            self.upload_page(page, &rgba)?;
        }
        Ok(())
    }

    pub fn mark_resident(
        &mut self,
        virtual_page: u32,
        physical_page: u32,
    ) -> Result<(), RuntimeError> {
        validate_sparse_virtual_page(&self.desc, virtual_page)?;
        validate_sparse_physical_page(&self.desc, physical_page)?;
        let entry =
            SPARSE_TEXTURE_ENTRY_RESIDENT | (physical_page & SPARSE_TEXTURE_ENTRY_PHYSICAL_MASK);
        self.buffer.upload_range(
            sparse_texture_page_table_offset(virtual_page)?,
            &entry.to_le_bytes(),
        )
    }

    pub fn mark_missing(&mut self, virtual_page: u32) -> Result<(), RuntimeError> {
        validate_sparse_virtual_page(&self.desc, virtual_page)?;
        self.buffer.upload_range(
            sparse_texture_page_table_offset(virtual_page)?,
            &0u32.to_le_bytes(),
        )
    }

    pub fn set_feedback_enabled(&mut self, enabled: bool) -> Result<(), RuntimeError> {
        let flags = if enabled {
            SPARSE_TEXTURE_FEEDBACK_ENABLED
        } else {
            0
        };
        self.buffer.upload_range(
            SPARSE_TEXTURE_HEADER_FEEDBACK_FLAGS_U32 * 4,
            &flags.to_le_bytes(),
        )
    }

    pub fn clear_feedback(&mut self) -> Result<(), RuntimeError> {
        let len = sparse_texture_feedback_byte_len(&self.desc)?;
        let zeros = vec![0u8; len];
        self.buffer
            .upload_range(sparse_texture_feedback_offset(&self.desc)?, &zeros)
    }

    pub fn download_feedback(&self) -> Result<Vec<u32>, RuntimeError> {
        let len = sparse_texture_feedback_byte_len(&self.desc)?;
        let mut bytes = vec![0u8; len];
        self.buffer
            .download_range(sparse_texture_feedback_offset(&self.desc)?, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("feedback chunk is u32")))
            .collect())
    }

    pub fn feedback_summary(&self) -> Result<SparseTextureFeedbackSummary, RuntimeError> {
        summarize_sparse_texture_feedback(&self.download_feedback()?)
    }

    pub fn desc(&self) -> SparseTextureDesc {
        self.desc
    }

    pub fn page_dims(&self) -> [u32; 2] {
        self.page_dims
    }

    pub fn virtual_page_count(&self) -> u32 {
        self.page_dims[0] * self.page_dims[1]
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        self.buffer.device_ptr_arg()
    }

    fn validate_page_bytes(&self, rgba: &[u8]) -> Result<(), RuntimeError> {
        let expected = sparse_texture_page_bytes(&self.desc)?;
        if rgba.len() != expected {
            return Err(RuntimeError::SparseTexture(format!(
                "expected {expected} bytes for one sparse texture page, got {}",
                rgba.len()
            )));
        }
        Ok(())
    }
}

impl MaterialStream {
    pub fn upload(ctx: &Context, material_ids: &[u32]) -> Result<Self, RuntimeError> {
        let desc = MaterialStreamDesc {
            material_count: u32::try_from(material_ids.len())
                .map_err(|_| RuntimeError::HostBufferTooLarge)?,
        };
        let blob = pack_material_stream(&desc, material_ids)?;
        let byte_len = blob.len();
        let buffer = DeviceBuffer::upload(ctx, &blob)?;
        Ok(Self {
            buffer,
            desc,
            byte_len,
        })
    }

    pub fn pack(desc: &MaterialStreamDesc, material_ids: &[u32]) -> Result<Vec<u8>, RuntimeError> {
        pack_material_stream(desc, material_ids)
    }

    pub fn desc(&self) -> MaterialStreamDesc {
        self.desc
    }

    pub fn material_count(&self) -> u32 {
        self.desc.material_count
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        self.buffer.device_ptr_arg()
    }
}

impl MeshBuffer {
    pub fn upload(
        ctx: &Context,
        desc: MeshBufferDesc,
        vertex_bytes: &[u8],
        index_bytes: &[u8],
    ) -> Result<Self, RuntimeError> {
        let blob = pack_mesh_buffer(&desc, vertex_bytes, index_bytes)?;
        let byte_len = blob.len();
        let buffer = DeviceBuffer::upload(ctx, &blob)?;
        Ok(Self {
            buffer,
            desc,
            byte_len,
        })
    }

    pub fn upload_typed<V, I>(
        ctx: &Context,
        desc: MeshBufferDesc,
        vertices: &[V],
        indices: &[I],
    ) -> Result<Self, RuntimeError>
    where
        V: Copy,
        I: Copy,
    {
        Self::upload(ctx, desc, slice_as_bytes(vertices), slice_as_bytes(indices))
    }

    pub fn desc(&self) -> &MeshBufferDesc {
        &self.desc
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

fn pack_mesh_buffer(
    desc: &MeshBufferDesc,
    vertex_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<Vec<u8>, RuntimeError> {
    validate_mesh_buffer(desc, vertex_bytes, index_bytes)?;
    let attr_count = desc.vertex_layout.attributes.len();
    let attr_bytes_offset = MESH_HEADER_BYTES;
    let vertex_bytes_offset =
        align_usize(attr_bytes_offset + attr_count * MESH_ATTRIBUTE_BYTES, 16);
    let index_bytes_offset = if index_bytes.is_empty() {
        0
    } else {
        align_usize(vertex_bytes_offset + vertex_bytes.len(), 4)
    };
    let total_bytes = if index_bytes.is_empty() {
        vertex_bytes_offset + vertex_bytes.len()
    } else {
        index_bytes_offset + index_bytes.len()
    };

    let mut blob = vec![0u8; total_bytes];
    let header = [
        MESH_MAGIC,
        MESH_VERSION,
        MESH_HEADER_BYTES as u32,
        desc.vertex_count,
        desc.vertex_layout.stride,
        vertex_bytes_offset as u32,
        desc.index_count,
        desc.index_format.code(),
        index_bytes_offset as u32,
        attr_count as u32,
        attr_bytes_offset as u32,
        desc.topology.code(),
    ];
    for (idx, value) in header.into_iter().enumerate() {
        write_u32_le(&mut blob, idx * 4, value);
    }
    for (idx, attr) in desc.vertex_layout.attributes.iter().enumerate() {
        let offset = attr_bytes_offset + idx * MESH_ATTRIBUTE_BYTES;
        write_u32_le(&mut blob, offset, attr.semantic.code());
        write_u32_le(&mut blob, offset + 4, attr.format.code());
        write_u32_le(&mut blob, offset + 8, attr.offset);
        write_u32_le(&mut blob, offset + 12, 0);
    }
    blob[vertex_bytes_offset..vertex_bytes_offset + vertex_bytes.len()]
        .copy_from_slice(vertex_bytes);
    if !index_bytes.is_empty() {
        blob[index_bytes_offset..index_bytes_offset + index_bytes.len()]
            .copy_from_slice(index_bytes);
    }
    Ok(blob)
}

fn validate_mesh_buffer(
    desc: &MeshBufferDesc,
    vertex_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<(), RuntimeError> {
    if desc.vertex_layout.stride == 0 {
        return Err(RuntimeError::Mesh(
            "vertex stride must be greater than zero".to_string(),
        ));
    }
    if desc.topology != PrimitiveTopology::TriangleList {
        return Err(RuntimeError::Mesh(
            "v1 only supports triangle-list meshes".to_string(),
        ));
    }
    let expected_vertex_bytes = usize::try_from(desc.vertex_count)
        .ok()
        .and_then(|count| count.checked_mul(desc.vertex_layout.stride as usize))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    if vertex_bytes.len() != expected_vertex_bytes {
        return Err(RuntimeError::Mesh(format!(
            "expected {expected_vertex_bytes} vertex bytes, got {}",
            vertex_bytes.len()
        )));
    }

    let mut seen = Vec::new();
    for attr in &desc.vertex_layout.attributes {
        if seen.contains(&attr.semantic) {
            return Err(RuntimeError::Mesh(format!(
                "duplicate vertex semantic {:?}",
                attr.semantic
            )));
        }
        seen.push(attr.semantic);
        let end = attr
            .offset
            .checked_add(attr.format.byte_len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > desc.vertex_layout.stride {
            return Err(RuntimeError::Mesh(format!(
                "vertex attribute {:?} extends past stride {}",
                attr.semantic, desc.vertex_layout.stride
            )));
        }
    }

    let expected_index_bytes = desc
        .index_format
        .byte_len()
        .checked_mul(desc.index_count as usize)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    if index_bytes.len() != expected_index_bytes {
        return Err(RuntimeError::Mesh(format!(
            "expected {expected_index_bytes} index bytes, got {}",
            index_bytes.len()
        )));
    }
    if desc.index_format == IndexFormat::None && desc.index_count != 0 {
        return Err(RuntimeError::Mesh(
            "index count must be zero when index format is none".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
fn pack_instance_buffer(
    desc: &InstanceBufferDesc,
    instance_bytes: &[u8],
) -> Result<Vec<u8>, RuntimeError> {
    pack_instance_buffer_with_layout(desc, instance_bytes, DataLayout::AoS)
}

fn pack_instance_buffer_with_layout(
    desc: &InstanceBufferDesc,
    instance_bytes: &[u8],
    data_layout: DataLayout,
) -> Result<Vec<u8>, RuntimeError> {
    validate_instance_buffer(desc, instance_bytes)?;
    let structured = StructuredBufferDesc {
        element_count: desc.instance_count,
        source_stride: desc.instance_layout.stride,
        layout: data_layout,
        fields: desc
            .instance_layout
            .attributes
            .iter()
            .map(|attr| BufferField {
                semantic: attr.semantic.code(),
                format: attr.format.into(),
                offset: attr.offset,
            })
            .collect(),
    };
    pack_structured_buffer(&structured, instance_bytes)
}

fn pack_structured_buffer(
    desc: &StructuredBufferDesc,
    source_bytes: &[u8],
) -> Result<Vec<u8>, RuntimeError> {
    validate_structured_buffer(desc, source_bytes)?;
    let attr_count = desc.fields.len();
    let attr_bytes_offset = INSTANCE_HEADER_BYTES;
    let data_bytes_offset = align_usize(
        attr_bytes_offset + attr_count * INSTANCE_ATTRIBUTE_BYTES,
        16,
    );
    let stream_offsets = structured_stream_offsets(desc)?;
    let data_len = structured_data_len(desc, &stream_offsets)?;
    let total_bytes = data_bytes_offset
        .checked_add(data_len)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let mut blob = vec![0u8; total_bytes];
    let header = [
        INSTANCE_MAGIC,
        INSTANCE_VERSION,
        INSTANCE_HEADER_BYTES as u32,
        desc.element_count,
        desc.source_stride,
        data_bytes_offset as u32,
        attr_count as u32,
        attr_bytes_offset as u32,
        desc.layout.code(),
        desc.layout.group_size(),
    ];
    for (idx, value) in header.into_iter().enumerate() {
        write_u32_le(&mut blob, idx * 4, value);
    }
    for (idx, field) in desc.fields.iter().enumerate() {
        let offset = attr_bytes_offset + idx * INSTANCE_ATTRIBUTE_BYTES;
        let device_offset = match desc.layout {
            DataLayout::AoS => field.offset,
            DataLayout::SoA | DataLayout::AoSoA { .. } => stream_offsets[idx] as u32,
        };
        write_u32_le(&mut blob, offset, field.semantic);
        write_u32_le(&mut blob, offset + 4, field.format.code());
        write_u32_le(&mut blob, offset + 8, device_offset);
        write_u32_le(&mut blob, offset + 12, field.offset);
    }
    match desc.layout {
        DataLayout::AoS => blob[data_bytes_offset..data_bytes_offset + source_bytes.len()]
            .copy_from_slice(source_bytes),
        DataLayout::SoA | DataLayout::AoSoA { .. } => {
            copy_structured_streams(
                desc,
                source_bytes,
                &stream_offsets,
                &mut blob[data_bytes_offset..],
            )?;
        }
    }
    Ok(blob)
}

fn validate_structured_buffer(
    desc: &StructuredBufferDesc,
    source_bytes: &[u8],
) -> Result<(), RuntimeError> {
    if desc.source_stride == 0 {
        return Err(RuntimeError::Instance(
            "structured source stride must be greater than zero".to_string(),
        ));
    }
    if desc.layout.group_size() == 0 {
        return Err(RuntimeError::Instance(
            "AoSoA group size must be greater than zero".to_string(),
        ));
    }
    let expected_bytes = usize::try_from(desc.element_count)
        .ok()
        .and_then(|count| count.checked_mul(desc.source_stride as usize))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    if source_bytes.len() != expected_bytes {
        return Err(RuntimeError::Instance(format!(
            "expected {expected_bytes} structured source bytes, got {}",
            source_bytes.len()
        )));
    }
    let mut seen = Vec::new();
    for field in &desc.fields {
        if seen.contains(&field.semantic) {
            return Err(RuntimeError::Instance(format!(
                "duplicate buffer semantic {}",
                field.semantic
            )));
        }
        seen.push(field.semantic);
        let end = field
            .offset
            .checked_add(field.format.byte_len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > desc.source_stride {
            return Err(RuntimeError::Instance(format!(
                "buffer field semantic {} extends past stride {}",
                field.semantic, desc.source_stride
            )));
        }
    }
    Ok(())
}

fn structured_stream_offsets(desc: &StructuredBufferDesc) -> Result<Vec<usize>, RuntimeError> {
    let mut offsets = Vec::with_capacity(desc.fields.len());
    let mut cursor = 0usize;
    for field in &desc.fields {
        cursor = align_usize(cursor, 4);
        offsets.push(cursor);
        cursor = cursor
            .checked_add(structured_stream_byte_len(desc, field.format)?)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
    }
    Ok(offsets)
}

fn structured_data_len(
    desc: &StructuredBufferDesc,
    stream_offsets: &[usize],
) -> Result<usize, RuntimeError> {
    match desc.layout {
        DataLayout::AoS => usize::try_from(desc.element_count)
            .ok()
            .and_then(|count| count.checked_mul(desc.source_stride as usize))
            .ok_or(RuntimeError::HostBufferTooLarge),
        DataLayout::SoA | DataLayout::AoSoA { .. } => {
            let Some((last_index, last_field)) = desc.fields.iter().enumerate().next_back() else {
                return Ok(0);
            };
            stream_offsets[last_index]
                .checked_add(structured_stream_byte_len(desc, last_field.format)?)
                .ok_or(RuntimeError::HostBufferTooLarge)
        }
    }
}

fn structured_stream_byte_len(
    desc: &StructuredBufferDesc,
    format: BufferFormat,
) -> Result<usize, RuntimeError> {
    let element_size = format.byte_len() as usize;
    match desc.layout {
        DataLayout::AoS => usize::try_from(desc.element_count)
            .ok()
            .and_then(|count| count.checked_mul(desc.source_stride as usize))
            .ok_or(RuntimeError::HostBufferTooLarge),
        DataLayout::SoA => usize::try_from(desc.element_count)
            .ok()
            .and_then(|count| count.checked_mul(element_size))
            .ok_or(RuntimeError::HostBufferTooLarge),
        DataLayout::AoSoA { group_size } => {
            let groups = desc.element_count.div_ceil(group_size);
            usize::try_from(groups)
                .ok()
                .and_then(|groups| groups.checked_mul(group_size as usize))
                .and_then(|slots| slots.checked_mul(element_size))
                .ok_or(RuntimeError::HostBufferTooLarge)
        }
    }
}

fn copy_structured_streams(
    desc: &StructuredBufferDesc,
    source_bytes: &[u8],
    stream_offsets: &[usize],
    dst: &mut [u8],
) -> Result<(), RuntimeError> {
    for element in 0..desc.element_count as usize {
        for (field_index, field) in desc.fields.iter().enumerate() {
            let element_size = field.format.byte_len() as usize;
            let src_offset = element
                .checked_mul(desc.source_stride as usize)
                .and_then(|offset| offset.checked_add(field.offset as usize))
                .ok_or(RuntimeError::HostBufferTooLarge)?;
            let dst_offset = match desc.layout {
                DataLayout::SoA => stream_offsets[field_index]
                    .checked_add(element * element_size)
                    .ok_or(RuntimeError::HostBufferTooLarge)?,
                DataLayout::AoSoA { group_size } => {
                    let group_size = group_size as usize;
                    let group = element / group_size;
                    let lane = element % group_size;
                    stream_offsets[field_index]
                        .checked_add(group * group_size * element_size)
                        .and_then(|offset| offset.checked_add(lane * element_size))
                        .ok_or(RuntimeError::HostBufferTooLarge)?
                }
                DataLayout::AoS => unreachable!("AoS does not use stream copy"),
            };
            dst[dst_offset..dst_offset + element_size]
                .copy_from_slice(&source_bytes[src_offset..src_offset + element_size]);
        }
    }
    Ok(())
}

fn validate_instance_buffer(
    desc: &InstanceBufferDesc,
    instance_bytes: &[u8],
) -> Result<(), RuntimeError> {
    if desc.instance_layout.stride == 0 {
        return Err(RuntimeError::Instance(
            "instance stride must be greater than zero".to_string(),
        ));
    }
    let expected_instance_bytes = usize::try_from(desc.instance_count)
        .ok()
        .and_then(|count| count.checked_mul(desc.instance_layout.stride as usize))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    if instance_bytes.len() != expected_instance_bytes {
        return Err(RuntimeError::Instance(format!(
            "expected {expected_instance_bytes} instance bytes, got {}",
            instance_bytes.len()
        )));
    }

    let mut seen = Vec::new();
    for attr in &desc.instance_layout.attributes {
        if seen.contains(&attr.semantic) {
            return Err(RuntimeError::Instance(format!(
                "duplicate instance semantic {:?}",
                attr.semantic
            )));
        }
        seen.push(attr.semantic);
        let end = attr
            .offset
            .checked_add(attr.format.byte_len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > desc.instance_layout.stride {
            return Err(RuntimeError::Instance(format!(
                "instance attribute {:?} extends past stride {}",
                attr.semantic, desc.instance_layout.stride
            )));
        }
    }
    Ok(())
}

fn visibility_macrocell_dims(desc: &VisibilityGridDesc) -> Result<[u32; 3], RuntimeError> {
    validate_visibility_grid_desc(desc)?;
    Ok([
        desc.cells[0].div_ceil(desc.macrocell_size),
        desc.cells[1].div_ceil(desc.macrocell_size),
        desc.cells[2].div_ceil(desc.macrocell_size),
    ])
}

fn visibility_macrocell_count(dims: [u32; 3]) -> Result<u32, RuntimeError> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn visibility_bitset_words(macrocell_count: u32) -> Result<u32, RuntimeError> {
    Ok(macrocell_count.div_ceil(32))
}

fn visibility_grid_u32_len(desc: &VisibilityGridDesc) -> Result<usize, RuntimeError> {
    let dims = visibility_macrocell_dims(desc)?;
    let count = visibility_macrocell_count(dims)?;
    let bitset_words = visibility_bitset_words(count)?;
    count
        .checked_mul(VISIBILITY_GRID_RECORD_U32S as u32)
        .and_then(|records| records.checked_add(VISIBILITY_GRID_HEADER_U32S as u32))
        .and_then(|records_and_header| records_and_header.checked_add(bitset_words))
        .and_then(|with_occupancy| with_occupancy.checked_add(bitset_words))
        .map(|values| values as usize)
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn pack_visibility_grid(desc: &VisibilityGridDesc) -> Result<Vec<u8>, RuntimeError> {
    let dims = visibility_macrocell_dims(desc)?;
    let count = visibility_macrocell_count(dims)?;
    let bitset_words = visibility_bitset_words(count)?;
    let record_offset = VISIBILITY_GRID_HEADER_U32S as u32;
    let occupancy_offset = record_offset
        .checked_add(
            count
                .checked_mul(VISIBILITY_GRID_RECORD_U32S as u32)
                .ok_or(RuntimeError::HostBufferTooLarge)?,
        )
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let relevance_offset = occupancy_offset
        .checked_add(bitset_words)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let mut values = vec![0u32; visibility_grid_u32_len(desc)?];
    values[0] = VISIBILITY_GRID_MAGIC;
    values[1] = desc.macrocell_size;
    values[2] = dims[0];
    values[3] = dims[1];
    values[4] = dims[2];
    values[5] = count;
    values[6] = occupancy_offset;
    values[7] = relevance_offset;

    let mut record_index = record_offset as usize;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let min_x = x * desc.macrocell_size;
                let min_y = y * desc.macrocell_size;
                let min_z = z * desc.macrocell_size;
                let max_x = (min_x + desc.macrocell_size - 1).min(desc.cells[0] - 1);
                let max_y = (min_y + desc.macrocell_size - 1).min(desc.cells[1] - 1);
                let max_z = (min_z + desc.macrocell_size - 1).min(desc.cells[2] - 1);
                values[record_index..record_index + VISIBILITY_GRID_RECORD_U32S]
                    .copy_from_slice(&[min_x, max_x, min_y, max_y, min_z, max_z]);
                record_index += VISIBILITY_GRID_RECORD_U32S;
            }
        }
    }

    for id in 0..count {
        let word = (id / 32) as usize;
        let bit = 1u32 << (id % 32);
        values[occupancy_offset as usize + word] |= bit;
        values[relevance_offset as usize + word] |= bit;
    }

    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn validate_visibility_grid_desc(desc: &VisibilityGridDesc) -> Result<(), RuntimeError> {
    if desc.macrocell_size == 0 {
        return Err(RuntimeError::VisibilityGrid(
            "macrocell size must be greater than zero".to_string(),
        ));
    }
    if desc.cells.contains(&0) {
        return Err(RuntimeError::VisibilityGrid(
            "visibility grid cell dimensions must be nonzero".to_string(),
        ));
    }
    Ok(())
}

fn sparse_texture_page_dims(desc: &SparseTextureDesc) -> Result<[u32; 2], RuntimeError> {
    validate_sparse_texture_desc(desc)?;
    Ok([
        desc.virtual_width.div_ceil(desc.page_size),
        desc.virtual_height.div_ceil(desc.page_size),
    ])
}

fn sparse_texture_virtual_page_count(desc: &SparseTextureDesc) -> Result<u32, RuntimeError> {
    let dims = sparse_texture_page_dims(desc)?;
    dims[0]
        .checked_mul(dims[1])
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_page_bytes(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    desc.page_size
        .checked_mul(desc.page_size)
        .and_then(|pixels| pixels.checked_mul(desc.format.bytes_per_pixel()))
        .map(|bytes| bytes as usize)
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_page_table_offset(virtual_page: u32) -> Result<usize, RuntimeError> {
    let page_offset = usize::try_from(virtual_page)
        .ok()
        .and_then(|page| page.checked_mul(SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S * 4))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    SPARSE_TEXTURE_HEADER_U32S
        .checked_mul(4)
        .and_then(|offset| offset.checked_add(page_offset))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_pages_offset(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    let page_table_bytes = usize::try_from(sparse_texture_virtual_page_count(desc)?)
        .ok()
        .and_then(|pages| pages.checked_mul(SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S * 4))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    Ok(align_usize(
        SPARSE_TEXTURE_HEADER_U32S * 4 + page_table_bytes,
        16,
    ))
}

fn sparse_texture_fallback_page_offset(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    let pages_offset = sparse_texture_pages_offset(desc)?;
    let page_bytes = sparse_texture_page_bytes(desc)?;
    let physical_bytes = usize::try_from(desc.physical_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(page_bytes))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    pages_offset
        .checked_add(physical_bytes)
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_feedback_offset(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    let fallback_offset = sparse_texture_fallback_page_offset(desc)?;
    let page_bytes = sparse_texture_page_bytes(desc)?;
    fallback_offset
        .checked_add(page_bytes)
        .map(|offset| align_usize(offset, 16))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_feedback_byte_len(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    usize::try_from(sparse_texture_virtual_page_count(desc)?)
        .ok()
        .and_then(|pages| pages.checked_mul(4))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn sparse_texture_physical_page_offset(
    desc: &SparseTextureDesc,
    page_index: u32,
) -> Result<usize, RuntimeError> {
    validate_sparse_physical_page(desc, page_index)?;
    let page_bytes = sparse_texture_page_bytes(desc)?;
    let page_offset = usize::try_from(page_index)
        .ok()
        .and_then(|page| page.checked_mul(page_bytes))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    sparse_texture_pages_offset(desc)?
        .checked_add(page_offset)
        .ok_or(RuntimeError::HostBufferTooLarge)
}

fn pack_sparse_texture(desc: &SparseTextureDesc) -> Result<Vec<u8>, RuntimeError> {
    validate_sparse_texture_desc(desc)?;
    let page_dims = sparse_texture_page_dims(desc)?;
    let virtual_pages = sparse_texture_virtual_page_count(desc)?;
    let pages_offset = sparse_texture_pages_offset(desc)?;
    let fallback_offset = sparse_texture_fallback_page_offset(desc)?;
    let feedback_offset = sparse_texture_feedback_offset(desc)?;
    let page_bytes = sparse_texture_page_bytes(desc)?;
    let feedback_bytes = sparse_texture_feedback_byte_len(desc)?;
    let total_bytes = feedback_offset
        .checked_add(feedback_bytes)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let mut blob = vec![0u8; total_bytes];
    let header = [
        SPARSE_TEXTURE_MAGIC,
        SPARSE_TEXTURE_VERSION,
        SPARSE_TEXTURE_HEADER_U32S as u32 * 4,
        desc.virtual_width,
        desc.virtual_height,
        desc.page_size,
        page_dims[0],
        page_dims[1],
        desc.mip_count,
        desc.format.code(),
        virtual_pages,
        desc.physical_pages,
        SPARSE_TEXTURE_HEADER_U32S as u32 * 4,
        pages_offset as u32,
        fallback_offset as u32,
        desc.gutter,
        feedback_offset as u32,
        virtual_pages,
        0,
        0,
    ];
    for (idx, value) in header.into_iter().enumerate() {
        write_u32_le(&mut blob, idx * 4, value);
    }
    fill_sparse_fallback_page(
        desc,
        &mut blob[fallback_offset..fallback_offset + page_bytes],
    )?;
    Ok(blob)
}

fn validate_sparse_texture_desc(desc: &SparseTextureDesc) -> Result<(), RuntimeError> {
    if desc.virtual_width == 0 || desc.virtual_height == 0 {
        return Err(RuntimeError::SparseTexture(
            "sparse texture dimensions must be greater than zero".to_string(),
        ));
    }
    if desc.page_size == 0 {
        return Err(RuntimeError::SparseTexture(
            "sparse texture page size must be greater than zero".to_string(),
        ));
    }
    if desc.mip_count != 1 {
        return Err(RuntimeError::SparseTexture(
            "v1 sparse textures support exactly one mip level".to_string(),
        ));
    }
    if desc.physical_pages == 0 {
        return Err(RuntimeError::SparseTexture(
            "sparse texture physical page count must be greater than zero".to_string(),
        ));
    }
    if desc.gutter >= desc.page_size / 2 {
        return Err(RuntimeError::SparseTexture(
            "sparse texture gutter must leave drawable page texels".to_string(),
        ));
    }
    let _ = sparse_texture_page_bytes(desc)?;
    Ok(())
}

fn validate_sparse_virtual_page(
    desc: &SparseTextureDesc,
    virtual_page: u32,
) -> Result<(), RuntimeError> {
    let pages = sparse_texture_virtual_page_count(desc)?;
    if virtual_page >= pages {
        return Err(RuntimeError::SparseTexture(format!(
            "virtual sparse page {virtual_page} is out of range for {pages} pages"
        )));
    }
    Ok(())
}

fn validate_sparse_physical_page(
    desc: &SparseTextureDesc,
    physical_page: u32,
) -> Result<(), RuntimeError> {
    validate_sparse_texture_desc(desc)?;
    if physical_page >= desc.physical_pages {
        return Err(RuntimeError::SparseTexture(format!(
            "physical sparse page {physical_page} is out of range for {} pages",
            desc.physical_pages
        )));
    }
    Ok(())
}

fn fill_sparse_checker_page(
    desc: &SparseTextureDesc,
    page_index: u32,
    dst: &mut [u8],
) -> Result<(), RuntimeError> {
    let expected = sparse_texture_page_bytes(desc)?;
    if dst.len() != expected {
        return Err(RuntimeError::SparseTexture(format!(
            "expected {expected} checker page bytes, got {}",
            dst.len()
        )));
    }
    let size = desc.page_size as usize;
    for y in 0..size {
        for x in 0..size {
            let tile = ((x / 16) ^ (y / 16) ^ page_index as usize) & 1;
            let base = (y * size + x) * 4;
            let hue = page_index.wrapping_mul(73);
            dst[base] = if tile == 0 {
                hue as u8
            } else {
                255u8.wrapping_sub(hue as u8)
            };
            dst[base + 1] = if tile == 0 {
                255u8.wrapping_sub((hue >> 1) as u8)
            } else {
                (hue >> 1) as u8
            };
            dst[base + 2] = if tile == 0 { (hue >> 2) as u8 } else { 255 };
            dst[base + 3] = 255;
        }
    }
    Ok(())
}

fn fill_sparse_fallback_page(desc: &SparseTextureDesc, dst: &mut [u8]) -> Result<(), RuntimeError> {
    let expected = sparse_texture_page_bytes(desc)?;
    if dst.len() != expected {
        return Err(RuntimeError::SparseTexture(format!(
            "expected {expected} fallback page bytes, got {}",
            dst.len()
        )));
    }
    let size = desc.page_size as usize;
    for y in 0..size {
        for x in 0..size {
            let checker = ((x / 8) ^ (y / 8)) & 1;
            let base = (y * size + x) * 4;
            dst[base] = if checker == 0 { 255 } else { 0 };
            dst[base + 1] = 0;
            dst[base + 2] = if checker == 0 { 255 } else { 0 };
            dst[base + 3] = 255;
        }
    }
    Ok(())
}

fn summarize_sparse_texture_feedback(
    counters: &[u32],
) -> Result<SparseTextureFeedbackSummary, RuntimeError> {
    let mut active_pages = 0u32;
    let mut total_requests = 0u64;
    let mut hottest_page = None;
    let mut hottest_requests = 0u32;
    for (page, requests) in counters.iter().copied().enumerate() {
        if requests != 0 {
            active_pages = active_pages.saturating_add(1);
            total_requests = total_requests.saturating_add(u64::from(requests));
            if requests > hottest_requests {
                hottest_requests = requests;
                hottest_page =
                    Some(u32::try_from(page).map_err(|_| RuntimeError::HostBufferTooLarge)?);
            }
        }
    }
    Ok(SparseTextureFeedbackSummary {
        active_pages,
        total_requests,
        hottest_page,
        hottest_requests,
    })
}

fn pack_material_stream(
    desc: &MaterialStreamDesc,
    material_ids: &[u32],
) -> Result<Vec<u8>, RuntimeError> {
    validate_material_stream(desc, material_ids)?;
    let data_offset = MATERIAL_STREAM_HEADER_U32S * 4;
    let data_bytes = material_ids
        .len()
        .checked_mul(4)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let total_bytes = data_offset
        .checked_add(data_bytes)
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    let mut blob = vec![0u8; total_bytes];
    let header = [
        MATERIAL_STREAM_MAGIC,
        MATERIAL_STREAM_VERSION,
        MATERIAL_STREAM_HEADER_U32S as u32 * 4,
        desc.material_count,
        data_offset as u32,
        0,
        0,
        0,
    ];
    for (idx, value) in header.into_iter().enumerate() {
        write_u32_le(&mut blob, idx * 4, value);
    }
    for (idx, value) in material_ids.iter().copied().enumerate() {
        write_u32_le(&mut blob, data_offset + idx * 4, value);
    }
    Ok(blob)
}

fn validate_material_stream(
    desc: &MaterialStreamDesc,
    material_ids: &[u32],
) -> Result<(), RuntimeError> {
    if desc.material_count == 0 {
        return Err(RuntimeError::MaterialStream(
            "material stream count must be greater than zero".to_string(),
        ));
    }
    if material_ids.len() != desc.material_count as usize {
        return Err(RuntimeError::MaterialStream(format!(
            "expected {} material IDs, got {}",
            desc.material_count,
            material_ids.len()
        )));
    }
    Ok(())
}

fn align_usize(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn write_u32_le(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn slice_as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

impl VertexSemantic {
    fn code(self) -> u32 {
        match self {
            Self::Position => MESH_SEMANTIC_POSITION,
            Self::Normal => MESH_SEMANTIC_NORMAL,
            Self::Uv0 => MESH_SEMANTIC_UV0,
            Self::Color0 => MESH_SEMANTIC_COLOR0,
        }
    }
}

impl VertexFormat {
    fn code(self) -> u32 {
        match self {
            Self::F32x2 => MESH_FORMAT_F32X2,
            Self::F32x3 => MESH_FORMAT_F32X3,
            Self::F32x4 => MESH_FORMAT_F32X4,
            Self::U8x4Unorm => MESH_FORMAT_U8X4_UNORM,
        }
    }

    fn byte_len(self) -> u32 {
        match self {
            Self::F32x2 => 8,
            Self::F32x3 => 12,
            Self::F32x4 => 16,
            Self::U8x4Unorm => 4,
        }
    }
}

impl InstanceSemantic {
    fn code(self) -> u32 {
        match self {
            Self::Position => INSTANCE_SEMANTIC_POSITION,
            Self::Rotation => INSTANCE_SEMANTIC_ROTATION,
            Self::Scale => INSTANCE_SEMANTIC_SCALE,
            Self::Color0 => INSTANCE_SEMANTIC_COLOR0,
        }
    }
}

impl InstanceFormat {
    fn byte_len(self) -> u32 {
        match self {
            Self::F32x2 => 8,
            Self::F32x3 => 12,
            Self::F32x4 => 16,
            Self::U8x4Unorm => 4,
        }
    }
}

impl From<InstanceFormat> for BufferFormat {
    fn from(value: InstanceFormat) -> Self {
        match value {
            InstanceFormat::F32x2 => Self::F32x2,
            InstanceFormat::F32x3 => Self::F32x3,
            InstanceFormat::F32x4 => Self::F32x4,
            InstanceFormat::U8x4Unorm => Self::U8x4Unorm,
        }
    }
}

impl IndexFormat {
    fn code(self) -> u32 {
        match self {
            Self::None => MESH_INDEX_NONE,
            Self::U16 => MESH_INDEX_U16,
            Self::U32 => MESH_INDEX_U32,
        }
    }

    fn byte_len(self) -> usize {
        match self {
            Self::None => 0,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

impl PrimitiveTopology {
    fn code(self) -> u32 {
        match self {
            Self::TriangleList => MESH_TOPOLOGY_TRIANGLE_LIST,
        }
    }
}

fn runtime_cuda_prelude() -> String {
    format!(
        r#"
struct NeoMeshHeader {{
    unsigned int magic;
    unsigned int version;
    unsigned int header_bytes;
    unsigned int vertex_count;
    unsigned int vertex_stride;
    unsigned int vertex_bytes_offset;
    unsigned int index_count;
    unsigned int index_format;
    unsigned int index_bytes_offset;
    unsigned int attr_count;
    unsigned int attr_bytes_offset;
    unsigned int topology;
}};

struct NeoMeshAttribute {{
    unsigned int semantic;
    unsigned int format;
    unsigned int offset;
    unsigned int reserved;
}};

__device__ __forceinline__ const NeoMeshHeader* neo_mesh_header(const unsigned char* mesh) {{
    return (const NeoMeshHeader*)mesh;
}}

__device__ __forceinline__ const NeoMeshAttribute* neo_mesh_attributes(const unsigned char* mesh) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    return (const NeoMeshAttribute*)(mesh + header->attr_bytes_offset);
}}

__device__ __forceinline__ unsigned int neo_mesh_vertex_count(const unsigned char* mesh) {{
    return neo_mesh_header(mesh)->vertex_count;
}}

__device__ __forceinline__ unsigned int neo_mesh_index_count(const unsigned char* mesh) {{
    return neo_mesh_header(mesh)->index_count;
}}

__device__ __forceinline__ unsigned int neo_mesh_triangle_count(const unsigned char* mesh) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    unsigned int element_count = header->index_count == 0u ? header->vertex_count : header->index_count;
    return element_count / 3u;
}}

__device__ __forceinline__ unsigned int neo_mesh_index(const unsigned char* mesh, unsigned int i) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    if (header->index_format == {MESH_INDEX_NONE}u) {{
        return i;
    }}
    const unsigned char* bytes = mesh + header->index_bytes_offset;
    if (header->index_format == {MESH_INDEX_U16}u) {{
        return ((const unsigned short*)bytes)[i];
    }}
    return ((const unsigned int*)bytes)[i];
}}

__device__ __forceinline__ const NeoMeshAttribute* neo_mesh_find_attr(const unsigned char* mesh, unsigned int semantic) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    const NeoMeshAttribute* attrs = neo_mesh_attributes(mesh);
    for (unsigned int i = 0u; i < header->attr_count; ++i) {{
        if (attrs[i].semantic == semantic) {{
            return &attrs[i];
        }}
    }}
    return 0;
}}

__device__ __forceinline__ float3 neo_mesh_position3f(const unsigned char* mesh, unsigned int vertex_id) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    if (vertex_id >= header->vertex_count) {{
        return make_float3(0.0f, 0.0f, 0.0f);
    }}
    const NeoMeshAttribute* attr = neo_mesh_find_attr(mesh, {MESH_SEMANTIC_POSITION}u);
    if (attr == 0) {{
        return make_float3(0.0f, 0.0f, 0.0f);
    }}
    const unsigned char* vertex = mesh + header->vertex_bytes_offset + vertex_id * header->vertex_stride + attr->offset;
    const float* values = (const float*)vertex;
    if (attr->format == {MESH_FORMAT_F32X2}u) {{
        return make_float3(values[0], values[1], 0.0f);
    }}
    if (attr->format == {MESH_FORMAT_F32X3}u || attr->format == {MESH_FORMAT_F32X4}u) {{
        return make_float3(values[0], values[1], values[2]);
    }}
    return make_float3(0.0f, 0.0f, 0.0f);
}}

__device__ __forceinline__ unsigned int neo_mesh_color4u8(const unsigned char* mesh, unsigned int vertex_id) {{
    const NeoMeshHeader* header = neo_mesh_header(mesh);
    if (vertex_id >= header->vertex_count) {{
        return 0xffffffffu;
    }}
    const NeoMeshAttribute* attr = neo_mesh_find_attr(mesh, {MESH_SEMANTIC_COLOR0}u);
    if (attr == 0 || attr->format != {MESH_FORMAT_U8X4_UNORM}u) {{
        return 0xffffffffu;
    }}
    const unsigned char* vertex = mesh + header->vertex_bytes_offset + vertex_id * header->vertex_stride + attr->offset;
    return *((const unsigned int*)vertex);
}}

struct NeoInstanceHeader {{
    unsigned int magic;
    unsigned int version;
    unsigned int header_bytes;
    unsigned int instance_count;
    unsigned int instance_stride;
    unsigned int instance_bytes_offset;
    unsigned int attr_count;
    unsigned int attr_bytes_offset;
    unsigned int layout_kind;
    unsigned int group_size;
}};

struct NeoInstanceAttribute {{
    unsigned int semantic;
    unsigned int format;
    unsigned int offset;
    unsigned int reserved;
}};

__device__ __forceinline__ const NeoInstanceHeader* neo_instance_header(const unsigned char* instances) {{
    return (const NeoInstanceHeader*)instances;
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_attributes(const unsigned char* instances) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    return (const NeoInstanceAttribute*)(instances + header->attr_bytes_offset);
}}

__device__ __forceinline__ unsigned int neo_instance_count(const unsigned char* instances) {{
    return neo_instance_header(instances)->instance_count;
}}

__device__ __forceinline__ unsigned int neo_instance_stride(const unsigned char* instances) {{
    return neo_instance_header(instances)->instance_stride;
}}

__device__ __forceinline__ unsigned int neo_instance_bytes_offset(const unsigned char* instances) {{
    return neo_instance_header(instances)->instance_bytes_offset;
}}

__device__ __forceinline__ unsigned int neo_instance_layout_kind(const unsigned char* instances) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    return header->version >= 2u ? header->layout_kind : {DATA_LAYOUT_AOS}u;
}}

__device__ __forceinline__ unsigned int neo_instance_group_size(const unsigned char* instances) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    unsigned int group_size = header->version >= 2u ? header->group_size : 1u;
    return group_size == 0u ? 1u : group_size;
}}

__device__ __forceinline__ const unsigned char* neo_instance_payload(const unsigned char* instances, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    if (neo_instance_layout_kind(instances) != {DATA_LAYOUT_AOS}u) {{
        return 0;
    }}
    return instances + header->instance_bytes_offset + instance_id * header->instance_stride;
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_find_attr(const unsigned char* instances, unsigned int semantic) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    const NeoInstanceAttribute* attrs = neo_instance_attributes(instances);
    for (unsigned int i = 0u; i < header->attr_count; ++i) {{
        if (attrs[i].semantic == semantic) {{
            return &attrs[i];
        }}
    }}
    return 0;
}}

__device__ __forceinline__ unsigned int neo_instance_format_size(unsigned int format);

__device__ __forceinline__ const unsigned char* neo_instance_attr_bytes(const unsigned char* instances, const NeoInstanceAttribute* attr, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    unsigned int layout_kind = neo_instance_layout_kind(instances);
    if (layout_kind == {DATA_LAYOUT_SOA}u) {{
        unsigned int element_size = neo_instance_format_size(attr->format);
        return instances + header->instance_bytes_offset + attr->offset + instance_id * element_size;
    }}
    if (layout_kind == {DATA_LAYOUT_AOSOA}u) {{
        unsigned int group_size = neo_instance_group_size(instances);
        unsigned int element_size = neo_instance_format_size(attr->format);
        unsigned int group = instance_id / group_size;
        unsigned int lane = instance_id - group * group_size;
        return instances + header->instance_bytes_offset + attr->offset + group * group_size * element_size + lane * element_size;
    }}
    return instances + header->instance_bytes_offset + instance_id * header->instance_stride + attr->offset;
}}

__device__ __forceinline__ unsigned int neo_instance_format_size(unsigned int format) {{
    if (format == {INSTANCE_FORMAT_F32X2}u) return 8u;
    if (format == {INSTANCE_FORMAT_F32X3}u) return 12u;
    if (format == {INSTANCE_FORMAT_F32X4}u) return 16u;
    if (format == {INSTANCE_FORMAT_U8X4_UNORM}u) return 4u;
    return 0u;
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_position_attr(const unsigned char* instances) {{
    return neo_instance_find_attr(instances, {INSTANCE_SEMANTIC_POSITION}u);
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_rotation_attr(const unsigned char* instances) {{
    return neo_instance_find_attr(instances, {INSTANCE_SEMANTIC_ROTATION}u);
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_scale_attr(const unsigned char* instances) {{
    return neo_instance_find_attr(instances, {INSTANCE_SEMANTIC_SCALE}u);
}}

__device__ __forceinline__ const NeoInstanceAttribute* neo_instance_color_attr(const unsigned char* instances) {{
    return neo_instance_find_attr(instances, {INSTANCE_SEMANTIC_COLOR0}u);
}}

__device__ __forceinline__ float3 neo_instance_position3f_attr(const unsigned char* instances, const NeoInstanceAttribute* attr, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    if (instance_id >= header->instance_count) {{
        return make_float3(0.0f, 0.0f, 0.0f);
    }}
    if (attr == 0) {{
        return make_float3(0.0f, 0.0f, 0.0f);
    }}
    const float* values = (const float*)neo_instance_attr_bytes(instances, attr, instance_id);
    if (attr->format == {INSTANCE_FORMAT_F32X2}u) {{
        return make_float3(values[0], values[1], 0.0f);
    }}
    if (attr->format == {INSTANCE_FORMAT_F32X3}u || attr->format == {INSTANCE_FORMAT_F32X4}u) {{
        return make_float3(values[0], values[1], values[2]);
    }}
    return make_float3(0.0f, 0.0f, 0.0f);
}}

__device__ __forceinline__ float3 neo_instance_position3f(const unsigned char* instances, unsigned int instance_id) {{
    return neo_instance_position3f_attr(instances, neo_instance_position_attr(instances), instance_id);
}}

__device__ __forceinline__ float4 neo_instance_rotation4f_attr(const unsigned char* instances, const NeoInstanceAttribute* attr, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    if (instance_id >= header->instance_count) {{
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }}
    if (attr == 0 || attr->format != {INSTANCE_FORMAT_F32X4}u) {{
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }}
    const float* values = (const float*)neo_instance_attr_bytes(instances, attr, instance_id);
    return make_float4(values[0], values[1], values[2], values[3]);
}}

__device__ __forceinline__ float4 neo_instance_rotation4f(const unsigned char* instances, unsigned int instance_id) {{
    return neo_instance_rotation4f_attr(instances, neo_instance_rotation_attr(instances), instance_id);
}}

__device__ __forceinline__ float2 neo_instance_scale2f_attr(const unsigned char* instances, const NeoInstanceAttribute* attr, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    if (instance_id >= header->instance_count) {{
        return make_float2(1.0f, 1.0f);
    }}
    if (attr == 0) {{
        return make_float2(1.0f, 1.0f);
    }}
    const float* values = (const float*)neo_instance_attr_bytes(instances, attr, instance_id);
    if (attr->format == {INSTANCE_FORMAT_F32X2}u || attr->format == {INSTANCE_FORMAT_F32X3}u || attr->format == {INSTANCE_FORMAT_F32X4}u) {{
        return make_float2(values[0], values[1]);
    }}
    return make_float2(1.0f, 1.0f);
}}

__device__ __forceinline__ float2 neo_instance_scale2f(const unsigned char* instances, unsigned int instance_id) {{
    return neo_instance_scale2f_attr(instances, neo_instance_scale_attr(instances), instance_id);
}}

__device__ __forceinline__ unsigned int neo_instance_color4u8_attr(const unsigned char* instances, const NeoInstanceAttribute* attr, unsigned int instance_id) {{
    const NeoInstanceHeader* header = neo_instance_header(instances);
    if (instance_id >= header->instance_count) {{
        return 0xffffffffu;
    }}
    if (attr == 0 || attr->format != {INSTANCE_FORMAT_U8X4_UNORM}u) {{
        return 0xffffffffu;
    }}
    return *((const unsigned int*)neo_instance_attr_bytes(instances, attr, instance_id));
}}

__device__ __forceinline__ unsigned int neo_instance_color4u8(const unsigned char* instances, unsigned int instance_id) {{
    return neo_instance_color4u8_attr(instances, neo_instance_color_attr(instances), instance_id);
}}

struct NeoSparseTextureHeader {{
    unsigned int magic;
    unsigned int version;
    unsigned int header_bytes;
    unsigned int virtual_width;
    unsigned int virtual_height;
    unsigned int page_size;
    unsigned int page_count_x;
    unsigned int page_count_y;
    unsigned int mip_count;
    unsigned int format;
    unsigned int virtual_page_count;
    unsigned int physical_page_count;
    unsigned int page_table_offset;
    unsigned int physical_pages_offset;
    unsigned int fallback_page_offset;
    unsigned int gutter;
    unsigned int feedback_offset;
    unsigned int feedback_count;
    unsigned int feedback_flags;
    unsigned int reserved0;
}};

struct NeoMaterialStreamHeader {{
    unsigned int magic;
    unsigned int version;
    unsigned int header_bytes;
    unsigned int material_count;
    unsigned int material_ids_offset;
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
}};

__device__ __forceinline__ const NeoSparseTextureHeader* neo_sparse_texture_header(const unsigned char* texture) {{
    return (const NeoSparseTextureHeader*)texture;
}}

__device__ __forceinline__ unsigned int neo_sparse_texture_width(const unsigned char* texture) {{
    return neo_sparse_texture_header(texture)->virtual_width;
}}

__device__ __forceinline__ unsigned int neo_sparse_texture_height(const unsigned char* texture) {{
    return neo_sparse_texture_header(texture)->virtual_height;
}}

__device__ __forceinline__ unsigned int neo_sparse_material_tile(const unsigned char* materials, unsigned int id) {{
    const NeoMaterialStreamHeader* header = (const NeoMaterialStreamHeader*)materials;
    if (id >= header->material_count) {{
        return 0u;
    }}
    return ((const unsigned int*)(materials + header->material_ids_offset))[id];
}}

__device__ __forceinline__ unsigned int neo_sparse_texture_page_entry(const unsigned char* texture, unsigned int page_id) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    if (page_id >= header->virtual_page_count) {{
        return 0u;
    }}
    return ((const unsigned int*)(texture + header->page_table_offset))[page_id];
}}

__device__ __forceinline__ unsigned int neo_sparse_texture_page_id(const unsigned char* texture, unsigned int material_id) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    unsigned int virtual_page_count = header->virtual_page_count == 0u ? 1u : header->virtual_page_count;
    return material_id % virtual_page_count;
}}

__device__ __forceinline__ unsigned int neo_sparse_page_id(const unsigned char* texture, unsigned int material_id) {{
    return neo_sparse_texture_page_id(texture, material_id);
}}

__device__ __forceinline__ unsigned int neo_sparse_page_resident(const unsigned char* texture, unsigned int page_id) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    unsigned int entry = neo_sparse_texture_page_entry(texture, page_id);
    unsigned int physical_page = entry & {SPARSE_TEXTURE_ENTRY_PHYSICAL_MASK}u;
    return ((entry & {SPARSE_TEXTURE_ENTRY_RESIDENT}u) != 0u && physical_page < header->physical_page_count) ? 1u : 0u;
}}

__device__ __forceinline__ void neo_sparse_texture_record_feedback(const unsigned char* texture, unsigned int page_id) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    if ((header->feedback_flags & {SPARSE_TEXTURE_FEEDBACK_ENABLED}u) == 0u || page_id >= header->feedback_count || header->feedback_offset == 0u) {{
        return;
    }}
    unsigned int* feedback = (unsigned int*)(texture + header->feedback_offset);
    atomicAdd(feedback + page_id, 1u);
}}

__device__ __forceinline__ unsigned int neo_sparse_feedback_hash(unsigned int page_id, unsigned int x, unsigned int y, unsigned int frame) {{
    unsigned int h = page_id * 2654435761u ^ x * 2246822519u ^ y * 3266489917u ^ frame * 668265263u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    h *= 3266489917u;
    h ^= h >> 16u;
    return h;
}}

__device__ __forceinline__ void neo_sparse_record_feedback_sampled(const unsigned char* texture, unsigned int page_id, unsigned int x, unsigned int y, unsigned int frame, unsigned int sample_rate) {{
    unsigned int rate = sample_rate == 0u ? 16u : sample_rate;
    if (rate <= 1u || (neo_sparse_feedback_hash(page_id, x, y, frame) % rate) == 0u) {{
        neo_sparse_texture_record_feedback(texture, page_id);
    }}
}}

__device__ __forceinline__ void neo_sparse_record_feedback_missing(const unsigned char* texture, unsigned int page_id) {{
    if (neo_sparse_page_resident(texture, page_id) == 0u) {{
        neo_sparse_texture_record_feedback(texture, page_id);
    }}
}}

__device__ __forceinline__ const unsigned char* neo_sparse_texture_page_bytes(const unsigned char* texture, unsigned int entry) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    unsigned int page_bytes = header->page_size * header->page_size * 4u;
    if ((entry & {SPARSE_TEXTURE_ENTRY_RESIDENT}u) == 0u) {{
        return texture + header->fallback_page_offset;
    }}
    unsigned int physical_page = entry & {SPARSE_TEXTURE_ENTRY_PHYSICAL_MASK}u;
    if (physical_page >= header->physical_page_count) {{
        return texture + header->fallback_page_offset;
    }}
    return texture + header->physical_pages_offset + physical_page * page_bytes;
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8_entry(const unsigned char* texture, unsigned int entry, float2 uv) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    float wrapped_u = uv.x - floorf(uv.x);
    float wrapped_v = uv.y - floorf(uv.y);
    const unsigned char* page = neo_sparse_texture_page_bytes(texture, entry);
    unsigned int gutter = header->gutter;
    unsigned int usable = header->page_size > gutter * 2u ? header->page_size - gutter * 2u : header->page_size;
    unsigned int sample_x = (unsigned int)(wrapped_u * (float)usable);
    unsigned int sample_y = (unsigned int)(wrapped_v * (float)usable);
    if (sample_x >= usable) sample_x = usable - 1u;
    if (sample_y >= usable) sample_y = usable - 1u;
    unsigned int texel_x = gutter + sample_x;
    unsigned int texel_y = gutter + sample_y;
    unsigned int offset = (texel_y * header->page_size + texel_x) * 4u;
    unsigned int r = page[offset + 0u];
    unsigned int g = page[offset + 1u];
    unsigned int b = page[offset + 2u];
    unsigned int a = page[offset + 3u];
    return b | (g << 8u) | (r << 16u) | (a << 24u);
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8_page(const unsigned char* texture, unsigned int page_id, float2 uv) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    unsigned int page_x = page_id % header->page_count_x;
    unsigned int page_y = page_id / header->page_count_x;
    unsigned int entry = neo_sparse_texture_page_entry(texture, page_y * header->page_count_x + page_x);
    return neo_sparse_sample_bgra8_entry(texture, entry, uv);
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8(const unsigned char* texture, unsigned int material_id, float2 uv) {{
    unsigned int page_id = neo_sparse_texture_page_id(texture, material_id);
    return neo_sparse_sample_bgra8_page(texture, page_id, uv);
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8_feedback(const unsigned char* texture, unsigned int material_id, float2 uv) {{
    unsigned int page_id = neo_sparse_texture_page_id(texture, material_id);
    neo_sparse_texture_record_feedback(texture, page_id);
    return neo_sparse_sample_bgra8_page(texture, page_id, uv);
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8_feedback_mode(const unsigned char* texture, unsigned int material_id, float2 uv, unsigned int x, unsigned int y, unsigned int frame, unsigned int feedback_mode, unsigned int sample_rate) {{
    unsigned int page_id = neo_sparse_texture_page_id(texture, material_id);
    if (feedback_mode == 1u) {{
        neo_sparse_record_feedback_sampled(texture, page_id, x, y, frame, sample_rate == 0u ? 16u : sample_rate);
    }} else if (feedback_mode == 2u) {{
        neo_sparse_record_feedback_sampled(texture, page_id, x, y, frame, sample_rate < 64u ? 64u : sample_rate);
    }} else if (feedback_mode == 3u) {{
        neo_sparse_record_feedback_missing(texture, page_id);
    }} else if (feedback_mode == 4u) {{
        neo_sparse_texture_record_feedback(texture, page_id);
    }}
    return neo_sparse_sample_bgra8_page(texture, page_id, uv);
}}

__device__ __forceinline__ float4 neo_sparse_sample_rgba8(const unsigned char* texture, unsigned int material_id, float2 uv) {{
    unsigned int bgra = neo_sparse_sample_bgra8(texture, material_id, uv);
    return make_float4(
        (float)((bgra >> 16u) & 255u) / 255.0f,
        (float)((bgra >> 8u) & 255u) / 255.0f,
        (float)(bgra & 255u) / 255.0f,
        (float)((bgra >> 24u) & 255u) / 255.0f);
}}

__device__ __forceinline__ float4 neo_sparse_sample_rgba8_feedback(const unsigned char* texture, unsigned int material_id, float2 uv) {{
    unsigned int bgra = neo_sparse_sample_bgra8_feedback(texture, material_id, uv);
    return make_float4(
        (float)((bgra >> 16u) & 255u) / 255.0f,
        (float)((bgra >> 8u) & 255u) / 255.0f,
        (float)(bgra & 255u) / 255.0f,
        (float)((bgra >> 24u) & 255u) / 255.0f);
}}

__device__ __forceinline__ float4 neo_sparse_sample_rgba8_feedback_mode(const unsigned char* texture, unsigned int material_id, float2 uv, unsigned int x, unsigned int y, unsigned int frame, unsigned int feedback_mode, unsigned int sample_rate) {{
    unsigned int bgra = neo_sparse_sample_bgra8_feedback_mode(texture, material_id, uv, x, y, frame, feedback_mode, sample_rate);
    return make_float4(
        (float)((bgra >> 16u) & 255u) / 255.0f,
        (float)((bgra >> 8u) & 255u) / 255.0f,
        (float)(bgra & 255u) / 255.0f,
        (float)((bgra >> 24u) & 255u) / 255.0f);
}}
"#
    )
}

pub fn nvrtc_available() -> bool {
    RuntimeDiagnostics::collect().nvrtc_loadable
}

fn compile_cuda_image_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    match compile_cubin_for_context_checked(ctx, cuda_source, diagnostics) {
        Ok(cubin) => return Ok(Ptx::from_binary(cubin)),
        Err(err) => {
            let _ = err;
        }
    }
    compile_ptx_checked(cuda_source, diagnostics)
}

fn compile_ptx_checked(
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_ptx(cuda_source)));
    std::panic::set_hook(panic_hook);
    result
        .map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))
}

fn compile_cubin_for_context_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let (major, minor) = ctx.inner.compute_capability()?;
    let arch = format!("sm_{major}{minor}");
    compile_cubin_checked(cuda_source, &arch, diagnostics)
}

fn compile_cubin_checked(
    cuda_source: &str,
    arch: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_cubin(cuda_source, arch)));
    std::panic::set_hook(panic_hook);
    result.map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
}

fn compile_cubin(cuda_source: &str, arch: &str) -> Result<Vec<u8>, RuntimeError> {
    use std::ffi::{CStr, CString};

    let src =
        CString::new(cuda_source.as_bytes()).expect("CUDA source cannot contain null terminators");
    let program = nvrtc_result::create_program(src.as_c_str(), None)
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let options = vec![format!("--gpu-architecture={arch}")];
    let compile_result = unsafe { nvrtc_result::compile_program(program, &options) };
    if let Err(err) = compile_result {
        let log = unsafe { nvrtc_result::get_program_log(program) }
            .ok()
            .map(|raw| {
                unsafe { CStr::from_ptr(raw.as_ptr()) }
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_default();
        unsafe {
            let _ = nvrtc_result::destroy_program(program);
        }
        return Err(RuntimeError::Nvrtc(format!(
            "native CUBIN compile failed for {arch}: {err}\n{log}"
        )));
    }
    let cubin = unsafe { nvrtc_get_cubin(program) };
    unsafe {
        let _ = nvrtc_result::destroy_program(program);
    }
    cubin
}

unsafe fn nvrtc_get_cubin(
    program: cudarc::nvrtc::sys::nvrtcProgram,
) -> Result<Vec<u8>, RuntimeError> {
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let mut cubin = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    Ok(cubin)
}

fn load_cuda_module_checked(
    ctx: &Context,
    image: Ptx,
) -> Result<Arc<cudarc::driver::CudaModule>, RuntimeError> {
    ctx.inner
        .load_module(image)
        .map_err(|err| unsupported_ptx_error(err).unwrap_or(RuntimeError::Driver(err)))
}

fn unsupported_ptx_error(err: DriverError) -> Option<RuntimeError> {
    (err.0 == sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION).then(|| {
        RuntimeError::Nvrtc(
            "CUDA driver rejected the compiled PTX because it was produced by a newer CUDA Toolkit than this driver supports. Update the NVIDIA driver, install a CUDA Toolkit matching the driver's reported CUDA version, or use Neo's native CUBIN path for the current GPU.".to_string(),
        )
    })
}

fn nvrtc_panic_help(payload: Box<dyn Any + Send>, diagnostics: &RuntimeDiagnostics) -> String {
    let panic_message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("cudarc panicked while loading NVRTC");
    format!("{panic_message}\n\n{}", diagnostics.nvrtc_loader_help())
}

#[cfg(windows)]
fn nvrtc_candidates() -> Vec<PathBuf> {
    let names = [
        "nvrtc.dll",
        "nvrtc64.dll",
        "nvrtc64_13.dll",
        "nvrtc64_130.dll",
        "nvrtc64_130_0.dll",
        "nvrtc64_12.dll",
        "nvrtc64_120.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_11.dll",
        "nvrtc64_112_0.dll",
    ];
    let mut dirs = Vec::new();
    let mut versioned_roots = std::env::vars_os()
        .filter_map(|(key, root)| {
            key.to_string_lossy()
                .starts_with("CUDA_PATH_V")
                .then(|| PathBuf::from(root))
        })
        .collect::<Vec<_>>();
    versioned_roots.sort_by(|left, right| right.cmp(left));
    for root in versioned_roots {
        push_cuda_root_bin_dirs(&mut dirs, root);
    }
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = std::env::var_os(key) {
            push_cuda_root_bin_dirs(&mut dirs, PathBuf::from(root));
        }
    }
    let mut toolkit_dirs = cuda_toolkit_bin_dirs();
    toolkit_dirs.sort_by(|left, right| right.cmp(left));
    for dir in toolkit_dirs {
        push_unique_path(&mut dirs, dir);
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            push_unique_path(&mut dirs, dir);
        }
    }
    for dir in nvidia_app_nvrtc_dirs() {
        push_unique_path(&mut dirs, dir);
    }

    dirs.into_iter()
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .collect()
}

#[cfg(windows)]
fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

#[cfg(windows)]
fn push_cuda_root_bin_dirs(paths: &mut Vec<PathBuf>, root: PathBuf) {
    let bin = root.join("bin");
    push_unique_path(paths, bin.join("x64"));
    push_unique_path(paths, bin);
}

#[cfg(windows)]
fn compatible_nvrtc_candidate(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    matches!(
        name.to_ascii_lowercase().as_str(),
        "nvrtc.dll" | "nvrtc64.dll" | "nvrtc64_13.dll" | "nvrtc64_130.dll" | "nvrtc64_130_0.dll"
    )
}

#[cfg(not(windows))]
fn compatible_nvrtc_candidate(_path: &Path) -> bool {
    true
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
    pub nvrtc_compatible: Vec<PathBuf>,
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
        let nvrtc_compatible = nvrtc_found
            .iter()
            .filter(|candidate| compatible_nvrtc_candidate(candidate))
            .cloned()
            .collect::<Vec<_>>();
        let nvrtc_loadable = !nvrtc_compatible.is_empty();
        Self {
            cuda_driver_available,
            cuda_driver_error,
            nvrtc_candidates,
            nvrtc_found,
            nvrtc_compatible,
            nvrtc_loadable,
        }
    }

    pub fn nvrtc_help(&self) -> String {
        if !self.nvrtc_compatible.is_empty() {
            return format!(
                "NVRTC was found, but the dynamic loader could not use it.\n\n{}",
                self.nvrtc_loader_help()
            );
        }
        if !self.nvrtc_found.is_empty() {
            return format!(
                "NVRTC was found, but not a CUDA 13-compatible NVRTC for this Neo build.\n\n{}",
                self.nvrtc_loader_help()
            );
        }
        "NVRTC shared library was not found. Install the NVIDIA CUDA Toolkit or add the directory containing nvrtc64_130_0.dll/nvrtc64_13.dll, nvrtc64_120_0.dll/nvrtc64_12.dll, or the matching NVRTC DLL to PATH.".to_string()
    }

    pub fn nvrtc_loader_help(&self) -> String {
        if let Some(found) = self.nvrtc_compatible.first() {
            return format!(
                "Neo found NVRTC at {} and tried to register {} with the process DLL loader. If this still fails, launch Neo from a shell where that CUDA bin directory is on PATH, or set CUDA_PATH/CUDA_PATH_V13_0 to the CUDA Toolkit root before starting Neo.",
                found.display(),
                found.parent().unwrap_or_else(|| Path::new("")).display()
            );
        }
        if let Some(found) = self.nvrtc_found.first() {
            let checked = self
                .nvrtc_candidates
                .iter()
                .filter(|candidate| compatible_nvrtc_candidate(candidate))
                .take(16)
                .map(|candidate| format!("  - {}", candidate.display()))
                .collect::<Vec<_>>()
                .join("\n");
            let checked = if checked.is_empty() {
                "  - no CUDA 13-compatible candidate names were generated".to_string()
            } else {
                checked
            };
            return format!(
                "Neo found NVRTC at {}, but this build expects CUDA 13-compatible NVRTC names such as nvrtc64_130_0.dll, nvrtc64_13.dll, or nvrtc.dll from the CUDA Toolkit.\nSet CUDA_PATH_V13_0 or CUDA_PATH to your CUDA 13 Toolkit root before starting Neo.\nChecked CUDA 13-compatible candidates:\n{}",
                found.display(),
                checked
            );
        }
        let checked = self
            .nvrtc_candidates
            .iter()
            .filter(|candidate| compatible_nvrtc_candidate(candidate))
            .take(16)
            .map(|candidate| format!("  - {}", candidate.display()))
            .collect::<Vec<_>>()
            .join("\n");
        if checked.is_empty() {
            "Neo did not generate any CUDA 13-compatible NVRTC candidate paths. Set CUDA_PATH_V13_0 or CUDA_PATH to your CUDA 13 Toolkit root before starting Neo.".to_string()
        } else {
            format!(
                "Neo could not find a CUDA 13-compatible NVRTC DLL. Set CUDA_PATH_V13_0 or CUDA_PATH to your CUDA 13 Toolkit root before starting Neo.\nChecked CUDA 13-compatible candidates:\n{checked}"
            )
        }
    }
}

#[cfg(windows)]
fn configure_nvrtc_search_path(diagnostics: &RuntimeDiagnostics) {
    let Some(dir) = diagnostics
        .nvrtc_compatible
        .first()
        .and_then(|path| path.parent())
    else {
        return;
    };

    register_windows_dll_directory(dir);

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

#[cfg(windows)]
fn register_windows_dll_directory(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::System::LibraryLoader::SetDllDirectoryW;
    use windows::core::PCWSTR;

    let wide = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    unsafe {
        let _ = SetDllDirectoryW(PCWSTR(wide.as_ptr()));
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
        .flat_map(|entry| {
            let mut dirs = Vec::new();
            push_cuda_root_bin_dirs(&mut dirs, entry.path());
            dirs
        })
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
        self.record_on_stream(&ctx.default_stream())
    }

    pub fn record_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        unsafe {
            sys::cuEventRecord(self.event, stream.inner.cu_stream())
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

    pub fn is_complete(&self) -> Result<bool, RuntimeError> {
        match unsafe { sys::cuEventQuery(self.event) } {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            err => Err(RuntimeError::Driver(cudarc::driver::DriverError(err))),
        }
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
        Self::new_on_stream(&ctx.default_stream(), len)
    }

    pub fn new_on_stream(stream: &Stream, len: usize) -> Result<Self, RuntimeError> {
        let inner = stream.inner.alloc_zeros(len)?;
        Ok(Self { inner })
    }
}

impl<T> DeviceBuffer<T>
where
    T: DeviceRepr,
{
    pub fn upload(ctx: &Context, values: &[T]) -> Result<Self, RuntimeError> {
        Self::upload_on_stream(&ctx.default_stream(), values)
    }

    pub fn upload_on_stream(stream: &Stream, values: &[T]) -> Result<Self, RuntimeError> {
        let inner = stream.inner.clone_htod(values)?;
        Ok(Self { inner })
    }

    pub fn download(&self) -> Result<Vec<T>, RuntimeError> {
        Ok(self.inner.stream().clone_dtoh(&self.inner)?)
    }

    pub fn download_into(&self, dst: &mut [T]) -> Result<(), RuntimeError> {
        self.inner.stream().memcpy_dtoh(&self.inner, dst)?;
        Ok(())
    }

    pub fn download_range(&self, byte_offset: usize, dst: &mut [u8]) -> Result<(), RuntimeError> {
        self.download_range_on_stream(
            &Stream {
                inner: self.inner.stream().clone(),
            },
            byte_offset,
            dst,
        )
    }

    pub fn download_range_on_stream(
        &self,
        stream: &Stream,
        byte_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtr as _;

        let end = byte_offset
            .checked_add(dst.len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > self.inner.num_bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let (src, _record_read) = self.inner.device_ptr(&stream.inner);
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr().cast(),
                src + byte_offset as u64,
                dst.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        stream.synchronize()?;
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
        let stream = Stream {
            inner: self.inner.stream().clone(),
        };
        self.download_into_readable_pinned_on_stream(&stream, dst)
    }

    pub fn download_into_readable_pinned_on_stream(
        &self,
        stream: &Stream,
        dst: &mut ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtr as _;

        let (src, _record_read) = self.inner.device_ptr(&stream.inner);
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.ptr.cast(),
                src,
                self.inner.num_bytes(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_from_readable_pinned_on_stream(
        &mut self,
        stream: &Stream,
        src: &ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        let byte_len = self.inner.num_bytes();
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.ptr.cast(), byte_len, stream.inner.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_from_on_stream(
        &mut self,
        stream: &Stream,
        src: &[T],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        if src.len() != self.inner.len() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let byte_len = self.inner.num_bytes();
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr().cast(), byte_len, stream.inner.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_range(&mut self, byte_offset: usize, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.upload_range_on_stream(
            &Stream {
                inner: self.inner.stream().clone(),
            },
            byte_offset,
            bytes,
        )
    }

    pub fn upload_range_on_stream(
        &mut self,
        stream: &Stream,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        let end = byte_offset
            .checked_add(bytes.len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > self.inner.num_bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst + byte_offset as u64,
                bytes.as_ptr().cast(),
                bytes.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        use cudarc::driver::DevicePtr as _;

        let (ptr, _record_read) = self.inner.device_ptr(self.inner.stream());
        CudaDevicePtrArg::new(ptr)
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

    pub fn on_stream(&self, stream: &Stream) -> Self {
        Self {
            function: self.function.clone(),
            stream: stream.inner.clone(),
        }
    }
}

pub struct KernelLaunch<'a> {
    inner: LaunchArgs<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDevicePtrArg {
    ptr: sys::CUdeviceptr,
}

impl CudaDevicePtrArg {
    pub fn new(ptr: sys::CUdeviceptr) -> Self {
        Self { ptr }
    }
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

    pub fn arg_device_ptr(&mut self, value: &'a CudaDevicePtrArg) -> &mut Self {
        self.inner.arg(&value.ptr);
        self
    }

    pub fn arg_mesh(&mut self, value: &'a MeshBuffer) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_instances(&mut self, value: &'a InstanceBuffer) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_visibility_grid(&mut self, value: &'a VisibilityGrid) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_sparse_texture(&mut self, value: &'a SparseTextureAtlas) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_materials(&mut self, value: &'a MaterialStream) -> &mut Self {
        self.arg_buffer(&value.buffer)
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

impl Kernel {
    /// Launches the current live image ABI with a raw CUDA device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `pixels` is valid for writes of the kernel's
    /// full output, that its lifetime extends until the launched work is
    /// complete, and that no other GPU queue aliases it without explicit
    /// synchronization.
    pub unsafe fn launch_image_raw_ptr(
        &self,
        dims: LaunchDims,
        pixels: CudaDevicePtrArg,
        width: u32,
        height: u32,
        time: f32,
        frame: u32,
    ) -> Result<(), RuntimeError> {
        let pixel_ptr = pixels.ptr;
        unsafe {
            self.launcher()
                .arg(&pixel_ptr)
                .arg(&width)
                .arg(&height)
                .arg(&time)
                .arg(&frame)
                .launch(dims)?;
        }
        Ok(())
    }
}

#[cfg(windows)]
pub struct NeoD3d12InteropDevice {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
}

#[cfg(windows)]
impl NeoD3d12InteropDevice {
    pub fn new(ctx: &Context) -> Result<Self, RuntimeError> {
        use windows::Win32::Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_11_0,
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                D3D12CreateDevice, ID3D12CommandQueue, ID3D12Device,
            },
            Dxgi::{
                CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
                IDXGIAdapter1, IDXGIFactory1,
            },
        };

        let cuda_luid = cuda_device_luid(ctx)?;
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let mut adapter_index = 0;
        let mut matched: Option<IDXGIAdapter1> = None;
        loop {
            let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
                Ok(adapter) => adapter,
                Err(_) => break,
            };
            let desc = unsafe { adapter.GetDesc1()? };
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0
                && dxgi_luid_bytes(desc.AdapterLuid) == cuda_luid
            {
                matched = Some(adapter);
                break;
            }
            adapter_index += 1;
        }
        let adapter = matched.ok_or_else(|| {
            RuntimeError::D3d12Interop(
                "could not find a DXGI adapter matching CUDA device 0 LUID".to_string(),
            )
        })?;
        let mut device: Option<ID3D12Device> = None;
        unsafe {
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)?;
        }
        let device = device.ok_or_else(|| {
            RuntimeError::D3d12Interop("D3D12CreateDevice returned no device".to_string())
        })?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc)? };
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Device {
        &self.device
    }

    pub fn queue(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue {
        &self.queue
    }

    pub fn create_shared_frame_ring(
        &self,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<SharedFrameRing, RuntimeError> {
        SharedFrameRing::new(&self.device, width, height, slots)
    }

    pub fn create_shared_gpu_buffer(&self, byte_len: u64) -> Result<SharedGpuBuffer, RuntimeError> {
        SharedGpuBuffer::new(&self.device, byte_len)
    }
}

#[cfg(windows)]
#[derive(Clone)]
pub struct DrawDevice {
    interop: Arc<NeoD3d12InteropDevice>,
}

#[cfg(windows)]
pub type RasterDevice = DrawDevice;

#[cfg(windows)]
impl DrawDevice {
    pub fn new(ctx: &Context) -> Result<Self, RuntimeError> {
        Ok(Self {
            interop: Arc::new(NeoD3d12InteropDevice::new(ctx)?),
        })
    }

    pub fn from_interop(interop: NeoD3d12InteropDevice) -> Self {
        Self {
            interop: Arc::new(interop),
        }
    }

    pub fn interop(&self) -> &NeoD3d12InteropDevice {
        &self.interop
    }

    pub fn create_shared_gpu_buffer(&self, byte_len: u64) -> Result<SharedGpuBuffer, RuntimeError> {
        self.interop.create_shared_gpu_buffer(byte_len)
    }
}

#[cfg(windows)]
pub struct DrawPipeline {
    label: String,
}

#[cfg(windows)]
pub type RasterPipeline = DrawPipeline;

#[cfg(windows)]
impl DrawPipeline {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct GeometryStream<'a> {
    mesh: &'a MeshBuffer,
}

#[cfg(windows)]
impl<'a> GeometryStream<'a> {
    pub fn from_mesh(mesh: &'a MeshBuffer) -> Self {
        Self { mesh }
    }

    pub fn mesh(&self) -> &'a MeshBuffer {
        self.mesh
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct InstanceStream<'a> {
    instances: &'a InstanceBuffer,
}

#[cfg(windows)]
impl<'a> InstanceStream<'a> {
    pub fn from_instances(instances: &'a InstanceBuffer) -> Self {
        Self { instances }
    }

    pub fn instances(&self) -> &'a InstanceBuffer {
        self.instances
    }
}

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

#[cfg(windows)]
pub const DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS: u32 = 850;

#[cfg(windows)]
pub const DEFAULT_MIN_PROJECTED_MILLIPIXELS: u32 = DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS;

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawPolicyConfig {
    pub policy: DrawPolicy,
    pub depth: DrawDepthMode,
    pub cull_order: CullOrder,
    pub visibility: VisibilityMode,
    pub min_projected_millipixels: u32,
}

#[cfg(windows)]
impl DrawPolicyConfig {
    pub fn draw_all() -> Self {
        Self {
            policy: DrawPolicy::DrawAll,
            depth: DrawDepthMode::Auto,
            cull_order: CullOrder::StableDense,
            visibility: VisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn compute_culled(cull_order: CullOrder) -> Self {
        Self {
            policy: DrawPolicy::ComputeCulled,
            depth: DrawDepthMode::Auto,
            cull_order,
            visibility: VisibilityMode::Frustum,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn compute_culled_with_visibility(
        cull_order: CullOrder,
        visibility: VisibilityMode,
    ) -> Self {
        Self {
            policy: DrawPolicy::ComputeCulled,
            depth: DrawDepthMode::Auto,
            cull_order,
            visibility,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn cuda_tiled() -> Self {
        Self {
            policy: DrawPolicy::CudaTiled,
            depth: DrawDepthMode::Auto,
            cull_order: CullOrder::StableDense,
            visibility: VisibilityMode::ProjectedSize,
            min_projected_millipixels: DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS,
        }
    }

    pub fn with_min_projected_millipixels(mut self, min_projected_millipixels: u32) -> Self {
        self.min_projected_millipixels = min_projected_millipixels;
        self
    }

    pub fn with_depth(mut self, depth: DrawDepthMode) -> Self {
        self.depth = depth;
        self
    }

    pub fn backend(self) -> DrawBackend {
        self.policy.backend()
    }

    pub fn policy_label(self) -> &'static str {
        self.policy.label()
    }

    pub fn backend_label(self) -> &'static str {
        self.backend().label()
    }

    pub fn cull_order_label(self) -> &'static str {
        self.cull_order.label()
    }

    pub fn depth_label(self) -> &'static str {
        self.depth.label()
    }

    pub fn uses_depth(self) -> bool {
        self.depth.uses_depth(self.policy)
    }

    pub fn visibility_label(self) -> &'static str {
        self.visibility.label()
    }

    pub fn min_projected_pixels(self) -> f32 {
        self.min_projected_millipixels as f32 / 1000.0
    }
}

#[cfg(windows)]
impl From<DrawPolicy> for DrawPolicyConfig {
    fn from(policy: DrawPolicy) -> Self {
        match policy {
            DrawPolicy::DrawAll => Self::draw_all(),
            DrawPolicy::ComputeCulled => Self::compute_culled(CullOrder::AtomicCompact),
            DrawPolicy::CudaTiled => Self::cuda_tiled(),
        }
    }
}

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

#[cfg(windows)]
pub struct DrawExecution<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
pub type RasterDraw<'a> = DrawExecution<'a>;

#[cfg(windows)]
pub type RasterDrawBuilder<'a> = DrawExecutionBuilder<'a>;

#[cfg(windows)]
impl<'a> DrawExecution<'a> {
    pub fn execution_builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> DrawExecutionBuilder<'a> {
        Self::builder(geometry, material, target)
    }

    pub fn builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> DrawExecutionBuilder<'a> {
        DrawExecutionBuilder {
            geometry,
            instances: None,
            material,
            target,
            policy: DrawPolicyConfig::draw_all(),
        }
    }

    pub fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    pub fn instances(&self) -> Option<InstanceStream<'a>> {
        self.instances
    }

    pub fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn policy(&self) -> DrawPolicy {
        self.policy.policy
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }

    pub fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    pub fn contract(&self) -> DrawContract {
        DrawRecipe::contract(self)
    }
}

#[cfg(windows)]
impl<'a> DrawRecipe<'a> for DrawExecution<'a> {
    fn backend(&self) -> DrawBackend {
        DrawBackend::HardwareRaster
    }

    fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    fn instances(&self) -> Option<InstanceStream<'a>> {
        self.instances
    }

    fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    fn target(&self) -> Target {
        self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }
}

#[cfg(windows)]
pub struct DrawExecutionBuilder<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> DrawExecutionBuilder<'a> {
    pub fn instance_stream(mut self, instances: InstanceStream<'a>) -> Self {
        self.instances = Some(instances);
        self
    }

    pub fn draw_policy(mut self, policy: DrawPolicy) -> Self {
        self.policy = policy.into();
        self
    }

    pub fn draw_policy_config(mut self, policy: DrawPolicyConfig) -> Self {
        self.policy = policy;
        self
    }

    pub fn compute_culled(mut self, cull_order: CullOrder) -> Self {
        self.policy = DrawPolicyConfig::compute_culled(cull_order);
        self
    }

    pub fn compute_culled_with_visibility(
        mut self,
        cull_order: CullOrder,
        visibility: VisibilityMode,
    ) -> Self {
        self.policy = DrawPolicyConfig::compute_culled_with_visibility(cull_order, visibility);
        self
    }

    pub fn compute_culled_projected(
        mut self,
        cull_order: CullOrder,
        min_projected_millipixels: u32,
    ) -> Self {
        self.policy = DrawPolicyConfig::compute_culled_with_visibility(
            cull_order,
            VisibilityMode::ProjectedSize,
        )
        .with_min_projected_millipixels(min_projected_millipixels);
        self
    }

    pub fn try_build(self) -> Result<DrawExecution<'a>, RuntimeError> {
        let abi = self.material.abi();
        if abi.is_cuda_tiled() {
            return Err(RuntimeError::Raster(format!(
                "DrawExecution requires a draw-execution MaterialKernel, got CUDA tiled material `{}`",
                self.material.label()
            )));
        }
        let policy = self.policy.policy;
        if policy == DrawPolicy::CudaTiled {
            return Err(RuntimeError::Raster(format!(
                "DrawExecution material `{}` cannot use DrawPolicy::CudaTiled",
                self.material.label()
            )));
        }
        if abi.requires_instance_stream() && self.instances.is_none() {
            return Err(RuntimeError::Raster(format!(
                "raster material `{}` requires an explicit InstanceStream",
                self.material.label()
            )));
        }
        if abi.requires_compute_culling() && policy != DrawPolicy::ComputeCulled {
            return Err(RuntimeError::Raster(format!(
                "raster material `{}` requires DrawPolicy::ComputeCulled",
                self.material.label()
            )));
        }
        if policy == DrawPolicy::ComputeCulled && !abi.requires_compute_culling() {
            return Err(RuntimeError::Raster(format!(
                "DrawPolicy::ComputeCulled requires material `{}` to read the visible InstanceStream",
                self.material.label()
            )));
        }
        if policy == DrawPolicy::ComputeCulled && self.instances.is_none() {
            return Err(RuntimeError::Raster(
                "DrawPolicy::ComputeCulled requires an explicit InstanceStream".to_string(),
            ));
        }
        Ok(DrawExecution {
            geometry: self.geometry,
            instances: self.instances,
            material: self.material,
            target: self.target,
            policy: self.policy,
        })
    }

    pub fn build(self) -> DrawExecution<'a> {
        self.try_build()
            .expect("invalid draw execution recipe; use try_build for recoverable validation")
    }
}

#[cfg(windows)]
pub struct CudaDraw<'a> {
    geometry: GeometryStream<'a>,
    instances: InstanceStream<'a>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> CudaDraw<'a> {
    pub fn builder(
        geometry: GeometryStream<'a>,
        material: &'a MaterialKernel,
        target: Target,
    ) -> CudaDrawBuilder<'a> {
        CudaDrawBuilder {
            geometry,
            instances: None,
            material,
            target,
            policy: DrawPolicyConfig::cuda_tiled(),
        }
    }

    pub fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    pub fn instances(&self) -> InstanceStream<'a> {
        self.instances
    }

    pub fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn policy(&self) -> DrawPolicy {
        self.policy.policy
    }

    pub fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }

    pub fn backend(&self) -> DrawBackend {
        DrawBackend::CudaTiled
    }

    pub fn contract(&self) -> DrawContract {
        DrawRecipe::contract(self)
    }
}

#[cfg(windows)]
impl<'a> DrawRecipe<'a> for CudaDraw<'a> {
    fn backend(&self) -> DrawBackend {
        DrawBackend::CudaTiled
    }

    fn geometry(&self) -> GeometryStream<'a> {
        self.geometry
    }

    fn instances(&self) -> Option<InstanceStream<'a>> {
        Some(self.instances)
    }

    fn material(&self) -> &'a MaterialKernel {
        self.material
    }

    fn target(&self) -> Target {
        self.target
    }

    fn policy_config(&self) -> DrawPolicyConfig {
        self.policy
    }
}

#[cfg(windows)]
pub struct CudaDrawBuilder<'a> {
    geometry: GeometryStream<'a>,
    instances: Option<InstanceStream<'a>>,
    material: &'a MaterialKernel,
    target: Target,
    policy: DrawPolicyConfig,
}

#[cfg(windows)]
impl<'a> CudaDrawBuilder<'a> {
    pub fn instance_stream(mut self, instances: InstanceStream<'a>) -> Self {
        self.instances = Some(instances);
        self
    }

    pub fn draw_policy_config(mut self, policy: DrawPolicyConfig) -> Self {
        self.policy = policy;
        self
    }

    pub fn try_build(self) -> Result<CudaDraw<'a>, RuntimeError> {
        let abi = self.material.abi();
        if !abi.is_cuda_tiled() {
            return Err(RuntimeError::Raster(format!(
                "CudaDraw requires a CUDA tiled MaterialKernel, got hardware raster material `{}`",
                self.material.label()
            )));
        }
        if self.policy.policy != DrawPolicy::CudaTiled {
            return Err(RuntimeError::Raster(format!(
                "CudaDraw material `{}` requires DrawPolicy::CudaTiled",
                self.material.label()
            )));
        }
        let instances = self.instances.ok_or_else(|| {
            RuntimeError::Raster(format!(
                "CudaDraw material `{}` requires an explicit InstanceStream",
                self.material.label()
            ))
        })?;
        Ok(CudaDraw {
            geometry: self.geometry,
            instances,
            material: self.material,
            target: self.target,
            policy: self.policy,
        })
    }

    pub fn build(self) -> CudaDraw<'a> {
        self.try_build()
            .expect("invalid CUDA draw recipe; use try_build for recoverable validation")
    }
}

#[cfg(windows)]
pub struct IndirectDrawBuffer {
    buffer: SharedGpuBuffer,
    command_capacity: u32,
}

#[cfg(windows)]
impl IndirectDrawBuffer {
    pub fn new(
        device: &NeoD3d12InteropDevice,
        command_capacity: u32,
    ) -> Result<Self, RuntimeError> {
        if command_capacity == 0 {
            return Err(RuntimeError::Raster(
                "indirect draw command capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(command_capacity)
            .checked_mul(std::mem::size_of::<DrawIndexedIndirectCommand>() as u64)
            .ok_or_else(|| {
                RuntimeError::Raster("indirect draw buffer size overflow".to_string())
            })?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            command_capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn command_capacity(&self) -> u32 {
        self.command_capacity
    }
}

#[cfg(windows)]
pub struct VisibleInstanceStream {
    buffer: SharedGpuBuffer,
    capacity: u32,
}

#[cfg(windows)]
impl VisibleInstanceStream {
    pub fn new(device: &NeoD3d12InteropDevice, capacity: u32) -> Result<Self, RuntimeError> {
        if capacity == 0 {
            return Err(RuntimeError::Raster(
                "visible instance stream capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(capacity)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(windows)]
pub struct SharedInstanceStream {
    buffer: SharedGpuBuffer,
    desc: InstanceBufferDesc,
    data_layout: DataLayout,
    byte_len: usize,
}

#[cfg(windows)]
impl SharedInstanceStream {
    pub fn upload_typed<I>(
        ctx: &Context,
        device: &NeoD3d12InteropDevice,
        desc: InstanceBufferDesc,
        instances: &[I],
        data_layout: DataLayout,
    ) -> Result<Self, RuntimeError>
    where
        I: Copy,
    {
        let packed = InstanceBuffer::pack_typed_with_layout(&desc, instances, data_layout)?;
        let byte_len = packed.len();
        let mut buffer = device.create_shared_gpu_buffer(byte_len as u64)?;
        let stream = ctx.default_stream();
        buffer.upload_bytes_on_stream(&stream, &packed)?;
        ctx.synchronize()?;
        Ok(Self {
            buffer,
            desc,
            data_layout,
            byte_len,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn desc(&self) -> &InstanceBufferDesc {
        &self.desc
    }

    pub fn data_layout(&self) -> DataLayout {
        self.data_layout
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count_per_instance: u32,
    pub instance_count: u32,
    pub start_index_location: u32,
    pub base_vertex_location: i32,
    pub start_instance_location: u32,
}

impl DrawIndexedIndirectCommand {
    pub const BYTE_LEN: usize = std::mem::size_of::<Self>();

    pub fn indexed_quad(instance_count: u32) -> Self {
        Self {
            index_count_per_instance: 6,
            instance_count,
            start_index_location: 0,
            base_vertex_location: 0,
            start_instance_location: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[cfg(windows)]
pub struct SharedGpuBuffer {
    slot: SharedFrameSlot,
}

#[cfg(windows)]
impl SharedGpuBuffer {
    pub fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        byte_len: u64,
    ) -> Result<Self, RuntimeError> {
        if byte_len == 0 {
            return Err(RuntimeError::Raster(
                "shared GPU buffer size must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            slot: SharedFrameSlot::new(device, 0, byte_len)?,
        })
    }

    pub fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        self.slot.resource()
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        self.slot.device_ptr_arg()
    }

    pub fn bytes(&self) -> u64 {
        self.slot.bytes()
    }

    pub fn upload_bytes_on_stream(
        &mut self,
        stream: &Stream,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        if bytes.len() as u64 > self.bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                self.slot.device_ptr,
                bytes.as_ptr().cast(),
                bytes.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn wait_available_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        self.slot.wait_available_on_stream(stream)
    }

    pub fn signal_cuda_complete_on_stream(&mut self, stream: &Stream) -> Result<u64, RuntimeError> {
        self.slot.signal_cuda_complete_on_stream(stream)
    }

    pub fn wait_d3d_for_value(
        &self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
        value: u64,
    ) -> Result<(), RuntimeError> {
        self.slot.wait_d3d_for_value(queue, value)
    }

    pub fn is_fence_complete(&self, value: u64) -> bool {
        self.slot.is_fence_complete(value)
    }

    pub fn signal_available_on_d3d(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<u64, RuntimeError> {
        self.slot.signal_available_on_d3d(queue)
    }
}

#[cfg(windows)]
pub struct SharedFrameRing {
    width: u32,
    height: u32,
    pitch_bytes: u32,
    slots: Vec<SharedFrameSlot>,
}

#[cfg(windows)]
impl SharedFrameRing {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<Self, RuntimeError> {
        let row_bytes = width
            .checked_mul(4)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let pitch_bytes = align_u32(row_bytes, 256);
        let total_bytes = u64::from(pitch_bytes)
            .checked_mul(u64::from(height))
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let mut ring = Vec::with_capacity(slots);
        for _ in 0..slots {
            ring.push(SharedFrameSlot::new(device, pitch_bytes, total_bytes)?);
        }
        Ok(Self {
            width,
            height,
            pitch_bytes,
            slots: ring,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pitch_bytes(&self) -> u32 {
        self.pitch_bytes
    }

    pub fn kernel_width(&self) -> u32 {
        self.pitch_bytes / 4
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn slot(&self, index: usize) -> Option<&SharedFrameSlot> {
        self.slots.get(index)
    }

    pub fn slot_mut(&mut self, index: usize) -> Option<&mut SharedFrameSlot> {
        self.slots.get_mut(index)
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn wait_idle(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<(), RuntimeError> {
        for slot in &mut self.slots {
            slot.wait_cpu(queue)?;
        }
        Ok(())
    }
}

#[cfg(windows)]
pub struct SharedFrameSlot {
    resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    memory_handle: windows::Win32::Foundation::HANDLE,
    fence_handle: windows::Win32::Foundation::HANDLE,
    external_memory: sys::CUexternalMemory,
    external_semaphore: sys::CUexternalSemaphore,
    device_ptr: sys::CUdeviceptr,
    bytes: u64,
    fence_value: u64,
}

#[cfg(windows)]
impl SharedFrameSlot {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        pitch_bytes: u32,
        total_bytes: u64,
    ) -> Result<Self, RuntimeError> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_FENCE_FLAG_SHARED, D3D12_HEAP_FLAG_SHARED,
                D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN,
                D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Fence,
                ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: total_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(|| {
            RuntimeError::D3d12Interop("D3D12 returned no shared frame resource".to_string())
        })?;
        let memory_handle = unsafe {
            device.CreateSharedHandle(
                &resource,
                None,
                GENERIC_ALL_ACCESS,
                windows::core::PCWSTR::null(),
            )?
        };
        let fence: ID3D12Fence = unsafe { device.CreateFence(1, D3D12_FENCE_FLAG_SHARED)? };
        let fence_handle = unsafe {
            device.CreateSharedHandle(
                &fence,
                None,
                GENERIC_ALL_ACCESS,
                windows::core::PCWSTR::null(),
            )?
        };
        let external_memory = unsafe { import_d3d12_resource_memory(memory_handle, total_bytes)? };
        let device_ptr = unsafe { map_external_buffer(external_memory, total_bytes)? };
        let external_semaphore = unsafe { import_d3d12_fence(fence_handle)? };
        let _ = pitch_bytes;
        Ok(Self {
            resource,
            fence,
            memory_handle,
            fence_handle,
            external_memory,
            external_semaphore,
            device_ptr,
            bytes: total_bytes,
            fence_value: 1,
        })
    }

    pub fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        &self.resource
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        CudaDevicePtrArg::new(self.device_ptr)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn wait_available_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        unsafe { wait_external_fence(self.external_semaphore, self.fence_value, stream) }
    }

    pub fn signal_cuda_complete_on_stream(&mut self, stream: &Stream) -> Result<u64, RuntimeError> {
        self.fence_value += 1;
        unsafe { signal_external_fence(self.external_semaphore, self.fence_value, stream)? };
        Ok(self.fence_value)
    }

    pub fn is_fence_complete(&self, value: u64) -> bool {
        unsafe { self.fence.GetCompletedValue() >= value }
    }

    pub fn wait_d3d_for_value(
        &self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
        value: u64,
    ) -> Result<(), RuntimeError> {
        unsafe {
            queue.Wait(&self.fence, value)?;
        }
        Ok(())
    }

    pub fn signal_available_on_d3d(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<u64, RuntimeError> {
        self.fence_value += 1;
        unsafe {
            queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    pub fn wait_cpu(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<(), RuntimeError> {
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

        let wait_value = self.fence_value + 1;
        unsafe {
            queue.Signal(&self.fence, wait_value)?;
            if self.fence.GetCompletedValue() < wait_value {
                let event = CreateEventW(None, false, false, windows::core::PCWSTR::null())?;
                self.fence.SetEventOnCompletion(wait_value, event)?;
                WaitForSingleObject(event, INFINITE);
                let _ = windows::Win32::Foundation::CloseHandle(event);
            }
        }
        self.fence_value = wait_value;
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for SharedFrameSlot {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuMemFree_v2(self.device_ptr).result();
            let _ = sys::cuDestroyExternalMemory(self.external_memory).result();
            let _ = sys::cuDestroyExternalSemaphore(self.external_semaphore).result();
            let _ = windows::Win32::Foundation::CloseHandle(self.memory_handle);
            let _ = windows::Win32::Foundation::CloseHandle(self.fence_handle);
        }
    }
}

#[cfg(windows)]
const GENERIC_ALL_ACCESS: u32 = 0x1000_0000;

#[cfg(windows)]
fn align_u32(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[cfg(windows)]
fn cuda_device_luid(ctx: &Context) -> Result<[u8; 8], RuntimeError> {
    let mut device = 0;
    unsafe {
        sys::cuDeviceGet(&mut device, ctx.inner.ordinal() as i32).result()?;
        let mut luid = [0i8; 8];
        let mut node_mask = 0u32;
        sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut node_mask, device).result()?;
        Ok(luid.map(|byte| byte as u8))
    }
}

#[cfg(windows)]
fn dxgi_luid_bytes(luid: windows::Win32::Foundation::LUID) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
    bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
    bytes
}

#[cfg(windows)]
unsafe fn import_d3d12_resource_memory(
    handle: windows::Win32::Foundation::HANDLE,
    size: u64,
) -> Result<sys::CUexternalMemory, RuntimeError> {
    let mut external_memory = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &desc).result()?;
        Ok(external_memory.assume_init())
    }
}

#[cfg(windows)]
unsafe fn map_external_buffer(
    external_memory: sys::CUexternalMemory,
    size: u64,
) -> Result<sys::CUdeviceptr, RuntimeError> {
    let mut device_ptr = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
        offset: 0,
        size,
        flags: 0,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuExternalMemoryGetMappedBuffer(device_ptr.as_mut_ptr(), external_memory, &desc)
            .result()?;
        Ok(device_ptr.assume_init())
    }
}

#[cfg(windows)]
unsafe fn import_d3d12_fence(
    handle: windows::Win32::Foundation::HANDLE,
) -> Result<sys::CUexternalSemaphore, RuntimeError> {
    let mut external_semaphore = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
        handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        flags: 0,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuImportExternalSemaphore(external_semaphore.as_mut_ptr(), &desc).result()?;
        Ok(external_semaphore.assume_init())
    }
}

#[cfg(windows)]
unsafe fn wait_external_fence(
    semaphore: sys::CUexternalSemaphore,
    value: u64,
    stream: &Stream,
) -> Result<(), RuntimeError> {
    let semaphores = [semaphore];
    let params = [sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
                value,
            },
            nvSciSync: unsafe { std::mem::zeroed() },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
                timeoutMs: 0,
            },
            reserved: [0; 10],
        },
        flags: 0,
        reserved: [0; 16],
    }];
    unsafe {
        sys::cuWaitExternalSemaphoresAsync(
            semaphores.as_ptr(),
            params.as_ptr(),
            1,
            stream.inner.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[cfg(windows)]
unsafe fn signal_external_fence(
    semaphore: sys::CUexternalSemaphore,
    value: u64,
    stream: &Stream,
) -> Result<(), RuntimeError> {
    let semaphores = [semaphore];
    let params = [sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
                value,
            },
            nvSciSync: unsafe { std::mem::zeroed() },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
            },
            reserved: [0; 12],
        },
        flags: 0,
        reserved: [0; 16],
    }];
    unsafe {
        sys::cuSignalExternalSemaphoresAsync(
            semaphores.as_ptr(),
            params.as_ptr(),
            1,
            stream.inner.cu_stream(),
        )
        .result()?;
    }
    Ok(())
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

    fn mesh_test_desc(index_format: IndexFormat, index_count: u32) -> MeshBufferDesc {
        MeshBufferDesc {
            vertex_count: 4,
            vertex_layout: VertexLayout {
                stride: 16,
                attributes: vec![
                    VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    },
                    VertexAttribute {
                        semantic: VertexSemantic::Color0,
                        format: VertexFormat::U8x4Unorm,
                        offset: 12,
                    },
                ],
            },
            index_format,
            index_count,
            topology: PrimitiveTopology::TriangleList,
        }
    }

    fn instance_test_desc(instance_count: u32) -> InstanceBufferDesc {
        InstanceBufferDesc {
            instance_count,
            instance_layout: InstanceLayout {
                stride: 40,
                attributes: vec![
                    InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Rotation,
                        format: InstanceFormat::F32x4,
                        offset: 12,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Scale,
                        format: InstanceFormat::F32x2,
                        offset: 28,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Color0,
                        format: InstanceFormat::U8x4Unorm,
                        offset: 36,
                    },
                ],
            },
        }
    }

    #[cfg(windows)]
    fn assert_draw_recipe_contract<'a, D: DrawRecipe<'a>>(
        draw: &D,
        backend: DrawBackend,
        policy: DrawPolicy,
        target: Target,
        material_label: &str,
    ) {
        assert_eq!(draw.backend(), backend);
        assert_eq!(draw.policy(), policy);
        assert_eq!(draw.policy_config().policy, policy);
        assert_eq!(draw.target(), target);
        assert_eq!(draw.material().label(), material_label);
        assert_eq!(draw.geometry().mesh().desc().vertex_count, 1);
        assert!(draw.instances().is_some());
        let contract = draw.contract();
        assert_eq!(contract.backend, backend);
        assert_eq!(contract.policy, policy);
        assert_eq!(contract.policy_config, draw.policy_config());
        assert_eq!(contract.backend_label(), backend.label());
        assert_eq!(contract.policy_label(), policy.label());
        assert_eq!(contract.depth_label(), draw.policy_config().depth_label());
        assert_eq!(contract.uses_depth(), draw.policy_config().uses_depth());
        assert_eq!(contract.material_kernel, material_label);
        assert_eq!(contract.material_label(), material_label);
        assert_eq!(contract.material_kind_label(), draw.material().kind_label());
        assert_eq!(contract.target_width, target.width);
        assert_eq!(contract.target_height, target.height);
        assert_eq!(contract.geometry_vertex_count, 1);
        assert!(contract.instance_count.is_some());
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

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
    fn mesh_header_packing_is_stable_and_aligned() {
        let desc = mesh_test_desc(IndexFormat::U16, 6);
        let vertex_bytes = vec![0u8; 4 * 16];
        let index_bytes = vec![0u8; 6 * 2];
        let blob = pack_mesh_buffer(&desc, &vertex_bytes, &index_bytes).unwrap();
        assert_eq!(read_u32_le(&blob, 0), MESH_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), MESH_VERSION);
        assert_eq!(read_u32_le(&blob, 8), MESH_HEADER_BYTES as u32);
        assert_eq!(read_u32_le(&blob, 12), 4);
        assert_eq!(read_u32_le(&blob, 16), 16);
        assert_eq!(read_u32_le(&blob, 20), 80);
        assert_eq!(read_u32_le(&blob, 24), 6);
        assert_eq!(read_u32_le(&blob, 28), MESH_INDEX_U16);
        assert_eq!(read_u32_le(&blob, 32), 144);
        assert_eq!(read_u32_le(&blob, 36), 2);
        assert_eq!(read_u32_le(&blob, 40), MESH_HEADER_BYTES as u32);
        assert_eq!(read_u32_le(&blob, 44), MESH_TOPOLOGY_TRIANGLE_LIST);
        assert_eq!(blob.len(), 156);
    }

    #[test]
    fn mesh_rejects_attribute_extending_past_stride() {
        let mut desc = mesh_test_desc(IndexFormat::None, 0);
        desc.vertex_layout.attributes[0].offset = 8;
        let err = pack_mesh_buffer(&desc, &[0u8; 64], &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("extends past stride"));
    }

    #[test]
    fn mesh_packs_u16_and_u32_index_formats() {
        let vertex_bytes = vec![0u8; 4 * 16];
        let u16_blob = pack_mesh_buffer(
            &mesh_test_desc(IndexFormat::U16, 6),
            &vertex_bytes,
            &[0u8; 12],
        )
        .unwrap();
        assert_eq!(read_u32_le(&u16_blob, 28), MESH_INDEX_U16);
        let u32_blob = pack_mesh_buffer(
            &mesh_test_desc(IndexFormat::U32, 6),
            &vertex_bytes,
            &[0u8; 24],
        )
        .unwrap();
        assert_eq!(read_u32_le(&u32_blob, 28), MESH_INDEX_U32);
        assert_eq!(read_u32_le(&u32_blob, 32), 144);
    }

    #[test]
    fn instance_header_packing_is_stable_and_aligned() {
        let desc = instance_test_desc(3);
        let blob = pack_instance_buffer(&desc, &[0u8; 3 * 40]).unwrap();
        assert_eq!(read_u32_le(&blob, 0), INSTANCE_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), INSTANCE_VERSION);
        assert_eq!(read_u32_le(&blob, 8), INSTANCE_HEADER_BYTES as u32);
        assert_eq!(read_u32_le(&blob, 12), 3);
        assert_eq!(read_u32_le(&blob, 16), 40);
        assert_eq!(read_u32_le(&blob, 20), 112);
        assert_eq!(read_u32_le(&blob, 24), 4);
        assert_eq!(read_u32_le(&blob, 28), INSTANCE_HEADER_BYTES as u32);
        assert_eq!(read_u32_le(&blob, 32), DATA_LAYOUT_AOS);
        assert_eq!(read_u32_le(&blob, 36), 1);
        assert_eq!(blob.len(), 232);
    }

    #[test]
    fn structured_buffer_packs_soa_and_aosoa_offsets() {
        let desc = StructuredBufferDesc {
            element_count: 3,
            source_stride: 16,
            layout: DataLayout::SoA,
            fields: vec![
                BufferField {
                    semantic: INSTANCE_SEMANTIC_POSITION,
                    format: BufferFormat::F32x3,
                    offset: 0,
                },
                BufferField {
                    semantic: INSTANCE_SEMANTIC_COLOR0,
                    format: BufferFormat::U8x4Unorm,
                    offset: 12,
                },
            ],
        };
        let mut source = vec![0u8; 3 * 16];
        source[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        source[16..20].copy_from_slice(&2.0f32.to_le_bytes());
        source[32..36].copy_from_slice(&3.0f32.to_le_bytes());
        let soa = pack_structured_buffer(&desc, &source).unwrap();
        assert_eq!(read_u32_le(&soa, 32), DATA_LAYOUT_SOA);
        assert_eq!(read_u32_le(&soa, 40 + 8), 0);
        assert_eq!(read_u32_le(&soa, 56 + 8), 36);
        assert_eq!(f32::from_le_bytes(soa[80..84].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(soa[92..96].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(soa[104..108].try_into().unwrap()), 3.0);

        let mut aosoa_desc = desc;
        aosoa_desc.layout = DataLayout::AoSoA { group_size: 32 };
        let aosoa = pack_structured_buffer(&aosoa_desc, &source).unwrap();
        assert_eq!(read_u32_le(&aosoa, 32), DATA_LAYOUT_AOSOA);
        assert_eq!(read_u32_le(&aosoa, 36), 32);
        assert_eq!(read_u32_le(&aosoa, 56 + 8), 384);

        aosoa_desc.layout = DataLayout::aosoa64();
        let aosoa64 = pack_structured_buffer(&aosoa_desc, &source).unwrap();
        assert_eq!(read_u32_le(&aosoa64, 32), DATA_LAYOUT_AOSOA);
        assert_eq!(read_u32_le(&aosoa64, 36), 64);
        assert_eq!(read_u32_le(&aosoa64, 56 + 8), 768);
    }

    #[test]
    fn visibility_grid_packs_macrocell_records_and_bitsets() {
        let desc = VisibilityGridDesc::macrocell_lattice([256, 256, 128]);
        let blob = VisibilityGrid::pack(&desc).unwrap();
        assert_eq!(read_u32_le(&blob, 0), VISIBILITY_GRID_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), DEFAULT_MACROCELL_SIZE);
        assert_eq!(read_u32_le(&blob, 8), 32);
        assert_eq!(read_u32_le(&blob, 12), 32);
        assert_eq!(read_u32_le(&blob, 16), 16);
        assert_eq!(read_u32_le(&blob, 20), 32 * 32 * 16);
        assert_eq!(
            read_u32_le(&blob, 24),
            VISIBILITY_GRID_HEADER_U32S as u32 + 32 * 32 * 16 * VISIBILITY_GRID_RECORD_U32S as u32
        );
        assert_eq!(
            read_u32_le(&blob, 28),
            read_u32_le(&blob, 24) + (32 * 32 * 16u32).div_ceil(32)
        );

        let first = VISIBILITY_GRID_HEADER_U32S * 4;
        assert_eq!(
            [
                read_u32_le(&blob, first),
                read_u32_le(&blob, first + 4),
                read_u32_le(&blob, first + 8),
                read_u32_le(&blob, first + 12),
                read_u32_le(&blob, first + 16),
                read_u32_le(&blob, first + 20),
            ],
            [0, 7, 0, 7, 0, 7]
        );
        let occupancy_offset = read_u32_le(&blob, 24) as usize * 4;
        let relevance_offset = read_u32_le(&blob, 28) as usize * 4;
        assert_eq!(read_u32_le(&blob, occupancy_offset), u32::MAX);
        assert_eq!(read_u32_le(&blob, relevance_offset), u32::MAX);
    }

    #[test]
    fn visibility_grid_rounds_up_and_rejects_invalid_descs() {
        let desc = VisibilityGridDesc::macrocell_lattice([17, 9, 1]);
        let blob = VisibilityGrid::pack(&desc).unwrap();
        assert_eq!(read_u32_le(&blob, 8), 3);
        assert_eq!(read_u32_le(&blob, 12), 2);
        assert_eq!(read_u32_le(&blob, 16), 1);
        let last = read_u32_le(&blob, 24) as usize * 4 - VISIBILITY_GRID_RECORD_U32S * 4;
        assert_eq!(
            [
                read_u32_le(&blob, last),
                read_u32_le(&blob, last + 4),
                read_u32_le(&blob, last + 8),
                read_u32_le(&blob, last + 12),
                read_u32_le(&blob, last + 16),
                read_u32_le(&blob, last + 20),
            ],
            [16, 16, 8, 8, 0, 0]
        );
        assert!(VisibilityGrid::pack(&VisibilityGridDesc::macrocell_lattice([0, 1, 1])).is_err());
        assert!(
            VisibilityGrid::pack(&VisibilityGridDesc {
                cells: [1, 1, 1],
                macrocell_size: 0,
            })
            .is_err()
        );
    }

    #[test]
    fn sparse_texture_header_and_missing_fallback_are_stable() {
        let desc = SparseTextureDesc {
            virtual_width: 256,
            virtual_height: 128,
            page_size: 64,
            mip_count: 1,
            format: SparseTextureFormat::Rgba8Unorm,
            physical_pages: 3,
            gutter: 1,
        };
        let blob = SparseTextureAtlas::pack(&desc).unwrap();
        let pages_offset = sparse_texture_pages_offset(&desc).unwrap();
        let fallback_offset = sparse_texture_fallback_page_offset(&desc).unwrap();

        assert_eq!(read_u32_le(&blob, 0), SPARSE_TEXTURE_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), SPARSE_TEXTURE_VERSION);
        assert_eq!(read_u32_le(&blob, 12), 256);
        assert_eq!(read_u32_le(&blob, 16), 128);
        assert_eq!(read_u32_le(&blob, 20), 64);
        assert_eq!(read_u32_le(&blob, 24), 4);
        assert_eq!(read_u32_le(&blob, 28), 2);
        assert_eq!(read_u32_le(&blob, 40), 8);
        assert_eq!(read_u32_le(&blob, 44), 3);
        assert_eq!(
            read_u32_le(&blob, 48),
            SPARSE_TEXTURE_HEADER_U32S as u32 * 4
        );
        assert_eq!(read_u32_le(&blob, 52), pages_offset as u32);
        assert_eq!(read_u32_le(&blob, 56), fallback_offset as u32);
        assert_eq!(
            read_u32_le(&blob, 64),
            sparse_texture_feedback_offset(&desc).unwrap() as u32
        );
        assert_eq!(read_u32_le(&blob, 68), 8);
        assert_eq!(read_u32_le(&blob, 72), 0);
        assert_eq!(
            read_u32_le(&blob, sparse_texture_page_table_offset(0).unwrap()),
            0
        );
        assert_eq!(
            blob[fallback_offset..fallback_offset + 4],
            [255, 0, 255, 255]
        );
        assert!(
            blob[sparse_texture_feedback_offset(&desc).unwrap()..]
                .iter()
                .all(|byte| *byte == 0)
        );
        assert_eq!(
            blob.len(),
            sparse_texture_feedback_offset(&desc).unwrap()
                + sparse_texture_feedback_byte_len(&desc).unwrap()
        );
    }

    #[test]
    fn sparse_texture_feedback_summary_reports_hot_pages() {
        let summary = summarize_sparse_texture_feedback(&[0, 5, 1, 9, 0]).unwrap();
        assert_eq!(summary.active_pages, 3);
        assert_eq!(summary.total_requests, 15);
        assert_eq!(summary.hottest_page, Some(3));
        assert_eq!(summary.hottest_requests, 9);

        let summary = summarize_sparse_texture_feedback(&[0, 0, 0]).unwrap();
        assert_eq!(summary.active_pages, 0);
        assert_eq!(summary.total_requests, 0);
        assert_eq!(summary.hottest_page, None);
        assert_eq!(summary.hottest_requests, 0);
    }

    #[test]
    fn sparse_texture_validation_rejects_invalid_descs_and_pages() {
        assert!(SparseTextureAtlas::pack(&SparseTextureDesc::rgba8(0, 128, 1)).is_err());
        assert!(SparseTextureAtlas::pack(&SparseTextureDesc::rgba8(128, 128, 0)).is_err());
        let mut desc = SparseTextureDesc::rgba8(128, 128, 1);
        desc.page_size = 0;
        assert!(SparseTextureAtlas::pack(&desc).is_err());
        desc.page_size = 128;
        desc.gutter = 64;
        assert!(SparseTextureAtlas::pack(&desc).is_err());
        desc.gutter = 1;
        desc.mip_count = 2;
        assert!(SparseTextureAtlas::pack(&desc).is_err());
        desc.mip_count = 1;
        assert!(validate_sparse_virtual_page(&desc, 1).is_err());
        assert!(validate_sparse_physical_page(&desc, 1).is_err());
    }

    #[test]
    fn sparse_checker_pages_are_deterministic() {
        let desc = SparseTextureDesc {
            virtual_width: 128,
            virtual_height: 128,
            page_size: 8,
            mip_count: 1,
            format: SparseTextureFormat::Rgba8Unorm,
            physical_pages: 2,
            gutter: 1,
        };
        let mut page0 = vec![0u8; sparse_texture_page_bytes(&desc).unwrap()];
        let mut page1 = page0.clone();
        fill_sparse_checker_page(&desc, 0, &mut page0).unwrap();
        fill_sparse_checker_page(&desc, 1, &mut page1).unwrap();
        assert_ne!(page0, page1);
        assert_eq!(&page0[0..4], &[0, 255, 0, 255]);
        assert_eq!(&page1[0..4], &[182, 36, 255, 255]);
    }

    #[test]
    fn material_stream_packing_is_stable() {
        let desc = MaterialStreamDesc { material_count: 3 };
        let blob = MaterialStream::pack(&desc, &[7, 9, 11]).unwrap();
        assert_eq!(read_u32_le(&blob, 0), MATERIAL_STREAM_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), MATERIAL_STREAM_VERSION);
        assert_eq!(
            read_u32_le(&blob, 8),
            MATERIAL_STREAM_HEADER_U32S as u32 * 4
        );
        assert_eq!(read_u32_le(&blob, 12), 3);
        let data = MATERIAL_STREAM_HEADER_U32S * 4;
        assert_eq!(read_u32_le(&blob, data), 7);
        assert_eq!(read_u32_le(&blob, data + 4), 9);
        assert_eq!(read_u32_le(&blob, data + 8), 11);
        let err = MaterialStream::pack(&desc, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 3 material IDs"));
        assert!(MaterialStream::pack(&MaterialStreamDesc { material_count: 0 }, &[]).is_err());
    }

    #[test]
    fn instance_rejects_attribute_extending_past_stride() {
        let mut desc = instance_test_desc(1);
        desc.instance_layout.attributes[1].offset = 32;
        let err = pack_instance_buffer(&desc, &[0u8; 40])
            .unwrap_err()
            .to_string();
        assert!(err.contains("extends past stride"));
    }

    #[test]
    fn runtime_compile_includes_mesh_prelude_without_changing_language_lowering() {
        let source =
            "kernel fn inspect(global u8* mesh) { let count: u32 = neo_mesh_vertex_count(mesh); }";
        let lowered = neo_lang::lower_to_cuda(source).unwrap();
        assert!(!lowered.contains("NeoMeshHeader"));
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping mesh prelude compile test without CUDA: {err}");
                return;
            }
        };
        let module = Module::from_neo_source(&ctx, source, &["inspect"]).unwrap();
        assert!(module.cuda_source.contains("NeoMeshHeader"));
        assert!(module.cuda_source.contains("neo_mesh_vertex_count"));
    }

    #[test]
    fn runtime_compile_includes_instance_prelude_without_changing_language_lowering() {
        let source = "kernel fn inspect(global u8* instances) { let count: u32 = neo_instance_count(instances); }";
        let lowered = neo_lang::lower_to_cuda(source).unwrap();
        assert!(!lowered.contains("NeoInstanceHeader"));
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping instance prelude compile test without CUDA: {err}");
                return;
            }
        };
        let module = Module::from_neo_source(&ctx, source, &["inspect"]).unwrap();
        assert!(module.cuda_source.contains("NeoInstanceHeader"));
        assert!(module.cuda_source.contains("neo_instance_count"));
        assert!(module.cuda_source.contains("neo_instance_stride"));
        assert!(module.cuda_source.contains("neo_instance_bytes_offset"));
        assert!(module.cuda_source.contains("neo_instance_payload"));
    }

    #[test]
    fn runtime_cuda_prelude_includes_sparse_texture_helpers() {
        let prelude = runtime_cuda_prelude();
        assert!(prelude.contains("NeoSparseTextureHeader"));
        assert!(prelude.contains("NeoMaterialStreamHeader"));
        assert!(prelude.contains("neo_sparse_texture_width"));
        assert!(prelude.contains("neo_sparse_material_tile"));
        assert!(prelude.contains("neo_sparse_texture_record_feedback"));
        assert!(prelude.contains("neo_sparse_page_id"));
        assert!(prelude.contains("neo_sparse_page_resident"));
        assert!(prelude.contains("neo_sparse_record_feedback_sampled"));
        assert!(prelude.contains("neo_sparse_record_feedback_missing"));
        assert!(prelude.contains("neo_sparse_sample_rgba8"));
        assert!(prelude.contains("neo_sparse_sample_bgra8"));
        assert!(prelude.contains("neo_sparse_sample_bgra8_feedback"));
        assert!(prelude.contains("neo_sparse_sample_bgra8_feedback_mode"));
    }

    #[test]
    fn arg_mesh_launches_without_taking_mesh_ownership() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping arg_mesh launch test without CUDA: {err}");
                return;
            }
        };
        let vertices = [[0.0f32; 4]; 3];
        let mesh = MeshBuffer::upload_typed(
            &ctx,
            MeshBufferDesc {
                vertex_count: 3,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &vertices,
            &[] as &[u16],
        )
        .unwrap();
        let module = Module::from_neo_source(
            &ctx,
            "kernel fn inspect(global u8* mesh) { let count: u32 = neo_mesh_vertex_count(mesh); }",
            &["inspect"],
        )
        .unwrap();
        let kernel = module.kernel("inspect").unwrap();
        let mut launch = kernel.launcher();
        launch.arg_mesh(&mesh);
        unsafe {
            launch.launch(LaunchDims::for_2d(1, 1, (1, 1))).unwrap();
        }
        ctx.synchronize().unwrap();
        assert_eq!(mesh.desc().vertex_count, 3);
    }

    #[test]
    fn arg_instances_launches_without_taking_instance_ownership() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping arg_instances launch test without CUDA: {err}");
                return;
            }
        };
        let bytes = [0u8; 40];
        let instances = InstanceBuffer::upload(
            &ctx,
            InstanceBufferDesc {
                instance_count: 1,
                instance_layout: InstanceLayout {
                    stride: 40,
                    attributes: vec![InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    }],
                },
            },
            &bytes,
        )
        .unwrap();
        let module = Module::from_neo_source(
            &ctx,
            "kernel fn inspect(global u8* instances) { let count: u32 = neo_instance_count(instances); }",
            &["inspect"],
        )
        .unwrap();
        let kernel = module.kernel("inspect").unwrap();
        let mut launch = kernel.launcher();
        launch.arg_instances(&instances);
        unsafe {
            launch.launch(LaunchDims::for_2d(1, 1, (1, 1))).unwrap();
        }
        ctx.synchronize().unwrap();
        assert_eq!(instances.desc().instance_count, 1);
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
    fn module_entrypoint_validation_ignores_graphics_stages() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping graphics entrypoint validation test without CUDA: {err}");
                return;
            }
        };
        let err = match Module::from_neo_source(&ctx, "vertex fn image() {}", &["image"]) {
            Ok(_) => panic!("expected missing CUDA entrypoint error"),
            Err(err) => err,
        };
        assert!(matches!(err, RuntimeError::MissingEntrypoint(name) if name == "image"));
    }

    #[test]
    fn indirect_draw_command_packing_is_stable() {
        assert_eq!(DrawIndexedIndirectCommand::BYTE_LEN, 20);
        let command = DrawIndexedIndirectCommand::indexed_quad(123);
        assert_eq!(
            command,
            DrawIndexedIndirectCommand {
                index_count_per_instance: 6,
                instance_count: 123,
                start_index_location: 0,
                base_vertex_location: 0,
                start_instance_location: 0,
            }
        );
        assert_eq!(
            command.as_bytes().len(),
            DrawIndexedIndirectCommand::BYTE_LEN
        );
        assert_eq!(&command.as_bytes()[0..4], &6u32.to_le_bytes());
        assert_eq!(&command.as_bytes()[4..8], &123u32.to_le_bytes());
    }

    #[cfg(windows)]
    #[test]
    fn draw_execution_vocabulary_is_explicit() {
        let pipeline = DrawPipeline::new("quad-pipeline");
        let legacy_pipeline = RasterPipeline::new("legacy-quad-pipeline");
        let material = MaterialKernel::from_stages("quad-material", "quad_vs", "quad_fs");
        let cuda_material = MaterialKernel::from_cuda_tiled("cuda-material", "instance_raster");
        assert_eq!(pipeline.label(), "quad-pipeline");
        assert_eq!(legacy_pipeline.label(), "legacy-quad-pipeline");
        assert_eq!(material.label(), "quad-material");
        assert_eq!(material.kind_label(), "draw-execution");
        assert_eq!(material.vertex_entrypoint(), "quad_vs");
        assert_eq!(material.fragment_entrypoint(), "quad_fs");
        assert_eq!(material.backend(), DrawBackend::HardwareRaster);
        assert_eq!(material.abi().backend(), DrawBackend::HardwareRaster);
        assert_eq!(material.abi().kind_label(), "draw-execution");
        assert!(material.kernel_entrypoint().is_none());
        assert_eq!(cuda_material.label(), "cuda-material");
        assert_eq!(cuda_material.kind_label(), "cuda-tiled");
        assert_eq!(cuda_material.kernel_entrypoint(), Some("instance_raster"));
        assert_eq!(cuda_material.backend(), DrawBackend::CudaTiled);
        assert_eq!(cuda_material.abi().backend(), DrawBackend::CudaTiled);
        assert_eq!(cuda_material.abi().kind_label(), "cuda-tiled");
        assert_eq!(
            MaterialKernelKind::DrawExecution.backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(MaterialKernelKind::DrawExecution.label(), "draw-execution");
        assert_eq!(
            MaterialKernelKind::DrawExecution.to_string(),
            "draw-execution"
        );
        assert_eq!(
            MaterialKernelKind::HardwareRaster.backend(),
            DrawBackend::HardwareRaster
        );
        assert!(MaterialKernelKind::HardwareRaster.is_draw_execution());
        assert_eq!(
            MaterialKernelKind::HardwareRaster.label(),
            "hardware-raster"
        );
        assert_eq!(
            MaterialKernelKind::CudaTiled.backend(),
            DrawBackend::CudaTiled
        );
        assert_eq!(MaterialKernelKind::CudaTiled.label(), "cuda-tiled");
        assert!(cuda_material.abi().is_cuda_tiled());
        assert!(cuda_material.abi().requires_instance_stream());
        assert!(!cuda_material.abi().requires_compute_culling());
        assert_eq!(
            material.abi().vertex_requirements,
            vec![
                MaterialVertexRequirement::ClipPositionOutput,
                MaterialVertexRequirement::VertexColorOutput
            ]
        );
        assert_eq!(DrawPolicy::DrawAll, DrawPolicy::DrawAll);
        assert_ne!(DrawPolicy::DrawAll, DrawPolicy::ComputeCulled);
        assert_eq!(DrawPolicy::DrawAll.backend(), DrawBackend::HardwareRaster);
        assert_eq!(DrawPolicy::DrawAll.label(), "draw-all");
        assert_eq!(DrawPolicy::DrawAll.to_string(), "draw-all");
        assert_eq!(
            DrawPolicy::ComputeCulled.backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(DrawPolicy::ComputeCulled.label(), "compute-culled");
        assert_eq!(DrawPolicy::CudaTiled.backend(), DrawBackend::CudaTiled);
        assert_eq!(DrawPolicy::CudaTiled.label(), "cuda-tiled");
        assert_eq!(DrawBackend::primary_neo(), DrawBackend::CudaTiled);
        assert!(DrawBackend::CudaTiled.is_primary_neo());
        assert!(!DrawBackend::HardwareRaster.is_primary_neo());
        assert_eq!(DrawBackend::HardwareRaster.label(), "hardware-raster");
        assert_eq!(DrawBackend::HardwareRaster.to_string(), "hardware-raster");
        assert_eq!(DrawBackend::CudaTiled.label(), "cuda-tiled");
        assert_eq!(CullOrder::AtomicCompact.label(), "atomic-compact");
        assert_eq!(CullOrder::StableDense.to_string(), "stable-dense");
        let neutral_cull_order: CullOrder = CullOrder::StableDense;
        assert_eq!(neutral_cull_order.label(), "stable-dense");
        let legacy_cull_order: RasterCullOrder = RasterCullOrder::StableDense;
        assert_eq!(legacy_cull_order, neutral_cull_order);
        assert_eq!(VisibilityMode::Frustum.label(), "frustum");
        assert_eq!(VisibilityMode::ProjectedSize.to_string(), "projected-size");
        let neutral_visibility: VisibilityMode = VisibilityMode::ProjectedSize;
        assert_eq!(neutral_visibility.label(), "projected-size");
        let legacy_visibility: RasterVisibilityMode = RasterVisibilityMode::ProjectedSize;
        assert_eq!(legacy_visibility, neutral_visibility);
        assert_eq!(DrawDepthMode::Auto.label(), "auto");
        assert_eq!(DrawDepthMode::Auto.to_string(), "auto");
        assert!(!DrawDepthMode::Auto.uses_depth(DrawPolicy::DrawAll));
        assert!(DrawDepthMode::Auto.uses_depth(DrawPolicy::ComputeCulled));
        assert!(DrawDepthMode::On.uses_depth(DrawPolicy::DrawAll));
        assert!(!DrawDepthMode::Off.uses_depth(DrawPolicy::ComputeCulled));
        assert_eq!(
            DEFAULT_MIN_PROJECTED_MILLIPIXELS,
            DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
        );
        assert_eq!(
            DrawPolicyConfig::draw_all().backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(DrawPolicyConfig::draw_all().policy_label(), "draw-all");
        assert_eq!(
            DrawPolicyConfig::draw_all().backend_label(),
            "hardware-raster"
        );
        assert_eq!(DrawPolicyConfig::draw_all().depth_label(), "auto");
        assert!(!DrawPolicyConfig::draw_all().uses_depth());
        assert!(DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).uses_depth());
        assert!(
            DrawPolicyConfig::draw_all()
                .with_depth(DrawDepthMode::On)
                .uses_depth()
        );
        assert_eq!(
            DrawPolicyConfig::draw_all().cull_order_label(),
            "stable-dense"
        );
        assert_eq!(DrawPolicyConfig::draw_all().visibility_label(), "frustum");
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).backend(),
            DrawBackend::HardwareRaster
        );
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).policy_label(),
            "compute-culled"
        );
        assert_eq!(
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact).cull_order_label(),
            "atomic-compact"
        );
        assert_eq!(
            DrawPolicyConfig::cuda_tiled().backend(),
            DrawBackend::CudaTiled
        );
        assert_eq!(DrawPolicyConfig::cuda_tiled().policy_label(), "cuda-tiled");
        assert_eq!(
            DrawPolicyConfig::cuda_tiled().visibility_label(),
            "projected-size"
        );
        assert_eq!(DrawPolicyConfig::cuda_tiled().min_projected_pixels(), 0.85);
    }

    #[test]
    fn material_kernel_abi_describes_cuda_tiled_instance_material() {
        let abi = MaterialKernelAbi::cuda_tiled_instance_color("instance_raster");
        assert!(abi.is_cuda_tiled());
        assert!(!abi.is_draw_execution());
        assert!(!abi.is_hardware_raster());
        assert_eq!(abi.vertex_entrypoint(), None);
        assert_eq!(abi.fragment_entrypoint(), None);
        assert_eq!(abi.kernel_entrypoint(), Some("instance_raster"));
        assert_eq!(abi.backend(), DrawBackend::CudaTiled);
        assert!(abi.requires_instance_stream());
        assert!(!abi.requires_compute_culling());
        assert_eq!(
            abi.binding(MaterialBindingKind::DrawParams).unwrap().kind,
            MaterialBindingKind::DrawParams
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::RasterParams)
                .unwrap()
                .root_parameter_index,
            0
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::InstanceStream)
                .unwrap()
                .root_parameter_index,
            1
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::GeometryStream)
                .unwrap()
                .root_parameter_index,
            2
        );
    }

    #[test]
    fn material_kernel_abi_describes_compute_culled_instance_material() {
        let abi = MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs");
        assert_eq!(abi.vertex_entrypoint, "quad_vs");
        assert_eq!(abi.fragment_entrypoint, "quad_fs");
        assert_eq!(abi.backend(), DrawBackend::HardwareRaster);
        assert!(abi.is_draw_execution());
        assert!(abi.is_hardware_raster());
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::VisibleInstanceStream)
        );
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::InstancePosition)
        );
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::ClipPositionOutput)
        );
        assert!(
            abi.fragment_requirements
                .contains(&MaterialFragmentRequirement::InterpolatedColorInput)
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::DrawParams).unwrap().kind,
            MaterialBindingKind::DrawParams
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::RasterParams)
                .unwrap()
                .root_parameter_index,
            0
        );

        let material = MaterialKernel::new("material").with_abi(abi);
        assert_eq!(material.vertex_entrypoint(), "quad_vs");
        assert_eq!(material.fragment_entrypoint(), "quad_fs");
        assert_eq!(material.backend(), DrawBackend::HardwareRaster);
        assert!(
            material
                .abi()
                .vertex_requirements
                .contains(&MaterialVertexRequirement::VisibleInstanceStream)
        );
    }

    #[test]
    fn material_kernel_abi_describes_direct_instance_material() {
        let abi = MaterialKernelAbi::direct_instance_color("quad_vs_direct", "quad_fs");
        assert_eq!(abi.vertex_entrypoint, "quad_vs_direct");
        assert_eq!(abi.fragment_entrypoint, "quad_fs");
        assert_eq!(abi.backend(), DrawBackend::HardwareRaster);
        assert!(abi.is_draw_execution());
        assert!(abi.is_hardware_raster());
        assert!(!abi.requires_compute_culling());
        assert!(abi.requires_instance_stream());
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::DirectInstanceId)
        );
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::InstancePosition)
        );
        assert!(
            abi.vertex_requirements
                .contains(&MaterialVertexRequirement::GeometryPosition)
        );
        assert!(
            abi.binding(MaterialBindingKind::VisibleInstanceStream)
                .is_none()
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::DrawParams).unwrap().kind,
            MaterialBindingKind::DrawParams
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::RasterParams)
                .unwrap()
                .root_parameter_index,
            0
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::InstanceStream)
                .unwrap()
                .root_parameter_index,
            1
        );
        assert_eq!(
            abi.binding(MaterialBindingKind::GeometryStream)
                .unwrap()
                .root_parameter_index,
            2
        );
    }

    #[cfg(windows)]
    #[test]
    fn target_is_the_primary_render_destination_vocabulary() {
        let target = Target::new(64, 32).unwrap();
        let raster_alias = RasterTarget::new(64, 32).unwrap();
        let render_alias = RenderTarget::new(64, 32).unwrap();
        let draw_pass = DrawPass { target };
        let raster_pass = RasterPass { target };

        assert_eq!(target.width, 64);
        assert_eq!(target.height, 32);
        assert_eq!(target, raster_alias);
        assert_eq!(target, render_alias);
        assert_eq!(draw_pass.target, target);
        assert_eq!(raster_pass, draw_pass);

        let err = Target::new(0, 32).unwrap_err();
        assert!(err.to_string().contains("target width and height"));
    }

    #[cfg(windows)]
    #[test]
    fn draw_execution_recipe_composes_streams_material_target_and_policy() {
        let mesh_bytes = [0u8; 16];
        let instance_bytes = [0u8; 40];
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping raster draw recipe test without CUDA: {err}");
                return;
            }
        };
        let mesh = MeshBuffer::upload(
            &ctx,
            MeshBufferDesc {
                vertex_count: 1,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &mesh_bytes,
            &[],
        )
        .unwrap();
        let instances = InstanceBuffer::upload(
            &ctx,
            InstanceBufferDesc {
                instance_count: 1,
                instance_layout: InstanceLayout {
                    stride: 40,
                    attributes: vec![InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    }],
                },
            },
            &instance_bytes,
        )
        .unwrap();
        let material = MaterialKernel::new("material").with_abi(
            MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs"),
        );
        let draw = DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .draw_policy(DrawPolicy::ComputeCulled)
        .try_build()
        .unwrap();
        assert_eq!(draw.target(), Target::new(64, 32).unwrap());
        assert_eq!(draw.backend(), DrawBackend::HardwareRaster);
        assert_eq!(draw.policy(), DrawPolicy::ComputeCulled);
        assert_eq!(
            draw.policy_config(),
            DrawPolicyConfig::compute_culled(CullOrder::AtomicCompact)
        );
        assert_eq!(draw.policy_config().visibility, VisibilityMode::Frustum);
        assert_eq!(
            draw.policy_config().min_projected_millipixels,
            DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS
        );
        assert!(draw.instances().is_some());
        assert_eq!(draw.material().label(), "material");
        assert_eq!(draw.geometry().mesh().desc().vertex_count, 1);
        assert_draw_recipe_contract(
            &draw,
            DrawBackend::HardwareRaster,
            DrawPolicy::ComputeCulled,
            Target::new(64, 32).unwrap(),
            "material",
        );

        let neutral_draw: DrawExecution<'_> = DrawExecution::execution_builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .draw_policy(DrawPolicy::ComputeCulled)
        .try_build()
        .unwrap();
        let _neutral_builder: DrawExecutionBuilder<'_> = DrawExecution::execution_builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        );
        let legacy_draw: RasterDraw<'_> = RasterDraw::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .draw_policy(DrawPolicy::ComputeCulled)
        .try_build()
        .unwrap();
        assert_eq!(neutral_draw.contract(), draw.contract());
        assert_eq!(legacy_draw.contract(), draw.contract());
    }

    #[cfg(windows)]
    #[test]
    fn cuda_draw_recipe_composes_streams_material_target_and_policy() {
        let mesh_bytes = [0u8; 16];
        let instance_bytes = [0u8; 40];
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping CUDA draw recipe test without CUDA: {err}");
                return;
            }
        };
        let mesh = MeshBuffer::upload(
            &ctx,
            MeshBufferDesc {
                vertex_count: 1,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &mesh_bytes,
            &[],
        )
        .unwrap();
        let instances = InstanceBuffer::upload(
            &ctx,
            InstanceBufferDesc {
                instance_count: 1,
                instance_layout: InstanceLayout {
                    stride: 40,
                    attributes: vec![InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    }],
                },
            },
            &instance_bytes,
        )
        .unwrap();
        let material = MaterialKernel::from_cuda_tiled("cuda-material", "instance_raster");
        let draw = CudaDraw::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .try_build()
        .unwrap();

        assert_eq!(draw.target(), Target::new(64, 32).unwrap());
        assert_eq!(draw.backend(), DrawBackend::CudaTiled);
        assert_eq!(draw.policy(), DrawPolicy::CudaTiled);
        assert_eq!(draw.policy_config(), DrawPolicyConfig::cuda_tiled());
        assert_eq!(draw.material().kernel_entrypoint(), Some("instance_raster"));
        assert_eq!(draw.instances().instances().desc().instance_count, 1);
        assert_eq!(draw.geometry().mesh().desc().vertex_count, 1);
        assert_draw_recipe_contract(
            &draw,
            DrawBackend::CudaTiled,
            DrawPolicy::CudaTiled,
            Target::new(64, 32).unwrap(),
            "cuda-material",
        );
    }

    #[cfg(windows)]
    #[test]
    fn draw_execution_recipe_preserves_explicit_policy_config() {
        let mesh_bytes = [0u8; 16];
        let instance_bytes = [0u8; 40];
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping raster draw policy config test without CUDA: {err}");
                return;
            }
        };
        let mesh = MeshBuffer::upload(
            &ctx,
            MeshBufferDesc {
                vertex_count: 1,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &mesh_bytes,
            &[],
        )
        .unwrap();
        let instances = InstanceBuffer::upload(
            &ctx,
            InstanceBufferDesc {
                instance_count: 1,
                instance_layout: InstanceLayout {
                    stride: 40,
                    attributes: vec![InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    }],
                },
            },
            &instance_bytes,
        )
        .unwrap();
        let material = MaterialKernel::new("material").with_abi(
            MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs"),
        );
        let draw = DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .compute_culled_with_visibility(CullOrder::StableDense, VisibilityMode::ProjectedSize)
        .draw_policy_config(
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize,
            )
            .with_min_projected_millipixels(500),
        )
        .try_build()
        .unwrap();

        assert_eq!(draw.policy(), DrawPolicy::ComputeCulled);
        assert_eq!(
            draw.policy_config(),
            DrawPolicyConfig::compute_culled_with_visibility(
                CullOrder::StableDense,
                VisibilityMode::ProjectedSize
            )
            .with_min_projected_millipixels(500)
        );

        let helper_draw = DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .compute_culled_projected(CullOrder::StableDense, 500)
        .try_build()
        .unwrap();
        assert_eq!(helper_draw.policy_config(), draw.policy_config());
    }

    #[cfg(windows)]
    #[test]
    fn draw_execution_recipe_rejects_missing_required_instance_stream() {
        let mesh_bytes = [0u8; 16];
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping raster draw validation test without CUDA: {err}");
                return;
            }
        };
        let mesh = MeshBuffer::upload(
            &ctx,
            MeshBufferDesc {
                vertex_count: 1,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &mesh_bytes,
            &[],
        )
        .unwrap();
        let material = MaterialKernel::new("material").with_abi(
            MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs"),
        );
        let err = match DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &material,
            Target::new(64, 32).unwrap(),
        )
        .draw_policy(DrawPolicy::ComputeCulled)
        .try_build()
        {
            Ok(_) => panic!("expected missing InstanceStream validation error"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires an explicit InstanceStream"));
    }

    #[cfg(windows)]
    #[test]
    fn draw_execution_recipe_rejects_mismatched_draw_policy_and_material() {
        let mesh_bytes = [0u8; 16];
        let instance_bytes = [0u8; 40];
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping raster draw policy validation test without CUDA: {err}");
                return;
            }
        };
        let mesh = MeshBuffer::upload(
            &ctx,
            MeshBufferDesc {
                vertex_count: 1,
                vertex_layout: VertexLayout {
                    stride: 16,
                    attributes: vec![VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    }],
                },
                index_format: IndexFormat::None,
                index_count: 0,
                topology: PrimitiveTopology::TriangleList,
            },
            &mesh_bytes,
            &[],
        )
        .unwrap();
        let instances = InstanceBuffer::upload(
            &ctx,
            InstanceBufferDesc {
                instance_count: 1,
                instance_layout: InstanceLayout {
                    stride: 40,
                    attributes: vec![InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    }],
                },
            },
            &instance_bytes,
        )
        .unwrap();

        let compute_material = MaterialKernel::new("compute-material").with_abi(
            MaterialKernelAbi::compute_culled_instance_color("quad_vs", "quad_fs"),
        );
        let err = match DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &compute_material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .try_build()
        {
            Ok(_) => panic!("expected missing compute culling policy validation error"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires DrawPolicy::ComputeCulled"));

        let simple_material = MaterialKernel::new("simple-material");
        let err = match DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &simple_material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .draw_policy(DrawPolicy::ComputeCulled)
        .try_build()
        {
            Ok(_) => panic!("expected visible InstanceStream material validation error"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("read the visible InstanceStream"));

        let cuda_material = MaterialKernel::from_cuda_tiled("cuda-material", "instance_raster");
        let err = match DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &cuda_material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .try_build()
        {
            Ok(_) => panic!("expected CUDA MaterialKernel rejection"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires a draw-execution MaterialKernel"));

        let err = match DrawExecution::builder(
            GeometryStream::from_mesh(&mesh),
            &simple_material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .draw_policy(DrawPolicy::CudaTiled)
        .try_build()
        {
            Ok(_) => panic!("expected DrawExecution CUDA policy rejection"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("cannot use DrawPolicy::CudaTiled"));

        let err = match CudaDraw::builder(
            GeometryStream::from_mesh(&mesh),
            &simple_material,
            Target::new(64, 32).unwrap(),
        )
        .instance_stream(InstanceStream::from_instances(&instances))
        .try_build()
        {
            Ok(_) => panic!("expected CudaDraw hardware material rejection"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires a CUDA tiled MaterialKernel"));

        let err = match CudaDraw::builder(
            GeometryStream::from_mesh(&mesh),
            &cuda_material,
            Target::new(64, 32).unwrap(),
        )
        .draw_policy_config(DrawPolicyConfig::draw_all())
        .instance_stream(InstanceStream::from_instances(&instances))
        .try_build()
        {
            Ok(_) => panic!("expected CudaDraw policy rejection"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires DrawPolicy::CudaTiled"));

        let err = match CudaDraw::builder(
            GeometryStream::from_mesh(&mesh),
            &cuda_material,
            Target::new(64, 32).unwrap(),
        )
        .try_build()
        {
            Ok(_) => panic!("expected CudaDraw missing InstanceStream rejection"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires an explicit InstanceStream"));
    }

    #[cfg(windows)]
    #[test]
    fn shared_raster_stream_wrappers_skip_without_interop() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TestInstance {
            position: [f32; 3],
            color: u32,
        }

        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping shared raster stream test without CUDA: {err}");
                return;
            }
        };
        let interop = match NeoD3d12InteropDevice::new(&ctx) {
            Ok(interop) => interop,
            Err(err) => {
                eprintln!("skipping shared raster stream test without D3D12 interop: {err}");
                return;
            }
        };
        let desc = InstanceBufferDesc {
            instance_count: 1,
            instance_layout: InstanceLayout {
                stride: std::mem::size_of::<TestInstance>() as u32,
                attributes: vec![
                    InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Color0,
                        format: InstanceFormat::U8x4Unorm,
                        offset: 12,
                    },
                ],
            },
        };
        let instances = [TestInstance {
            position: [1.0, 2.0, 3.0],
            color: 0xff00_ffff,
        }];
        let stream = SharedInstanceStream::upload_typed(
            &ctx,
            &interop,
            desc.clone(),
            &instances,
            DataLayout::aosoa32(),
        )
        .unwrap();
        let visible = VisibleInstanceStream::new(&interop, 4).unwrap();
        let indirect = IndirectDrawBuffer::new(&interop, 1).unwrap();
        assert_eq!(stream.desc(), &desc);
        assert_eq!(stream.data_layout(), DataLayout::aosoa32());
        assert!(stream.byte_len() >= INSTANCE_HEADER_BYTES);
        assert_eq!(visible.capacity(), 4);
        assert_eq!(visible.buffer().bytes(), 16);
        assert_eq!(indirect.command_capacity(), 1);
    }

    #[test]
    fn diagnostics_collect_without_panicking() {
        let diagnostics = RuntimeDiagnostics::collect();
        if diagnostics.nvrtc_loadable {
            assert!(!diagnostics.nvrtc_compatible.is_empty());
        }
    }

    #[test]
    fn runtime_can_compile_native_cuda_image_smoke() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping native CUDA image smoke test without CUDA: {err}");
                return;
            }
        };
        if !nvrtc_available() {
            eprintln!("skipping native CUDA image smoke test without NVRTC");
            return;
        }
        let module = Module::from_cuda_source(
            &ctx,
            "extern \"C\" __global__ void noop_kernel() {}".to_string(),
        )
        .unwrap();
        module.kernel("noop_kernel").unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn cuda_root_expands_to_x64_bin_before_plain_bin() {
        let mut dirs = Vec::new();
        push_cuda_root_bin_dirs(
            &mut dirs,
            PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"),
        );
        assert_eq!(
            dirs,
            vec![
                PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"),
                PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"),
            ]
        );
    }

    #[test]
    fn runtime_smoke_test_skips_without_cuda() {
        match Context::new_default_device() {
            Ok(ctx) => ctx.synchronize().unwrap(),
            Err(err) => eprintln!("skipping CUDA smoke test: {err}"),
        }
    }

    #[test]
    fn cuda_fence_query_skips_without_cuda() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping CUDA fence query test: {err}");
                return;
            }
        };
        let fence = ctx.create_fence().unwrap();
        fence.record(&ctx).unwrap();
        let _ = fence.is_complete().unwrap();
        fence.synchronize().unwrap();
        assert!(fence.is_complete().unwrap());
    }

    #[test]
    fn cuda_stream_fence_query_skips_without_cuda() {
        let ctx = match Context::new_default_device() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skipping CUDA stream fence query test: {err}");
                return;
            }
        };
        let stream = ctx.create_stream().unwrap();
        let fence = stream.create_fence().unwrap();
        fence.record_on_stream(&stream).unwrap();
        let _ = fence.is_complete().unwrap();
        fence.synchronize().unwrap();
        assert!(fence.is_complete().unwrap());
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
