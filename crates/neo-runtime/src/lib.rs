use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaGraph as CudarcCudaGraph, CudaSlice, CudaStream, DeviceRepr,
        DriverError, LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits,
        sys,
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
    #[error("mesh buffer error: {0}")]
    Mesh(String),
    #[error("instance buffer error: {0}")]
    Instance(String),
    #[cfg(windows)]
    #[error("D3D12 interop error: {0}")]
    D3d12Interop(String),
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
            if !program
                .kernels
                .iter()
                .any(|kernel| kernel.name == *entrypoint)
            {
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
        let ptx = compile_ptx(&cuda_source).map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
        let inner = ctx.inner.load_module(ptx)?;
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

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
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
"#
    )
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
