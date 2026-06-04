use std::{path::Path, sync::Arc};

#[cfg(all(feature = "cuda-12060", feature = "cuda-13000"))]
compile_error!("Enable exactly one Neo CUDA build feature: cuda-12060 or cuda-13000, not both.");

#[cfg(not(any(feature = "cuda-12060", feature = "cuda-13000")))]
compile_error!("Enable exactly one Neo CUDA build feature: cuda-12060 or cuda-13000.");

use cudarc::driver::{
    CudaContext, CudaGraph as CudarcCudaGraph, CudaStream, DeviceRepr, DriverError, LaunchConfig,
    ValidAsZeroBits, sys,
};
use thiserror::Error;

mod cuda;
mod diagnostics;
mod draw;
mod interop;
mod resources;

pub use cuda::{
    CudaDevicePtrArg, CudaFence, DeviceBuffer, Kernel, KernelLaunch, PinnedHostBuffer,
    ReadablePinnedHostBuffer,
};
pub use diagnostics::{RuntimeDiagnostics, nvrtc_available};
#[cfg(test)]
use diagnostics::{compatible_nvrtc_candidate, expected_cuda_build_label, push_cuda_root_bin_dirs};
use diagnostics::{
    compile_cuda_image_checked, configure_nvrtc_search_path, load_cuda_module_checked,
};
#[cfg(windows)]
pub use draw::{
    CudaDraw, CudaDrawBuilder, CullOrder, DEFAULT_MIN_PROJECTED_MILLIPIXELS,
    DEFAULT_RASTER_MIN_PROJECTED_MILLIPIXELS, DrawBackend, DrawContract, DrawDepthMode, DrawDevice,
    DrawExecution, DrawExecutionBuilder, DrawIndexedIndirectCommand, DrawPass, DrawPipeline,
    DrawPolicy, DrawPolicyConfig, DrawRecipe, GeometryStream, IndirectDrawBuffer, InstanceStream,
    MaterialBinding, MaterialBindingKind, MaterialFragmentRequirement, MaterialKernel,
    MaterialKernelAbi, MaterialKernelKind, MaterialVertexRequirement, RasterCullOrder,
    RasterDevice, RasterDraw, RasterDrawBuilder, RasterPass, RasterPipeline, RasterTarget,
    RasterVisibilityMode, RenderTarget, SharedInstanceStream, Target, VisibilityMode,
    VisibleInstanceStream,
};
#[cfg(windows)]
pub use interop::NeoD3d12InteropDevice;
#[cfg(windows)]
use interop::{
    import_d3d12_fence, import_d3d12_resource_memory, map_external_buffer, signal_external_fence,
    wait_external_fence,
};
use resources::*;
pub use resources::{
    AccelerationGrid, BufferField, BufferFormat, DEFAULT_AOSOA_GROUP_SIZE, DEFAULT_MACROCELL_SIZE,
    DEFAULT_SPARSE_TEXTURE_GUTTER, DEFAULT_SPARSE_TEXTURE_PAGE_SIZE, DataLayout, IndexFormat,
    InstanceAttribute, InstanceBuffer, InstanceBufferDesc, InstanceFormat, InstanceLayout,
    InstanceSemantic, MATERIAL_STREAM_HEADER_U32S, MATERIAL_STREAM_MAGIC, MATERIAL_STREAM_VERSION,
    MaterialStream, MaterialStreamDesc, MaterialStreamFormat, MeshBuffer, MeshBufferDesc,
    PrimitiveTopology, SPARSE_TEXTURE_HEADER_U32S, SPARSE_TEXTURE_MAGIC,
    SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S, SPARSE_TEXTURE_VERSION, SparseTextureAtlas,
    SparseTextureDesc, SparseTextureFeedbackSummary, SparseTextureFormat, StructuredBuffer,
    StructuredBufferDesc, VISIBILITY_GRID_HEADER_U32S, VISIBILITY_GRID_MAGIC,
    VISIBILITY_GRID_RECORD_U32S, VertexAttribute, VertexFormat, VertexLayout, VertexSemantic,
    VisibilityGrid, VisibilityGridDesc,
};
#[cfg(test)]
use std::path::PathBuf;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
}

impl DeviceInfo {
    pub fn sm_label(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
    }

    pub fn is_pascal_sm61(&self) -> bool {
        self.compute_capability == (6, 1)
    }
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

    pub fn device_info(&self) -> Result<DeviceInfo, RuntimeError> {
        Ok(DeviceInfo {
            ordinal: self.inner.ordinal(),
            name: self.inner.name()?,
            compute_capability: self.inner.compute_capability()?,
        })
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
    unsigned int flags;
}};

struct NeoMaterialStreamHeader {{
    unsigned int magic;
    unsigned int version;
    unsigned int header_bytes;
    unsigned int material_count;
    unsigned int material_ids_offset;
    unsigned int format;
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
    if (header->format == {MATERIAL_STREAM_FORMAT_U16}u) {{
        return (unsigned int)((const unsigned short*)(materials + header->material_ids_offset))[id];
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

__device__ __forceinline__ const unsigned char* neo_sparse_texture_identity_page_bytes(const unsigned char* texture, unsigned int page_id) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    unsigned int page_bytes = header->page_size * header->page_size * 4u;
    return texture + header->physical_pages_offset + page_id * page_bytes;
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
    unsigned int rgba = ((const unsigned int*)(page + offset))[0];
    unsigned int r = rgba & 255u;
    unsigned int g = (rgba >> 8u) & 255u;
    unsigned int b = (rgba >> 16u) & 255u;
    unsigned int a = (rgba >> 24u) & 255u;
    return b | (g << 8u) | (r << 16u) | (a << 24u);
}}

__device__ __forceinline__ unsigned int neo_sparse_sample_bgra8_identity_resident_page(const unsigned char* texture, unsigned int page_id, float2 uv) {{
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    float wrapped_u = uv.x - floorf(uv.x);
    float wrapped_v = uv.y - floorf(uv.y);
    const unsigned char* page = neo_sparse_texture_identity_page_bytes(texture, page_id);
    unsigned int gutter = header->gutter;
    unsigned int usable = header->page_size > gutter * 2u ? header->page_size - gutter * 2u : header->page_size;
    unsigned int sample_x = (unsigned int)(wrapped_u * (float)usable);
    unsigned int sample_y = (unsigned int)(wrapped_v * (float)usable);
    if (sample_x >= usable) sample_x = usable - 1u;
    if (sample_y >= usable) sample_y = usable - 1u;
    unsigned int texel_x = gutter + sample_x;
    unsigned int texel_y = gutter + sample_y;
    unsigned int offset = (texel_y * header->page_size + texel_x) * 4u;
    unsigned int rgba = ((const unsigned int*)(page + offset))[0];
    unsigned int r = rgba & 255u;
    unsigned int g = (rgba >> 8u) & 255u;
    unsigned int b = (rgba >> 16u) & 255u;
    unsigned int a = (rgba >> 24u) & 255u;
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
    const NeoSparseTextureHeader* header = neo_sparse_texture_header(texture);
    if ((header->flags & {SPARSE_TEXTURE_FLAG_IDENTITY_RESIDENT}u) != 0u && page_id < header->physical_page_count) {{
        return neo_sparse_sample_bgra8_identity_resident_page(texture, page_id, uv);
    }}
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

    fn module_from_neo_source_or_skip(
        ctx: &Context,
        source: &str,
        entrypoints: &[&str],
        label: &str,
    ) -> Option<Module> {
        match Module::from_neo_source(ctx, source, entrypoints) {
            Ok(module) => Some(module),
            Err(RuntimeError::Nvrtc(err)) => {
                eprintln!("skipping {label} without usable NVRTC: {err}");
                None
            }
            Err(err) => panic!("{label} failed unexpectedly: {err}"),
        }
    }

    fn module_from_cuda_source_or_skip(
        ctx: &Context,
        source: String,
        label: &str,
    ) -> Option<Module> {
        match Module::from_cuda_source(ctx, source) {
            Ok(module) => Some(module),
            Err(RuntimeError::Nvrtc(err)) => {
                eprintln!("skipping {label} without usable NVRTC: {err}");
                None
            }
            Err(err) => panic!("{label} failed unexpectedly: {err}"),
        }
    }

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
        let desc = MaterialStreamDesc {
            material_count: 3,
            format: MaterialStreamFormat::U32,
        };
        let blob = MaterialStream::pack(&desc, &[7, 9, 11]).unwrap();
        assert_eq!(read_u32_le(&blob, 0), MATERIAL_STREAM_MAGIC);
        assert_eq!(read_u32_le(&blob, 4), MATERIAL_STREAM_VERSION);
        assert_eq!(
            read_u32_le(&blob, 8),
            MATERIAL_STREAM_HEADER_U32S as u32 * 4
        );
        assert_eq!(read_u32_le(&blob, 12), 3);
        assert_eq!(read_u32_le(&blob, 20), MATERIAL_STREAM_FORMAT_U32);
        let data = MATERIAL_STREAM_HEADER_U32S * 4;
        assert_eq!(read_u32_le(&blob, data), 7);
        assert_eq!(read_u32_le(&blob, data + 4), 9);
        assert_eq!(read_u32_le(&blob, data + 8), 11);

        let u16_desc = MaterialStreamDesc {
            material_count: 3,
            format: MaterialStreamFormat::U16,
        };
        let u16_blob = MaterialStream::pack(&u16_desc, &[7, 9, 11]).unwrap();
        assert_eq!(read_u32_le(&u16_blob, 20), MATERIAL_STREAM_FORMAT_U16);
        assert_eq!(
            u16::from_le_bytes(u16_blob[data..data + 2].try_into().unwrap()),
            7
        );
        assert_eq!(
            u16::from_le_bytes(u16_blob[data + 2..data + 4].try_into().unwrap()),
            9
        );
        assert!(MaterialStream::pack(&u16_desc, &[u32::from(u16::MAX) + 1, 0, 0]).is_err());
        let err = MaterialStream::pack(&desc, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 3 material IDs"));
        assert!(
            MaterialStream::pack(
                &MaterialStreamDesc {
                    material_count: 0,
                    format: MaterialStreamFormat::U32,
                },
                &[],
            )
            .is_err()
        );
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
        let Some(module) =
            module_from_neo_source_or_skip(&ctx, source, &["inspect"], "mesh prelude compile test")
        else {
            return;
        };
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
        let Some(module) = module_from_neo_source_or_skip(
            &ctx,
            source,
            &["inspect"],
            "instance prelude compile test",
        ) else {
            return;
        };
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
        assert!(prelude.contains("neo_sparse_texture_identity_page_bytes"));
        assert!(prelude.contains("neo_sparse_sample_bgra8_identity_resident_page"));
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
        let Some(module) = module_from_neo_source_or_skip(
            &ctx,
            "kernel fn inspect(global u8* mesh) { let count: u32 = neo_mesh_vertex_count(mesh); }",
            &["inspect"],
            "arg_mesh launch test",
        ) else {
            return;
        };
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
        let Some(module) = module_from_neo_source_or_skip(
            &ctx,
            "kernel fn inspect(global u8* instances) { let count: u32 = neo_instance_count(instances); }",
            &["inspect"],
            "arg_instances launch test",
        ) else {
            return;
        };
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
        let Some(module) = module_from_cuda_source_or_skip(
            &ctx,
            "extern \"C\" __global__ void noop_kernel() {}".to_string(),
            "native CUDA image smoke test",
        ) else {
            return;
        };
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
    fn diagnostics_mentions_active_cuda_build() {
        let diagnostics = RuntimeDiagnostics {
            cuda_driver_available: false,
            cuda_driver_error: None,
            nvrtc_candidates: Vec::new(),
            nvrtc_found: Vec::new(),
            nvrtc_compatible: Vec::new(),
            nvrtc_loadable: false,
        };
        assert!(
            diagnostics
                .nvrtc_help()
                .contains(expected_cuda_build_label())
        );
    }

    #[cfg(all(windows, feature = "cuda-12060"))]
    #[test]
    fn cuda_126_nvrtc_names_are_compatible_for_cuda_126_builds() {
        assert!(compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\x64\nvrtc64_120_0.dll"
        )));
        assert!(compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\x64\nvrtc64_12.dll"
        )));
        assert!(!compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\nvrtc64_130_0.dll"
        )));
        assert!(!compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\nvrtc.dll"
        )));
    }

    #[cfg(all(windows, feature = "cuda-13000"))]
    #[test]
    fn cuda_13_nvrtc_names_are_compatible_for_cuda_13_builds() {
        assert!(compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\nvrtc64_130_0.dll"
        )));
        assert!(compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\nvrtc64_13.dll"
        )));
        assert!(!compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\x64\nvrtc64_120_0.dll"
        )));
        assert!(!compatible_nvrtc_candidate(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\x64\nvrtc.dll"
        )));
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
