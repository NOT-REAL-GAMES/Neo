use crate::{Context, CudaDevicePtrArg, DeviceBuffer, RuntimeError};

pub(crate) const MESH_MAGIC: u32 = 0x4d48_454e;
pub(crate) const MESH_VERSION: u32 = 1;
pub(crate) const MESH_HEADER_BYTES: usize = 48;
pub(crate) const MESH_ATTRIBUTE_BYTES: usize = 16;

pub(crate) const MESH_SEMANTIC_POSITION: u32 = 1;
pub(crate) const MESH_SEMANTIC_NORMAL: u32 = 2;
pub(crate) const MESH_SEMANTIC_UV0: u32 = 3;
pub(crate) const MESH_SEMANTIC_COLOR0: u32 = 4;

pub(crate) const MESH_FORMAT_F32X2: u32 = 1;
pub(crate) const MESH_FORMAT_F32X3: u32 = 2;
pub(crate) const MESH_FORMAT_F32X4: u32 = 3;
pub(crate) const MESH_FORMAT_U8X4_UNORM: u32 = 4;

pub(crate) const MESH_INDEX_NONE: u32 = 0;
pub(crate) const MESH_INDEX_U16: u32 = 1;
pub(crate) const MESH_INDEX_U32: u32 = 2;
pub(crate) const MESH_TOPOLOGY_TRIANGLE_LIST: u32 = 1;

pub(crate) const INSTANCE_MAGIC: u32 = 0x4948_454e;
pub(crate) const INSTANCE_VERSION: u32 = 2;
pub(crate) const INSTANCE_HEADER_BYTES: usize = 40;
pub(crate) const INSTANCE_ATTRIBUTE_BYTES: usize = 16;

pub(crate) const INSTANCE_SEMANTIC_POSITION: u32 = 1;
pub(crate) const INSTANCE_SEMANTIC_ROTATION: u32 = 2;
pub(crate) const INSTANCE_SEMANTIC_SCALE: u32 = 3;
pub(crate) const INSTANCE_SEMANTIC_COLOR0: u32 = 4;

pub(crate) const INSTANCE_FORMAT_F32X2: u32 = 1;
pub(crate) const INSTANCE_FORMAT_F32X3: u32 = 2;
pub(crate) const INSTANCE_FORMAT_F32X4: u32 = 3;
pub(crate) const INSTANCE_FORMAT_U8X4_UNORM: u32 = 4;

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
pub(crate) const SPARSE_TEXTURE_FORMAT_RGBA8_UNORM: u32 = 1;
pub(crate) const SPARSE_TEXTURE_ENTRY_RESIDENT: u32 = 1 << 31;
pub(crate) const SPARSE_TEXTURE_ENTRY_PHYSICAL_MASK: u32 = 0x00ff_ffff;
pub(crate) const SPARSE_TEXTURE_HEADER_FEEDBACK_FLAGS_U32: usize = 18;
pub(crate) const SPARSE_TEXTURE_HEADER_FLAGS_U32: usize = 19;
pub(crate) const SPARSE_TEXTURE_FEEDBACK_ENABLED: u32 = 1;
pub(crate) const SPARSE_TEXTURE_FLAG_IDENTITY_RESIDENT: u32 = 1;

pub const MATERIAL_STREAM_MAGIC: u32 = 0x4d53_584e;
pub const MATERIAL_STREAM_VERSION: u32 = 1;
pub const MATERIAL_STREAM_HEADER_U32S: usize = 8;
pub(crate) const MATERIAL_STREAM_FORMAT_U32: u32 = 0;
pub(crate) const MATERIAL_STREAM_FORMAT_U16: u32 = 1;

pub const DEFAULT_AOSOA_GROUP_SIZE: u32 = 32;
pub(crate) const DATA_LAYOUT_AOS: u32 = 0;
pub(crate) const DATA_LAYOUT_SOA: u32 = 1;
pub(crate) const DATA_LAYOUT_AOSOA: u32 = 2;

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
    pub(crate) buffer: DeviceBuffer<u8>,
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
    pub(crate) buffer: DeviceBuffer<u8>,
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
    pub(crate) buffer: DeviceBuffer<u8>,
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
    pub(crate) buffer: DeviceBuffer<u8>,
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
    pub format: MaterialStreamFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialStreamFormat {
    U32,
    U16,
}

impl MaterialStreamFormat {
    fn code(self) -> u32 {
        match self {
            Self::U32 => MATERIAL_STREAM_FORMAT_U32,
            Self::U16 => MATERIAL_STREAM_FORMAT_U16,
        }
    }

    fn byte_len(self) -> usize {
        match self {
            Self::U32 => 4,
            Self::U16 => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::U16 => "u16",
        }
    }
}

pub struct MaterialStream {
    pub(crate) buffer: DeviceBuffer<u8>,
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

    pub fn set_identity_resident_fast_path(&mut self, enabled: bool) -> Result<(), RuntimeError> {
        let flags = if enabled {
            SPARSE_TEXTURE_FLAG_IDENTITY_RESIDENT
        } else {
            0
        };
        self.buffer
            .upload_range(SPARSE_TEXTURE_HEADER_FLAGS_U32 * 4, &flags.to_le_bytes())
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
        Self::upload_with_format(ctx, material_ids, MaterialStreamFormat::U32)
    }

    pub fn upload_u16(ctx: &Context, material_ids: &[u32]) -> Result<Self, RuntimeError> {
        Self::upload_with_format(ctx, material_ids, MaterialStreamFormat::U16)
    }

    pub fn upload_with_format(
        ctx: &Context,
        material_ids: &[u32],
        format: MaterialStreamFormat,
    ) -> Result<Self, RuntimeError> {
        let desc = MaterialStreamDesc {
            material_count: u32::try_from(material_ids.len())
                .map_err(|_| RuntimeError::HostBufferTooLarge)?,
            format,
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

    pub fn format(&self) -> MaterialStreamFormat {
        self.desc.format
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

pub(crate) fn pack_mesh_buffer(
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

pub(crate) fn validate_mesh_buffer(
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
pub(crate) fn pack_instance_buffer(
    desc: &InstanceBufferDesc,
    instance_bytes: &[u8],
) -> Result<Vec<u8>, RuntimeError> {
    pack_instance_buffer_with_layout(desc, instance_bytes, DataLayout::AoS)
}

pub(crate) fn pack_instance_buffer_with_layout(
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

pub(crate) fn pack_structured_buffer(
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

pub(crate) fn validate_structured_buffer(
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

pub(crate) fn structured_stream_offsets(
    desc: &StructuredBufferDesc,
) -> Result<Vec<usize>, RuntimeError> {
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

pub(crate) fn structured_data_len(
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

pub(crate) fn structured_stream_byte_len(
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

pub(crate) fn validate_instance_buffer(
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

pub(crate) fn visibility_macrocell_dims(
    desc: &VisibilityGridDesc,
) -> Result<[u32; 3], RuntimeError> {
    validate_visibility_grid_desc(desc)?;
    Ok([
        desc.cells[0].div_ceil(desc.macrocell_size),
        desc.cells[1].div_ceil(desc.macrocell_size),
        desc.cells[2].div_ceil(desc.macrocell_size),
    ])
}

pub(crate) fn visibility_macrocell_count(dims: [u32; 3]) -> Result<u32, RuntimeError> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn visibility_bitset_words(macrocell_count: u32) -> Result<u32, RuntimeError> {
    Ok(macrocell_count.div_ceil(32))
}

pub(crate) fn visibility_grid_u32_len(desc: &VisibilityGridDesc) -> Result<usize, RuntimeError> {
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

pub(crate) fn pack_visibility_grid(desc: &VisibilityGridDesc) -> Result<Vec<u8>, RuntimeError> {
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

pub(crate) fn validate_visibility_grid_desc(desc: &VisibilityGridDesc) -> Result<(), RuntimeError> {
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

pub(crate) fn sparse_texture_page_dims(desc: &SparseTextureDesc) -> Result<[u32; 2], RuntimeError> {
    validate_sparse_texture_desc(desc)?;
    Ok([
        desc.virtual_width.div_ceil(desc.page_size),
        desc.virtual_height.div_ceil(desc.page_size),
    ])
}

pub(crate) fn sparse_texture_virtual_page_count(
    desc: &SparseTextureDesc,
) -> Result<u32, RuntimeError> {
    let dims = sparse_texture_page_dims(desc)?;
    dims[0]
        .checked_mul(dims[1])
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn sparse_texture_page_bytes(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    desc.page_size
        .checked_mul(desc.page_size)
        .and_then(|pixels| pixels.checked_mul(desc.format.bytes_per_pixel()))
        .map(|bytes| bytes as usize)
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn sparse_texture_page_table_offset(virtual_page: u32) -> Result<usize, RuntimeError> {
    let page_offset = usize::try_from(virtual_page)
        .ok()
        .and_then(|page| page.checked_mul(SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S * 4))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    SPARSE_TEXTURE_HEADER_U32S
        .checked_mul(4)
        .and_then(|offset| offset.checked_add(page_offset))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn sparse_texture_pages_offset(desc: &SparseTextureDesc) -> Result<usize, RuntimeError> {
    let page_table_bytes = usize::try_from(sparse_texture_virtual_page_count(desc)?)
        .ok()
        .and_then(|pages| pages.checked_mul(SPARSE_TEXTURE_PAGE_TABLE_ENTRY_U32S * 4))
        .ok_or(RuntimeError::HostBufferTooLarge)?;
    Ok(align_usize(
        SPARSE_TEXTURE_HEADER_U32S * 4 + page_table_bytes,
        16,
    ))
}

pub(crate) fn sparse_texture_fallback_page_offset(
    desc: &SparseTextureDesc,
) -> Result<usize, RuntimeError> {
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

pub(crate) fn sparse_texture_feedback_offset(
    desc: &SparseTextureDesc,
) -> Result<usize, RuntimeError> {
    let fallback_offset = sparse_texture_fallback_page_offset(desc)?;
    let page_bytes = sparse_texture_page_bytes(desc)?;
    fallback_offset
        .checked_add(page_bytes)
        .map(|offset| align_usize(offset, 16))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn sparse_texture_feedback_byte_len(
    desc: &SparseTextureDesc,
) -> Result<usize, RuntimeError> {
    usize::try_from(sparse_texture_virtual_page_count(desc)?)
        .ok()
        .and_then(|pages| pages.checked_mul(4))
        .ok_or(RuntimeError::HostBufferTooLarge)
}

pub(crate) fn sparse_texture_physical_page_offset(
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

pub(crate) fn pack_sparse_texture(desc: &SparseTextureDesc) -> Result<Vec<u8>, RuntimeError> {
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

pub(crate) fn validate_sparse_texture_desc(desc: &SparseTextureDesc) -> Result<(), RuntimeError> {
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

pub(crate) fn validate_sparse_virtual_page(
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

pub(crate) fn validate_sparse_physical_page(
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

pub(crate) fn fill_sparse_checker_page(
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

pub(crate) fn fill_sparse_fallback_page(
    desc: &SparseTextureDesc,
    dst: &mut [u8],
) -> Result<(), RuntimeError> {
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

pub(crate) fn summarize_sparse_texture_feedback(
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

pub(crate) fn pack_material_stream(
    desc: &MaterialStreamDesc,
    material_ids: &[u32],
) -> Result<Vec<u8>, RuntimeError> {
    validate_material_stream(desc, material_ids)?;
    let data_offset = MATERIAL_STREAM_HEADER_U32S * 4;
    let data_bytes = material_ids
        .len()
        .checked_mul(desc.format.byte_len())
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
        desc.format.code(),
        0,
        0,
    ];
    for (idx, value) in header.into_iter().enumerate() {
        write_u32_le(&mut blob, idx * 4, value);
    }
    match desc.format {
        MaterialStreamFormat::U32 => {
            for (idx, value) in material_ids.iter().copied().enumerate() {
                write_u32_le(&mut blob, data_offset + idx * 4, value);
            }
        }
        MaterialStreamFormat::U16 => {
            for (idx, value) in material_ids.iter().copied().enumerate() {
                let value = u16::try_from(value).map_err(|_| {
                    RuntimeError::MaterialStream(format!(
                        "material ID {value} is too large for u16 material stream"
                    ))
                })?;
                let offset = data_offset + idx * 2;
                blob[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(blob)
}

pub(crate) fn validate_material_stream(
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
    if desc.format == MaterialStreamFormat::U16
        && let Some(value) = material_ids
            .iter()
            .copied()
            .find(|value| *value > u16::MAX as u32)
    {
        return Err(RuntimeError::MaterialStream(format!(
            "material ID {value} is too large for u16 material stream"
        )));
    }
    Ok(())
}

pub(crate) fn align_usize(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

pub(crate) fn write_u32_le(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(crate) fn slice_as_bytes<T: Copy>(values: &[T]) -> &[u8] {
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
