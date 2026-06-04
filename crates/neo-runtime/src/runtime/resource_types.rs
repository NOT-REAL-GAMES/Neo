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
const SPARSE_TEXTURE_HEADER_FLAGS_U32: usize = 19;
const SPARSE_TEXTURE_FEEDBACK_ENABLED: u32 = 1;
const SPARSE_TEXTURE_FLAG_IDENTITY_RESIDENT: u32 = 1;

pub const MATERIAL_STREAM_MAGIC: u32 = 0x4d53_584e;
pub const MATERIAL_STREAM_VERSION: u32 = 1;
pub const MATERIAL_STREAM_HEADER_U32S: usize = 8;
const MATERIAL_STREAM_FORMAT_U32: u32 = 0;
const MATERIAL_STREAM_FORMAT_U16: u32 = 1;

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
    buffer: DeviceBuffer<u8>,
    desc: MaterialStreamDesc,
    byte_len: usize,
}
