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
