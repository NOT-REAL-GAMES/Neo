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
