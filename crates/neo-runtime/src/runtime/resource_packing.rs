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
