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
