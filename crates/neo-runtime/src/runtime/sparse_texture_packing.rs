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
