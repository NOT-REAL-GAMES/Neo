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
