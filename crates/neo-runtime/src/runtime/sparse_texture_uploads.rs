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
