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
