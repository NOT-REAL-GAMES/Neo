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
