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
