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
