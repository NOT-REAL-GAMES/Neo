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
