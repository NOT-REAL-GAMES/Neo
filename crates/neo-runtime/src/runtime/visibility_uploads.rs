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
