#[cfg(windows)]
pub struct IndirectDrawBuffer {
    buffer: SharedGpuBuffer,
    command_capacity: u32,
}

#[cfg(windows)]
impl IndirectDrawBuffer {
    pub fn new(
        device: &NeoD3d12InteropDevice,
        command_capacity: u32,
    ) -> Result<Self, RuntimeError> {
        if command_capacity == 0 {
            return Err(RuntimeError::Raster(
                "indirect draw command capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(command_capacity)
            .checked_mul(std::mem::size_of::<DrawIndexedIndirectCommand>() as u64)
            .ok_or_else(|| {
                RuntimeError::Raster("indirect draw buffer size overflow".to_string())
            })?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            command_capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn command_capacity(&self) -> u32 {
        self.command_capacity
    }
}

#[cfg(windows)]
pub struct VisibleInstanceStream {
    buffer: SharedGpuBuffer,
    capacity: u32,
}

#[cfg(windows)]
impl VisibleInstanceStream {
    pub fn new(device: &NeoD3d12InteropDevice, capacity: u32) -> Result<Self, RuntimeError> {
        if capacity == 0 {
            return Err(RuntimeError::Raster(
                "visible instance stream capacity must be greater than zero".to_string(),
            ));
        }
        let byte_len = u64::from(capacity)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        Ok(Self {
            buffer: device.create_shared_gpu_buffer(byte_len)?,
            capacity,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(windows)]
pub struct SharedInstanceStream {
    buffer: SharedGpuBuffer,
    desc: InstanceBufferDesc,
    data_layout: DataLayout,
    byte_len: usize,
}

#[cfg(windows)]
impl SharedInstanceStream {
    pub fn upload_typed<I>(
        ctx: &Context,
        device: &NeoD3d12InteropDevice,
        desc: InstanceBufferDesc,
        instances: &[I],
        data_layout: DataLayout,
    ) -> Result<Self, RuntimeError>
    where
        I: Copy,
    {
        let packed = InstanceBuffer::pack_typed_with_layout(&desc, instances, data_layout)?;
        let byte_len = packed.len();
        let mut buffer = device.create_shared_gpu_buffer(byte_len as u64)?;
        let stream = ctx.default_stream();
        buffer.upload_bytes_on_stream(&stream, &packed)?;
        ctx.synchronize()?;
        Ok(Self {
            buffer,
            desc,
            data_layout,
            byte_len,
        })
    }

    pub fn buffer(&self) -> &SharedGpuBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut SharedGpuBuffer {
        &mut self.buffer
    }

    pub fn desc(&self) -> &InstanceBufferDesc {
        &self.desc
    }

    pub fn data_layout(&self) -> DataLayout {
        self.data_layout
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count_per_instance: u32,
    pub instance_count: u32,
    pub start_index_location: u32,
    pub base_vertex_location: i32,
    pub start_instance_location: u32,
}

impl DrawIndexedIndirectCommand {
    pub const BYTE_LEN: usize = std::mem::size_of::<Self>();

    pub fn indexed_quad(instance_count: u32) -> Self {
        Self {
            index_count_per_instance: 6,
            instance_count,
            start_index_location: 0,
            base_vertex_location: 0,
            start_instance_location: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}
