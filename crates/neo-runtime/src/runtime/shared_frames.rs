#[cfg(windows)]
pub struct SharedGpuBuffer {
    slot: SharedFrameSlot,
}

#[cfg(windows)]
impl SharedGpuBuffer {
    pub fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        byte_len: u64,
    ) -> Result<Self, RuntimeError> {
        if byte_len == 0 {
            return Err(RuntimeError::Raster(
                "shared GPU buffer size must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            slot: SharedFrameSlot::new(device, 0, byte_len)?,
        })
    }

    pub fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        self.slot.resource()
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        self.slot.device_ptr_arg()
    }

    pub fn bytes(&self) -> u64 {
        self.slot.bytes()
    }

    pub fn upload_bytes_on_stream(
        &mut self,
        stream: &Stream,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        if bytes.len() as u64 > self.bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                self.slot.device_ptr,
                bytes.as_ptr().cast(),
                bytes.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn wait_available_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        self.slot.wait_available_on_stream(stream)
    }

    pub fn signal_cuda_complete_on_stream(&mut self, stream: &Stream) -> Result<u64, RuntimeError> {
        self.slot.signal_cuda_complete_on_stream(stream)
    }

    pub fn wait_d3d_for_value(
        &self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
        value: u64,
    ) -> Result<(), RuntimeError> {
        self.slot.wait_d3d_for_value(queue, value)
    }

    pub fn is_fence_complete(&self, value: u64) -> bool {
        self.slot.is_fence_complete(value)
    }

    pub fn signal_available_on_d3d(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<u64, RuntimeError> {
        self.slot.signal_available_on_d3d(queue)
    }
}

#[cfg(windows)]
pub struct SharedFrameRing {
    width: u32,
    height: u32,
    pitch_bytes: u32,
    slots: Vec<SharedFrameSlot>,
}

#[cfg(windows)]
impl SharedFrameRing {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<Self, RuntimeError> {
        let row_bytes = width
            .checked_mul(4)
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let pitch_bytes = align_u32(row_bytes, 256);
        let total_bytes = u64::from(pitch_bytes)
            .checked_mul(u64::from(height))
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let mut ring = Vec::with_capacity(slots);
        for _ in 0..slots {
            ring.push(SharedFrameSlot::new(device, pitch_bytes, total_bytes)?);
        }
        Ok(Self {
            width,
            height,
            pitch_bytes,
            slots: ring,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pitch_bytes(&self) -> u32 {
        self.pitch_bytes
    }

    pub fn kernel_width(&self) -> u32 {
        self.pitch_bytes / 4
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn slot(&self, index: usize) -> Option<&SharedFrameSlot> {
        self.slots.get(index)
    }

    pub fn slot_mut(&mut self, index: usize) -> Option<&mut SharedFrameSlot> {
        self.slots.get_mut(index)
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn wait_idle(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<(), RuntimeError> {
        for slot in &mut self.slots {
            slot.wait_cpu(queue)?;
        }
        Ok(())
    }
}

#[cfg(windows)]
pub struct SharedFrameSlot {
    resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    memory_handle: windows::Win32::Foundation::HANDLE,
    fence_handle: windows::Win32::Foundation::HANDLE,
    external_memory: sys::CUexternalMemory,
    external_semaphore: sys::CUexternalSemaphore,
    device_ptr: sys::CUdeviceptr,
    bytes: u64,
    fence_value: u64,
}

#[cfg(windows)]
impl SharedFrameSlot {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        pitch_bytes: u32,
        total_bytes: u64,
    ) -> Result<Self, RuntimeError> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_FENCE_FLAG_SHARED, D3D12_HEAP_FLAG_SHARED,
                D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN,
                D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Fence,
                ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: total_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(|| {
            RuntimeError::D3d12Interop("D3D12 returned no shared frame resource".to_string())
        })?;
        let memory_handle = unsafe {
            device.CreateSharedHandle(
                &resource,
                None,
                GENERIC_ALL_ACCESS,
                windows::core::PCWSTR::null(),
            )?
        };
        let fence: ID3D12Fence = unsafe { device.CreateFence(1, D3D12_FENCE_FLAG_SHARED)? };
        let fence_handle = unsafe {
            device.CreateSharedHandle(
                &fence,
                None,
                GENERIC_ALL_ACCESS,
                windows::core::PCWSTR::null(),
            )?
        };
        let external_memory = unsafe { import_d3d12_resource_memory(memory_handle, total_bytes)? };
        let device_ptr = unsafe { map_external_buffer(external_memory, total_bytes)? };
        let external_semaphore = unsafe { import_d3d12_fence(fence_handle)? };
        let _ = pitch_bytes;
        Ok(Self {
            resource,
            fence,
            memory_handle,
            fence_handle,
            external_memory,
            external_semaphore,
            device_ptr,
            bytes: total_bytes,
            fence_value: 1,
        })
    }

    pub fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        &self.resource
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        CudaDevicePtrArg::new(self.device_ptr)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn wait_available_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        unsafe { wait_external_fence(self.external_semaphore, self.fence_value, stream) }
    }

    pub fn signal_cuda_complete_on_stream(&mut self, stream: &Stream) -> Result<u64, RuntimeError> {
        self.fence_value += 1;
        unsafe { signal_external_fence(self.external_semaphore, self.fence_value, stream)? };
        Ok(self.fence_value)
    }

    pub fn is_fence_complete(&self, value: u64) -> bool {
        unsafe { self.fence.GetCompletedValue() >= value }
    }

    pub fn wait_d3d_for_value(
        &self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
        value: u64,
    ) -> Result<(), RuntimeError> {
        unsafe {
            queue.Wait(&self.fence, value)?;
        }
        Ok(())
    }

    pub fn signal_available_on_d3d(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<u64, RuntimeError> {
        self.fence_value += 1;
        unsafe {
            queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    pub fn wait_cpu(
        &mut self,
        queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    ) -> Result<(), RuntimeError> {
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

        let wait_value = self.fence_value + 1;
        unsafe {
            queue.Signal(&self.fence, wait_value)?;
            if self.fence.GetCompletedValue() < wait_value {
                let event = CreateEventW(None, false, false, windows::core::PCWSTR::null())?;
                self.fence.SetEventOnCompletion(wait_value, event)?;
                WaitForSingleObject(event, INFINITE);
                let _ = windows::Win32::Foundation::CloseHandle(event);
            }
        }
        self.fence_value = wait_value;
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for SharedFrameSlot {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuMemFree_v2(self.device_ptr).result();
            let _ = sys::cuDestroyExternalMemory(self.external_memory).result();
            let _ = sys::cuDestroyExternalSemaphore(self.external_semaphore).result();
            let _ = windows::Win32::Foundation::CloseHandle(self.memory_handle);
            let _ = windows::Win32::Foundation::CloseHandle(self.fence_handle);
        }
    }
}

