#[cfg(windows)]
pub struct NeoD3d12InteropDevice {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
}

#[cfg(windows)]
impl NeoD3d12InteropDevice {
    pub fn new(ctx: &Context) -> Result<Self, RuntimeError> {
        use windows::Win32::Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_11_0,
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                D3D12CreateDevice, ID3D12CommandQueue, ID3D12Device,
            },
            Dxgi::{
                CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
                IDXGIAdapter1, IDXGIFactory1,
            },
        };

        let cuda_luid = cuda_device_luid(ctx)?;
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let mut adapter_index = 0;
        let mut matched: Option<IDXGIAdapter1> = None;
        loop {
            let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
                Ok(adapter) => adapter,
                Err(_) => break,
            };
            let desc = unsafe { adapter.GetDesc1()? };
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0
                && dxgi_luid_bytes(desc.AdapterLuid) == cuda_luid
            {
                matched = Some(adapter);
                break;
            }
            adapter_index += 1;
        }
        let adapter = matched.ok_or_else(|| {
            RuntimeError::D3d12Interop(
                "could not find a DXGI adapter matching CUDA device 0 LUID".to_string(),
            )
        })?;
        let mut device: Option<ID3D12Device> = None;
        unsafe {
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)?;
        }
        let device = device.ok_or_else(|| {
            RuntimeError::D3d12Interop("D3D12CreateDevice returned no device".to_string())
        })?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc)? };
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Device {
        &self.device
    }

    pub fn queue(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue {
        &self.queue
    }

    pub fn create_shared_frame_ring(
        &self,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<SharedFrameRing, RuntimeError> {
        SharedFrameRing::new(&self.device, width, height, slots)
    }

    pub fn create_shared_gpu_buffer(&self, byte_len: u64) -> Result<SharedGpuBuffer, RuntimeError> {
        SharedGpuBuffer::new(&self.device, byte_len)
    }
}

#[cfg(windows)]
#[derive(Clone)]
pub struct DrawDevice {
    interop: Arc<NeoD3d12InteropDevice>,
}

#[cfg(windows)]
pub type RasterDevice = DrawDevice;

#[cfg(windows)]
impl DrawDevice {
    pub fn new(ctx: &Context) -> Result<Self, RuntimeError> {
        Ok(Self {
            interop: Arc::new(NeoD3d12InteropDevice::new(ctx)?),
        })
    }

    pub fn from_interop(interop: NeoD3d12InteropDevice) -> Self {
        Self {
            interop: Arc::new(interop),
        }
    }

    pub fn interop(&self) -> &NeoD3d12InteropDevice {
        &self.interop
    }

    pub fn create_shared_gpu_buffer(&self, byte_len: u64) -> Result<SharedGpuBuffer, RuntimeError> {
        self.interop.create_shared_gpu_buffer(byte_len)
    }
}

#[cfg(windows)]
pub struct DrawPipeline {
    label: String,
}

#[cfg(windows)]
pub type RasterPipeline = DrawPipeline;

#[cfg(windows)]
impl DrawPipeline {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}
