#[cfg(windows)]
use cudarc::driver::sys;

#[cfg(windows)]
use crate::{Context, RuntimeError, SharedFrameRing, SharedGpuBuffer, Stream};

#[cfg(windows)]
pub struct NeoD3d12InteropDevice {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
}

fn cuda_device_luid(ctx: &Context) -> Result<[u8; 8], RuntimeError> {
    let mut device = 0;
    unsafe {
        sys::cuDeviceGet(&mut device, ctx.inner.ordinal() as i32).result()?;
        let mut luid = [0i8; 8];
        let mut node_mask = 0u32;
        sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut node_mask, device).result()?;
        Ok(luid.map(|byte| byte as u8))
    }
}

#[cfg(windows)]
fn dxgi_luid_bytes(luid: windows::Win32::Foundation::LUID) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
    bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
    bytes
}

#[cfg(windows)]
pub(crate) unsafe fn import_d3d12_resource_memory(
    handle: windows::Win32::Foundation::HANDLE,
    size: u64,
) -> Result<sys::CUexternalMemory, RuntimeError> {
    let mut external_memory = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &desc).result()?;
        Ok(external_memory.assume_init())
    }
}

#[cfg(windows)]
pub(crate) unsafe fn map_external_buffer(
    external_memory: sys::CUexternalMemory,
    size: u64,
) -> Result<sys::CUdeviceptr, RuntimeError> {
    let mut device_ptr = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
        offset: 0,
        size,
        flags: 0,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuExternalMemoryGetMappedBuffer(device_ptr.as_mut_ptr(), external_memory, &desc)
            .result()?;
        Ok(device_ptr.assume_init())
    }
}

#[cfg(windows)]
pub(crate) unsafe fn import_d3d12_fence(
    handle: windows::Win32::Foundation::HANDLE,
) -> Result<sys::CUexternalSemaphore, RuntimeError> {
    let mut external_semaphore = std::mem::MaybeUninit::uninit();
    let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
        handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        flags: 0,
        reserved: [0; 16],
    };
    unsafe {
        sys::cuImportExternalSemaphore(external_semaphore.as_mut_ptr(), &desc).result()?;
        Ok(external_semaphore.assume_init())
    }
}

#[cfg(windows)]
pub(crate) unsafe fn wait_external_fence(
    semaphore: sys::CUexternalSemaphore,
    value: u64,
    stream: &Stream,
) -> Result<(), RuntimeError> {
    let semaphores = [semaphore];
    let params = [sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
                value,
            },
            nvSciSync: unsafe { std::mem::zeroed() },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
                timeoutMs: 0,
            },
            reserved: [0; 10],
        },
        flags: 0,
        reserved: [0; 16],
    }];
    unsafe {
        sys::cuWaitExternalSemaphoresAsync(
            semaphores.as_ptr(),
            params.as_ptr(),
            1,
            stream.inner.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) unsafe fn signal_external_fence(
    semaphore: sys::CUexternalSemaphore,
    value: u64,
    stream: &Stream,
) -> Result<(), RuntimeError> {
    let semaphores = [semaphore];
    let params = [sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
                value,
            },
            nvSciSync: unsafe { std::mem::zeroed() },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
            },
            reserved: [0; 12],
        },
        flags: 0,
        reserved: [0; 16],
    }];
    unsafe {
        sys::cuSignalExternalSemaphoresAsync(
            semaphores.as_ptr(),
            params.as_ptr(),
            1,
            stream.inner.cu_stream(),
        )
        .result()?;
    }
    Ok(())
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
