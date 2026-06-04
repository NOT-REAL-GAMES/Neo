#[cfg(windows)]
const GENERIC_ALL_ACCESS: u32 = 0x1000_0000;

#[cfg(windows)]
fn align_u32(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[cfg(windows)]
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
unsafe fn import_d3d12_resource_memory(
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
unsafe fn map_external_buffer(
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
unsafe fn import_d3d12_fence(
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
unsafe fn wait_external_fence(
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
unsafe fn signal_external_fence(
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
