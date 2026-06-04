pub fn nvrtc_available() -> bool {
    RuntimeDiagnostics::collect().nvrtc_loadable
}

fn compile_cuda_image_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    match compile_cubin_for_context_checked(ctx, cuda_source, diagnostics) {
        Ok(cubin) => return Ok(Ptx::from_binary(cubin)),
        Err(err) => {
            let _ = err;
        }
    }
    compile_ptx_checked(cuda_source, diagnostics)
}

fn compile_ptx_checked(
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_ptx(cuda_source)));
    std::panic::set_hook(panic_hook);
    result
        .map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))
}

fn compile_cubin_for_context_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let (major, minor) = ctx.inner.compute_capability()?;
    let arch = format!("sm_{major}{minor}");
    compile_cubin_checked(cuda_source, &arch, diagnostics)
}

fn compile_cubin_checked(
    cuda_source: &str,
    arch: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_cubin(cuda_source, arch)));
    std::panic::set_hook(panic_hook);
    result.map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
}

fn compile_cubin(cuda_source: &str, arch: &str) -> Result<Vec<u8>, RuntimeError> {
    use std::ffi::{CStr, CString};

    let src =
        CString::new(cuda_source.as_bytes()).expect("CUDA source cannot contain null terminators");
    let program = nvrtc_result::create_program(src.as_c_str(), None)
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let options = vec![format!("--gpu-architecture={arch}")];
    let compile_result = unsafe { nvrtc_result::compile_program(program, &options) };
    if let Err(err) = compile_result {
        let log = unsafe { nvrtc_result::get_program_log(program) }
            .ok()
            .map(|raw| {
                unsafe { CStr::from_ptr(raw.as_ptr()) }
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_default();
        unsafe {
            let _ = nvrtc_result::destroy_program(program);
        }
        return Err(RuntimeError::Nvrtc(format!(
            "native CUBIN compile failed for {arch}: {err}\n{log}"
        )));
    }
    let cubin = unsafe { nvrtc_get_cubin(program) };
    unsafe {
        let _ = nvrtc_result::destroy_program(program);
    }
    cubin
}

unsafe fn nvrtc_get_cubin(
    program: cudarc::nvrtc::sys::nvrtcProgram,
) -> Result<Vec<u8>, RuntimeError> {
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let mut cubin = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    Ok(cubin)
}

fn load_cuda_module_checked(
    ctx: &Context,
    image: Ptx,
) -> Result<Arc<cudarc::driver::CudaModule>, RuntimeError> {
    ctx.inner
        .load_module(image)
        .map_err(|err| unsupported_ptx_error(err).unwrap_or(RuntimeError::Driver(err)))
}

fn unsupported_ptx_error(err: DriverError) -> Option<RuntimeError> {
    (err.0 == sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION).then(|| {
        RuntimeError::Nvrtc(
            "CUDA driver rejected the compiled PTX because it was produced by a newer CUDA Toolkit than this driver supports. Update the NVIDIA driver, install a CUDA Toolkit matching the driver's reported CUDA version, or use Neo's native CUBIN path for the current GPU.".to_string(),
        )
    })
}

fn nvrtc_panic_help(payload: Box<dyn Any + Send>, diagnostics: &RuntimeDiagnostics) -> String {
    let panic_message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("cudarc panicked while loading NVRTC");
    format!("{panic_message}\n\n{}", diagnostics.nvrtc_loader_help())
}
