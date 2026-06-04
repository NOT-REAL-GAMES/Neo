pub struct CudaFence {
    event: sys::CUevent,
}

impl CudaFence {
    fn new() -> Result<Self, RuntimeError> {
        let mut event = std::ptr::null_mut();
        unsafe {
            sys::cuEventCreate(
                &mut event,
                sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC as u32,
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(Self { event })
    }

    pub fn record(&self, ctx: &Context) -> Result<(), RuntimeError> {
        self.record_on_stream(&ctx.default_stream())
    }

    pub fn record_on_stream(&self, stream: &Stream) -> Result<(), RuntimeError> {
        unsafe {
            sys::cuEventRecord(self.event, stream.inner.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        unsafe {
            sys::cuEventSynchronize(self.event)
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn is_complete(&self) -> Result<bool, RuntimeError> {
        match unsafe { sys::cuEventQuery(self.event) } {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            err => Err(RuntimeError::Driver(cudarc::driver::DriverError(err))),
        }
    }
}

impl Drop for CudaFence {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = unsafe { sys::cuEventDestroy_v2(self.event).result() };
        }
    }
}
