pub struct Context {
    inner: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
}

impl DeviceInfo {
    pub fn sm_label(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
    }

    pub fn is_pascal_sm61(&self) -> bool {
        self.compute_capability == (6, 1)
    }
}

impl Context {
    pub fn new_default_device() -> Result<Self, RuntimeError> {
        let inner = CudaContext::new(0)?;
        let stream = inner.default_stream();
        Ok(Self { inner, stream })
    }

    pub fn compile_neo_module(&self, source: &str) -> Result<Module, RuntimeError> {
        Module::from_neo_source(self, source, &[])
    }

    pub fn alloc_zeros<T>(&self, len: usize) -> Result<DeviceBuffer<T>, RuntimeError>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        DeviceBuffer::new(self, len)
    }

    pub fn upload<T>(&self, values: &[T]) -> Result<DeviceBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        DeviceBuffer::upload(self, values)
    }

    pub fn alloc_pinned<T>(&self, len: usize) -> Result<PinnedHostBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        PinnedHostBuffer::new(self, len)
    }

    pub fn alloc_readable_pinned<T>(
        &self,
        len: usize,
    ) -> Result<ReadablePinnedHostBuffer<T>, RuntimeError>
    where
        T: DeviceRepr,
    {
        ReadablePinnedHostBuffer::new(self, len)
    }

    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn create_fence(&self) -> Result<CudaFence, RuntimeError> {
        CudaFence::new()
    }

    pub fn create_stream(&self) -> Result<Stream, RuntimeError> {
        Ok(Stream {
            inner: self.inner.new_stream()?,
        })
    }

    pub fn default_stream(&self) -> Stream {
        Stream {
            inner: self.stream.clone(),
        }
    }

    pub fn device_info(&self) -> Result<DeviceInfo, RuntimeError> {
        Ok(DeviceInfo {
            ordinal: self.inner.ordinal(),
            name: self.inner.name()?,
            compute_capability: self.inner.compute_capability()?,
        })
    }

    /// Disables cudarc's automatic multi-stream event tracking for future allocations.
    ///
    /// Callers that use this must provide their own stream/fence lifetime ordering.
    ///
    /// # Safety
    ///
    /// The caller must ensure all buffers allocated after this call are not used
    /// concurrently across streams unless explicit CUDA stream waits, fences, or
    /// other ordering guarantees protect those accesses.
    pub unsafe fn disable_automatic_event_tracking(&self) {
        unsafe {
            self.inner.disable_event_tracking();
        }
    }
}
