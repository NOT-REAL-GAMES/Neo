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

#[derive(Clone)]
pub struct Stream {
    inner: Arc<CudaStream>,
}

impl Stream {
    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.inner.synchronize()?;
        Ok(())
    }

    pub fn create_fence(&self) -> Result<CudaFence, RuntimeError> {
        CudaFence::new()
    }

    pub fn begin_graph_capture(&self) -> Result<(), RuntimeError> {
        self.inner
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        Ok(())
    }

    pub fn end_graph_capture(&self) -> Result<Option<CudaGraph>, RuntimeError> {
        let no_flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0) };
        let graph = self.inner.end_capture(no_flags)?;
        Ok(graph.map(|inner| CudaGraph { inner }))
    }
}

pub struct CudaGraph {
    inner: CudarcCudaGraph,
}

impl CudaGraph {
    pub fn launch(&self) -> Result<(), RuntimeError> {
        self.inner.launch()?;
        Ok(())
    }

    pub fn upload(&self) -> Result<(), RuntimeError> {
        self.inner.upload()?;
        Ok(())
    }
}

pub struct Module {
    inner: Arc<cudarc::driver::CudaModule>,
    stream: Arc<CudaStream>,
    pub cuda_source: String,
}

impl Module {
    pub fn from_neo_source(
        ctx: &Context,
        source: &str,
        entrypoints: &[&str],
    ) -> Result<Self, RuntimeError> {
        let program = neo_lang::parse(source)?;
        for entrypoint in entrypoints {
            if !program.kernels.iter().any(|kernel| {
                kernel.kind == neo_lang::EntryPointKind::Kernel && kernel.name == *entrypoint
            }) {
                return Err(RuntimeError::MissingEntrypoint((*entrypoint).to_string()));
            }
        }
        let cuda_source = format!(
            "{}\n{}",
            runtime_cuda_prelude(),
            neo_lang::lower_program(&program)
        );
        let diagnostics = RuntimeDiagnostics::collect();
        if !diagnostics.nvrtc_loadable {
            return Err(RuntimeError::Nvrtc(diagnostics.nvrtc_help()));
        }
        configure_nvrtc_search_path(&diagnostics);
        let ptx = compile_cuda_image_checked(ctx, &cuda_source, &diagnostics)?;
        let inner = load_cuda_module_checked(ctx, ptx)?;
        Ok(Self {
            inner,
            stream: ctx.stream.clone(),
            cuda_source,
        })
    }

    pub fn from_cuda_source(ctx: &Context, cuda_source: String) -> Result<Self, RuntimeError> {
        let diagnostics = RuntimeDiagnostics::collect();
        if !diagnostics.nvrtc_loadable {
            return Err(RuntimeError::Nvrtc(diagnostics.nvrtc_help()));
        }
        configure_nvrtc_search_path(&diagnostics);
        let ptx = compile_cuda_image_checked(ctx, &cuda_source, &diagnostics)?;
        let inner = load_cuda_module_checked(ctx, ptx)?;
        Ok(Self {
            inner,
            stream: ctx.stream.clone(),
            cuda_source,
        })
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, RuntimeError> {
        let function = self.inner.load_function(name)?;
        Ok(Kernel {
            function,
            stream: self.stream.clone(),
        })
    }

    pub fn kernel_on_stream(&self, name: &str, stream: &Stream) -> Result<Kernel, RuntimeError> {
        let function = self.inner.load_function(name)?;
        Ok(Kernel {
            function,
            stream: stream.inner.clone(),
        })
    }
}
