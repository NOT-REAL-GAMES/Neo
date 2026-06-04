pub struct Kernel {
    function: CudaFunction,
    stream: Arc<CudaStream>,
}

impl Kernel {
    pub fn launcher(&self) -> KernelLaunch<'_> {
        KernelLaunch {
            inner: self.stream.launch_builder(&self.function),
        }
    }

    pub fn on_stream(&self, stream: &Stream) -> Self {
        Self {
            function: self.function.clone(),
            stream: stream.inner.clone(),
        }
    }
}

pub struct KernelLaunch<'a> {
    inner: LaunchArgs<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDevicePtrArg {
    ptr: sys::CUdeviceptr,
}

impl CudaDevicePtrArg {
    pub fn new(ptr: sys::CUdeviceptr) -> Self {
        Self { ptr }
    }
}

impl<'a> KernelLaunch<'a> {
    pub fn arg<T>(&mut self, value: &'a T) -> &mut Self
    where
        T: DeviceRepr,
    {
        self.inner.arg(value);
        self
    }

    pub fn arg_buffer<T>(&mut self, value: &'a DeviceBuffer<T>) -> &mut Self {
        self.inner.arg(&value.inner);
        self
    }

    pub fn arg_buffer_mut<T>(&mut self, value: &'a mut DeviceBuffer<T>) -> &mut Self {
        self.inner.arg(&mut value.inner);
        self
    }

    pub fn arg_device_ptr(&mut self, value: &'a CudaDevicePtrArg) -> &mut Self {
        self.inner.arg(&value.ptr);
        self
    }

    pub fn arg_mesh(&mut self, value: &'a MeshBuffer) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_instances(&mut self, value: &'a InstanceBuffer) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_visibility_grid(&mut self, value: &'a VisibilityGrid) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_sparse_texture(&mut self, value: &'a SparseTextureAtlas) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    pub fn arg_materials(&mut self, value: &'a MaterialStream) -> &mut Self {
        self.arg_buffer(&value.buffer)
    }

    /// Launches the configured kernel with explicit grid/block dimensions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pushed arguments exactly match the CUDA
    /// kernel signature, that mutable buffers are not aliased by concurrent GPU
    /// work, and that the kernel does not read or write outside the provided
    /// device allocations.
    pub unsafe fn launch(&mut self, dims: LaunchDims) -> Result<(), RuntimeError> {
        unsafe {
            self.inner.launch(dims.into())?;
        }
        Ok(())
    }
}

impl Kernel {
    /// Launches the current live image ABI with a raw CUDA device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `pixels` is valid for writes of the kernel's
    /// full output, that its lifetime extends until the launched work is
    /// complete, and that no other GPU queue aliases it without explicit
    /// synchronization.
    pub unsafe fn launch_image_raw_ptr(
        &self,
        dims: LaunchDims,
        pixels: CudaDevicePtrArg,
        width: u32,
        height: u32,
        time: f32,
        frame: u32,
    ) -> Result<(), RuntimeError> {
        let pixel_ptr = pixels.ptr;
        unsafe {
            self.launcher()
                .arg(&pixel_ptr)
                .arg(&width)
                .arg(&height)
                .arg(&time)
                .arg(&frame)
                .launch(dims)?;
        }
        Ok(())
    }
}
