pub struct DeviceBuffer<T> {
    inner: CudaSlice<T>,
}

pub struct PinnedHostBuffer<T> {
    inner: PinnedHostSlice<T>,
}

pub struct ReadablePinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for ReadablePinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for ReadablePinnedHostBuffer<T> {}

impl<T> ReadablePinnedHostBuffer<T>
where
    T: DeviceRepr,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        ctx.inner.bind_to_thread()?;
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        let mut ptr = std::ptr::null_mut();
        unsafe {
            sys::cuMemAllocHost_v2(&mut ptr, byte_len)
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for ReadablePinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { sys::cuMemFreeHost(self.ptr.cast()).result() };
        }
    }
}

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

impl<T> PinnedHostBuffer<T>
where
    T: DeviceRepr,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        let inner = unsafe { ctx.inner.alloc_pinned(len)? };
        Ok(Self { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T> PinnedHostBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn as_slice(&self) -> Result<&[T], RuntimeError> {
        Ok(self.inner.as_slice()?)
    }
}

impl<T> DeviceBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn new(ctx: &Context, len: usize) -> Result<Self, RuntimeError> {
        Self::new_on_stream(&ctx.default_stream(), len)
    }

    pub fn new_on_stream(stream: &Stream, len: usize) -> Result<Self, RuntimeError> {
        let inner = stream.inner.alloc_zeros(len)?;
        Ok(Self { inner })
    }
}

impl<T> DeviceBuffer<T>
where
    T: DeviceRepr,
{
    pub fn upload(ctx: &Context, values: &[T]) -> Result<Self, RuntimeError> {
        Self::upload_on_stream(&ctx.default_stream(), values)
    }

    pub fn upload_on_stream(stream: &Stream, values: &[T]) -> Result<Self, RuntimeError> {
        let inner = stream.inner.clone_htod(values)?;
        Ok(Self { inner })
    }

    pub fn download(&self) -> Result<Vec<T>, RuntimeError> {
        Ok(self.inner.stream().clone_dtoh(&self.inner)?)
    }

    pub fn download_into(&self, dst: &mut [T]) -> Result<(), RuntimeError> {
        self.inner.stream().memcpy_dtoh(&self.inner, dst)?;
        Ok(())
    }

    pub fn download_range(&self, byte_offset: usize, dst: &mut [u8]) -> Result<(), RuntimeError> {
        self.download_range_on_stream(
            &Stream {
                inner: self.inner.stream().clone(),
            },
            byte_offset,
            dst,
        )
    }

    pub fn download_range_on_stream(
        &self,
        stream: &Stream,
        byte_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtr as _;

        let end = byte_offset
            .checked_add(dst.len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > self.inner.num_bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let (src, _record_read) = self.inner.device_ptr(&stream.inner);
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr().cast(),
                src + byte_offset as u64,
                dst.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        stream.synchronize()?;
        Ok(())
    }

    pub fn download_into_pinned(&self, dst: &mut PinnedHostBuffer<T>) -> Result<(), RuntimeError> {
        self.inner
            .stream()
            .memcpy_dtoh(&self.inner, &mut dst.inner)?;
        Ok(())
    }

    pub fn download_into_readable_pinned(
        &self,
        dst: &mut ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        let stream = Stream {
            inner: self.inner.stream().clone(),
        };
        self.download_into_readable_pinned_on_stream(&stream, dst)
    }

    pub fn download_into_readable_pinned_on_stream(
        &self,
        stream: &Stream,
        dst: &mut ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtr as _;

        let (src, _record_read) = self.inner.device_ptr(&stream.inner);
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.ptr.cast(),
                src,
                self.inner.num_bytes(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_from_readable_pinned_on_stream(
        &mut self,
        stream: &Stream,
        src: &ReadablePinnedHostBuffer<T>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        let byte_len = self.inner.num_bytes();
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.ptr.cast(), byte_len, stream.inner.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_from_on_stream(
        &mut self,
        stream: &Stream,
        src: &[T],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        if src.len() != self.inner.len() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let byte_len = self.inner.num_bytes();
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr().cast(), byte_len, stream.inner.cu_stream())
                .result()
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn upload_range(&mut self, byte_offset: usize, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.upload_range_on_stream(
            &Stream {
                inner: self.inner.stream().clone(),
            },
            byte_offset,
            bytes,
        )
    }

    pub fn upload_range_on_stream(
        &mut self,
        stream: &Stream,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::DevicePtrMut as _;

        let end = byte_offset
            .checked_add(bytes.len())
            .ok_or(RuntimeError::HostBufferTooLarge)?;
        if end > self.inner.num_bytes() {
            return Err(RuntimeError::HostBufferTooLarge);
        }
        let (dst, _record_write) = self.inner.device_ptr_mut(&stream.inner);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst + byte_offset as u64,
                bytes.as_ptr().cast(),
                bytes.len(),
                stream.inner.cu_stream(),
            )
            .result()
            .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn device_ptr_arg(&self) -> CudaDevicePtrArg {
        use cudarc::driver::DevicePtr as _;

        let (ptr, _record_read) = self.inner.device_ptr(self.inner.stream());
        CudaDevicePtrArg::new(ptr)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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

