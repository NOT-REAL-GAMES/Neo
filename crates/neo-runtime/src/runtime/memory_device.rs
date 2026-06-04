pub struct DeviceBuffer<T> {
    inner: CudaSlice<T>,
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
