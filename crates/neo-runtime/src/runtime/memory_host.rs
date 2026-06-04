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
