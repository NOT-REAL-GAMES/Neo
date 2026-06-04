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
