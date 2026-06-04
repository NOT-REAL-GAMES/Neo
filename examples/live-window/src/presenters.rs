use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PresentTimings {
    pub(crate) map_copy: Duration,
    pub(crate) draw: Duration,
    pub(crate) swap_present: Duration,
    pub(crate) total: Duration,
}

pub(crate) enum PresentSink {
    Direct(WindowPresenter),
    Threaded(ThreadedPresenter),
}

impl PresentSink {
    pub(crate) fn threaded(presenter: WindowPresenter) -> Self {
        Self::Threaded(ThreadedPresenter::new(presenter))
    }

    pub(crate) fn present_sync(
        &mut self,
        size: PhysicalSize<u32>,
        bgra: &[u8],
    ) -> Result<PresentTimings> {
        match self {
            Self::Direct(presenter) => presenter.present(size, bgra),
            Self::Threaded(_) => {
                bail!("synchronous presentation is not available on present thread")
            }
        }
    }

    pub(crate) fn present_sampled(
        &mut self,
        size: PhysicalSize<u32>,
        bgra: &[u8],
        sample_download: Duration,
    ) -> Result<Option<PresentTimings>> {
        match self {
            Self::Direct(presenter) => Ok(Some(presenter.present(size, bgra)?)),
            Self::Threaded(presenter) => {
                presenter.submit(PresentFrame {
                    size,
                    bgra: bgra.to_vec(),
                    sample_download,
                })?;
                Ok(None)
            }
        }
    }

    pub(crate) fn drain_results(&mut self) -> Result<ThroughputBatchStats> {
        match self {
            Self::Direct(_) => Ok(ThroughputBatchStats::default()),
            Self::Threaded(presenter) => presenter.drain_results(),
        }
    }

    pub(crate) fn kind(&self) -> PresenterKind {
        match self {
            Self::Direct(presenter) => presenter.kind(),
            Self::Threaded(presenter) => presenter.kind(),
        }
    }
}

struct PresentFrame {
    size: PhysicalSize<u32>,
    bgra: Vec<u8>,
    sample_download: Duration,
}

struct PresentThreadResult {
    size: PhysicalSize<u32>,
    sample_download: Duration,
    timings: PresentTimings,
}

pub(crate) struct ThreadedPresenter {
    kind: PresenterKind,
    shared: Arc<ThreadedPresenterShared>,
    results: Receiver<std::result::Result<PresentThreadResult, String>>,
    worker: Option<thread::JoinHandle<()>>,
}

struct ThreadedPresenterShared {
    latest: Mutex<Option<PresentFrame>>,
    available: Condvar,
    shutdown: AtomicBool,
}

impl ThreadedPresenter {
    fn new(mut presenter: WindowPresenter) -> Self {
        let kind = presenter.kind();
        let shared = Arc::new(ThreadedPresenterShared {
            latest: Mutex::new(None),
            available: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let worker_shared = shared.clone();
        let (result_tx, results) = mpsc::channel();
        let worker = thread::spawn(move || {
            loop {
                let frame = {
                    let mut latest = worker_shared
                        .latest
                        .lock()
                        .expect("present thread queue mutex poisoned");
                    while latest.is_none() && !worker_shared.shutdown.load(Ordering::Acquire) {
                        latest = worker_shared
                            .available
                            .wait(latest)
                            .expect("present thread queue mutex poisoned");
                    }
                    if latest.is_none() && worker_shared.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    latest.take()
                };

                let Some(frame) = frame else {
                    continue;
                };
                let result = presenter
                    .present(frame.size, &frame.bgra)
                    .map(|timings| PresentThreadResult {
                        size: frame.size,
                        sample_download: frame.sample_download,
                        timings,
                    })
                    .map_err(|err| format!("{err:#}"));
                if result_tx.send(result).is_err() {
                    break;
                }
            }
        });
        Self {
            kind,
            shared,
            results,
            worker: Some(worker),
        }
    }

    fn submit(&self, frame: PresentFrame) -> Result<()> {
        let mut latest = self
            .shared
            .latest
            .lock()
            .map_err(|_| anyhow!("present thread queue mutex poisoned"))?;
        *latest = Some(frame);
        self.shared.available.notify_one();
        Ok(())
    }

    fn drain_results(&mut self) -> Result<ThroughputBatchStats> {
        let mut stats = ThroughputBatchStats::default();
        for result in self.results.try_iter() {
            let result = result.map_err(|err| anyhow!("{err}"))?;
            stats.sample_download += result.sample_download;
            stats.present += result.timings.total;
            stats.map_copy += result.timings.map_copy;
            stats.draw += result.timings.draw;
            stats.swap_present += result.timings.swap_present;
            stats.sampled_frames += 1;
            stats.presented_frames += 1;
            let bytes = frame_byte_len(result.size.width, result.size.height)?;
            stats.sampled_bytes += bytes;
            stats.uploaded_bytes += bytes;
        }
        Ok(stats)
    }

    fn kind(&self) -> PresenterKind {
        self.kind
    }
}

impl Drop for ThreadedPresenter {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.available.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(windows)]
pub(crate) struct WindowPresenter {
    inner: PresenterImpl,
}

// The presenter is moved once onto the present worker and then used only there.
// The wrapped Win32/D3D handles are not accessed concurrently after the move.
#[cfg(windows)]
unsafe impl Send for WindowPresenter {}

#[cfg(windows)]
enum PresenterImpl {
    D3d12Interop(D3d12InteropPresenter),
    D3d12(D3d12Presenter),
    D3d11(D3d11Presenter),
    Gdi(GdiPresenter),
}

#[cfg(windows)]
pub(crate) struct RasterPresentInput<'a> {
    pub(crate) size: PhysicalSize<u32>,
    pub(crate) args: &'a mut SharedGpuBuffer,
    pub(crate) visible_ids: &'a mut SharedGpuBuffer,
    pub(crate) raster_instances: &'a mut SharedGpuBuffer,
    pub(crate) material: &'a MaterialKernelAbi,
    pub(crate) geometry: &'a HardwareRasterGeometryStreamPlan,
    pub(crate) cuda_done_value: u64,
    pub(crate) frame: u32,
    pub(crate) grid: InstanceGrid,
    pub(crate) camera: CameraParams,
    pub(crate) shaders: &'a neo_lang::GraphicsShaders,
    pub(crate) use_depth: bool,
}

#[cfg(windows)]
impl WindowPresenter {
    pub(crate) fn new(
        window: &Window,
        kind: PresenterKind,
        upload_ring: usize,
        d3d_upload: D3dUploadMode,
        interop_device: Option<&NeoD3d12InteropDevice>,
    ) -> Result<Self> {
        let inner = match kind {
            PresenterKind::D3d12Interop => {
                let device = interop_device
                    .context("d3d12-interop presenter requires a Neo D3D12 interop device")?;
                PresenterImpl::D3d12Interop(D3d12InteropPresenter::new(window, device)?)
            }
            PresenterKind::D3d12 => PresenterImpl::D3d12(D3d12Presenter::new(window, upload_ring)?),
            PresenterKind::D3d11 => {
                PresenterImpl::D3d11(D3d11Presenter::new(window, upload_ring, d3d_upload)?)
            }
            PresenterKind::Gdi => PresenterImpl::Gdi(GdiPresenter::new(window)?),
        };
        Ok(Self { inner })
    }

    pub(crate) fn present(
        &mut self,
        size: PhysicalSize<u32>,
        bgra: &[u8],
    ) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(_) => {
                bail!("d3d12-interop presenter requires a shared Neo frame slot")
            }
            PresenterImpl::D3d12(presenter) => presenter.present(size, bgra),
            PresenterImpl::D3d11(presenter) => presenter.present(size, bgra),
            PresenterImpl::Gdi(presenter) => presenter.present(size, bgra),
        }
    }

    #[cfg(windows)]
    pub(crate) fn present_shared(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        slot: &mut neo_runtime::SharedFrameSlot,
        cuda_done_value: u64,
    ) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(presenter) => {
                presenter.present_shared(size, pitch_bytes, slot, cuda_done_value)
            }
            _ => bail!("shared frame presentation requires d3d12-interop presenter"),
        }
    }

    #[cfg(windows)]
    pub(crate) fn present_raster_indirect(
        &mut self,
        input: RasterPresentInput<'_>,
    ) -> Result<PresentTimings> {
        match &mut self.inner {
            PresenterImpl::D3d12Interop(presenter) => presenter.present_raster_indirect(input),
            _ => bail!("hardware raster presentation requires d3d12-interop presenter"),
        }
    }

    pub(crate) fn kind(&self) -> PresenterKind {
        match self.inner {
            PresenterImpl::D3d12Interop(_) => PresenterKind::D3d12Interop,
            PresenterImpl::D3d12(_) => PresenterKind::D3d12,
            PresenterImpl::D3d11(_) => PresenterKind::D3d11,
            PresenterImpl::Gdi(_) => PresenterKind::Gdi,
        }
    }
}

#[cfg(windows)]
struct GdiPresenter {
    hwnd: windows_sys::Win32::Foundation::HWND,
    hdc: windows_sys::Win32::Graphics::Gdi::HDC,
    width: u32,
    height: u32,
    bitmap_info: windows_sys::Win32::Graphics::Gdi::BITMAPINFO,
}

#[cfg(windows)]
impl GdiPresenter {
    fn new(window: &Window) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("no-interop presenter requires a Win32 window handle");
        };

        let hwnd = handle.hwnd.get() as windows_sys::Win32::Foundation::HWND;
        let hdc = unsafe { windows_sys::Win32::Graphics::Gdi::GetDC(hwnd) };
        if hdc.is_null() {
            bail!("GetDC failed for live window");
        }

        Ok(Self {
            hwnd,
            hdc,
            width: 0,
            height: 0,
            bitmap_info: bitmap_info(1, 1),
        })
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        if self.width != size.width || self.height != size.height {
            self.width = size.width;
            self.height = size.height;
            self.bitmap_info = bitmap_info(size.width, size.height);
        }

        unsafe {
            use windows_sys::Win32::Graphics::Gdi::{DIB_RGB_COLORS, SetDIBitsToDevice};

            let scanlines = SetDIBitsToDevice(
                self.hdc,
                0,
                0,
                size.width,
                size.height,
                0,
                0,
                0,
                size.height,
                bgra.as_ptr().cast(),
                &self.bitmap_info,
                DIB_RGB_COLORS,
            );
            if scanlines == 0 {
                bail!("SetDIBitsToDevice failed for live frame");
            }
        }
        let elapsed = start.elapsed();
        Ok(PresentTimings {
            swap_present: elapsed,
            total: elapsed,
            ..PresentTimings::default()
        })
    }
}

#[cfg(windows)]
impl Drop for GdiPresenter {
    fn drop(&mut self) {
        if !self.hdc.is_null() {
            unsafe {
                windows_sys::Win32::Graphics::Gdi::ReleaseDC(self.hwnd, self.hdc);
            }
        }
    }
}

#[cfg(windows)]
const D3D12_SWAPCHAIN_BUFFER_COUNT: usize = 3;

#[cfg(windows)]
pub(crate) const RASTER_ROOT_CONSTANT_DWORDS: u32 = 28;

#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HardwareRasterSubmitKind {
    DirectIndexedInstanced,
    IndirectIndexedInstanced,
}

#[cfg(windows)]
pub(crate) fn hardware_raster_submit_kind(
    material: &MaterialKernelAbi,
) -> HardwareRasterSubmitKind {
    if material.requires_compute_culling() {
        HardwareRasterSubmitKind::IndirectIndexedInstanced
    } else {
        HardwareRasterSubmitKind::DirectIndexedInstanced
    }
}

#[cfg(windows)]
pub(crate) fn raster_root_constants(
    grid: InstanceGrid,
    size: PhysicalSize<u32>,
    camera: CameraParams,
    geometry: &HardwareRasterGeometryStreamPlan,
    frame: u32,
) -> [u32; RASTER_ROOT_CONSTANT_DWORDS as usize] {
    [
        grid.x,
        grid.y,
        grid.z,
        frame,
        size.width,
        size.height,
        geometry.vertex_stride,
        geometry.color_offset,
        camera.origin[0].to_bits(),
        camera.origin[1].to_bits(),
        camera.origin[2].to_bits(),
        camera.origin[3].to_bits(),
        camera.right[0].to_bits(),
        camera.right[1].to_bits(),
        camera.right[2].to_bits(),
        camera.right[3].to_bits(),
        camera.up[0].to_bits(),
        camera.up[1].to_bits(),
        camera.up[2].to_bits(),
        camera.up[3].to_bits(),
        camera.forward[0].to_bits(),
        camera.forward[1].to_bits(),
        camera.forward[2].to_bits(),
        camera.forward[3].to_bits(),
        camera.view[0].to_bits(),
        camera.view[1].to_bits(),
        camera.view[2].to_bits(),
        camera.view[3].to_bits(),
    ]
}

#[cfg(windows)]
pub(crate) const D3D12_RASTER_VS_TARGET: &[u8; 7] = b"vs_5_1\0";
#[cfg(windows)]
pub(crate) const D3D12_RASTER_PS_TARGET: &[u8; 7] = b"ps_5_1\0";

#[cfg(windows)]
struct D3d12RasterState {
    root_signature: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pipeline: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    command_signature: windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature,
    geometry_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    index_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    index_view: windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_VIEW,
    shader_hash: u64,
}

#[cfg(windows)]
impl D3d12RasterState {
    fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        shaders: &neo_lang::GraphicsShaders,
        material: &MaterialKernelAbi,
        geometry: &HardwareRasterGeometryStreamPlan,
        use_depth: bool,
        shader_hash: u64,
    ) -> Result<Self> {
        use windows::{
            Win32::Graphics::{
                Direct3D::ID3DBlob,
                Direct3D12::{
                    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_COLOR_WRITE_ENABLE_ALL,
                    D3D12_COMMAND_SIGNATURE_DESC, D3D12_COMPARISON_FUNC_LESS,
                    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    D3D12_CULL_MODE_NONE, D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, D3D12_DEPTH_STENCIL_DESC,
                    D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_FILL_MODE_SOLID,
                    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_HEAP_FLAG_NONE,
                    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_UPLOAD, D3D12_INDEX_BUFFER_VIEW,
                    D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
                    D3D12_INPUT_LAYOUT_DESC, D3D12_LOGIC_OP_NOOP, D3D12_MEMORY_POOL_UNKNOWN,
                    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC,
                    D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOURCE_DESC,
                    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR,
                    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
                    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
                    D3D12_ROOT_SIGNATURE_DESC,
                    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                    D3D12_SHADER_BYTECODE, D3D12_SO_DECLARATION_ENTRY, D3D12_STATIC_SAMPLER_DESC,
                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12SerializeRootSignature,
                },
                Dxgi::Common::{
                    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16_UINT,
                    DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
                },
            },
            core::PCSTR,
        };

        validate_contiguous_material_root_bindings(material)?;

        let mut root_params = Vec::with_capacity(material.bindings.len());
        for binding in &material.bindings {
            let root_param = match binding.kind {
                MaterialBindingKind::DrawParams | MaterialBindingKind::RasterParams => {
                    D3D12_ROOT_PARAMETER {
                        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                        Anonymous: D3D12_ROOT_PARAMETER_0 {
                            Constants: D3D12_ROOT_CONSTANTS {
                                ShaderRegister: binding.shader_register,
                                RegisterSpace: binding.register_space,
                                Num32BitValues: RASTER_ROOT_CONSTANT_DWORDS,
                            },
                        },
                        ShaderVisibility:
                            windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
                    }
                }
                MaterialBindingKind::VisibleInstanceStream
                | MaterialBindingKind::InstanceStream
                | MaterialBindingKind::GeometryStream => D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: binding.shader_register,
                            RegisterSpace: binding.register_space,
                        },
                    },
                    ShaderVisibility:
                        windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
                },
            };
            root_params.push(root_param);
        }
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: root_params.len() as u32,
            pParameters: root_params.as_mut_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null::<D3D12_STATIC_SAMPLER_DESC>(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        let mut signature_blob: Option<ID3DBlob> = None;
        let mut signature_error: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut signature_blob,
                Some(&mut signature_error),
            )
            .map_err(|err| anyhow!("failed to serialize D3D12 root signature: {err:?}"))?;
        }
        let signature_blob =
            signature_blob.context("D3D12 root signature serialization returned no blob")?;
        let root_signature: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    signature_blob.GetBufferPointer().cast::<u8>(),
                    signature_blob.GetBufferSize(),
                ),
            )?
        };

        let vertex_entrypoint = std::ffi::CString::new(material.vertex_entrypoint.as_str())
            .context("hardware raster vertex entrypoint contains an interior NUL byte")?;
        let fragment_entrypoint = std::ffi::CString::new(material.fragment_entrypoint.as_str())
            .context("hardware raster fragment entrypoint contains an interior NUL byte")?;
        let vs = compile_hlsl(
            &shaders.vertex_source,
            PCSTR(vertex_entrypoint.as_ptr().cast()),
            PCSTR(D3D12_RASTER_VS_TARGET.as_ptr()),
        )?;
        let ps = compile_hlsl(
            &shaders.fragment_source,
            PCSTR(fragment_entrypoint.as_ptr().cast()),
            PCSTR(D3D12_RASTER_PS_TARGET.as_ptr()),
        )?;
        let blend_desc = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                LogicOpEnable: false.into(),
                SrcBlend: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ONE,
                DestBlend: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ZERO,
                BlendOp: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ONE,
                DestBlendAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_ZERO,
                BlendOpAlpha: windows::Win32::Graphics::Direct3D12::D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            }; 8],
        };
        let rasterizer_desc = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: false.into(),
            DepthBias: D3D12_DEFAULT_DEPTH_BIAS,
            DepthBiasClamp: D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
            SlopeScaledDepthBias: D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
            DepthClipEnable: true.into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        };
        let depth_desc = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: use_depth.into(),
            DepthWriteMask: if use_depth {
                D3D12_DEPTH_WRITE_MASK_ALL
            } else {
                D3D12_DEPTH_WRITE_MASK_ZERO
            },
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: Default::default(),
            BackFace: Default::default(),
        };
        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rtv_formats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        let (vs_ptr, vs_len, ps_ptr, ps_len) = unsafe {
            (
                vs.GetBufferPointer(),
                vs.GetBufferSize(),
                ps.GetBufferPointer(),
                ps.GetBufferSize(),
            )
        };
        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_ptr,
                BytecodeLength: vs_len,
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps_ptr,
                BytecodeLength: ps_len,
            },
            DS: D3D12_SHADER_BYTECODE::default(),
            HS: D3D12_SHADER_BYTECODE::default(),
            GS: D3D12_SHADER_BYTECODE::default(),
            StreamOutput: windows::Win32::Graphics::Direct3D12::D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: std::ptr::null::<D3D12_SO_DECLARATION_ENTRY>(),
                NumEntries: 0,
                pBufferStrides: std::ptr::null(),
                NumStrides: 0,
                RasterizedStream: 0,
            },
            BlendState: blend_desc,
            SampleMask: u32::MAX,
            RasterizerState: rasterizer_desc,
            DepthStencilState: depth_desc,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: std::ptr::null(),
                NumElements: 0,
            },
            IBStripCutValue:
                windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: rtv_formats,
            DSVFormat: if use_depth {
                DXGI_FORMAT_D32_FLOAT
            } else {
                DXGI_FORMAT_UNKNOWN
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: Default::default(),
            Flags: windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pipeline = unsafe { device.CreateGraphicsPipelineState(&pso_desc)? };

        let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            Anonymous: Default::default(),
        };
        let command_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: DrawIndexedIndirectCommand::BYTE_LEN as u32,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg_desc,
            NodeMask: 0,
        };
        let mut command_signature = None;
        unsafe {
            device.CreateCommandSignature(
                &command_desc,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12RootSignature>,
                &mut command_signature,
            )?;
        }
        let command_signature =
            command_signature.context("D3D12 returned no indirect command signature")?;

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let geometry_bytes = geometry.vertex_bytes.len() as u64;
        let mut geometry_buffer = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Alignment: 0,
                    Width: geometry_bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut geometry_buffer,
            )?;
        }
        let geometry_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource =
            geometry_buffer.context("D3D12 returned no geometry buffer")?;
        unsafe {
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            geometry_buffer.Map(0, Some(&read_range), Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                geometry.vertex_bytes.as_ptr(),
                mapped.cast::<u8>(),
                geometry.vertex_bytes.len(),
            );
            geometry_buffer.Unmap(0, None);
        }

        let indices = &geometry.indices_u16;
        let index_bytes = std::mem::size_of_val(indices.as_slice()) as u64;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: index_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut index_buffer = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut index_buffer,
            )?;
        }
        let index_buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource =
            index_buffer.context("D3D12 returned no index buffer")?;
        unsafe {
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            index_buffer.Map(0, Some(&read_range), Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                indices.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                index_bytes as usize,
            );
            index_buffer.Unmap(0, None);
        }
        let index_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { index_buffer.GetGPUVirtualAddress() },
            SizeInBytes: index_bytes as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };
        Ok(Self {
            root_signature,
            pipeline,
            command_signature,
            geometry_buffer,
            index_buffer,
            index_view,
            shader_hash,
        })
    }
}

#[cfg(windows)]
pub(crate) fn compile_hlsl(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    use windows::Win32::Graphics::{Direct3D::Fxc::D3DCompile, Direct3D::ID3DBlob};
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::PCSTR::null(),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
        .map_err(|err| {
            let message = errors
                .as_ref()
                .map(|blob| {
                    let bytes = std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    );
                    String::from_utf8_lossy(bytes).to_string()
                })
                .unwrap_or_default();
            anyhow!("failed to compile D3D12 raster HLSL: {err:?}\n{message}")
        })?;
    }
    code.context("D3DCompile returned no shader bytecode")
}

#[cfg(windows)]
fn raster_shader_hash(
    shaders: &neo_lang::GraphicsShaders,
    material: &MaterialKernelAbi,
    geometry: &HardwareRasterGeometryStreamPlan,
    use_depth: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    shaders.vertex_source.hash(&mut hasher);
    shaders.fragment_source.hash(&mut hasher);
    material.hash(&mut hasher);
    use_depth.hash(&mut hasher);
    geometry.vertex_bytes.hash(&mut hasher);
    geometry.vertex_stride.hash(&mut hasher);
    geometry.color_offset.hash(&mut hasher);
    geometry.indices_u16.hash(&mut hasher);
    hasher.finish()
}

#[cfg(windows)]
pub(crate) fn material_binding<'a>(
    material: &'a MaterialKernelAbi,
    kind: MaterialBindingKind,
    label: &str,
) -> Result<&'a neo_runtime::MaterialBinding> {
    material.binding(kind).with_context(|| {
        format!(
            "hardware raster MaterialKernel `{}`/`{}` is missing its {label} binding",
            material.vertex_entrypoint, material.fragment_entrypoint
        )
    })
}

#[cfg(windows)]
pub(crate) fn validate_contiguous_material_root_bindings(
    material: &MaterialKernelAbi,
) -> Result<()> {
    let mut expected = vec![(MaterialBindingKind::DrawParams, "draw params")];
    if material.requires_compute_culling() {
        expected.push((
            MaterialBindingKind::VisibleInstanceStream,
            "visible InstanceStream",
        ));
    }
    if material.requires_instance_stream() {
        expected.push((MaterialBindingKind::InstanceStream, "InstanceStream"));
    }
    if material
        .vertex_requirements
        .contains(&MaterialVertexRequirement::GeometryPosition)
    {
        expected.push((MaterialBindingKind::GeometryStream, "GeometryStream"));
    }

    for (expected_root, (kind, label)) in expected.into_iter().enumerate() {
        let binding = material_binding(material, kind, label)?;
        if binding.root_parameter_index != expected_root as u32 {
            bail!(
                "hardware raster MaterialKernel `{}`/`{}` binding `{label}` must use root parameter {expected_root}, got {}",
                material.vertex_entrypoint,
                material.fragment_entrypoint,
                binding.root_parameter_index
            );
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn graphics_bindings_for_material(
    material: &MaterialKernelAbi,
) -> Result<neo_lang::GraphicsBindings> {
    let draw_params = material_binding(material, MaterialBindingKind::DrawParams, "draw params")?;
    let visible_instances = material
        .binding(MaterialBindingKind::VisibleInstanceStream)
        .map(|binding| neo_lang::HlslRegister::new(binding.shader_register, binding.register_space))
        .unwrap_or_else(|| neo_lang::HlslRegister::new(0, 0));
    let instances = material_binding(
        material,
        MaterialBindingKind::InstanceStream,
        "InstanceStream",
    )?;
    let geometry = material_binding(
        material,
        MaterialBindingKind::GeometryStream,
        "GeometryStream",
    )?;
    Ok(neo_lang::GraphicsBindings {
        raster_params: neo_lang::HlslRegister::new(
            draw_params.shader_register,
            draw_params.register_space,
        ),
        visible_instances,
        instances: neo_lang::HlslRegister::new(instances.shader_register, instances.register_space),
        geometry: neo_lang::HlslRegister::new(geometry.shader_register, geometry.register_space),
    })
}

#[cfg(windows)]
struct D3d12InteropPresenter {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    command_frames: Vec<D3d12InteropCommandFrame>,
    command_frame_index: usize,
    command_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    back_buffers: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    rtv_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    dsv_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    depth_buffer: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    rtv_descriptor_size: u32,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    fence_value: u64,
    fence_event: windows::Win32::Foundation::HANDLE,
    width: u32,
    height: u32,
    tearing_supported: bool,
    raster_state: Option<D3d12RasterState>,
}

#[cfg(windows)]
struct D3d12InteropCommandFrame {
    command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    fence_value: u64,
}

#[cfg(windows)]
impl D3d12InteropPresenter {
    fn new(window: &Window, interop: &NeoD3d12InteropDevice) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::HWND,
                Graphics::{
                    Direct3D12::{
                        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_DESCRIPTOR_HEAP_DESC,
                        D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_FENCE_FLAG_NONE,
                        ID3D12CommandAllocator, ID3D12DescriptorHeap, ID3D12Fence,
                        ID3D12GraphicsCommandList,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_NONE,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2, IDXGISwapChain3,
                    },
                },
                System::Threading::CreateEventW,
            },
            core::{BOOL, Interface as _, PCWSTR},
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D12 interop presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let device = interop.device().clone();
        let queue = interop.queue().clone();
        let mut command_frames = Vec::with_capacity(D3D12_SWAPCHAIN_BUFFER_COUNT);
        for _ in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let command_allocator: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)? };
            command_frames.push(D3d12InteropCommandFrame {
                command_allocator,
                fence_value: 0,
            });
        }
        let command_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &command_frames[0].command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?
        };
        unsafe {
            command_list.Close()?;
        }

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D12 interop presenter tearing support: {tearing_supported}");
        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Scaling: DXGI_SCALING_NONE,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(&queue, HWND(handle.hwnd.get() as _), &desc, None, None)
        }?
        .cast::<IDXGISwapChain3>()?;
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let fence_event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }?;
        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&rtv_heap_desc)? };
        let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let dsv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&dsv_heap_desc)? };
        let rtv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
        let mut presenter = Self {
            device,
            queue,
            command_frames,
            command_frame_index: 0,
            command_list,
            swap_chain,
            back_buffers: Vec::new(),
            rtv_heap,
            dsv_heap,
            depth_buffer: None,
            rtv_descriptor_size,
            fence,
            fence_value: 0,
            fence_event,
            width,
            height,
            tearing_supported,
            raster_state: None,
        };
        presenter.recreate_backbuffers()?;
        Ok(presenter)
    }

    fn present_shared(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        slot: &mut neo_runtime::SharedFrameSlot,
        cuda_done_value: u64,
    ) -> Result<PresentTimings> {
        let total_start = Instant::now();
        self.ensure_size(size)?;
        let copy_start = Instant::now();
        slot.wait_d3d_for_value(&self.queue, cuda_done_value)?;
        self.copy_shared_to_backbuffer(size, pitch_bytes, slot.resource())?;
        let draw = copy_start.elapsed();
        let _ = slot.signal_available_on_d3d(&self.queue)?;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            draw,
            swap_present,
            total: total_start.elapsed(),
            ..PresentTimings::default()
        })
    }

    fn present_raster_indirect(&mut self, input: RasterPresentInput<'_>) -> Result<PresentTimings> {
        let RasterPresentInput {
            size,
            args,
            visible_ids,
            raster_instances,
            material,
            geometry,
            cuda_done_value,
            frame,
            grid,
            camera,
            shaders,
            use_depth,
        } = input;
        use windows::{
            Win32::Foundation::RECT,
            Win32::Graphics::{
                Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                Direct3D12::{
                    D3D12_CLEAR_FLAG_DEPTH, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_VIEWPORT, ID3D12CommandList,
                },
                Dxgi::DXGI_PRESENT,
            },
            core::Interface as _,
        };

        let total_start = Instant::now();
        self.ensure_size(size)?;
        let shader_hash = raster_shader_hash(shaders, material, geometry, use_depth);
        if self
            .raster_state
            .as_ref()
            .is_none_or(|state| state.shader_hash != shader_hash)
        {
            self.raster_state = Some(D3d12RasterState::new(
                &self.device,
                shaders,
                material,
                geometry,
                use_depth,
                shader_hash,
            )?);
        }
        let (command_frame, command_allocator) = self.acquire_command_frame()?;
        let raster = self
            .raster_state
            .as_ref()
            .context("missing D3D12 raster state")?;
        let _keep_index_buffer_alive = &raster.index_buffer;
        let draw_start = Instant::now();
        let submit_kind = hardware_raster_submit_kind(material);
        let uses_indirect_args = submit_kind == HardwareRasterSubmitKind::IndirectIndexedInstanced;
        if cuda_done_value != 0 {
            args.wait_d3d_for_value(&self.queue, cuda_done_value)?;
        }
        let direct_instance_count = grid
            .count()
            .context("hardware raster direct draw instance count overflow")?;
        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 interop backbuffer is not available")?;
        let rtv_handle = windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
                + back_index * self.rtv_descriptor_size as usize,
        };
        let _keep_depth_buffer_alive = if use_depth {
            Some(
                self.depth_buffer
                    .as_ref()
                    .context("D3D12 raster depth buffer is not available")?,
            )
        } else {
            None
        };
        let dsv_handle = unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe {
            command_allocator.Reset()?;
            self.command_list
                .Reset(&command_allocator, Some(&raster.pipeline))?;

            let mut back_to_rt = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_rt));
            drop_d3d12_transition_barrier(&mut back_to_rt);
            let mut args_to_indirect = if uses_indirect_args {
                Some(d3d12_transition(
                    args.resource(),
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ))
            } else {
                None
            };
            if let Some(barrier) = args_to_indirect.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = args_to_indirect.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut visible_to_srv = if material.requires_compute_culling() {
                Some(d3d12_transition(
                    visible_ids.resource(),
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ))
            } else {
                None
            };
            if let Some(barrier) = visible_to_srv.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = visible_to_srv.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut instances_to_srv = d3d12_transition(
                raster_instances.resource(),
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&instances_to_srv));
            drop_d3d12_transition_barrier(&mut instances_to_srv);

            self.command_list
                .SetGraphicsRootSignature(&raster.root_signature);
            self.command_list.SetPipelineState(&raster.pipeline);
            self.command_list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.command_list.IASetIndexBuffer(Some(&raster.index_view));
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size.width as f32,
                Height: size.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            self.command_list.RSSetViewports(&[viewport]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: size.width as i32,
                bottom: size.height as i32,
            };
            self.command_list.RSSetScissorRects(&[scissor]);
            self.command_list.OMSetRenderTargets(
                1,
                Some(&rtv_handle),
                false,
                use_depth.then_some(&dsv_handle),
            );
            let clear = [0.005, 0.006, 0.009, 1.0];
            self.command_list
                .ClearRenderTargetView(rtv_handle, &clear, None);
            if use_depth {
                self.command_list.ClearDepthStencilView(
                    dsv_handle,
                    D3D12_CLEAR_FLAG_DEPTH,
                    1.0,
                    0,
                    None,
                );
            }
            let constants = raster_root_constants(grid, size, camera, geometry, frame);
            for binding in &material.bindings {
                match binding.kind {
                    MaterialBindingKind::DrawParams | MaterialBindingKind::RasterParams => {
                        self.command_list.SetGraphicsRoot32BitConstants(
                            binding.root_parameter_index,
                            constants.len() as u32,
                            constants.as_ptr().cast(),
                            0,
                        );
                    }
                    MaterialBindingKind::VisibleInstanceStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            visible_ids.resource().GetGPUVirtualAddress(),
                        );
                    }
                    MaterialBindingKind::InstanceStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            raster_instances.resource().GetGPUVirtualAddress(),
                        );
                    }
                    MaterialBindingKind::GeometryStream => {
                        self.command_list.SetGraphicsRootShaderResourceView(
                            binding.root_parameter_index,
                            raster.geometry_buffer.GetGPUVirtualAddress(),
                        );
                    }
                }
            }
            if uses_indirect_args {
                self.command_list.ExecuteIndirect(
                    &raster.command_signature,
                    1,
                    args.resource(),
                    0,
                    None,
                    0,
                );
            } else {
                self.command_list.DrawIndexedInstanced(
                    geometry.index_count(),
                    direct_instance_count,
                    0,
                    0,
                    0,
                );
            }

            let mut args_to_common = if uses_indirect_args {
                Some(d3d12_transition(
                    args.resource(),
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                ))
            } else {
                None
            };
            if let Some(barrier) = args_to_common.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = args_to_common.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut visible_to_common = if material.requires_compute_culling() {
                Some(d3d12_transition(
                    visible_ids.resource(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                ))
            } else {
                None
            };
            if let Some(barrier) = visible_to_common.as_ref() {
                self.command_list
                    .ResourceBarrier(std::slice::from_ref(barrier));
            }
            if let Some(barrier) = visible_to_common.as_mut() {
                drop_d3d12_transition_barrier(barrier);
            }
            let mut instances_to_common = d3d12_transition(
                raster_instances.resource(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COMMON,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&instances_to_common));
            drop_d3d12_transition_barrier(&mut instances_to_common);
            let mut back_to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_present));
            drop_d3d12_transition_barrier(&mut back_to_present);
            self.command_list.Close()?;
            let list: ID3D12CommandList = self.command_list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
        }
        self.signal_command_frame(command_frame)?;
        let draw = draw_start.elapsed();
        if uses_indirect_args {
            let _ = args.signal_available_on_d3d(&self.queue)?;
        }
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            draw,
            swap_present,
            total: total_start.elapsed(),
            ..PresentTimings::default()
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.wait_for_gpu()?;
        self.back_buffers.clear();
        self.depth_buffer = None;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain.ResizeBuffers(
                D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
                width,
                height,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                flags,
            )?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffers()
    }

    fn recreate_backbuffers(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

        self.back_buffers.clear();
        let base = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for index in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let back_buffer: ID3D12Resource = unsafe { self.swap_chain.GetBuffer(index as u32)? };
            let handle = windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + index * self.rtv_descriptor_size as usize,
            };
            unsafe {
                self.device
                    .CreateRenderTargetView(&back_buffer, None, handle);
            }
            self.back_buffers.push(back_buffer);
        }
        self.recreate_depth_target()?;
        Ok(())
    }

    fn recreate_depth_target(&mut self) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                D3D12_DEPTH_STENCIL_VALUE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_SAMPLE_DESC},
        };

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: self.width as u64,
            Height: self.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };
        let mut depth_buffer: Option<ID3D12Resource> = None;
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&clear),
                &mut depth_buffer,
            )?;
        }
        let depth_buffer = depth_buffer.context("D3D12 returned no raster depth buffer")?;
        let dsv_handle = unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe {
            self.device
                .CreateDepthStencilView(&depth_buffer, None, dsv_handle);
        }
        self.depth_buffer = Some(depth_buffer);
        Ok(())
    }

    fn copy_shared_to_backbuffer(
        &mut self,
        size: PhysicalSize<u32>,
        pitch_bytes: u32,
        shared: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) -> Result<()> {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PRESENT, D3D12_SUBRESOURCE_FOOTPRINT,
                D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ID3D12CommandList,
            },
            core::Interface as _,
        };

        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let (command_frame, command_allocator) = self.acquire_command_frame()?;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 interop backbuffer is not available")?;
        unsafe {
            command_allocator.Reset()?;
            self.command_list.Reset(
                &command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            let mut shared_to_copy = d3d12_transition(
                shared,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&shared_to_copy));
            drop_d3d12_transition_barrier(&mut shared_to_copy);
            let mut back_to_copy = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_copy));
            drop_d3d12_transition_barrier(&mut back_to_copy);

            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                    Width: size.width,
                    Height: size.height,
                    Depth: 1,
                    RowPitch: pitch_bytes,
                },
            };
            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(shared.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(back_buffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            self.command_list
                .CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop_d3d12_texture_copy_location(&mut src);
            drop_d3d12_texture_copy_location(&mut dst);

            let mut shared_to_common = d3d12_transition(
                shared,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&shared_to_common));
            drop_d3d12_transition_barrier(&mut shared_to_common);
            let mut back_to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&back_to_present));
            drop_d3d12_transition_barrier(&mut back_to_present);
            self.command_list.Close()?;
            let list: ID3D12CommandList = self.command_list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
        }
        self.signal_command_frame(command_frame)?;
        Ok(())
    }

    fn acquire_command_frame(
        &mut self,
    ) -> Result<(
        usize,
        windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    )> {
        let index = self.command_frame_index;
        self.command_frame_index = (self.command_frame_index + 1) % self.command_frames.len();
        self.wait_for_fence(self.command_frames[index].fence_value)?;
        Ok((index, self.command_frames[index].command_allocator.clone()))
    }

    fn signal_command_frame(&mut self, index: usize) -> Result<u64> {
        let fence_value = self.signal_queue()?;
        self.command_frames[index].fence_value = fence_value;
        Ok(fence_value)
    }

    fn signal_queue(&mut self) -> Result<u64> {
        self.fence_value += 1;
        unsafe {
            self.queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    fn wait_for_fence(&self, fence_value: u64) -> Result<()> {
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        if fence_value == 0 {
            return Ok(());
        }
        unsafe {
            if self.fence.GetCompletedValue() < fence_value {
                self.fence
                    .SetEventOnCompletion(fence_value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<()> {
        let fence_value = self.signal_queue()?;
        self.wait_for_fence(fence_value)
    }
}

#[cfg(windows)]
impl Drop for D3d12InteropPresenter {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.fence_event) };
        }
    }
}

#[cfg(windows)]
struct D3d12Presenter {
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    command_queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    command_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    back_buffers: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    upload_slots: Vec<D3d12UploadSlot>,
    upload_index: usize,
    upload_ring: usize,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    fence_value: u64,
    fence_event: windows::Win32::Foundation::HANDLE,
    width: u32,
    height: u32,
    tearing_supported: bool,
}

#[cfg(windows)]
struct D3d12UploadSlot {
    command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    mapped: *mut u8,
    layout: windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    fence_value: u64,
}

#[cfg(windows)]
impl Drop for D3d12UploadSlot {
    fn drop(&mut self) {
        unsafe {
            self.resource.Unmap(0, None);
        }
    }
}

#[cfg(windows)]
impl D3d12Presenter {
    fn new(window: &Window, upload_ring: usize) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::HWND,
                Graphics::{
                    Direct3D::D3D_FEATURE_LEVEL_11_0,
                    Direct3D12::{
                        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                        D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                        D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandQueue,
                        ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_NONE,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2, IDXGISwapChain3,
                    },
                },
                System::Threading::CreateEventW,
            },
            core::{BOOL, Interface as _, PCWSTR},
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D12 presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut device: Option<ID3D12Device> = None;
        unsafe {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device)
                .context("failed to create D3D12 device")?;
        }
        let device = device.context("D3D12 did not return a device")?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let command_queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&queue_desc)
                .context("failed to create D3D12 command queue")?
        };
        let bootstrap_allocator: ID3D12CommandAllocator = unsafe {
            device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .context("failed to create D3D12 command allocator")?
        };
        let command_list: ID3D12GraphicsCommandList = unsafe {
            device
                .CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &bootstrap_allocator,
                    None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                )
                .context("failed to create D3D12 command list")?
        };
        unsafe {
            command_list.Close()?;
        }

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("failed to create DXGI factory")?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D12 flip presenter tearing support: {tearing_supported}");
        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
            Scaling: DXGI_SCALING_NONE,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(
                &command_queue,
                HWND(handle.hwnd.get() as _),
                &desc,
                None,
                None,
            )
        }
        .context("failed to create D3D12 flip-model swapchain")?
        .cast::<IDXGISwapChain3>()
        .context("failed to cast D3D12 swapchain to IDXGISwapChain3")?;

        let fence: ID3D12Fence = unsafe {
            device.CreateFence(
                0,
                windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE,
            )
        }
        .context("failed to create D3D12 fence")?;
        let fence_event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
            .context("failed to create D3D12 fence event")?;

        let mut presenter = Self {
            device,
            command_queue,
            command_list,
            swap_chain,
            back_buffers: Vec::new(),
            upload_slots: Vec::new(),
            upload_index: 0,
            upload_ring,
            fence,
            fence_value: 0,
            fence_event,
            width,
            height,
            tearing_supported,
        };
        presenter.recreate_backbuffers()?;
        presenter.recreate_upload_buffers(width, height, upload_ring)?;
        Ok(presenter)
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let total_start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        self.ensure_size(size)?;
        let upload_index = self.upload_index;
        self.upload_index = (self.upload_index + 1) % self.upload_slots.len();
        self.wait_for_upload_slot(upload_index)?;

        let map_copy_start = Instant::now();
        self.upload_bgra(upload_index, size.width, size.height, bgra)?;
        let map_copy = map_copy_start.elapsed();

        let copy_start = Instant::now();
        let fence_value = self.copy_upload_to_backbuffer(upload_index)?;
        self.upload_slots[upload_index].fence_value = fence_value;
        let draw = copy_start.elapsed();

        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            map_copy,
            draw,
            swap_present,
            total: total_start.elapsed(),
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.release_resize_references()?;
        self.back_buffers.clear();
        self.upload_slots.clear();
        self.upload_index = 0;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain
                .ResizeBuffers(
                    D3D12_SWAPCHAIN_BUFFER_COUNT as u32,
                    width,
                    height,
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                    flags,
                )
                .context("failed to resize D3D12 swapchain buffers")?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffers()?;
        self.recreate_upload_buffers(width, height, self.upload_ring)?;
        Ok(())
    }

    fn release_resize_references(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT;

        self.wait_for_gpu()?;
        for slot in &self.upload_slots {
            self.wait_for_fence(slot.fence_value)?;
        }

        if let Some(slot) = self.upload_slots.first() {
            unsafe {
                slot.command_allocator
                    .Reset()
                    .context("failed to reset D3D12 command allocator before resize")?;
                self.command_list
                    .Reset(
                        &slot.command_allocator,
                        None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                    )
                    .context("failed to reset D3D12 command list before resize")?;
                self.command_list
                    .Close()
                    .context("failed to close D3D12 command list before resize")?;
            }
        } else {
            let command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .context("failed to create D3D12 command allocator before resize")?
            };
            unsafe {
                self.command_list
                    .Reset(
                        &command_allocator,
                        None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
                    )
                    .context("failed to reset D3D12 command list before resize")?;
                self.command_list
                    .Close()
                    .context("failed to close D3D12 command list before resize")?;
            }
        }

        Ok(())
    }

    fn recreate_backbuffers(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

        self.back_buffers.clear();
        self.back_buffers.reserve(D3D12_SWAPCHAIN_BUFFER_COUNT);
        for index in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let back_buffer: ID3D12Resource = unsafe { self.swap_chain.GetBuffer(index as u32) }
                .with_context(|| format!("failed to get D3D12 swapchain backbuffer {index}"))?;
            self.back_buffers.push(back_buffer);
        }
        Ok(())
    }

    fn recreate_upload_buffers(
        &mut self,
        width: u32,
        height: u32,
        upload_ring: usize,
    ) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_UPLOAD,
                D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator, ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        };

        let texture_desc = d3d12_texture_desc(width, height);
        let mut layout =
            windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut row_count = 0;
        let mut _row_size_bytes = 0;
        let mut total_bytes = 0;
        unsafe {
            self.device.GetCopyableFootprints(
                &texture_desc,
                0,
                1,
                0,
                Some(&mut layout),
                Some(&mut row_count),
                Some(&mut _row_size_bytes),
                Some(&mut total_bytes),
            );
        }

        let upload_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: total_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };

        self.upload_slots.clear();
        self.upload_slots.reserve(upload_ring);
        for _ in 0..upload_ring {
            let command_allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .context("failed to create D3D12 upload command allocator")?
            };
            let mut resource: Option<ID3D12Resource> = None;
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &upload_desc,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                        None,
                        &mut resource,
                    )
                    .context("failed to create D3D12 upload resource")?;
            }
            let resource = resource.context("D3D12 did not return an upload resource")?;
            let mut mapped = std::ptr::null_mut();
            let read_range = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
            unsafe {
                resource
                    .Map(0, Some(&read_range), Some(&mut mapped))
                    .context("failed to persistently map D3D12 upload resource")?;
            }
            self.upload_slots.push(D3d12UploadSlot {
                command_allocator,
                resource,
                mapped: mapped.cast(),
                layout,
                fence_value: 0,
            });
        }
        Ok(())
    }

    fn upload_bgra(&self, slot_index: usize, width: u32, height: u32, bgra: &[u8]) -> Result<()> {
        let slot = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?;
        if slot.mapped.is_null() {
            bail!("D3D12 upload resource is not mapped");
        }
        unsafe {
            let dst_pitch = slot.layout.Footprint.RowPitch as usize;
            let _used_fast_path = copy_bgra_to_mapped(bgra, slot.mapped, width, height, dst_pitch);
        }
        Ok(())
    }

    fn copy_upload_to_backbuffer(&mut self, slot_index: usize) -> Result<u64> {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT,
                D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ID3D12CommandList,
            },
            core::Interface as _,
        };

        let slot = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?;
        let command_allocator = slot.command_allocator.clone();
        let upload_resource = slot.resource.clone();
        let upload_layout = slot.layout;
        let back_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 backbuffer is not available")?;

        unsafe {
            command_allocator.Reset()?;
            self.command_list.Reset(
                &command_allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            let mut to_copy = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&to_copy));
            drop_d3d12_transition_barrier(&mut to_copy);

            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(upload_resource)),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: upload_layout,
                },
            };
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::ManuallyDrop::new(Some(back_buffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            self.command_list
                .CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop_d3d12_texture_copy_location(&mut src);
            drop_d3d12_texture_copy_location(&mut dst);

            let mut to_present = d3d12_transition(
                back_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&to_present));
            drop_d3d12_transition_barrier(&mut to_present);
            self.command_list.Close()?;
            let command_list: ID3D12CommandList = self.command_list.cast()?;
            self.command_queue
                .ExecuteCommandLists(&[Some(command_list)]);
        }
        self.signal_queue()
    }

    fn signal_queue(&mut self) -> Result<u64> {
        self.fence_value += 1;
        unsafe {
            self.command_queue.Signal(&self.fence, self.fence_value)?;
        }
        Ok(self.fence_value)
    }

    fn wait_for_upload_slot(&self, slot_index: usize) -> Result<()> {
        let fence_value = self
            .upload_slots
            .get(slot_index)
            .context("D3D12 upload resource is not available")?
            .fence_value;
        self.wait_for_fence(fence_value)
    }

    fn wait_for_fence(&self, fence_value: u64) -> Result<()> {
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        if fence_value == 0 {
            return Ok(());
        }
        unsafe {
            if self.fence.GetCompletedValue() < fence_value {
                self.fence
                    .SetEventOnCompletion(fence_value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<()> {
        let fence_value = self.signal_queue()?;
        self.wait_for_fence(fence_value)
    }
}

#[cfg(windows)]
impl Drop for D3d12Presenter {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.fence_event) };
        }
    }
}

#[cfg(windows)]
fn d3d12_texture_desc(
    width: u32,
    height: u32,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC {
    use windows::Win32::Graphics::{
        Direct3D12::{
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
        },
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    };

    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

#[cfg(windows)]
fn d3d12_transition(
    resource: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_TRANSITION_BARRIER,
    };

    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[cfg(windows)]
fn drop_d3d12_texture_copy_location(
    location: &mut windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_COPY_LOCATION,
) {
    unsafe {
        std::mem::ManuallyDrop::drop(&mut location.pResource);
    }
}

#[cfg(windows)]
fn drop_d3d12_transition_barrier(
    barrier: &mut windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER,
) {
    unsafe {
        let transition = &mut *barrier.Anonymous.Transition;
        std::mem::ManuallyDrop::drop(&mut transition.pResource);
    }
}

#[cfg(windows)]
struct D3d11Presenter {
    device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain1,
    back_buffer: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    upload_slots: Vec<D3d11UploadSlot>,
    upload_index: usize,
    upload_mode: D3dUploadMode,
    width: u32,
    height: u32,
    tearing_supported: bool,
}

#[cfg(windows)]
struct D3d11UploadSlot {
    texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
}

#[cfg(windows)]
impl D3d11Presenter {
    fn new(window: &Window, upload_ring: usize, upload_mode: D3dUploadMode) -> Result<Self> {
        use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
        use windows::{
            Win32::{
                Foundation::{HMODULE, HWND},
                Graphics::{
                    Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0},
                    Direct3D11::{
                        D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice,
                        ID3D11Device, ID3D11DeviceContext,
                    },
                    Dxgi::{
                        Common::{
                            DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM,
                            DXGI_SAMPLE_DESC,
                        },
                        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_STRETCH,
                        DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        IDXGIFactory2,
                    },
                },
            },
            core::BOOL,
        };

        let handle = window.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            bail!("D3D11 presenter requires a Win32 window handle");
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .context("failed to create D3D11 device")?;
        }
        let device = device.context("D3D11 did not return a device")?;
        let context = context.context("D3D11 did not return an immediate context")?;

        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("failed to create DXGI factory")?;
        let tearing_supported = dxgi_tearing_supported(&factory);
        eprintln!("D3D11 flip presenter tearing support: {tearing_supported}");
        eprintln!("D3D11 upload mode: {upload_mode}");

        let swap_chain_flags = if tearing_supported {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 3,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: swap_chain_flags,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(&device, HWND(handle.hwnd.get() as _), &desc, None, None)
        }
        .context("failed to create D3D11 flip-model swapchain")?;

        let mut presenter = Self {
            device,
            context,
            swap_chain,
            back_buffer: None,
            upload_slots: Vec::new(),
            upload_index: 0,
            upload_mode,
            width: size.width.max(1),
            height: size.height.max(1),
            tearing_supported,
        };
        presenter.recreate_backbuffer()?;
        if upload_mode == D3dUploadMode::MappedCopy {
            presenter.recreate_upload_textures(width, height, upload_ring)?;
        }
        Ok(presenter)
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        let total_start = Instant::now();
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        self.ensure_size(size)?;
        let (map_copy, draw) = match self.upload_mode {
            D3dUploadMode::MappedCopy => {
                let upload_index = self.upload_index;
                self.upload_index = (self.upload_index + 1) % self.upload_slots.len();
                let map_copy_start = Instant::now();
                self.upload_bgra(upload_index, size.width, size.height, bgra)?;
                let map_copy = map_copy_start.elapsed();
                let copy_start = Instant::now();
                self.copy_upload_to_backbuffer(upload_index)?;
                (map_copy, copy_start.elapsed())
            }
            D3dUploadMode::UpdateSubresource => {
                let map_copy_start = Instant::now();
                self.update_backbuffer(size.width, bgra)?;
                (map_copy_start.elapsed(), Duration::ZERO)
            }
        };
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        let swap_start = Instant::now();
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        let swap_present = swap_start.elapsed();
        Ok(PresentTimings {
            map_copy,
            draw,
            swap_present,
            total: total_start.elapsed(),
        })
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.back_buffer = None;
        let upload_ring = self.upload_slots.len();
        self.upload_slots.clear();
        self.upload_index = 0;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe {
            self.swap_chain.ResizeBuffers(
                0,
                width,
                height,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                flags,
            )?;
        }
        self.width = width;
        self.height = height;
        self.recreate_backbuffer()?;
        if self.upload_mode == D3dUploadMode::MappedCopy {
            self.recreate_upload_textures(width, height, upload_ring)?;
        }
        Ok(())
    }

    fn recreate_backbuffer(&mut self) -> Result<()> {
        self.back_buffer = Some(unsafe { self.swap_chain.GetBuffer(0)? });
        Ok(())
    }

    fn recreate_upload_textures(
        &mut self,
        width: u32,
        height: u32,
        upload_ring: usize,
    ) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D11::{
                D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_DYNAMIC, ID3D11Texture2D,
            },
            Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
        };

        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
        };
        self.upload_slots.clear();
        self.upload_slots.reserve(upload_ring);
        for _ in 0..upload_ring {
            let mut texture: Option<ID3D11Texture2D> = None;
            unsafe {
                self.device
                    .CreateTexture2D(&desc, None, Some(&mut texture))?;
            }
            let texture = texture.context("D3D11 did not return an upload texture")?;
            self.upload_slots.push(D3d11UploadSlot { texture });
        }
        self.upload_index = 0;
        Ok(())
    }

    fn upload_bgra(&self, slot_index: usize, width: u32, height: u32, bgra: &[u8]) -> Result<()> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE,
        };

        let texture = &self
            .upload_slots
            .get(slot_index)
            .context("D3D11 upload texture is not available")?
            .texture;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(texture, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))?;
            let dst_pitch = mapped.RowPitch as usize;
            let dst_base = mapped.pData.cast::<u8>();
            let _used_fast_path = copy_bgra_to_mapped(bgra, dst_base, width, height, dst_pitch);
            self.context.Unmap(texture, 0);
        }
        Ok(())
    }

    fn copy_upload_to_backbuffer(&self, slot_index: usize) -> Result<()> {
        let upload_texture = self
            .upload_slots
            .get(slot_index)
            .context("D3D11 upload texture is not available")?
            .texture
            .clone();
        let back_buffer = self
            .back_buffer
            .as_ref()
            .context("D3D11 backbuffer is not available")?;
        unsafe {
            self.context.CopyResource(back_buffer, &upload_texture);
        }
        Ok(())
    }

    fn update_backbuffer(&self, width: u32, bgra: &[u8]) -> Result<()> {
        let back_buffer = self
            .back_buffer
            .as_ref()
            .context("D3D11 backbuffer is not available")?;
        unsafe {
            self.context.UpdateSubresource(
                back_buffer,
                0,
                None,
                bgra.as_ptr().cast(),
                width * 4,
                0,
            );
        }
        Ok(())
    }
}

#[cfg(windows)]
fn dxgi_tearing_supported(factory: &windows::Win32::Graphics::Dxgi::IDXGIFactory2) -> bool {
    use windows::{
        Win32::Graphics::Dxgi::{DXGI_FEATURE_PRESENT_ALLOW_TEARING, IDXGIFactory5},
        core::BOOL,
        core::Interface as _,
    };

    let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allow_tearing = BOOL(0);
    unsafe {
        factory5
            .CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                (&mut allow_tearing as *mut BOOL).cast(),
                std::mem::size_of::<BOOL>() as u32,
            )
            .is_ok()
            && allow_tearing.as_bool()
    }
}

#[cfg(windows)]
fn bitmap_info(width: u32, height: u32) -> windows_sys::Win32::Graphics::Gdi::BITMAPINFO {
    use windows_sys::Win32::Graphics::Gdi::{BI_RGB, BITMAPINFO, BITMAPINFOHEADER, RGBQUAD};

    BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: width.saturating_mul(height).saturating_mul(4),
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [RGBQUAD {
            rgbBlue: 0,
            rgbGreen: 0,
            rgbRed: 0,
            rgbReserved: 0,
        }],
    }
}

#[cfg(not(windows))]
pub(crate) struct WindowPresenter;

#[cfg(not(windows))]
impl WindowPresenter {
    pub(crate) fn new(
        _window: &Window,
        _kind: PresenterKind,
        _upload_ring: usize,
        _d3d_upload: D3dUploadMode,
        _interop_device: Option<&NeoD3d12InteropDevice>,
    ) -> Result<Self> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    pub(crate) fn present(
        &mut self,
        _size: PhysicalSize<u32>,
        _bgra: &[u8],
    ) -> Result<PresentTimings> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    pub(crate) fn kind(&self) -> PresenterKind {
        PresenterKind::Gdi
    }
}
