use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use neo_lang::{AddressSpace, TypeName};
use neo_runtime::{
    Context as NeoContext, CudaFence, CudaGraph, DeviceBuffer, IndexFormat, InstanceAttribute,
    InstanceBuffer, InstanceBufferDesc, InstanceFormat, InstanceLayout, InstanceSemantic, Kernel,
    LaunchDims, MeshBuffer, MeshBufferDesc, NeoD3d12InteropDevice, PrimitiveTopology,
    ReadablePinnedHostBuffer, SharedFrameRing, Stream as CudaStream, VertexAttribute, VertexFormat,
    VertexLayout, VertexSemantic,
};
use notify::{Event as NotifyEvent, RecursiveMode, Watcher as _};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes},
};

const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 540;
const BLOCK: (u32, u32) = (16, 16);
const INSTANCE_CULL_TILE: u32 = 8;
const TILE_CULL_RECORD_BYTES: usize = 16;
const EMPTY_IDLE_FPS: f32 = 15.0;
const UNFOCUSED_IDLE_FPS: f32 = 15.0;
const CAMERA_MOVE_UNITS_PER_SEC: f32 = 4.0;
const CAMERA_MAX_STEP_SECONDS: f32 = 1.0 / 30.0;
const DEFAULT_INSTANCE_GRID: InstanceGrid = InstanceGrid {
    x: 128,
    y: 128,
    z: 64,
};

pub fn main_entry() -> Result<()> {
    run_from_args(std::env::args().skip(1))
}

pub fn run_from_args(args: impl IntoIterator<Item = String>) -> Result<()> {
    run(LiveOptions::parse(args)?)
}

#[allow(deprecated)]
fn run(options: LiveOptions) -> Result<()> {
    let source_path = options
        .source_path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", options.source_path.display()))?;
    let instance_source_path = if options.mode == RunMode::InstanceStress {
        options.instance_stress_variant.source_path(&source_path)
    } else {
        source_path.clone()
    };
    let event_loop = EventLoop::new()?;
    let window = create_window(&event_loop, &options.title, options.width, options.height)?;
    let neo = NeoContext::new_default_device()?;
    let mut interop_device = None;
    let presenter_kind = if options.presenter == PresenterKind::D3d12Interop {
        match NeoD3d12InteropDevice::new(&neo) {
            Ok(device) => {
                interop_device = Some(device);
                PresenterKind::D3d12Interop
            }
            Err(err) if options.interop_fallback == InteropFallback::NoInterop => {
                eprintln!("D3D12/CUDA interop unavailable; falling back to d3d12: {err:#}");
                PresenterKind::D3d12
            }
            Err(err) => return Err(err).context("failed to initialize D3D12/CUDA interop"),
        }
    } else {
        options.presenter
    };
    if matches!(options.mode, RunMode::MeshDemo | RunMode::InstanceStress)
        && presenter_kind != PresenterKind::D3d12Interop
    {
        bail!(
            "{} requires D3D12/CUDA interop; use --presenter d3d12-interop --interop-fallback fail",
            options.mode
        );
    }
    let presenter = WindowPresenter::new(
        &window,
        presenter_kind,
        options.present_ring,
        options.d3d_upload,
        interop_device.as_ref(),
    )?;
    let mut presenter = if matches!(options.mode, RunMode::KernelThroughput)
        && presenter_kind.uses_present_thread()
    {
        PresentSink::threaded(presenter)
    } else {
        PresentSink::Direct(presenter)
    };

    if matches!(options.mode, RunMode::KernelThroughput) {
        unsafe {
            neo.disable_automatic_event_tracking();
        }
    }
    let mut live_reload = if matches!(options.mode, RunMode::MeshDemo | RunMode::InstanceStress) {
        None
    } else {
        Some(ReloadState::new(LiveKernel::compile(&neo, &source_path)?))
    };
    let mut mesh_reload = if options.mode == RunMode::MeshDemo {
        Some(ReloadState::new(MeshKernel::compile(&neo, &source_path)?))
    } else {
        None
    };
    let mut instance_reload = if options.mode == RunMode::InstanceStress {
        Some(ReloadState::new(InstanceKernel::compile(
            &neo,
            &instance_source_path,
            options.instance_stress_variant,
        )?))
    } else {
        None
    };
    let mesh_buffer = if options.mode == RunMode::MeshDemo {
        Some(create_demo_mesh(&neo)?)
    } else {
        None
    };
    let mut instance_assets = if options.mode == RunMode::InstanceStress {
        Some(create_instance_stress_assets(
            &neo,
            options.instance_grid,
            options.present_ring,
        )?)
    } else {
        None
    };
    let watched_source_path = if options.mode == RunMode::InstanceStress {
        &instance_source_path
    } else {
        &source_path
    };
    let (_watcher, reload_rx) = if options.hot_reload {
        let (watcher, rx) = watch_source(watched_source_path)?;
        (Some(watcher), Some(rx))
    } else {
        (None, None)
    };
    let mut frame_resources: Option<FrameResources> = None;
    let mut throughput_resources: Option<ThroughputResources> = None;
    let mut interop_throughput_resources: Option<InteropThroughputResources> = None;
    let mut throughput_generation = live_reload
        .as_ref()
        .map(|reload| reload.generation)
        .or_else(|| mesh_reload.as_ref().map(|reload| reload.generation))
        .or_else(|| instance_reload.as_ref().map(|reload| reload.generation))
        .unwrap_or(0);
    let mut fps = FpsCounter::new();
    let mut throughput = ThroughputCounter::new();
    let start = Instant::now();
    let mut frame = 0u32;
    let mut completed_kernels = 0u64;
    let mut next_sample_at = options.sample_every as u64;
    let mut next_present_at = options.present_interval().map(|interval| start + interval);
    let mut kernel_limiter =
        KernelRateLimiter::new(options.kernel_cap(), options.max_inflight, start);
    let mut unfocused_kernel_limiter = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
    let mut interop_present_limiter = PresentRateLimiter::new(options.present_target_fps, start);
    let mut camera = CameraController::new();
    let mut window_visibility = WindowVisibilityState::default();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                if let Some(reload_rx) = reload_rx.as_ref() {
                    let reload_result = match options.mode {
                        RunMode::MeshDemo => handle_mesh_reload_events(
                            &neo,
                            &source_path,
                            reload_rx,
                            mesh_reload
                                .as_mut()
                                .expect("mesh reload exists in mesh-demo mode"),
                        ),
                        RunMode::InstanceStress => handle_instance_reload_events(
                            &neo,
                            &instance_source_path,
                            options.instance_stress_variant,
                            reload_rx,
                            instance_reload
                                .as_mut()
                                .expect("instance reload exists in instance-stress mode"),
                        ),
                        _ => handle_reload_events(
                            &neo,
                            &source_path,
                            reload_rx,
                            live_reload
                                .as_mut()
                                .expect("live reload exists outside mesh-demo mode"),
                        ),
                    };
                    if let Err(err) = reload_result {
                        eprintln!("hot reload watcher error: {err:#}");
                    }
                }

                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    window_visibility.minimized = true;
                    elwt.set_control_flow(ControlFlow::Wait);
                    return;
                }
                window_visibility.minimized = false;

                if matches!(options.mode, RunMode::InstanceStress) {
                    let reload = instance_reload
                        .as_mut()
                        .expect("instance reload exists in instance-stress mode");
                    if presenter.kind() != PresenterKind::D3d12Interop {
                        eprintln!("instance-stress requires --presenter d3d12-interop");
                        elwt.exit();
                        return;
                    }
                    if throughput_generation != reload.generation {
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let camera_params =
                        camera.params(size, start.elapsed().as_secs_f32(), options.instance_grid);
                    let visibility = instance_render_visibility(
                        options.render_policy,
                        window_visibility,
                        &camera_params,
                    );
                    let max_inflight = grant_instance_inflight(
                        now,
                        visibility,
                        options.render_policy,
                        window_visibility.focused,
                        options.max_inflight,
                        &mut kernel_limiter,
                        &mut unfocused_kernel_limiter,
                    );
                    let has_live_interop_work = interop_throughput_resources
                        .as_ref()
                        .is_some_and(InteropThroughputResources::has_live_work);
                    let skip_idle_gpu_work = should_skip_idle_instance_gpu_work(
                        visibility,
                        options.render_policy,
                        max_inflight,
                        has_live_interop_work,
                    );
                    let result = if skip_idle_gpu_work {
                        Ok(ThroughputBatchStats::default())
                    } else {
                        match (interop_device.as_ref(), instance_assets.as_mut()) {
                            (Some(interop), Some(assets)) => {
                                run_instance_stress_batch(InstanceStressBatch {
                                    neo: &neo,
                                    interop,
                                    resources: &mut interop_throughput_resources,
                                    kernel: &reload.active,
                                    assets,
                                    camera: camera_params,
                                    presenter: &mut presenter,
                                    size,
                                    start,
                                    next_frame: &mut frame,
                                    completed_kernels: &mut completed_kernels,
                                    present_limiter: &mut interop_present_limiter,
                                    max_inflight,
                                    present_ring: options.present_ring,
                                })
                            }
                            _ => Err(anyhow!(
                                "missing D3D12 interop device or instance stress assets"
                            )),
                        }
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: Some(options.instance_stress_variant),
                                render_policy: options.render_policy,
                                visibility,
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                            if max_inflight == 0
                                && let Some(next_kernel_at) =
                                    next_instance_tick_at(InstanceTickPacing {
                                        now,
                                        visibility,
                                        kernel_limiter: &kernel_limiter,
                                        unfocused_limiter: &unfocused_kernel_limiter,
                                        render_policy: options.render_policy,
                                        focused: window_visibility.focused,
                                        wait_for_kernel_token: should_wait_for_kernel_token(
                                            presenter.kind(),
                                            options.present_target_fps,
                                        ),
                                    })
                            {
                                elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                            } else if max_inflight == 0
                                && options.render_policy != RenderPolicy::ForceRender
                                && matches!(
                                    visibility,
                                    RenderVisibility::Occluded | RenderVisibility::Minimized
                                )
                            {
                                elwt.set_control_flow(ControlFlow::Wait);
                            }
                        }
                        Err(err) => {
                            eprintln!("instance stress error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                if matches!(options.mode, RunMode::MeshDemo) {
                    let reload = mesh_reload
                        .as_mut()
                        .expect("mesh reload exists in mesh-demo mode");
                    if presenter.kind() != PresenterKind::D3d12Interop {
                        eprintln!("mesh-demo requires --presenter d3d12-interop");
                        elwt.exit();
                        return;
                    }
                    if throughput_generation != reload.generation {
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    let result = match (interop_device.as_ref(), mesh_buffer.as_ref()) {
                        (Some(interop), Some(mesh)) => run_mesh_demo_batch(MeshDemoBatch {
                            neo: &neo,
                            interop,
                            resources: &mut interop_throughput_resources,
                            kernel: &reload.active.kernel,
                            mesh,
                            presenter: &mut presenter,
                            size,
                            start,
                            next_frame: &mut frame,
                            completed_kernels: &mut completed_kernels,
                            present_limiter: &mut interop_present_limiter,
                            max_inflight,
                            present_ring: options.present_ring,
                        }),
                        _ => Err(anyhow!("missing D3D12 interop device or mesh buffer")),
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: None,
                                render_policy: options.render_policy,
                                visibility: window_visibility.render_visibility(),
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                        }
                        Err(err) => {
                            eprintln!("mesh demo error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                if matches!(options.mode, RunMode::KernelThroughput) {
                    let reload = live_reload
                        .as_mut()
                        .expect("live reload exists in kernel-throughput mode");
                    if throughput_generation != reload.generation {
                        throughput_resources = None;
                        wait_for_interop_idle(
                            &mut interop_throughput_resources,
                            interop_device.as_ref(),
                        );
                        interop_throughput_resources = None;
                        throughput_generation = reload.generation;
                    }
                    let now = Instant::now();
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    let result = if presenter.kind() == PresenterKind::D3d12Interop {
                        match interop_device.as_ref() {
                            Some(interop) => run_interop_throughput_batch(InteropThroughputBatch {
                                neo: &neo,
                                interop,
                                resources: &mut interop_throughput_resources,
                                kernel: &reload.active.kernel,
                                presenter: &mut presenter,
                                size,
                                start,
                                next_frame: &mut frame,
                                completed_kernels: &mut completed_kernels,
                                present_limiter: &mut interop_present_limiter,
                                max_inflight,
                                present_ring: options.present_ring,
                            }),
                            None => Err(anyhow!("missing D3D12 interop device")),
                        }
                    } else {
                        run_kernel_throughput_batch(ThroughputBatch {
                            neo: &neo,
                            resources: &mut throughput_resources,
                            graph_kernel: &reload.active.graph_kernel,
                            presenter: &mut presenter,
                            size,
                            start,
                            next_frame: &mut frame,
                            completed_kernels: &mut completed_kernels,
                            next_sample_at: &mut next_sample_at,
                            next_present_at: &mut next_present_at,
                            sample_every: options.sample_every,
                            present_interval: options.present_interval(),
                            max_inflight,
                            present_ring: options.present_ring,
                        })
                    };
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(ThroughputLogContext {
                                size,
                                frame,
                                presenter: presenter.kind(),
                                reload_error: reload.last_error.as_deref(),
                                kernel_cap: options.kernel_cap(),
                                instance_variant: None,
                                render_policy: options.render_policy,
                                visibility: window_visibility.render_visibility(),
                            });
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                wait_for_interop_idle(
                                    &mut interop_throughput_resources,
                                    interop_device.as_ref(),
                                );
                                elwt.exit();
                            }
                            if max_inflight == 0
                                && should_wait_for_kernel_token(
                                    presenter.kind(),
                                    options.present_target_fps,
                                )
                                && let Some(next_kernel_at) = kernel_limiter.next_token_at(now)
                            {
                                elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                            }
                        }
                        Err(err) => {
                            eprintln!("throughput error: {err:#}");
                            wait_for_interop_idle(
                                &mut interop_throughput_resources,
                                interop_device.as_ref(),
                            );
                            elwt.exit();
                        }
                    }
                    return;
                }

                let reload = live_reload
                    .as_ref()
                    .expect("live reload exists in live mode");
                let frame_start = Instant::now();
                let mut timings = match render_frame(RenderFrame {
                    neo: &neo,
                    resources: &mut frame_resources,
                    kernel: &reload.active.kernel,
                    size,
                    time: start.elapsed().as_secs_f32(),
                    frame,
                }) {
                    Ok(timings) => timings,
                    Err(err) => {
                        eprintln!("render error: {err:#}");
                        elwt.exit();
                        return;
                    }
                };

                let resources = frame_resources
                    .as_ref()
                    .expect("frame resources are created before presentation");
                let before_present = Instant::now();
                let present_timings =
                    match presenter.present_sync(size, resources.host_bgra.as_slice()) {
                        Ok(timings) => timings,
                        Err(err) => {
                            eprintln!("present error: {err:#}");
                            elwt.exit();
                            return;
                        }
                    };
                timings.render = before_present - frame_start;
                timings.present = present_timings.total;
                timings.total = frame_start.elapsed();

                frame = frame.wrapping_add(1);
                fps.tick(
                    size,
                    frame,
                    reload.last_error.as_deref(),
                    presenter.kind(),
                    timings,
                );
                if options.should_stop(frame, start.elapsed()) {
                    elwt.exit();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera.handle_raw_mouse_delta(delta.0 as f32, delta.1 as f32);
            }
            Event::WindowEvent { event, .. } => {
                match &event {
                    WindowEvent::Occluded(occluded) => {
                        window_visibility.occluded = *occluded;
                    }
                    WindowEvent::Focused(focused) => {
                        window_visibility.focused = *focused;
                    }
                    WindowEvent::Resized(size) => {
                        window_visibility.minimized = size.width == 0 || size.height == 0;
                    }
                    _ => {}
                }
                match camera.handle_window_event(&event) {
                    CameraWindowAction::None => {}
                    CameraWindowAction::CaptureMouse => capture_mouse(&window),
                    CameraWindowAction::ReleaseMouse => release_mouse(&window),
                    CameraWindowAction::RecenterMouse => recenter_mouse(&window),
                }
            }
            _ => {}
        }
    })?;
    Ok(())
}

fn wait_for_interop_idle(
    resources: &mut Option<InteropThroughputResources>,
    interop: Option<&NeoD3d12InteropDevice>,
) {
    if let (Some(resources), Some(interop)) = (resources.as_mut(), interop)
        && let Err(err) = resources.ring.wait_idle(interop.queue())
    {
        eprintln!("failed to wait for D3D12/CUDA interop idle: {err:#}");
    }
}

fn interop_trace(message: &str) {
    if std::env::var_os("NEO_INTEROP_TRACE").is_some() {
        eprintln!("[interop] {message}");
    }
}

fn capture_mouse(window: &Window) {
    if window.set_cursor_grab(CursorGrabMode::Locked).is_err()
        && let Err(err) = window.set_cursor_grab(CursorGrabMode::Confined)
    {
        eprintln!("mouse look capture unavailable: {err}");
    }
    window.set_cursor_visible(false);
    recenter_mouse(window);
}

fn release_mouse(window: &Window) {
    if let Err(err) = window.set_cursor_grab(CursorGrabMode::None) {
        eprintln!("failed to release mouse look capture: {err}");
    }
    window.set_cursor_visible(true);
}

fn recenter_mouse(window: &Window) {
    let size = window.inner_size();
    if size.width == 0 || size.height == 0 {
        return;
    }
    if let Err(err) = window.set_cursor_position(PhysicalPosition::new(
        f64::from(size.width) * 0.5,
        f64::from(size.height) * 0.5,
    )) {
        if std::env::var_os("NEO_MOUSE_TRACE").is_some() {
            eprintln!("failed to recenter mouse look cursor: {err}");
        }
    }
}

fn should_wait_for_kernel_token(presenter: PresenterKind, present_target_fps: Option<f32>) -> bool {
    const POLL_PRECISE_INTEROP_FPS: f32 = 120.0;
    !(presenter == PresenterKind::D3d12Interop
        && present_target_fps.is_some_and(|fps| fps >= POLL_PRECISE_INTEROP_FPS))
}

fn grant_instance_inflight(
    now: Instant,
    visibility: RenderVisibility,
    render_policy: RenderPolicy,
    focused: bool,
    max_inflight: u32,
    kernel_limiter: &mut KernelRateLimiter,
    unfocused_limiter: &mut KernelRateLimiter,
) -> u32 {
    let visibility_limit = visibility.limit_max_inflight(max_inflight);
    if visibility_limit == 0 {
        return 0;
    }
    if render_policy == RenderPolicy::Auto && !focused {
        let kernel_available = kernel_limiter.available(now, visibility_limit);
        let unfocused_available = unfocused_limiter.available(now, 1);
        let granted = kernel_available.min(unfocused_available);
        kernel_limiter.consume(granted);
        unfocused_limiter.consume(granted);
        granted
    } else {
        kernel_limiter.grant(now, visibility_limit)
    }
}

fn should_skip_idle_instance_gpu_work(
    visibility: RenderVisibility,
    render_policy: RenderPolicy,
    max_inflight: u32,
    has_live_interop_work: bool,
) -> bool {
    render_policy != RenderPolicy::ForceRender
        && max_inflight == 0
        && !has_live_interop_work
        && matches!(
            visibility,
            RenderVisibility::Empty | RenderVisibility::Occluded | RenderVisibility::Minimized
        )
}

struct InstanceTickPacing<'a> {
    now: Instant,
    visibility: RenderVisibility,
    kernel_limiter: &'a KernelRateLimiter,
    unfocused_limiter: &'a KernelRateLimiter,
    render_policy: RenderPolicy,
    focused: bool,
    wait_for_kernel_token: bool,
}

fn next_instance_tick_at(pacing: InstanceTickPacing<'_>) -> Option<Instant> {
    if pacing.render_policy == RenderPolicy::ForceRender {
        return if pacing.wait_for_kernel_token {
            pacing.kernel_limiter.next_token_at(pacing.now)
        } else {
            None
        };
    }
    match pacing.visibility {
        RenderVisibility::Visible => {
            if pacing.render_policy == RenderPolicy::Auto && !pacing.focused {
                return latest_token_time(
                    pacing.kernel_limiter.next_token_at(pacing.now),
                    pacing.unfocused_limiter.next_token_at(pacing.now),
                );
            }
            if pacing.wait_for_kernel_token {
                pacing.kernel_limiter.next_token_at(pacing.now)
            } else {
                None
            }
        }
        RenderVisibility::Empty => Some(pacing.now + Duration::from_secs_f32(1.0 / EMPTY_IDLE_FPS)),
        RenderVisibility::Occluded | RenderVisibility::Minimized => None,
    }
}

fn latest_token_time(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[derive(Debug, Clone)]
struct LiveOptions {
    source_path: PathBuf,
    title: String,
    width: u32,
    height: u32,
    max_frames: Option<u32>,
    max_seconds: Option<f32>,
    presenter: PresenterKind,
    mode: RunMode,
    sample_every: u32,
    present_target_fps: Option<f32>,
    kernel_target_fps: Option<f32>,
    max_inflight: u32,
    present_ring: usize,
    instance_grid: InstanceGrid,
    instance_stress_variant: InstanceStressVariant,
    render_policy: RenderPolicy,
    d3d_upload: D3dUploadMode,
    interop_fallback: InteropFallback,
    hot_reload: bool,
}

impl LiveOptions {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            source_path: PathBuf::from("examples/live-window/live.neo"),
            title: "Neo Live Window".to_string(),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            max_frames: None,
            max_seconds: None,
            presenter: PresenterKind::D3d11,
            mode: RunMode::Live,
            sample_every: 256,
            present_target_fps: None,
            kernel_target_fps: None,
            max_inflight: 2,
            present_ring: 6,
            instance_grid: DEFAULT_INSTANCE_GRID,
            instance_stress_variant: InstanceStressVariant::Tiled,
            render_policy: RenderPolicy::Auto,
            d3d_upload: D3dUploadMode::MappedCopy,
            interop_fallback: InteropFallback::NoInterop,
            hot_reload: true,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--width" => options.width = parse_next(&mut args, "--width")?,
                "--height" => options.height = parse_next(&mut args, "--height")?,
                "--title" => options.title = parse_next(&mut args, "--title")?,
                "--frames" => options.max_frames = Some(parse_next(&mut args, "--frames")?),
                "--seconds" => options.max_seconds = Some(parse_next(&mut args, "--seconds")?),
                "--presenter" => options.presenter = parse_next(&mut args, "--presenter")?,
                "--mode" => options.mode = parse_next(&mut args, "--mode")?,
                "--sample-every" => options.sample_every = parse_next(&mut args, "--sample-every")?,
                "--present-target-fps" => {
                    options.present_target_fps =
                        Some(parse_next(&mut args, "--present-target-fps")?)
                }
                "--kernel-target-fps" => {
                    options.kernel_target_fps = Some(parse_next(&mut args, "--kernel-target-fps")?)
                }
                "--max-inflight" => options.max_inflight = parse_next(&mut args, "--max-inflight")?,
                "--present-ring" => options.present_ring = parse_next(&mut args, "--present-ring")?,
                "--instance-grid" => {
                    options.instance_grid = parse_next(&mut args, "--instance-grid")?
                }
                "--instance-stress-variant" => {
                    options.instance_stress_variant =
                        parse_next(&mut args, "--instance-stress-variant")?
                }
                "--render-policy" => {
                    options.render_policy = parse_next(&mut args, "--render-policy")?
                }
                "--d3d-upload" => options.d3d_upload = parse_next(&mut args, "--d3d-upload")?,
                "--interop-fallback" => {
                    options.interop_fallback = parse_next(&mut args, "--interop-fallback")?
                }
                "--hot-reload" => options.hot_reload = true,
                "--no-hot-reload" => options.hot_reload = false,
                "--help" | "-h" => bail!(
                    "usage: neo-live-window [path.neo] [--title TEXT] [--width N] [--height N] [--frames N] [--seconds N] [--presenter d3d12-interop|d3d12|d3d11|gdi] [--mode live|kernel-throughput|mesh-demo|instance-stress] [--sample-every N] [--present-target-fps N] [--kernel-target-fps N] [--max-inflight N] [--present-ring N] [--instance-grid XxYxZ] [--instance-stress-variant baseline|fast|culled|tiled] [--render-policy auto|force-render|pause-when-empty] [--d3d-upload mapped-copy|update-subresource] [--interop-fallback no-interop|fail] [--hot-reload|--no-hot-reload]"
                ),
                value if value.starts_with('-') => bail!("unknown option `{value}`"),
                value => options.source_path = PathBuf::from(value),
            }
        }
        options.validate()?;
        Ok(options)
    }

    fn should_stop(&self, frame: u32, elapsed: Duration) -> bool {
        self.max_frames.is_some_and(|max| frame >= max)
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    fn should_stop_completed(&self, completed: u64, elapsed: Duration) -> bool {
        self.max_frames
            .is_some_and(|max| completed >= u64::from(max))
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    fn validate(&self) -> Result<()> {
        if self.width == 0 {
            bail!("--width must be greater than zero");
        }
        if self.height == 0 {
            bail!("--height must be greater than zero");
        }
        if self.sample_every == 0 {
            bail!("--sample-every must be greater than zero");
        }
        if self.max_inflight == 0 {
            bail!("--max-inflight must be greater than zero");
        }
        if self.present_ring == 0 {
            bail!("--present-ring must be greater than zero");
        }
        self.instance_grid.validate()?;
        if self
            .present_target_fps
            .is_some_and(|fps| !fps.is_finite() || fps <= 0.0)
        {
            bail!("--present-target-fps must be greater than zero");
        }
        if self
            .kernel_target_fps
            .is_some_and(|fps| !fps.is_finite() || fps < 0.0)
        {
            bail!("--kernel-target-fps must be finite and non-negative");
        }
        Ok(())
    }

    fn present_interval(&self) -> Option<Duration> {
        self.present_target_fps
            .map(|fps| Duration::from_secs_f32(1.0 / fps))
    }

    fn kernel_cap(&self) -> Option<f32> {
        self.kernel_target_fps.filter(|fps| *fps > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Live,
    KernelThroughput,
    MeshDemo,
    InstanceStress,
}

impl std::str::FromStr for RunMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "live" => Ok(Self::Live),
            "kernel-throughput" => Ok(Self::KernelThroughput),
            "mesh-demo" => Ok(Self::MeshDemo),
            "instance-stress" => Ok(Self::InstanceStress),
            _ => bail!(
                "unknown mode `{value}`; expected live, kernel-throughput, mesh-demo, or instance-stress"
            ),
        }
    }
}

impl std::fmt::Display for RunMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live => f.write_str("live"),
            Self::KernelThroughput => f.write_str("kernel-throughput"),
            Self::MeshDemo => f.write_str("mesh-demo"),
            Self::InstanceStress => f.write_str("instance-stress"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderPolicy {
    Auto,
    ForceRender,
    PauseWhenEmpty,
}

impl std::str::FromStr for RenderPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "force-render" => Ok(Self::ForceRender),
            "pause-when-empty" => Ok(Self::PauseWhenEmpty),
            _ => bail!(
                "unknown render policy `{value}`; expected auto, force-render, or pause-when-empty"
            ),
        }
    }
}

impl std::fmt::Display for RenderPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::ForceRender => f.write_str("force-render"),
            Self::PauseWhenEmpty => f.write_str("pause-when-empty"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderVisibility {
    Visible,
    Empty,
    Occluded,
    Minimized,
}

impl RenderVisibility {
    fn limit_max_inflight(self, max_inflight: u32) -> u32 {
        match self {
            Self::Visible => max_inflight,
            Self::Empty | Self::Occluded | Self::Minimized => 0,
        }
    }
}

impl std::fmt::Display for RenderVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Visible => f.write_str("visible"),
            Self::Empty => f.write_str("empty"),
            Self::Occluded => f.write_str("occluded"),
            Self::Minimized => f.write_str("minimized"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WindowVisibilityState {
    minimized: bool,
    occluded: bool,
    focused: bool,
}

impl Default for WindowVisibilityState {
    fn default() -> Self {
        Self {
            minimized: false,
            occluded: false,
            focused: true,
        }
    }
}

impl WindowVisibilityState {
    fn render_visibility(self) -> RenderVisibility {
        if self.minimized {
            RenderVisibility::Minimized
        } else if self.occluded {
            RenderVisibility::Occluded
        } else {
            RenderVisibility::Visible
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstanceGrid {
    x: u32,
    y: u32,
    z: u32,
}

impl InstanceGrid {
    fn validate(self) -> Result<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            bail!("--instance-grid dimensions must be greater than zero");
        }
        self.count()
            .ok_or_else(|| anyhow!("--instance-grid instance count overflow"))?;
        Ok(())
    }

    fn count(self) -> Option<u32> {
        self.x.checked_mul(self.y)?.checked_mul(self.z)
    }
}

impl std::str::FromStr for InstanceGrid {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let parts = value.split('x').collect::<Vec<_>>();
        if parts.len() != 3 {
            bail!("invalid --instance-grid `{value}`; expected XxYxZ");
        }
        let grid = Self {
            x: parts[0]
                .parse()
                .with_context(|| format!("invalid X dimension in --instance-grid `{value}`"))?,
            y: parts[1]
                .parse()
                .with_context(|| format!("invalid Y dimension in --instance-grid `{value}`"))?,
            z: parts[2]
                .parse()
                .with_context(|| format!("invalid Z dimension in --instance-grid `{value}`"))?,
        };
        grid.validate()?;
        Ok(grid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceStressVariant {
    Baseline,
    Fast,
    Culled,
    Tiled,
}

impl InstanceStressVariant {
    fn source_path(self, requested: &Path) -> PathBuf {
        if requested
            .file_name()
            .is_some_and(|name| name == "three_d_instances.neo")
        {
            match self {
                Self::Baseline => {
                    return requested.with_file_name("three_d_instances_baseline.neo");
                }
                Self::Fast => return requested.with_file_name("three_d_instances_fast.neo"),
                Self::Culled => {}
                Self::Tiled => return requested.with_file_name("three_d_instances_tiled.neo"),
            }
        }
        requested.to_path_buf()
    }
}

impl std::str::FromStr for InstanceStressVariant {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "fast" => Ok(Self::Fast),
            "culled" => Ok(Self::Culled),
            "tiled" => Ok(Self::Tiled),
            _ => bail!(
                "unknown instance stress variant `{value}`; expected baseline, fast, culled, or tiled"
            ),
        }
    }
}

impl std::fmt::Display for InstanceStressVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline => f.write_str("baseline"),
            Self::Fast => f.write_str("fast"),
            Self::Culled => f.write_str("culled"),
            Self::Tiled => f.write_str("tiled"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresenterKind {
    D3d12Interop,
    D3d12,
    D3d11,
    Gdi,
}

impl PresenterKind {
    fn uses_present_thread(self) -> bool {
        !matches!(self, Self::D3d12 | Self::D3d12Interop)
    }
}

impl std::str::FromStr for PresenterKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "d3d12-interop" => Ok(Self::D3d12Interop),
            "d3d12" => Ok(Self::D3d12),
            "d3d11" => Ok(Self::D3d11),
            "gdi" => Ok(Self::Gdi),
            _ => bail!("unknown presenter `{value}`; expected d3d12-interop, d3d12, d3d11, or gdi"),
        }
    }
}

impl std::fmt::Display for PresenterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D3d12Interop => f.write_str("d3d12-interop"),
            Self::D3d12 => f.write_str("d3d12-flip-upload"),
            Self::D3d11 => f.write_str("d3d11-flip-host"),
            Self::Gdi => f.write_str("win32-gdi"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteropFallback {
    NoInterop,
    Fail,
}

impl std::str::FromStr for InteropFallback {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "no-interop" => Ok(Self::NoInterop),
            "fail" => Ok(Self::Fail),
            _ => bail!("unknown interop fallback `{value}`; expected no-interop or fail"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum D3dUploadMode {
    MappedCopy,
    UpdateSubresource,
}

impl std::str::FromStr for D3dUploadMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "mapped-copy" => Ok(Self::MappedCopy),
            "update-subresource" => Ok(Self::UpdateSubresource),
            _ => bail!(
                "unknown D3D upload mode `{value}`; expected mapped-copy or update-subresource"
            ),
        }
    }
}

impl std::fmt::Display for D3dUploadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MappedCopy => f.write_str("mapped-copy"),
            Self::UpdateSubresource => f.write_str("update-subresource"),
        }
    }
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let value = args
        .next()
        .ok_or_else(|| anyhow!("missing value for {name}"))?;
    value
        .parse()
        .map_err(|err| anyhow!("invalid value for {name}: {err}"))
}

#[allow(deprecated)]
fn create_window(
    event_loop: &EventLoop<()>,
    title: &str,
    width: u32,
    height: u32,
) -> Result<Window> {
    let attrs = WindowAttributes::default()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(width.max(1), height.max(1)))
        .with_resizable(true);
    event_loop
        .create_window(attrs)
        .context("failed to create live window")
}

struct LiveKernel {
    kernel: Kernel,
    graph_kernel: Kernel,
}

impl LiveKernel {
    fn compile(ctx: &NeoContext, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        validate_live_kernel_abi(&source)?;
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &["image"])?;
        let graph_module =
            neo_runtime::Module::from_cuda_source(ctx, live_graph_cuda_source(&source)?)?;
        Ok(Self {
            kernel: module.kernel("image")?,
            graph_kernel: graph_module.kernel("image_graph")?,
        })
    }
}

fn live_graph_cuda_source(source: &str) -> Result<String> {
    let program = neo_lang::parse(source)?;
    let cuda = neo_lang::lower_program(&program);
    let signature = "extern \"C\" __global__ void image(unsigned char* pixels, unsigned int width, unsigned int height, float time, unsigned int frame)";
    let replacement = "__device__ __forceinline__ void neo_user_image(unsigned char* pixels, unsigned int width, unsigned int height, float time, unsigned int frame)";
    let device_source = cuda.replacen(signature, replacement, 1);
    if device_source == cuda {
        bail!("failed to build CUDA graph wrapper for live kernel ABI");
    }
    Ok(format!(
        "{device_source}\n\
struct NeoLiveGraphParams {{\n\
    float time;\n\
    unsigned int frame;\n\
}};\n\
\n\
extern \"C\" __global__ void image_graph(unsigned char* pixels, unsigned int width, unsigned int height, const NeoLiveGraphParams* params) {{\n\
    neo_user_image(pixels, width, height, params->time, params->frame);\n\
}}\n"
    ))
}

fn validate_live_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.name == "image")
        .ok_or_else(|| anyhow!("live kernel must define `kernel fn image(...)`"))?;
    let expected = [
        ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("width", None, TypeName::U32, 0),
        ("height", None, TypeName::U32, 0),
        ("time", None, TypeName::F32, 0),
        ("frame", None, TypeName::U32, 0),
    ];
    if kernel.params.len() != expected.len() {
        bail!(
            "live kernel `image` must have {} params: global u8* pixels, u32 width, u32 height, f32 time, u32 frame",
            expected.len()
        );
    }
    for (param, expected_param) in kernel.params.iter().zip(expected.iter()) {
        let (name, address_space, ty, pointer_depth) = expected_param;
        if param.name != *name
            || &param.address_space != address_space
            || &param.ty.base != ty
            || param.ty.pointer_depth != *pointer_depth
        {
            bail!(
                "invalid live kernel parameter `{}`; expected `{}` in ABI `global u8* pixels, u32 width, u32 height, f32 time, u32 frame`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct MeshKernel {
    kernel: Kernel,
}

impl MeshKernel {
    fn compile(ctx: &NeoContext, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        validate_mesh_kernel_abi(&source)?;
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &["raster"])?;
        Ok(Self {
            kernel: module.kernel("raster")?,
        })
    }
}

fn validate_mesh_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.name == "raster")
        .ok_or_else(|| anyhow!("mesh demo kernel must define `kernel fn raster(...)`"))?;
    let expected = [
        ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("width", None, TypeName::U32, 0),
        ("height", None, TypeName::U32, 0),
        ("mesh", Some(AddressSpace::Global), TypeName::U8, 1usize),
        ("time", None, TypeName::F32, 0),
        ("frame", None, TypeName::U32, 0),
    ];
    if kernel.params.len() != expected.len() {
        bail!(
            "mesh demo kernel `raster` must have {} params: global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame",
            expected.len()
        );
    }
    for (param, (name, address_space, ty, pointer_depth)) in kernel.params.iter().zip(expected) {
        if param.name != name
            || param.address_space != address_space
            || param.ty.base != ty
            || param.ty.pointer_depth != pointer_depth
        {
            bail!(
                "invalid mesh demo kernel parameter `{}`; expected `{}` in ABI `global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct InstanceKernel {
    raster: Kernel,
    cull: Option<Kernel>,
    tiled: bool,
}

impl InstanceKernel {
    fn compile(ctx: &NeoContext, path: &Path, variant: InstanceStressVariant) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        match variant {
            InstanceStressVariant::Baseline
            | InstanceStressVariant::Fast
            | InstanceStressVariant::Tiled => {
                validate_instance_kernel_abi(&source)?;
            }
            InstanceStressVariant::Culled => validate_culled_instance_kernel_abi(&source)?,
        }
        let entrypoints = match variant {
            InstanceStressVariant::Baseline
            | InstanceStressVariant::Fast
            | InstanceStressVariant::Tiled => {
                vec!["instance_raster"]
            }
            InstanceStressVariant::Culled => vec!["instance_cull", "instance_raster"],
        };
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &entrypoints)?;
        Ok(Self {
            raster: module.kernel("instance_raster")?,
            cull: if variant == InstanceStressVariant::Culled {
                Some(module.kernel("instance_cull")?)
            } else {
                None
            },
            tiled: variant == InstanceStressVariant::Tiled,
        })
    }
}

fn validate_instance_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "instance_raster",
        &[
            ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "instance stress kernel",
        "global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame",
    )
}

fn validate_culled_instance_kernel_abi(source: &str) -> Result<()> {
    let program = neo_lang::parse(source)?;
    validate_kernel_signature(
        &program,
        "instance_cull",
        &[
            ("cull", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "culled instance stress prepass kernel",
        "global u8* cull, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame",
    )?;
    validate_kernel_signature(
        &program,
        "instance_raster",
        &[
            ("pixels", Some(AddressSpace::Global), TypeName::U8, 1usize),
            ("width", None, TypeName::U32, 0),
            ("height", None, TypeName::U32, 0),
            ("mesh", Some(AddressSpace::Global), TypeName::U8, 1),
            ("instances", Some(AddressSpace::Global), TypeName::U8, 1),
            ("camera", Some(AddressSpace::Global), TypeName::U8, 1),
            ("cull", Some(AddressSpace::Global), TypeName::U8, 1),
            ("time", None, TypeName::F32, 0),
            ("frame", None, TypeName::U32, 0),
        ],
        "culled instance stress raster kernel",
        "global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, global u8* cull, f32 time, u32 frame",
    )
}

fn validate_kernel_signature(
    program: &neo_lang::Program,
    name: &str,
    expected: &[(&str, Option<AddressSpace>, TypeName, usize)],
    label: &str,
    abi: &str,
) -> Result<()> {
    let kernel = program
        .kernels
        .iter()
        .find(|kernel| kernel.name == name)
        .ok_or_else(|| anyhow!("{label} must define `kernel fn {name}(...)`"))?;
    if kernel.params.len() != expected.len() {
        bail!(
            "{label} `{name}` must have {} params: {abi}",
            expected.len(),
        );
    }
    for (param, (name, address_space, ty, pointer_depth)) in kernel.params.iter().zip(expected) {
        if param.name != *name
            || &param.address_space != address_space
            || &param.ty.base != ty
            || param.ty.pointer_depth != *pointer_depth
        {
            bail!(
                "invalid {label} parameter `{}`; expected `{}` in ABI `{abi}`",
                param.name,
                name
            );
        }
    }
    Ok(())
}

struct ReloadState<T> {
    active: T,
    generation: u64,
    last_error: Option<String>,
}

impl<T> ReloadState<T> {
    fn new(active: T) -> Self {
        Self {
            active,
            generation: 0,
            last_error: None,
        }
    }

    fn try_replace<E: std::fmt::Display>(&mut self, result: Result<T, E>) -> bool {
        match result {
            Ok(next) => {
                self.active = next;
                self.generation += 1;
                self.last_error = None;
                true
            }
            Err(err) => {
                self.last_error = Some(err.to_string());
                false
            }
        }
    }
}

fn watch_source(
    path: &Path,
) -> Result<(
    notify::RecommendedWatcher,
    Receiver<notify::Result<NotifyEvent>>,
)> {
    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    })?;
    watcher.watch(path, RecursiveMode::NonRecursive)?;
    Ok((watcher, rx))
}

fn handle_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<LiveKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(LiveKernel::compile(ctx, source_path));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good kernel:\n{err}");
        }
    }
    Ok(())
}

fn handle_mesh_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<MeshKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(MeshKernel::compile(ctx, source_path));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good mesh kernel:\n{err}");
        }
    }
    Ok(())
}

fn handle_instance_reload_events(
    ctx: &NeoContext,
    source_path: &Path,
    variant: InstanceStressVariant,
    rx: &Receiver<notify::Result<NotifyEvent>>,
    reload: &mut ReloadState<InstanceKernel>,
) -> Result<()> {
    let mut should_reload = false;
    for event in rx.try_iter() {
        let event = event?;
        if event.paths.iter().any(|path| path == source_path) {
            should_reload = true;
        }
    }

    if should_reload {
        let replaced = reload.try_replace(InstanceKernel::compile(ctx, source_path, variant));
        if replaced {
            println!("hot reload: compiled {}", source_path.display());
        } else if let Some(err) = &reload.last_error {
            eprintln!("hot reload failed; keeping last good instance kernel:\n{err}");
        }
    }
    Ok(())
}

struct FrameResources {
    width: u32,
    height: u32,
    device_pixels: DeviceBuffer<u8>,
    host_bgra: ReadablePinnedHostBuffer<u8>,
}

impl FrameResources {
    fn new(neo: &NeoContext, width: u32, height: u32) -> Result<Self> {
        let byte_len = frame_byte_len(width, height)?;
        Ok(Self {
            width,
            height,
            device_pixels: neo.alloc_zeros(byte_len)?,
            host_bgra: neo.alloc_readable_pinned(byte_len)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DemoVertex {
    position: [f32; 3],
    color_bgra: u32,
}

fn create_demo_mesh(neo: &NeoContext) -> Result<MeshBuffer> {
    let vertices = [
        DemoVertex {
            position: [-0.65, -0.65, 0.0],
            color_bgra: 0xffff_4040,
        },
        DemoVertex {
            position: [0.65, -0.65, 0.0],
            color_bgra: 0xff40_ff40,
        },
        DemoVertex {
            position: [0.65, 0.65, 0.0],
            color_bgra: 0xff40_40ff,
        },
        DemoVertex {
            position: [-0.65, 0.65, 0.0],
            color_bgra: 0xffff_ff40,
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    MeshBuffer::upload_typed(
        neo,
        MeshBufferDesc {
            vertex_count: vertices.len() as u32,
            vertex_layout: VertexLayout {
                stride: std::mem::size_of::<DemoVertex>() as u32,
                attributes: vec![
                    VertexAttribute {
                        semantic: VertexSemantic::Position,
                        format: VertexFormat::F32x3,
                        offset: 0,
                    },
                    VertexAttribute {
                        semantic: VertexSemantic::Color0,
                        format: VertexFormat::U8x4Unorm,
                        offset: 12,
                    },
                ],
            },
            index_format: IndexFormat::U16,
            index_count: indices.len() as u32,
            topology: PrimitiveTopology::TriangleList,
        },
        &vertices,
        &indices,
    )
    .context("failed to upload demo MeshBuffer")
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StressInstance {
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 2],
    color_bgra: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CameraParams {
    origin: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    grid: [u32; 4],
    view: [f32; 4],
}

struct InstanceStressAssets {
    mesh: MeshBuffer,
    instances: InstanceBuffer,
    camera_buffers: Vec<DeviceBuffer<u8>>,
    tile_cull_width: u32,
    tile_cull_height: u32,
    tile_cull_buffers: Vec<DeviceBuffer<u8>>,
}

fn create_instance_stress_assets(
    neo: &NeoContext,
    grid: InstanceGrid,
    present_ring: usize,
) -> Result<InstanceStressAssets> {
    let mesh = create_demo_mesh(neo)?;
    let instance_count = grid
        .count()
        .ok_or_else(|| anyhow!("instance grid count overflow"))?;
    let instances = create_stress_instances(grid)?;
    let instance_buffer = InstanceBuffer::upload_typed(
        neo,
        InstanceBufferDesc {
            instance_count,
            instance_layout: InstanceLayout {
                stride: std::mem::size_of::<StressInstance>() as u32,
                attributes: vec![
                    InstanceAttribute {
                        semantic: InstanceSemantic::Position,
                        format: InstanceFormat::F32x3,
                        offset: 0,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Rotation,
                        format: InstanceFormat::F32x4,
                        offset: 12,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Scale,
                        format: InstanceFormat::F32x2,
                        offset: 28,
                    },
                    InstanceAttribute {
                        semantic: InstanceSemantic::Color0,
                        format: InstanceFormat::U8x4Unorm,
                        offset: 36,
                    },
                ],
            },
        },
        &instances,
    )
    .context("failed to upload instance stress InstanceBuffer")?;
    let camera_len = std::mem::size_of::<CameraParams>();
    let mut camera_buffers = Vec::with_capacity(present_ring);
    for _ in 0..present_ring {
        camera_buffers.push(neo.alloc_zeros(camera_len)?);
    }
    Ok(InstanceStressAssets {
        mesh,
        instances: instance_buffer,
        camera_buffers,
        tile_cull_width: 0,
        tile_cull_height: 0,
        tile_cull_buffers: Vec::new(),
    })
}

fn tile_cull_grid_size(size: PhysicalSize<u32>) -> Result<(u32, u32)> {
    if size.width == 0 || size.height == 0 {
        bail!("tile cull size must be nonzero");
    }
    Ok((
        size.width.div_ceil(INSTANCE_CULL_TILE),
        size.height.div_ceil(INSTANCE_CULL_TILE),
    ))
}

fn tile_cull_byte_len(size: PhysicalSize<u32>) -> Result<usize> {
    let (tiles_x, tiles_y) = tile_cull_grid_size(size)?;
    tiles_x
        .checked_mul(tiles_y)
        .and_then(|tiles| tiles.checked_mul(TILE_CULL_RECORD_BYTES as u32))
        .map(|bytes| bytes as usize)
        .ok_or_else(|| {
            anyhow!(
                "tile cull buffer size overflow for {}x{}",
                size.width,
                size.height
            )
        })
}

fn ensure_tile_cull_buffers(
    neo: &NeoContext,
    assets: &mut InstanceStressAssets,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let (tiles_x, tiles_y) = tile_cull_grid_size(size)?;
    if assets.tile_cull_width == tiles_x
        && assets.tile_cull_height == tiles_y
        && assets.tile_cull_buffers.len() == present_ring
    {
        return Ok(());
    }
    let byte_len = tile_cull_byte_len(size)?;
    let mut buffers = Vec::with_capacity(present_ring);
    for _ in 0..present_ring {
        buffers.push(neo.alloc_zeros(byte_len)?);
    }
    assets.tile_cull_width = tiles_x;
    assets.tile_cull_height = tiles_y;
    assets.tile_cull_buffers = buffers;
    Ok(())
}

fn create_stress_instances(grid: InstanceGrid) -> Result<Vec<StressInstance>> {
    let count = grid
        .count()
        .ok_or_else(|| anyhow!("instance grid count overflow"))? as usize;
    let mut instances = Vec::with_capacity(count);
    let spacing = 0.085f32;
    let ox = (grid.x as f32 - 1.0) * 0.5;
    let oy = (grid.y as f32 - 1.0) * 0.5;
    let oz = (grid.z as f32 - 1.0) * 0.5;
    for z in 0..grid.z {
        for y in 0..grid.y {
            for x in 0..grid.x {
                let id = z * grid.x * grid.y + y * grid.x + x;
                let px = (x as f32 - ox) * spacing;
                let py = (y as f32 - oy) * spacing;
                let pz = (z as f32 - oz) * spacing;
                let yaw = (x as f32 * 0.17 + z as f32 * 0.11).sin() * 0.45;
                let pitch = (y as f32 * 0.13 + z as f32 * 0.07).cos() * 0.35;
                let rotation = quat_from_euler(pitch, yaw, 0.0);
                let hash = id.wrapping_mul(747_796_405).wrapping_add(0x9e37_79b9);
                let r = (hash >> 16) & 255;
                let g = (hash >> 8) & 255;
                let b = hash & 255;
                instances.push(StressInstance {
                    position: [px, py, pz],
                    rotation,
                    scale: [spacing * 0.32, spacing * 0.32],
                    color_bgra: b | (g << 8) | (r << 16) | 0xff00_0000,
                });
            }
        }
    }
    Ok(instances)
}

fn quat_from_euler(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    let (sx, cx) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let (sz, cz) = (roll * 0.5).sin_cos();
    [
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    ]
}

fn camera_params_bytes(params: &CameraParams) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (params as *const CameraParams).cast::<u8>(),
            std::mem::size_of::<CameraParams>(),
        )
    }
}

fn instance_render_visibility(
    policy: RenderPolicy,
    window: WindowVisibilityState,
    camera: &CameraParams,
) -> RenderVisibility {
    if policy == RenderPolicy::ForceRender {
        return RenderVisibility::Visible;
    }
    let window_visibility = window.render_visibility();
    if window_visibility != RenderVisibility::Visible {
        return window_visibility;
    }
    if instance_lattice_visible(camera) {
        RenderVisibility::Visible
    } else {
        RenderVisibility::Empty
    }
}

fn instance_lattice_visible(camera: &CameraParams) -> bool {
    let grid_x = camera.grid[0] as f32;
    let grid_y = camera.grid[1] as f32;
    let grid_z = camera.grid[2] as f32;
    if grid_x <= 0.0 || grid_y <= 0.0 || grid_z <= 0.0 {
        return false;
    }
    let spacing = camera.view[2];
    if spacing <= 0.0 {
        return false;
    }
    let pad = spacing * 0.5;
    let extent_x = (grid_x - 1.0) * 0.5 * spacing + pad;
    let extent_y = (grid_y - 1.0) * 0.5 * spacing + pad;
    let extent_z = (grid_z - 1.0) * 0.5 * spacing + pad;
    let forward = [camera.forward[0], camera.forward[1], camera.forward[2]];
    let right = [camera.right[0], camera.right[1], camera.right[2]];
    let up = [camera.up[0], camera.up[1], camera.up[2]];
    let origin = [camera.origin[0], camera.origin[1], camera.origin[2]];
    let tan_x = camera.view[0].max(0.001);
    let tan_y = camera.view[1].max(0.001);
    let edge_slack = spacing * 3.0;

    let mut any_in_front = false;
    let mut outside_left = true;
    let mut outside_right = true;
    let mut outside_bottom = true;
    let mut outside_top = true;
    for x in [-extent_x, extent_x] {
        for y in [-extent_y, extent_y] {
            for z in [-extent_z, extent_z] {
                let rel = [x - origin[0], y - origin[1], z - origin[2]];
                let camera_x = dot3(rel, right);
                let camera_y = dot3(rel, up);
                let camera_z = dot3(rel, forward);
                any_in_front |= camera_z > 0.0;
                outside_right &= camera_z * tan_x - camera_x + edge_slack < 0.0;
                outside_left &= camera_z * tan_x + camera_x + edge_slack < 0.0;
                outside_top &= camera_z * tan_y - camera_y + edge_slack < 0.0;
                outside_bottom &= camera_z * tan_y + camera_y + edge_slack < 0.0;
            }
        }
    }

    any_in_front && !(outside_left || outside_right || outside_bottom || outside_top)
}

#[derive(Default)]
struct CameraController {
    auto: bool,
    origin: [f32; 3],
    yaw: f32,
    pitch: f32,
    moving_forward: bool,
    moving_backward: bool,
    moving_left: bool,
    moving_right: bool,
    moving_up: bool,
    moving_down: bool,
    right_mouse_down: bool,
    raw_mouse_seen_during_capture: bool,
    last_cursor: Option<(f64, f64)>,
    last_manual_time: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraWindowAction {
    None,
    CaptureMouse,
    ReleaseMouse,
    RecenterMouse,
}

impl CameraController {
    fn new() -> Self {
        Self {
            auto: true,
            origin: [0.0, -14.0, 5.2],
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: -0.32,
            ..Self::default()
        }
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> CameraWindowAction {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    return self.handle_key_code(code, pressed);
                }
            }
            WindowEvent::Focused(false) => {
                if self.right_mouse_down {
                    self.end_mouse_look();
                    return CameraWindowAction::ReleaseMouse;
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                if *state == ElementState::Pressed {
                    self.begin_mouse_look();
                    return CameraWindowAction::CaptureMouse;
                } else {
                    self.end_mouse_look();
                    return CameraWindowAction::ReleaseMouse;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.right_mouse_down && !self.raw_mouse_seen_during_capture {
                    if let Some((last_x, last_y)) = self.last_cursor {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.apply_mouse_delta(dx, dy);
                        self.last_cursor = None;
                        return CameraWindowAction::RecenterMouse;
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            _ => {}
        }
        CameraWindowAction::None
    }

    fn handle_key_code(&mut self, code: KeyCode, pressed: bool) -> CameraWindowAction {
        if pressed && code == KeyCode::Escape {
            self.end_mouse_look();
            return CameraWindowAction::ReleaseMouse;
        }
        match code {
            KeyCode::KeyW => self.moving_forward = pressed,
            KeyCode::KeyS => self.moving_backward = pressed,
            KeyCode::KeyA => self.moving_left = pressed,
            KeyCode::KeyD => self.moving_right = pressed,
            KeyCode::Space => self.moving_up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.moving_down = pressed,
            _ => {}
        }
        if pressed {
            self.auto = false;
        }
        CameraWindowAction::None
    }

    fn begin_mouse_look(&mut self) {
        self.right_mouse_down = true;
        self.raw_mouse_seen_during_capture = false;
        self.last_cursor = None;
        self.auto = false;
    }

    fn end_mouse_look(&mut self) {
        self.right_mouse_down = false;
        self.raw_mouse_seen_during_capture = false;
        self.last_cursor = None;
    }

    fn handle_raw_mouse_delta(&mut self, dx: f32, dy: f32) {
        if self.right_mouse_down {
            self.raw_mouse_seen_during_capture = true;
            self.apply_mouse_delta(dx, dy);
        }
    }

    fn apply_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.003;
        self.pitch = (self.pitch - dy * 0.003).clamp(-1.35, 1.35);
    }

    fn params(&mut self, size: PhysicalSize<u32>, time: f32, grid: InstanceGrid) -> CameraParams {
        let aspect = size.width as f32 / size.height as f32;
        let tan_y = (60.0f32.to_radians() * 0.5).tan();
        let tan_x = tan_y * aspect;
        let origin = if self.auto {
            self.last_manual_time = None;
            let radius = 14.5;
            let angle = time * 0.23;
            [angle.cos() * radius, angle.sin() * radius, 5.3]
        } else {
            self.update_manual(time);
            self.origin
        };
        let forward = if self.auto {
            normalize3([-origin[0], -origin[1], -origin[2] * 0.72])
        } else {
            [
                self.yaw.cos() * self.pitch.cos(),
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
            ]
        };
        let world_up = [0.0, 0.0, 1.0];
        let right = normalize3(cross3(forward, world_up));
        let up = normalize3(cross3(right, forward));
        CameraParams {
            origin: [origin[0], origin[1], origin[2], 0.0],
            right: [right[0], right[1], right[2], 0.0],
            up: [up[0], up[1], up[2], 0.0],
            forward: [forward[0], forward[1], forward[2], 0.0],
            grid: [grid.x, grid.y, grid.z, grid.count().unwrap_or(0)],
            view: [tan_x, tan_y, 0.085, 0.0],
        }
    }

    fn update_manual(&mut self, time: f32) {
        let dt = self
            .last_manual_time
            .replace(time)
            .map(|last| (time - last).clamp(0.0, CAMERA_MAX_STEP_SECONDS))
            .unwrap_or(0.0);
        let step = CAMERA_MOVE_UNITS_PER_SEC * dt;
        let forward = [
            self.yaw.cos() * self.pitch.cos(),
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
        ];
        let right = normalize3(cross3(forward, [0.0, 0.0, 1.0]));
        let mut delta = [0.0, 0.0, 0.0];
        if self.moving_forward {
            add_scaled3(&mut delta, forward, step);
        }
        if self.moving_backward {
            add_scaled3(&mut delta, forward, -step);
        }
        if self.moving_right {
            add_scaled3(&mut delta, right, step);
        }
        if self.moving_left {
            add_scaled3(&mut delta, right, -step);
        }
        if self.moving_up {
            delta[2] += step;
        }
        if self.moving_down {
            delta[2] -= step;
        }
        self.origin[0] += delta[0];
        self.origin[1] += delta[1];
        self.origin[2] += delta[2];
    }
}

fn add_scaled3(dst: &mut [f32; 3], value: [f32; 3], scale: f32) {
    dst[0] += value[0] * scale;
    dst[1] += value[1] * scale;
    dst[2] += value[2] * scale;
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [1.0, 0.0, 0.0]
    }
}

struct ThroughputResources {
    width: u32,
    height: u32,
    slots: Vec<ThroughputSlot>,
}

struct ThroughputSlot {
    stream: CudaStream,
    device_pixels: DeviceBuffer<u8>,
    graph_params_device: DeviceBuffer<u8>,
    graph_params_host: ReadablePinnedHostBuffer<u8>,
    graph: Option<CudaGraph>,
    host_bgra: ReadablePinnedHostBuffer<u8>,
    ready: CudaFence,
    state: ThroughputSlotState,
    frame: u32,
    completed_kernels: u32,
    sample_started_at: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThroughputSlotState {
    Free,
    Pending { readback: bool },
}

impl ThroughputResources {
    fn new(neo: &NeoContext, width: u32, height: u32, slots: usize) -> Result<Self> {
        let byte_len = frame_byte_len(width, height)?;
        let mut ring = Vec::with_capacity(slots);
        for _ in 0..slots {
            let stream = neo.create_stream()?;
            ring.push(ThroughputSlot {
                device_pixels: DeviceBuffer::new_on_stream(&stream, byte_len)?,
                graph_params_device: DeviceBuffer::new_on_stream(
                    &stream,
                    std::mem::size_of::<LiveGraphParams>(),
                )?,
                graph_params_host: neo
                    .alloc_readable_pinned(std::mem::size_of::<LiveGraphParams>())?,
                graph: None,
                host_bgra: neo.alloc_readable_pinned(byte_len)?,
                ready: stream.create_fence()?,
                stream,
                state: ThroughputSlotState::Free,
                frame: 0,
                completed_kernels: 0,
                sample_started_at: None,
            });
        }
        Ok(Self {
            width,
            height,
            slots: ring,
        })
    }

    fn newest_completed_slot_index(&self) -> Result<Option<usize>> {
        let mut completed = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            if matches!(slot.state, ThroughputSlotState::Pending { readback: true })
                && slot.ready.is_complete()?
            {
                completed.push((index, slot.frame));
            }
        }
        Ok(choose_newest_completed_slot(completed))
    }

    fn drain_completed(
        &mut self,
        presenter: &mut PresentSink,
        size: PhysicalSize<u32>,
    ) -> Result<CompletedPresentation> {
        let present_index = self.newest_completed_slot_index()?;
        let mut stats = CompletedPresentation::default();

        for index in 0..self.slots.len() {
            let state = self.slots[index].state;
            if !matches!(state, ThroughputSlotState::Pending { .. })
                || !self.slots[index].ready.is_complete()?
            {
                continue;
            }

            stats.completed_kernels += self.slots[index].completed_kernels;
            if Some(index) == present_index {
                let slot = &mut self.slots[index];
                stats.sample_download += slot
                    .sample_started_at
                    .map(|started| started.elapsed())
                    .unwrap_or_default();
                let timings = presenter.present_sampled(
                    size,
                    slot.host_bgra.as_slice(),
                    stats.sample_download,
                )?;
                if let Some(timings) = timings {
                    stats.present = timings;
                    stats.sampled_frames = 1;
                    stats.presented_frames = 1;
                }
            }

            self.slots[index].state = ThroughputSlotState::Free;
            self.slots[index].completed_kernels = 0;
            self.slots[index].sample_started_at = None;
        }

        Ok(stats)
    }

    fn launch_graph_slot(&mut self, index: usize, input: GraphSlotLaunch<'_>) -> Result<()> {
        let slot = &mut self.slots[index];
        write_live_graph_params(
            &mut slot.graph_params_host,
            LiveGraphParams {
                time: input.time,
                frame: input.frame,
            },
        );
        if slot.graph.is_none() {
            let graph_kernel = input.kernel.on_stream(&slot.stream);
            slot.stream
                .synchronize()
                .context("failed to synchronize stream before CUDA graph capture")?;
            slot.stream
                .begin_graph_capture()
                .context("failed to begin CUDA graph capture")?;
            slot.graph_params_device
                .upload_from_readable_pinned_on_stream(&slot.stream, &slot.graph_params_host)
                .context("failed to capture CUDA graph parameter upload")?;
            {
                let mut launch = graph_kernel.launcher();
                launch
                    .arg_buffer_mut(&mut slot.device_pixels)
                    .arg(&input.width)
                    .arg(&input.height)
                    .arg_buffer(&slot.graph_params_device);
                unsafe {
                    launch
                        .launch(input.dims)
                        .context("failed to capture CUDA graph kernel launch")?;
                }
            }
            slot.graph = Some(
                slot.stream
                    .end_graph_capture()?
                    .ok_or_else(|| anyhow!("CUDA graph capture did not produce a graph"))
                    .context("failed to end CUDA graph capture")?,
            );
        }
        slot.graph
            .as_ref()
            .expect("graph was just initialized")
            .launch()
            .context("failed to launch CUDA graph")?;
        if input.readback {
            slot.device_pixels
                .download_into_readable_pinned_on_stream(&slot.stream, &mut slot.host_bgra)?;
            slot.sample_started_at = Some(Instant::now());
        } else {
            slot.sample_started_at = None;
        }
        slot.ready.record_on_stream(&slot.stream)?;
        slot.state = ThroughputSlotState::Pending {
            readback: input.readback,
        };
        slot.frame = input.frame;
        slot.completed_kernels = 1;
        Ok(())
    }
}

struct GraphSlotLaunch<'a> {
    kernel: &'a Kernel,
    dims: LaunchDims,
    width: u32,
    height: u32,
    time: f32,
    frame: u32,
    readback: bool,
}

struct InteropThroughputResources {
    width: u32,
    height: u32,
    ring: SharedFrameRing,
    slots: Vec<InteropThroughputSlot>,
}

struct InteropThroughputSlot {
    stream: CudaStream,
    state: InteropSlotState,
    frame: u32,
    cuda_done_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InteropSlotState {
    Free,
    Pending,
    Completed,
}

impl InteropThroughputResources {
    fn new(
        neo: &NeoContext,
        interop: &NeoD3d12InteropDevice,
        width: u32,
        height: u32,
        slots: usize,
    ) -> Result<Self> {
        interop_trace("create shared frame ring");
        let ring = interop
            .create_shared_frame_ring(width, height, slots)
            .context("failed to create D3D12/CUDA shared frame ring")?;
        interop_trace("created shared frame ring");
        let mut aux = Vec::with_capacity(slots);
        for _ in 0..slots {
            aux.push(InteropThroughputSlot {
                stream: neo.create_stream()?,
                state: InteropSlotState::Free,
                frame: 0,
                cuda_done_value: 0,
            });
        }
        Ok(Self {
            width,
            height,
            ring,
            slots: aux,
        })
    }

    fn has_live_work(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.state != InteropSlotState::Free)
    }

    fn drain_completed(
        &mut self,
        presenter: &mut WindowPresenter,
        size: PhysicalSize<u32>,
        present_limiter: &mut PresentRateLimiter,
    ) -> Result<ThroughputBatchStats> {
        let mut stats = ThroughputBatchStats::default();
        for index in 0..self.slots.len() {
            if self.slots[index].state != InteropSlotState::Pending {
                continue;
            }
            let complete = self
                .ring
                .slot(index)
                .is_some_and(|shared| shared.is_fence_complete(self.slots[index].cuda_done_value));
            if !complete {
                continue;
            }
            stats.completed_kernels += 1;
            self.slots[index].state = InteropSlotState::Completed;
        }

        let newest = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == InteropSlotState::Completed)
            .max_by_key(|(_, slot)| slot.frame)
            .map(|(index, _)| index);

        if newest.is_some() && present_limiter.try_consume(Instant::now()) {
            if let Some(index) = newest {
                self.present_completed_slot(index, presenter, size, &mut stats)?;
            }
        }

        let newest_to_retain = if stats.presented_frames > 0 {
            None
        } else {
            newest
        };
        for index in 0..self.slots.len() {
            if self.slots[index].state == InteropSlotState::Completed
                && Some(index) != newest_to_retain
            {
                self.slots[index].state = InteropSlotState::Free;
            }
        }
        Ok(stats)
    }

    fn present_completed_slot(
        &mut self,
        index: usize,
        presenter: &mut WindowPresenter,
        size: PhysicalSize<u32>,
        stats: &mut ThroughputBatchStats,
    ) -> Result<()> {
        interop_trace("present shared frame");
        let timings = presenter.present_shared(
            size,
            self.ring.pitch_bytes(),
            self.ring
                .slot_mut(index)
                .context("missing shared interop slot")?,
            self.slots[index].cuda_done_value,
        )?;
        stats.present += timings.total;
        stats.draw += timings.draw;
        stats.swap_present += timings.swap_present;
        stats.sampled_frames += 1;
        stats.presented_frames += 1;
        self.slots[index].state = InteropSlotState::Free;
        interop_trace("presented shared frame");
        Ok(())
    }
}

fn ensure_interop_throughput_resources(
    neo: &NeoContext,
    interop: &NeoD3d12InteropDevice,
    resources: &mut Option<InteropThroughputResources>,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let recreate = resources.as_ref().is_none_or(|resources| {
        resources.width != size.width
            || resources.height != size.height
            || resources.ring.len() != present_ring
    });
    if recreate {
        if let Some(resources) = resources.as_mut() {
            resources.ring.wait_idle(interop.queue())?;
        }
        *resources = Some(InteropThroughputResources::new(
            neo,
            interop,
            size.width,
            size.height,
            present_ring,
        )?);
    }
    Ok(())
}

struct InteropThroughputBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a Kernel,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_interop_throughput_batch(input: InteropThroughputBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, BLOCK);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("d3d12-interop does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        interop_trace("cuda wait slot available");
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        interop_trace("launch kernel into shared frame");
        unsafe {
            input.kernel.on_stream(stream).launch_image_raw_ptr(
                dims,
                shared.device_ptr_arg(),
                kernel_width,
                input.size.height,
                input.start.elapsed().as_secs_f32(),
                *input.next_frame,
            )
        }
        .context("failed to launch Neo kernel into D3D12 shared frame")?;
        interop_trace("cuda signal frame complete");
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = *input.next_frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

struct MeshDemoBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a Kernel,
    mesh: &'a MeshBuffer,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_mesh_demo_batch(input: MeshDemoBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, BLOCK);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("mesh-demo does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        let pixel_arg = shared.device_ptr_arg();
        let stream_kernel = input.kernel.on_stream(stream);
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        {
            let mut launch = stream_kernel.launcher();
            launch
                .arg_device_ptr(&pixel_arg)
                .arg(&kernel_width)
                .arg(&input.size.height)
                .arg_mesh(input.mesh)
                .arg(&time)
                .arg(&frame);
            unsafe {
                launch
                    .launch(dims)
                    .context("failed to launch mesh demo raster kernel")?;
            }
        }
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

struct InstanceStressBatch<'a> {
    neo: &'a NeoContext,
    interop: &'a NeoD3d12InteropDevice,
    resources: &'a mut Option<InteropThroughputResources>,
    kernel: &'a InstanceKernel,
    assets: &'a mut InstanceStressAssets,
    camera: CameraParams,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    present_limiter: &'a mut PresentRateLimiter,
    max_inflight: u32,
    present_ring: usize,
}

fn run_instance_stress_batch(input: InstanceStressBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_interop_throughput_resources(
        input.neo,
        input.interop,
        input.resources,
        input.size,
        input.present_ring,
    )?;
    let resources = input
        .resources
        .as_mut()
        .expect("interop resources were just created for nonzero size");
    let kernel_width = resources.ring.kernel_width();
    if input.kernel.cull.is_some() {
        ensure_tile_cull_buffers(
            input.neo,
            input.assets,
            PhysicalSize::new(kernel_width, input.size.height),
            resources.slots.len(),
        )?;
    }
    let raster_block = if input.kernel.tiled { (8, 8) } else { BLOCK };
    let dims = LaunchDims::for_2d(kernel_width, input.size.height, raster_block);
    let mut stats = ThroughputBatchStats::default();
    let direct_presenter = match input.presenter {
        PresentSink::Direct(presenter) => presenter,
        PresentSink::Threaded(_) => bail!("instance-stress does not use the present thread"),
    };

    stats += resources.drain_completed(direct_presenter, input.size, input.present_limiter)?;
    if stats.completed_kernels > 0 {
        *input.completed_kernels += u64::from(stats.completed_kernels);
    }
    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let camera_bytes = camera_params_bytes(&input.camera);
    let mut launched = 0u32;
    for index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[index].state != InteropSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let shared = resources
            .ring
            .slot_mut(index)
            .context("missing shared interop slot")?;
        let stream = &resources.slots[index].stream;
        shared
            .wait_available_on_stream(stream)
            .context("failed to make CUDA wait for D3D12 shared frame availability")?;
        input.assets.camera_buffers[index]
            .upload_from_on_stream(stream, camera_bytes)
            .context("failed to upload instance stress camera params")?;
        let pixel_arg = shared.device_ptr_arg();
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        if let Some(cull_kernel) = &input.kernel.cull {
            let (tiles_x, tiles_y) =
                tile_cull_grid_size(PhysicalSize::new(kernel_width, input.size.height))?;
            let cull_dims = LaunchDims::for_2d(tiles_x, tiles_y, BLOCK);
            let stream_kernel = cull_kernel.on_stream(stream);
            let mut launch = stream_kernel.launcher();
            launch
                .arg_buffer(&input.assets.tile_cull_buffers[index])
                .arg(&kernel_width)
                .arg(&input.size.height)
                .arg_mesh(&input.assets.mesh)
                .arg_instances(&input.assets.instances)
                .arg_buffer(&input.assets.camera_buffers[index])
                .arg(&time)
                .arg(&frame);
            unsafe {
                launch
                    .launch(cull_dims)
                    .context("failed to launch instance stress cull kernel")?;
            }
        }
        let stream_kernel = input.kernel.raster.on_stream(stream);
        {
            let mut launch = stream_kernel.launcher();
            launch
                .arg_device_ptr(&pixel_arg)
                .arg(&kernel_width)
                .arg(&input.size.height)
                .arg_mesh(&input.assets.mesh)
                .arg_instances(&input.assets.instances)
                .arg_buffer(&input.assets.camera_buffers[index]);
            if input.kernel.cull.is_some() {
                launch.arg_buffer(&input.assets.tile_cull_buffers[index]);
            }
            launch.arg(&time).arg(&frame);
            unsafe {
                launch
                    .launch(dims)
                    .context("failed to launch instance stress raster kernel")?;
            }
        }
        let cuda_done_value = shared
            .signal_cuda_complete_on_stream(stream)
            .context("failed to signal D3D12 shared frame completion from CUDA")?;
        resources.slots[index].state = InteropSlotState::Pending;
        resources.slots[index].frame = frame;
        resources.slots[index].cuda_done_value = cuda_done_value;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }
    Ok(stats)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LiveGraphParams {
    time: f32,
    frame: u32,
}

fn write_live_graph_params(dst: &mut ReadablePinnedHostBuffer<u8>, params: LiveGraphParams) {
    debug_assert_eq!(dst.len(), std::mem::size_of::<LiveGraphParams>());
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&params as *const LiveGraphParams).cast::<u8>(),
            std::mem::size_of::<LiveGraphParams>(),
        )
    };
    dst.as_mut_slice().copy_from_slice(bytes);
}

#[derive(Default)]
struct CompletedPresentation {
    completed_kernels: u32,
    sample_download: Duration,
    present: PresentTimings,
    sampled_frames: u32,
    presented_frames: u32,
}

fn frame_byte_len(width: u32, height: u32) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .map(|bytes| bytes as usize)
        .ok_or_else(|| anyhow!("framebuffer size overflow for {width}x{height}"))
}

fn ensure_frame_resources(
    neo: &NeoContext,
    resources: &mut Option<FrameResources>,
    size: PhysicalSize<u32>,
) -> Result<()> {
    let recreate = resources
        .as_ref()
        .is_none_or(|resources| resources.width != size.width || resources.height != size.height);
    if recreate {
        *resources = Some(FrameResources::new(neo, size.width, size.height)?);
    }
    Ok(())
}

fn ensure_throughput_resources(
    neo: &NeoContext,
    resources: &mut Option<ThroughputResources>,
    size: PhysicalSize<u32>,
    present_ring: usize,
) -> Result<()> {
    let recreate = resources.as_ref().is_none_or(|resources| {
        resources.width != size.width
            || resources.height != size.height
            || resources.slots.len() != present_ring
    });
    if recreate {
        *resources = Some(ThroughputResources::new(
            neo,
            size.width,
            size.height,
            present_ring,
        )?);
    }
    Ok(())
}

unsafe fn copy_bgra_rows(src: *const u8, dst: *mut u8, width: u32, height: u32, dst_pitch: usize) {
    let src_pitch = width as usize * 4;
    for y in 0..height as usize {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(y * src_pitch),
                dst.add(y * dst_pitch),
                src_pitch,
            );
        }
    }
}

fn choose_newest_completed_slot(slots: impl IntoIterator<Item = (usize, u32)>) -> Option<usize> {
    slots
        .into_iter()
        .max_by_key(|(_, frame)| *frame)
        .map(|(index, _)| index)
}

unsafe fn copy_bgra_to_mapped(
    src: &[u8],
    dst: *mut u8,
    width: u32,
    height: u32,
    dst_pitch: usize,
) -> bool {
    let src_pitch = width as usize * 4;
    if dst_pitch == src_pitch {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        true
    } else {
        unsafe {
            copy_bgra_rows(src.as_ptr(), dst, width, height, dst_pitch);
        }
        false
    }
}

struct RenderFrame<'a> {
    neo: &'a NeoContext,
    resources: &'a mut Option<FrameResources>,
    kernel: &'a Kernel,
    size: PhysicalSize<u32>,
    time: f32,
    frame: u32,
}

fn render_frame(input: RenderFrame<'_>) -> Result<FrameTimings> {
    ensure_frame_resources(input.neo, input.resources, input.size)?;
    let resources = input
        .resources
        .as_mut()
        .expect("frame resources were just created for nonzero size");
    let width = input.size.width;
    let height = input.size.height;
    let dims = LaunchDims::for_2d(width, height, BLOCK);
    let launch_start = Instant::now();
    {
        let mut launch = input.kernel.launcher();
        launch
            .arg_buffer_mut(&mut resources.device_pixels)
            .arg(&width)
            .arg(&height)
            .arg(&input.time)
            .arg(&input.frame);
        unsafe {
            launch.launch(dims)?;
        }
    }
    let download_start = Instant::now();
    resources
        .device_pixels
        .download_into_readable_pinned(&mut resources.host_bgra)?;
    let sync_start = Instant::now();
    input.neo.synchronize()?;
    Ok(FrameTimings {
        launch: download_start - launch_start,
        download: sync_start - download_start,
        sync: sync_start.elapsed(),
        ..FrameTimings::default()
    })
}

struct ThroughputBatch<'a> {
    neo: &'a NeoContext,
    resources: &'a mut Option<ThroughputResources>,
    graph_kernel: &'a Kernel,
    presenter: &'a mut PresentSink,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    next_sample_at: &'a mut u64,
    next_present_at: &'a mut Option<Instant>,
    sample_every: u32,
    present_interval: Option<Duration>,
    max_inflight: u32,
    present_ring: usize,
}

fn run_kernel_throughput_batch(input: ThroughputBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_throughput_resources(input.neo, input.resources, input.size, input.present_ring)?;
    let resources = input
        .resources
        .as_mut()
        .expect("frame resources were just created for nonzero size");
    let width = input.size.width;
    let height = input.size.height;
    let dims = LaunchDims::for_2d(width, height, BLOCK);
    let mut stats = ThroughputBatchStats::default();

    stats += input.presenter.drain_results()?;
    let completed = resources.drain_completed(input.presenter, input.size)?;
    if completed.completed_kernels > 0 {
        stats.completed_kernels += completed.completed_kernels;
        *input.completed_kernels += u64::from(completed.completed_kernels);
    }
    if completed.presented_frames > 0 {
        stats.sample_download += completed.sample_download;
        stats.present += completed.present.total;
        stats.map_copy += completed.present.map_copy;
        stats.draw += completed.present.draw;
        stats.swap_present += completed.present.swap_present;
        stats.sampled_frames += completed.sampled_frames;
        stats.presented_frames += completed.presented_frames;
        stats.sampled_bytes += frame_byte_len(width, height)?;
        stats.uploaded_bytes += frame_byte_len(width, height)?;
    }

    if input.max_inflight == 0 {
        return Ok(stats);
    }

    let should_sample_by_count =
        input.present_interval.is_none() && *input.completed_kernels >= *input.next_sample_at;
    let should_sample_by_time = input
        .next_present_at
        .is_some_and(|next_present_at| Instant::now() >= next_present_at);
    let mut should_readback = should_sample_by_count || should_sample_by_time;
    let mut launched = 0u32;
    for launch_slot_index in 0..resources.slots.len() {
        if launched >= input.max_inflight {
            break;
        }
        if resources.slots[launch_slot_index].state != ThroughputSlotState::Free {
            continue;
        }
        let launch_start = Instant::now();
        let time = input.start.elapsed().as_secs_f32();
        let frame = *input.next_frame;
        resources
            .launch_graph_slot(
                launch_slot_index,
                GraphSlotLaunch {
                    kernel: input.graph_kernel,
                    dims,
                    width,
                    height,
                    time,
                    frame,
                    readback: should_readback,
                },
            )
            .context("failed to submit CUDA graph slot")?;
        stats.launch += launch_start.elapsed();
        launched += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
        should_readback = false;
    }

    if launched == 0 {
        return Ok(stats);
    }

    let wait_start = Instant::now();
    stats.completion_wait += wait_start.elapsed();

    if should_sample_by_count || should_sample_by_time {
        if let Some(interval) = input.present_interval {
            let now = Instant::now();
            let next_present_at = input.next_present_at.get_or_insert(now + interval);
            while *next_present_at <= now {
                *next_present_at += interval;
            }
        } else {
            while *input.completed_kernels >= *input.next_sample_at {
                *input.next_sample_at += u64::from(input.sample_every);
            }
        }
    }

    Ok(stats)
}

#[derive(Clone, Copy, Debug, Default)]
struct ThroughputBatchStats {
    completed_kernels: u32,
    sampled_frames: u32,
    presented_frames: u32,
    sampled_bytes: usize,
    uploaded_bytes: usize,
    launch: Duration,
    completion_wait: Duration,
    sample_download: Duration,
    present: Duration,
    map_copy: Duration,
    draw: Duration,
    swap_present: Duration,
}

impl std::ops::AddAssign for ThroughputBatchStats {
    fn add_assign(&mut self, rhs: Self) {
        self.completed_kernels += rhs.completed_kernels;
        self.sampled_frames += rhs.sampled_frames;
        self.presented_frames += rhs.presented_frames;
        self.sampled_bytes += rhs.sampled_bytes;
        self.uploaded_bytes += rhs.uploaded_bytes;
        self.launch += rhs.launch;
        self.completion_wait += rhs.completion_wait;
        self.sample_download += rhs.sample_download;
        self.present += rhs.present;
        self.map_copy += rhs.map_copy;
        self.draw += rhs.draw;
        self.swap_present += rhs.swap_present;
    }
}

struct KernelRateLimiter {
    target_fps: Option<f64>,
    tokens: f64,
    max_burst: f64,
    last_refill: Instant,
}

impl KernelRateLimiter {
    fn new(target_fps: Option<f32>, max_inflight: u32, now: Instant) -> Self {
        Self {
            target_fps: target_fps.map(f64::from),
            tokens: 0.0,
            max_burst: f64::from(max_inflight.max(1).saturating_mul(4)),
            last_refill: now,
        }
    }

    fn grant(&mut self, now: Instant, max_inflight: u32) -> u32 {
        let granted = self.available(now, max_inflight);
        self.consume(granted);
        granted
    }

    fn available(&mut self, now: Instant, max_inflight: u32) -> u32 {
        if self.target_fps.is_none() {
            return max_inflight;
        }
        self.refill(now);
        self.tokens.floor().min(f64::from(max_inflight)) as u32
    }

    fn consume(&mut self, granted: u32) {
        if self.target_fps.is_some() {
            self.tokens -= f64::from(granted);
        }
    }

    fn next_token_at(&self, now: Instant) -> Option<Instant> {
        let target_fps = self.target_fps?;
        if self.tokens >= 1.0 {
            return Some(now);
        }
        let seconds_until_token = (1.0 - self.tokens) / target_fps;
        Some(now + Duration::from_secs_f64(seconds_until_token))
    }

    fn refill(&mut self, now: Instant) {
        let Some(target_fps) = self.target_fps else {
            return;
        };
        let elapsed = now.saturating_duration_since(self.last_refill);
        self.tokens = (self.tokens + elapsed.as_secs_f64() * target_fps).min(self.max_burst);
        self.last_refill = now;
    }
}

struct PresentRateLimiter {
    target_fps: Option<f64>,
    tokens: f64,
    last_refill: Instant,
}

impl PresentRateLimiter {
    fn new(target_fps: Option<f32>, now: Instant) -> Self {
        Self {
            target_fps: target_fps.map(f64::from),
            tokens: 0.0,
            last_refill: now,
        }
    }

    fn try_consume(&mut self, now: Instant) -> bool {
        let Some(target_fps) = self.target_fps else {
            return true;
        };
        let elapsed = now.saturating_duration_since(self.last_refill);
        self.tokens = (self.tokens + elapsed.as_secs_f64() * target_fps).min(1.0);
        self.last_refill = now;
        if self.tokens + 1.0e-9 >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

struct ThroughputCounter {
    last_log: Instant,
    completed_since_log: u64,
    sampled_since_log: u64,
    presented_since_log: u64,
    total_completed: u64,
    total_sampled: u64,
    total_presented: u64,
    sampled_bytes_since_log: u64,
    uploaded_bytes_since_log: u64,
    launch_accum: Duration,
    completion_wait_accum: Duration,
    sample_download_accum: Duration,
    present_accum: Duration,
    map_copy_accum: Duration,
    draw_accum: Duration,
    swap_present_accum: Duration,
}

struct ThroughputLogContext<'a> {
    size: PhysicalSize<u32>,
    frame: u32,
    presenter: PresenterKind,
    reload_error: Option<&'a str>,
    kernel_cap: Option<f32>,
    instance_variant: Option<InstanceStressVariant>,
    render_policy: RenderPolicy,
    visibility: RenderVisibility,
}

impl ThroughputCounter {
    fn new() -> Self {
        Self {
            last_log: Instant::now(),
            completed_since_log: 0,
            sampled_since_log: 0,
            presented_since_log: 0,
            total_completed: 0,
            total_sampled: 0,
            total_presented: 0,
            sampled_bytes_since_log: 0,
            uploaded_bytes_since_log: 0,
            launch_accum: Duration::ZERO,
            completion_wait_accum: Duration::ZERO,
            sample_download_accum: Duration::ZERO,
            present_accum: Duration::ZERO,
            map_copy_accum: Duration::ZERO,
            draw_accum: Duration::ZERO,
            swap_present_accum: Duration::ZERO,
        }
    }

    fn record(&mut self, batch: ThroughputBatchStats) {
        let completed = u64::from(batch.completed_kernels);
        let sampled = u64::from(batch.sampled_frames);
        let presented = u64::from(batch.presented_frames);
        self.completed_since_log += completed;
        self.sampled_since_log += sampled;
        self.presented_since_log += presented;
        self.total_completed += completed;
        self.total_sampled += sampled;
        self.total_presented += presented;
        self.sampled_bytes_since_log += batch.sampled_bytes as u64;
        self.uploaded_bytes_since_log += batch.uploaded_bytes as u64;
        self.launch_accum += batch.launch;
        self.completion_wait_accum += batch.completion_wait;
        self.sample_download_accum += batch.sample_download;
        self.present_accum += batch.present;
        self.map_copy_accum += batch.map_copy;
        self.draw_accum += batch.draw;
        self.swap_present_accum += batch.swap_present;
    }

    fn log_if_due(&mut self, context: ThroughputLogContext<'_>) {
        let elapsed = self.last_log.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let secs = elapsed.as_secs_f64();
        let kernel_fps = self.completed_since_log as f64 / secs;
        let sample_fps = self.sampled_since_log as f64 / secs;
        let present_fps = self.presented_since_log as f64 / secs;
        let completed = self.completed_since_log.max(1) as f64;
        let launch_us = self.launch_accum.as_secs_f64() * 1_000_000.0 / completed;
        let wait_us = self.completion_wait_accum.as_secs_f64() * 1_000_000.0 / completed;
        let sample_us = if self.sampled_since_log == 0 {
            0.0
        } else {
            self.sample_download_accum.as_secs_f64() * 1_000_000.0 / self.sampled_since_log as f64
        };
        let present_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.present_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let map_copy_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.map_copy_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let draw_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.draw_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let swap_us = if self.presented_since_log == 0 {
            0.0
        } else {
            self.swap_present_accum.as_secs_f64() * 1_000_000.0 / self.presented_since_log as f64
        };
        let mb_frame = frame_byte_len(context.size.width, context.size.height)
            .map(|bytes| bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let dtoh_gbps = self.sampled_bytes_since_log as f64 / secs / 1_000_000_000.0;
        let upload_gbps = self.uploaded_bytes_since_log as f64 / secs / 1_000_000_000.0;
        let reload_state = if context.reload_error.is_some() {
            "last-good"
        } else {
            "current"
        };
        let kernel_cap = context
            .kernel_cap
            .map(|fps| format!("{fps:.1} fps"))
            .unwrap_or_else(|| "uncapped".to_string());
        let interop_marker = if context.presenter == PresenterKind::D3d12Interop {
            " | interop d3d12-external-memory"
        } else {
            ""
        };
        let variant_marker = context
            .instance_variant
            .map(|variant| format!(" | instance_variant {variant}"))
            .unwrap_or_default();
        let presenter = context.presenter;
        let render_policy = context.render_policy;
        let visibility = context.visibility;
        let frame = context.frame;
        println!(
            "kernel_fps {kernel_fps:>9.1} | sample_fps {sample_fps:>6.1} | present_fps {present_fps:>6.1} | completed {:>10} | frame {frame:>8} | {}x{} | {mb_frame:>5.1} MB/frame | dtoh {dtoh_gbps:>5.1} GB/s {sample_us:>6.1} us | upload {upload_gbps:>5.1} GB/s map_copy {map_copy_us:>6.1} us | gpu_copy {draw_us:>6.1} us | swap {swap_us:>6.1} us | present {present_us:>6.1} us | launch {launch_us:>5.1} us/k | wait {wait_us:>5.1} us/k | presenter {presenter} | kernel_cap {kernel_cap} | render_policy {render_policy} | visibility {visibility} | kernel {reload_state}{interop_marker}{variant_marker}",
            self.total_completed, context.size.width, context.size.height
        );
        self.completed_since_log = 0;
        self.sampled_since_log = 0;
        self.presented_since_log = 0;
        self.sampled_bytes_since_log = 0;
        self.uploaded_bytes_since_log = 0;
        self.launch_accum = Duration::ZERO;
        self.completion_wait_accum = Duration::ZERO;
        self.sample_download_accum = Duration::ZERO;
        self.present_accum = Duration::ZERO;
        self.map_copy_accum = Duration::ZERO;
        self.draw_accum = Duration::ZERO;
        self.swap_present_accum = Duration::ZERO;
        self.last_log = Instant::now();
    }
}

struct FpsCounter {
    last_log: Instant,
    frames_since_log: u32,
    launch_accum: Duration,
    download_accum: Duration,
    sync_accum: Duration,
    render_accum: Duration,
    present_accum: Duration,
    total_accum: Duration,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last_log: Instant::now(),
            frames_since_log: 0,
            launch_accum: Duration::ZERO,
            download_accum: Duration::ZERO,
            sync_accum: Duration::ZERO,
            render_accum: Duration::ZERO,
            present_accum: Duration::ZERO,
            total_accum: Duration::ZERO,
        }
    }

    fn tick(
        &mut self,
        size: PhysicalSize<u32>,
        frame: u32,
        reload_error: Option<&str>,
        presenter: PresenterKind,
        timings: FrameTimings,
    ) {
        self.frames_since_log += 1;
        self.launch_accum += timings.launch;
        self.download_accum += timings.download;
        self.sync_accum += timings.sync;
        self.render_accum += timings.render;
        self.present_accum += timings.present;
        self.total_accum += timings.total;
        let elapsed = self.last_log.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let fps = self.frames_since_log as f32 / elapsed.as_secs_f32();
        let frame_count = self.frames_since_log as f64;
        let launch_us = self.launch_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let download_us = self.download_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let sync_us = self.sync_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let render_us = self.render_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let present_us = self.present_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let total_us = self.total_accum.as_secs_f64() * 1_000_000.0 / frame_count;
        let reload_state = if reload_error.is_some() {
            "last-good"
        } else {
            "current"
        };
        println!(
            "fps {fps:>7.1} | frame {frame:>8} | {}x{} | launch {launch_us:>5.1} us | dtoh {download_us:>5.1} us | sync {sync_us:>5.1} us | render {render_us:>6.1} us | present {present_us:>6.1} us | total {total_us:>6.1} us | presenter {presenter} | kernel {reload_state}",
            size.width, size.height
        );
        self.frames_since_log = 0;
        self.launch_accum = Duration::ZERO;
        self.download_accum = Duration::ZERO;
        self.sync_accum = Duration::ZERO;
        self.render_accum = Duration::ZERO;
        self.present_accum = Duration::ZERO;
        self.total_accum = Duration::ZERO;
        self.last_log = Instant::now();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameTimings {
    launch: Duration,
    download: Duration,
    sync: Duration,
    render: Duration,
    present: Duration,
    total: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct PresentTimings {
    map_copy: Duration,
    draw: Duration,
    swap_present: Duration,
    total: Duration,
}

enum PresentSink {
    Direct(WindowPresenter),
    Threaded(ThreadedPresenter),
}

impl PresentSink {
    fn threaded(presenter: WindowPresenter) -> Self {
        Self::Threaded(ThreadedPresenter::new(presenter))
    }

    fn present_sync(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
        match self {
            Self::Direct(presenter) => presenter.present(size, bgra),
            Self::Threaded(_) => {
                bail!("synchronous presentation is not available on present thread")
            }
        }
    }

    fn present_sampled(
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

    fn drain_results(&mut self) -> Result<ThroughputBatchStats> {
        match self {
            Self::Direct(_) => Ok(ThroughputBatchStats::default()),
            Self::Threaded(presenter) => presenter.drain_results(),
        }
    }

    fn kind(&self) -> PresenterKind {
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

struct ThreadedPresenter {
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
struct WindowPresenter {
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
impl WindowPresenter {
    fn new(
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

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<PresentTimings> {
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
    fn present_shared(
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

    fn kind(&self) -> PresenterKind {
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
struct D3d12InteropPresenter {
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    command_allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    command_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    back_buffers: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    fence_value: u64,
    fence_event: windows::Win32::Foundation::HANDLE,
    width: u32,
    height: u32,
    tearing_supported: bool,
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
                        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_NONE,
                        ID3D12CommandAllocator, ID3D12Fence, ID3D12GraphicsCommandList,
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
        let command_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)? };
        let command_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &command_allocator,
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
        let mut presenter = Self {
            queue,
            command_allocator,
            command_list,
            swap_chain,
            back_buffers: Vec::new(),
            fence,
            fence_value: 0,
            fence_event,
            width,
            height,
            tearing_supported,
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

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.wait_for_gpu()?;
        self.back_buffers.clear();
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
        for index in 0..D3D12_SWAPCHAIN_BUFFER_COUNT {
            let back_buffer: ID3D12Resource = unsafe { self.swap_chain.GetBuffer(index as u32)? };
            self.back_buffers.push(back_buffer);
        }
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
        let back_buffer = self
            .back_buffers
            .get(back_index)
            .context("D3D12 interop backbuffer is not available")?;
        unsafe {
            self.command_allocator.Reset()?;
            self.command_list.Reset(
                &self.command_allocator,
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
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<()> {
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        self.fence_value += 1;
        unsafe {
            self.queue.Signal(&self.fence, self.fence_value)?;
            if self.fence.GetCompletedValue() < self.fence_value {
                self.fence
                    .SetEventOnCompletion(self.fence_value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
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
struct WindowPresenter;

#[cfg(not(windows))]
impl WindowPresenter {
    fn new(
        _window: &Window,
        _kind: PresenterKind,
        _upload_ring: usize,
        _d3d_upload: D3dUploadMode,
        _interop_device: Option<&NeoD3d12InteropDevice>,
    ) -> Result<Self> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    fn present(&mut self, _size: PhysicalSize<u32>, _bgra: &[u8]) -> Result<PresentTimings> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    fn kind(&self) -> PresenterKind {
        PresenterKind::Gdi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_kernel_abi_accepts_expected_signature() {
        let source =
            "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        validate_live_kernel_abi(source).unwrap();
    }

    #[test]
    fn live_graph_cuda_source_wraps_image_kernel() {
        let source =
            "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        let cuda = live_graph_cuda_source(source).unwrap();
        assert!(cuda.contains("__device__ __forceinline__ void neo_user_image"));
        assert!(cuda.contains("extern \"C\" __global__ void image_graph"));
        assert!(cuda.contains("params->time"));
        assert!(cuda.contains("params->frame"));
    }

    #[test]
    fn live_kernel_abi_rejects_missing_frame() {
        let source = "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time) {}";
        let err = validate_live_kernel_abi(source).unwrap_err().to_string();
        assert!(err.contains("must have 5 params"));
    }

    #[test]
    fn quad_stress_kernel_keeps_live_abi() {
        validate_live_kernel_abi(include_str!("../../stress-quads/million_quads.neo")).unwrap();
    }

    #[test]
    fn mesh_kernel_abi_accepts_expected_signature() {
        let source = "kernel fn raster(global u8* pixels, u32 width, u32 height, global u8* mesh, f32 time, u32 frame) {}";
        validate_mesh_kernel_abi(source).unwrap();
    }

    #[test]
    fn mesh_kernel_abi_rejects_missing_mesh_pointer() {
        let source =
            "kernel fn raster(global u8* pixels, u32 width, u32 height, f32 time, u32 frame) {}";
        let err = validate_mesh_kernel_abi(source).unwrap_err().to_string();
        assert!(err.contains("must have 6 params"));
    }

    #[test]
    fn instance_kernel_abi_accepts_expected_signature() {
        let source = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, global u8* camera, f32 time, u32 frame) {}";
        validate_instance_kernel_abi(source).unwrap();
    }

    #[test]
    fn instance_kernel_abi_rejects_missing_camera_pointer() {
        let source = "kernel fn instance_raster(global u8* pixels, u32 width, u32 height, global u8* mesh, global u8* instances, f32 time, u32 frame) {}";
        let err = validate_instance_kernel_abi(source)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must have 8 params"));
    }

    #[test]
    fn instance_stress_kernel_keeps_expected_abi() {
        validate_culled_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_fast.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_baseline.neo"
        ))
        .unwrap();
        validate_instance_kernel_abi(include_str!(
            "../../stress-quads/three_d_instances_tiled.neo"
        ))
        .unwrap();
        assert!(
            !include_str!("../../stress-quads/three_d_instances_tiled.neo")
                .contains("instance_cull")
        );
    }

    #[test]
    fn reload_state_keeps_last_good_after_failure() {
        let mut reload = ReloadState::new(7);
        assert!(!reload.try_replace::<&str>(Err("compile failed")));
        assert_eq!(reload.active, 7);
        assert_eq!(reload.generation, 0);
        assert!(
            reload
                .last_error
                .as_deref()
                .unwrap()
                .contains("compile failed")
        );
        assert!(reload.try_replace::<&str>(Ok(9)));
        assert_eq!(reload.active, 9);
        assert_eq!(reload.generation, 1);
        assert!(reload.last_error.is_none());
    }

    #[test]
    fn live_options_accept_bounded_run_controls() {
        let options = LiveOptions::parse(
            [
                "custom.neo",
                "--title",
                "Tiny Stress",
                "--width",
                "320",
                "--height",
                "180",
                "--frames",
                "12",
                "--seconds",
                "0.5",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.source_path, PathBuf::from("custom.neo"));
        assert_eq!(options.title, "Tiny Stress");
        assert_eq!(options.width, 320);
        assert_eq!(options.height, 180);
        assert_eq!(options.max_frames, Some(12));
        assert_eq!(options.max_seconds, Some(0.5));
        assert!(options.should_stop(12, Duration::from_millis(1)));
        assert!(options.should_stop(1, Duration::from_millis(500)));
    }

    #[test]
    fn live_options_accept_kernel_throughput_controls() {
        let options = LiveOptions::parse(
            [
                "custom.neo",
                "--mode",
                "kernel-throughput",
                "--sample-every",
                "128",
                "--max-inflight",
                "8",
                "--presenter",
                "gdi",
                "--present-target-fps",
                "60",
                "--kernel-target-fps",
                "240",
                "--present-ring",
                "5",
                "--d3d-upload",
                "update-subresource",
                "--interop-fallback",
                "fail",
                "--no-hot-reload",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::KernelThroughput);
        assert_eq!(options.sample_every, 128);
        assert_eq!(options.max_inflight, 8);
        assert_eq!(options.presenter, PresenterKind::Gdi);
        assert_eq!(options.present_target_fps, Some(60.0));
        assert_eq!(options.kernel_target_fps, Some(240.0));
        assert_eq!(options.kernel_cap(), Some(240.0));
        assert_eq!(options.present_ring, 5);
        assert_eq!(options.d3d_upload, D3dUploadMode::UpdateSubresource);
        assert_eq!(options.interop_fallback, InteropFallback::Fail);
        assert!(!options.hot_reload);
        assert_eq!(
            options.present_interval(),
            Some(Duration::from_secs_f32(1.0 / 60.0))
        );
    }

    #[test]
    fn live_options_accept_mesh_demo_mode() {
        let options = LiveOptions::parse(
            [
                "examples/mesh-buffer/raster.neo",
                "--mode",
                "mesh-demo",
                "--presenter",
                "d3d12-interop",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::MeshDemo);
        assert_eq!(options.presenter, PresenterKind::D3d12Interop);
    }

    #[test]
    fn live_options_accept_instance_stress_mode_and_grid() {
        let options = LiveOptions::parse(
            [
                "examples/stress-quads/three_d_instances.neo",
                "--mode",
                "instance-stress",
                "--presenter",
                "d3d12-interop",
                "--instance-grid",
                "128x128x64",
                "--instance-stress-variant",
                "baseline",
                "--render-policy",
                "pause-when-empty",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(options.mode, RunMode::InstanceStress);
        assert_eq!(options.presenter, PresenterKind::D3d12Interop);
        assert_eq!(options.instance_grid, DEFAULT_INSTANCE_GRID);
        assert_eq!(options.instance_grid.count(), Some(1_048_576));
        assert_eq!(
            options.instance_stress_variant,
            InstanceStressVariant::Baseline
        );
        assert_eq!(options.render_policy, RenderPolicy::PauseWhenEmpty);
    }

    #[test]
    fn render_policy_accepts_expected_values() {
        assert_eq!("auto".parse::<RenderPolicy>().unwrap(), RenderPolicy::Auto);
        assert_eq!(
            "force-render".parse::<RenderPolicy>().unwrap(),
            RenderPolicy::ForceRender
        );
        assert_eq!(
            "pause-when-empty".parse::<RenderPolicy>().unwrap(),
            RenderPolicy::PauseWhenEmpty
        );
        let err = "maybe".parse::<RenderPolicy>().unwrap_err().to_string();
        assert!(err.contains("expected auto, force-render, or pause-when-empty"));
    }

    #[test]
    fn instance_stress_variant_accepts_baseline_fast_culled_and_tiled() {
        assert_eq!(
            "baseline".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Baseline
        );
        assert_eq!(
            "fast".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Fast
        );
        assert_eq!(
            "culled".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Culled
        );
        assert_eq!(
            "tiled".parse::<InstanceStressVariant>().unwrap(),
            InstanceStressVariant::Tiled
        );
        let err = "turbo"
            .parse::<InstanceStressVariant>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected baseline, fast, culled, or tiled"));
    }

    #[test]
    fn instance_stress_variant_resolves_stock_sources() {
        let requested = Path::new("D:/Neo/examples/stress-quads/three_d_instances.neo");
        assert_eq!(
            InstanceStressVariant::Baseline.source_path(requested),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_baseline.neo")
        );
        assert_eq!(
            InstanceStressVariant::Fast.source_path(requested),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_fast.neo")
        );
        assert_eq!(
            InstanceStressVariant::Culled.source_path(requested),
            requested.to_path_buf()
        );
        assert_eq!(
            InstanceStressVariant::Tiled.source_path(requested),
            PathBuf::from("D:/Neo/examples/stress-quads/three_d_instances_tiled.neo")
        );
    }

    #[test]
    fn tile_cull_size_covers_non_multiple_framebuffers() {
        assert_eq!(
            tile_cull_grid_size(PhysicalSize::new(16, 8)).unwrap(),
            (2, 1)
        );
        assert_eq!(
            tile_cull_grid_size(PhysicalSize::new(17, 9)).unwrap(),
            (3, 2)
        );
        assert_eq!(
            tile_cull_byte_len(PhysicalSize::new(17, 9)).unwrap(),
            3 * 2 * TILE_CULL_RECORD_BYTES
        );
    }

    #[test]
    fn tile_cull_size_rejects_zero_and_overflow() {
        let err = tile_cull_byte_len(PhysicalSize::new(0, 8))
            .unwrap_err()
            .to_string();
        assert!(err.contains("nonzero"));
        let err = tile_cull_byte_len(PhysicalSize::new(u32::MAX, u32::MAX))
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"));
    }

    #[test]
    fn instance_grid_rejects_zero_and_malformed_values() {
        let err = "128x0x64".parse::<InstanceGrid>().unwrap_err().to_string();
        assert!(err.contains("greater than zero"));
        let err = "128,128,64"
            .parse::<InstanceGrid>()
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected XxYxZ"));
    }

    #[test]
    fn camera_defaults_to_auto_orbit() {
        let mut camera = CameraController::new();
        let params = camera.params(PhysicalSize::new(1280, 720), 1.0, DEFAULT_INSTANCE_GRID);
        assert!(camera.auto);
        assert_eq!(params.grid, [128, 128, 64, 1_048_576]);
        assert!(params.view[0] > params.view[1]);
    }

    #[test]
    fn window_visibility_state_prioritizes_minimized_and_occluded() {
        let mut state = WindowVisibilityState::default();
        assert_eq!(state.render_visibility(), RenderVisibility::Visible);
        state.occluded = true;
        assert_eq!(state.render_visibility(), RenderVisibility::Occluded);
        state.minimized = true;
        assert_eq!(state.render_visibility(), RenderVisibility::Minimized);
    }

    #[test]
    fn instance_lattice_visibility_detects_visible_and_empty_views() {
        let mut camera = CameraController::new();
        let visible = camera.params(PhysicalSize::new(1280, 720), 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&visible));

        let mut empty = visible;
        empty.forward = [
            -visible.forward[0],
            -visible.forward[1],
            -visible.forward[2],
            0.0,
        ];
        assert!(!instance_lattice_visible(&empty));
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Empty
        );
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::ForceRender,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Visible
        );
    }

    #[test]
    fn instance_lattice_visibility_stays_stable_at_edges() {
        let mut camera = CameraController::new();
        camera.auto = false;
        let size = PhysicalSize::new(1280, 720);
        camera.yaw = std::f32::consts::FRAC_PI_2 + 0.9;
        let edge_visible = camera.params(size, 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&edge_visible));

        camera.yaw = std::f32::consts::FRAC_PI_2 + 2.4;
        let fully_empty = camera.params(size, 1.01, DEFAULT_INSTANCE_GRID);
        assert!(!instance_lattice_visible(&fully_empty));
    }

    #[test]
    fn empty_idle_ticks_skip_gpu_work_without_live_interop_slots() {
        assert!(should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::Auto,
            0,
            false
        ));
        assert!(should_skip_idle_instance_gpu_work(
            RenderVisibility::Occluded,
            RenderPolicy::PauseWhenEmpty,
            0,
            false
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::Auto,
            0,
            true
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Empty,
            RenderPolicy::ForceRender,
            0,
            false
        ));
        assert!(!should_skip_idle_instance_gpu_work(
            RenderVisibility::Visible,
            RenderPolicy::Auto,
            0,
            false
        ));
    }

    #[test]
    fn mouse_look_capture_transitions_are_explicit() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        assert!(camera.right_mouse_down);
        assert!(!camera.auto);
        assert!(camera.last_cursor.is_none());

        camera.end_mouse_look();
        assert!(!camera.right_mouse_down);
        assert!(camera.last_cursor.is_none());
    }

    #[test]
    fn escape_exits_mouse_look_capture() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        let action = camera.handle_key_code(KeyCode::Escape, true);
        assert_eq!(action, CameraWindowAction::ReleaseMouse);
        assert!(!camera.right_mouse_down);
    }

    #[test]
    fn raw_mouse_delta_can_turn_beyond_full_rotation() {
        let mut camera = CameraController::new();
        let start_yaw = camera.yaw;
        camera.handle_raw_mouse_delta(10_000.0, 0.0);
        assert_eq!(camera.yaw, start_yaw);

        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta((std::f32::consts::TAU / 0.003) + 100.0, 0.0);
        assert!(camera.raw_mouse_seen_during_capture);
        assert!(camera.yaw - start_yaw > std::f32::consts::TAU);
    }

    #[test]
    fn raw_mouse_delta_can_turn_both_directions() {
        let mut camera = CameraController::new();
        let start_yaw = camera.yaw;
        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta(250.0, 0.0);
        assert!(camera.yaw > start_yaw);
        camera.handle_raw_mouse_delta(-500.0, 0.0);
        assert!(camera.yaw < start_yaw);
    }

    #[test]
    fn cursor_fallback_recenters_after_bounded_mouse_delta() {
        let mut camera = CameraController::new();
        camera.begin_mouse_look();
        let first = WindowEvent::CursorMoved {
            device_id: winit::event::DeviceId::dummy(),
            position: PhysicalPosition::new(100.0, 100.0),
        };
        assert_eq!(camera.handle_window_event(&first), CameraWindowAction::None);
        let second = WindowEvent::CursorMoved {
            device_id: winit::event::DeviceId::dummy(),
            position: PhysicalPosition::new(90.0, 100.0),
        };
        assert_eq!(
            camera.handle_window_event(&second),
            CameraWindowAction::RecenterMouse
        );
        assert!(camera.last_cursor.is_none());
    }

    #[test]
    fn empty_visibility_keeps_input_alive_until_view_returns() {
        let mut camera = CameraController::new();
        camera.auto = false;
        let size = PhysicalSize::new(1280, 720);
        let visible = camera.params(size, 1.0, DEFAULT_INSTANCE_GRID);
        assert!(instance_lattice_visible(&visible));

        camera.begin_mouse_look();
        camera.handle_raw_mouse_delta(std::f32::consts::PI / 0.003, 0.0);
        let empty = camera.params(size, 1.01, DEFAULT_INSTANCE_GRID);
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &empty,
            ),
            RenderVisibility::Empty
        );
        assert_eq!(RenderVisibility::Empty.limit_max_inflight(8), 0);

        camera.handle_raw_mouse_delta(std::f32::consts::PI / 0.003, 0.0);
        let visible_again = camera.params(size, 1.02, DEFAULT_INSTANCE_GRID);
        assert_eq!(
            instance_render_visibility(
                RenderPolicy::Auto,
                WindowVisibilityState::default(),
                &visible_again,
            ),
            RenderVisibility::Visible
        );
    }

    #[test]
    fn empty_or_occluded_visibility_prevents_kernel_launches() {
        assert_eq!(RenderVisibility::Empty.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Occluded.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Minimized.limit_max_inflight(8), 0);
        assert_eq!(RenderVisibility::Visible.limit_max_inflight(8), 8);
    }

    #[test]
    fn unfocused_pacing_does_not_consume_main_tokens_until_soft_limiter_is_ready() {
        let start = Instant::now();
        let mut kernel = KernelRateLimiter::new(Some(1000.0), 8, start);
        let mut unfocused = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
        let first = start + Duration::from_millis(1);
        assert_eq!(
            grant_instance_inflight(
                first,
                RenderVisibility::Visible,
                RenderPolicy::Auto,
                false,
                8,
                &mut kernel,
                &mut unfocused,
            ),
            0
        );
        let ready = start + Duration::from_millis(67);
        assert_eq!(
            grant_instance_inflight(
                ready,
                RenderVisibility::Visible,
                RenderPolicy::Auto,
                false,
                8,
                &mut kernel,
                &mut unfocused,
            ),
            1
        );
    }

    #[test]
    fn unfocused_next_tick_waits_for_both_limiters() {
        let start = Instant::now();
        let kernel = KernelRateLimiter::new(Some(1000.0), 8, start);
        let unfocused = KernelRateLimiter::new(Some(UNFOCUSED_IDLE_FPS), 1, start);
        let next = next_instance_tick_at(InstanceTickPacing {
            now: start,
            visibility: RenderVisibility::Visible,
            kernel_limiter: &kernel,
            unfocused_limiter: &unfocused,
            render_policy: RenderPolicy::Auto,
            focused: false,
            wait_for_kernel_token: false,
        })
        .unwrap();
        assert!(next.duration_since(start) >= Duration::from_millis(66));
    }

    #[test]
    fn lattice_traversal_math_handles_positive_negative_and_miss() {
        fn range(origin_z: f32, dir_z: f32, spacing: f32, grid_z: u32) -> Option<(i32, i32, i32)> {
            if dir_z.abs() <= 0.00001 {
                return None;
            }
            let half_z = (grid_z as f32 - 1.0) * 0.5;
            let min_z = -half_z * spacing;
            let max_z = half_z * spacing;
            let t0 = (min_z - origin_z) / dir_z;
            let t1 = (max_z - origin_z) / dir_z;
            let enter = t0.min(t1);
            let exit = t0.max(t1);
            if exit < 0.0 {
                return None;
            }
            let begin = enter.max(0.0);
            if begin > exit {
                return None;
            }
            let to_layer =
                |z: f32| ((z / spacing + half_z + 0.5).floor() as i32).clamp(0, grid_z as i32 - 1);
            let start = to_layer(origin_z + dir_z * begin);
            let end = to_layer(origin_z + dir_z * exit);
            let step = if start > end { -1 } else { 1 };
            Some((start, end, step))
        }

        assert_eq!(range(2.0, -1.0, 0.5, 8), Some((7, 0, -1)));
        assert_eq!(range(-2.0, 1.0, 0.5, 8), Some((0, 7, 1)));
        assert_eq!(range(2.0, 1.0, 0.5, 8), None);
    }

    #[test]
    fn manual_camera_movement_is_delta_time_based_and_clamped() {
        let mut camera = CameraController::new();
        camera.auto = false;
        camera.moving_forward = true;
        let start_y = camera.origin[1];
        camera.params(PhysicalSize::new(1280, 720), 10.0, DEFAULT_INSTANCE_GRID);
        assert_eq!(camera.origin[1], start_y);

        camera.params(PhysicalSize::new(1280, 720), 10.016, DEFAULT_INSTANCE_GRID);
        let small_step = camera.origin[1] - start_y;
        assert!(small_step > 0.05);
        assert!(small_step < 0.08);

        camera.params(PhysicalSize::new(1280, 720), 11.0, DEFAULT_INSTANCE_GRID);
        let clamped_step = camera.origin[1] - start_y - small_step;
        assert!(clamped_step > 0.12);
        assert!(clamped_step < 0.14);
    }

    #[test]
    fn live_options_enable_hot_reload_by_default() {
        let options = LiveOptions::parse(std::iter::empty::<String>()).unwrap();
        assert!(options.hot_reload);
        assert_eq!(options.present_ring, 6);

        let options =
            LiveOptions::parse(["--no-hot-reload"].into_iter().map(String::from)).unwrap();
        assert!(!options.hot_reload);

        let options = LiveOptions::parse(
            ["--no-hot-reload", "--hot-reload"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(options.hot_reload);
    }

    #[test]
    fn presenter_kind_accepts_d3d11_and_gdi() {
        assert_eq!(
            "d3d12-interop".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d12Interop
        );
        assert_eq!(
            "d3d12".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d12
        );
        assert_eq!(
            "d3d11".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d11
        );
        assert_eq!("gdi".parse::<PresenterKind>().unwrap(), PresenterKind::Gdi);
        assert_eq!(PresenterKind::D3d12Interop.to_string(), "d3d12-interop");
        assert_eq!(PresenterKind::D3d12.to_string(), "d3d12-flip-upload");
        assert_eq!(PresenterKind::D3d11.to_string(), "d3d11-flip-host");
        assert_eq!(PresenterKind::Gdi.to_string(), "win32-gdi");
        assert!(!PresenterKind::D3d12Interop.uses_present_thread());
        assert!(!PresenterKind::D3d12.uses_present_thread());
        assert!(PresenterKind::D3d11.uses_present_thread());
        assert!(PresenterKind::Gdi.uses_present_thread());
    }

    #[test]
    fn interop_fallback_accepts_no_interop_and_fail() {
        assert_eq!(
            "no-interop".parse::<InteropFallback>().unwrap(),
            InteropFallback::NoInterop
        );
        assert_eq!(
            "fail".parse::<InteropFallback>().unwrap(),
            InteropFallback::Fail
        );
        let err = "maybe".parse::<InteropFallback>().unwrap_err().to_string();
        assert!(err.contains("expected no-interop or fail"));
    }

    #[test]
    fn interop_stats_do_not_count_host_transfer_bytes() {
        let mut counter = ThroughputCounter::new();
        counter.record(ThroughputBatchStats {
            completed_kernels: 3,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 0,
            uploaded_bytes: 0,
            launch: Duration::from_micros(30),
            completion_wait: Duration::ZERO,
            sample_download: Duration::ZERO,
            present: Duration::from_micros(50),
            map_copy: Duration::ZERO,
            draw: Duration::from_micros(20),
            swap_present: Duration::from_micros(30),
        });
        assert_eq!(counter.sampled_bytes_since_log, 0);
        assert_eq!(counter.uploaded_bytes_since_log, 0);
        assert_eq!(counter.total_presented, 1);
    }

    #[test]
    fn interop_high_rate_presenter_does_not_sleep_on_kernel_tokens() {
        assert!(!should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(1000.0)
        ));
        assert!(!should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(240.0)
        ));
        assert!(should_wait_for_kernel_token(
            PresenterKind::D3d12Interop,
            Some(60.0)
        ));
        assert!(should_wait_for_kernel_token(
            PresenterKind::D3d12,
            Some(1000.0)
        ));
    }

    #[test]
    fn present_rate_limiter_caps_average_rate_without_sleeping() {
        let start = Instant::now();
        let mut limiter = PresentRateLimiter::new(Some(1000.0), start);
        let mut presented = 0;
        for tick in 0..10_000 {
            let now = start + Duration::from_micros(tick * 100);
            if limiter.try_consume(now) {
                presented += 1;
            }
        }
        assert!((999..=1000).contains(&presented));
    }

    #[test]
    fn d3d_upload_mode_accepts_mapped_copy_and_update_subresource() {
        assert_eq!(
            "mapped-copy".parse::<D3dUploadMode>().unwrap(),
            D3dUploadMode::MappedCopy
        );
        assert_eq!(
            "update-subresource".parse::<D3dUploadMode>().unwrap(),
            D3dUploadMode::UpdateSubresource
        );
        let err = "wat".parse::<D3dUploadMode>().unwrap_err().to_string();
        assert!(err.contains("expected mapped-copy or update-subresource"));
    }

    #[test]
    fn frame_byte_len_validates_overflow() {
        assert_eq!(frame_byte_len(2, 3).unwrap(), 24);
        assert!(frame_byte_len(u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn live_options_accept_zero_kernel_target_as_uncapped() {
        let options =
            LiveOptions::parse(["--kernel-target-fps", "0"].into_iter().map(String::from)).unwrap();
        assert_eq!(options.kernel_target_fps, Some(0.0));
        assert_eq!(options.kernel_cap(), None);
    }

    #[test]
    fn live_options_reject_zero_throughput_values() {
        let err = LiveOptions::parse(["--sample-every", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sample-every must be greater than zero"));

        let err = LiveOptions::parse(["--max-inflight", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--max-inflight must be greater than zero"));

        let err = LiveOptions::parse(["--present-target-fps", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--present-target-fps must be greater than zero"));

        let err = LiveOptions::parse(["--present-ring", "0"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--present-ring must be greater than zero"));
    }

    #[test]
    fn live_options_reject_invalid_kernel_target_fps() {
        for value in ["-1", "NaN", "inf"] {
            let err =
                LiveOptions::parse(["--kernel-target-fps", value].into_iter().map(String::from))
                    .unwrap_err()
                    .to_string();
            assert!(err.contains("--kernel-target-fps must be finite and non-negative"));
        }
    }

    #[test]
    fn kernel_rate_limiter_grants_target_rate_over_time() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(240.0), 8, start);
        let mut granted = 0;
        for tick in 1..=100 {
            granted += limiter.grant(start + Duration::from_millis(tick * 10), 8);
        }
        assert!((239..=241).contains(&granted));
    }

    #[test]
    fn kernel_rate_limiter_clamps_burst() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(10.0), 4, start);
        let granted = limiter.grant(start + Duration::from_secs(10), 100);
        assert_eq!(granted, 16);
    }

    #[test]
    fn kernel_rate_limiter_returns_zero_without_tokens() {
        let start = Instant::now();
        let mut limiter = KernelRateLimiter::new(Some(240.0), 8, start);
        assert_eq!(limiter.grant(start, 8), 0);
        assert!(
            limiter
                .next_token_at(start)
                .is_some_and(|wake| wake > start)
        );
    }

    #[test]
    fn presentation_ring_chooses_newest_completed_slot() {
        let newest = choose_newest_completed_slot([(0, 10), (1, 12), (2, 11)]);
        assert_eq!(newest, Some(1));
        assert_eq!(choose_newest_completed_slot([]), None);
    }

    #[test]
    fn bgra_copy_reports_fast_path_only_for_matching_pitch() {
        let src: Vec<u8> = (0..16).collect();
        let mut tight = vec![0u8; 16];
        let used_fast_path = unsafe { copy_bgra_to_mapped(&src, tight.as_mut_ptr(), 2, 2, 8) };
        assert!(used_fast_path);
        assert_eq!(tight, src);

        let mut padded = vec![0u8; 24];
        let used_fast_path = unsafe { copy_bgra_to_mapped(&src, padded.as_mut_ptr(), 2, 2, 12) };
        assert!(!used_fast_path);
        assert_eq!(&padded[0..8], &src[0..8]);
        assert_eq!(&padded[12..20], &src[8..16]);
    }

    #[test]
    fn throughput_counter_tracks_completed_separately_from_presented() {
        let mut counter = ThroughputCounter::new();
        counter.record(ThroughputBatchStats {
            completed_kernels: 256,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 1024,
            uploaded_bytes: 1024,
            launch: Duration::from_micros(256),
            completion_wait: Duration::from_micros(512),
            sample_download: Duration::from_micros(200),
            present: Duration::from_micros(100),
            map_copy: Duration::from_micros(40),
            draw: Duration::from_micros(30),
            swap_present: Duration::from_micros(30),
        });
        assert_eq!(counter.total_completed, 256);
        assert_eq!(counter.total_sampled, 1);
        assert_eq!(counter.total_presented, 1);
        assert_ne!(counter.total_completed, counter.total_presented);
    }

    #[test]
    fn throughput_batch_stats_accumulate_threaded_present_results() {
        let mut stats = ThroughputBatchStats {
            completed_kernels: 4,
            sampled_frames: 0,
            presented_frames: 0,
            sampled_bytes: 0,
            uploaded_bytes: 0,
            launch: Duration::from_micros(4),
            completion_wait: Duration::ZERO,
            sample_download: Duration::ZERO,
            present: Duration::ZERO,
            map_copy: Duration::ZERO,
            draw: Duration::ZERO,
            swap_present: Duration::ZERO,
        };
        stats += ThroughputBatchStats {
            completed_kernels: 0,
            sampled_frames: 1,
            presented_frames: 1,
            sampled_bytes: 16,
            uploaded_bytes: 16,
            launch: Duration::ZERO,
            completion_wait: Duration::ZERO,
            sample_download: Duration::from_micros(10),
            present: Duration::from_micros(20),
            map_copy: Duration::from_micros(6),
            draw: Duration::from_micros(7),
            swap_present: Duration::from_micros(8),
        };
        assert_eq!(stats.completed_kernels, 4);
        assert_eq!(stats.sampled_frames, 1);
        assert_eq!(stats.presented_frames, 1);
        assert_eq!(stats.sampled_bytes, 16);
        assert_eq!(stats.uploaded_bytes, 16);
        assert_eq!(stats.sample_download, Duration::from_micros(10));
        assert_eq!(stats.present, Duration::from_micros(20));
    }
}
