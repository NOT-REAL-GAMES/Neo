use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use neo_lang::{AddressSpace, TypeName};
use neo_runtime::{
    Context as NeoContext, DeviceBuffer, Kernel, LaunchDims, ReadablePinnedHostBuffer,
};
use notify::{Event as NotifyEvent, RecursiveMode, Watcher as _};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 540;
const BLOCK: (u32, u32) = (16, 16);

fn main() -> Result<()> {
    run(LiveOptions::parse(std::env::args().skip(1))?)
}

#[allow(deprecated)]
fn run(options: LiveOptions) -> Result<()> {
    let source_path = options
        .source_path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", options.source_path.display()))?;
    let event_loop = EventLoop::new()?;
    let window = create_window(&event_loop, options.width, options.height)?;
    let mut presenter = WindowPresenter::new(&window, options.presenter)?;

    let neo = NeoContext::new_default_device()?;
    let live = LiveKernel::compile(&neo, &source_path)?;
    let mut reload = ReloadState::new(live);
    let (_watcher, reload_rx) = watch_source(&source_path)?;
    let mut frame_resources: Option<FrameResources> = None;
    let mut fps = FpsCounter::new();
    let mut throughput = ThroughputCounter::new();
    let start = Instant::now();
    let mut frame = 0u32;
    let mut completed_kernels = 0u64;
    let mut next_sample_at = options.sample_every as u64;
    let mut next_present_at = options.present_interval().map(|interval| start + interval);
    let mut kernel_limiter =
        KernelRateLimiter::new(options.kernel_cap(), options.max_inflight, start);

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                if let Err(err) = handle_reload_events(&neo, &source_path, &reload_rx, &mut reload)
                {
                    eprintln!("hot reload watcher error: {err:#}");
                }

                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }

                if matches!(options.mode, RunMode::KernelThroughput) {
                    let now = Instant::now();
                    let max_inflight = kernel_limiter.grant(now, options.max_inflight);
                    if max_inflight == 0 {
                        if let Some(next_kernel_at) = kernel_limiter.next_token_at(now) {
                            elwt.set_control_flow(ControlFlow::WaitUntil(next_kernel_at));
                        }
                        return;
                    }
                    let result = run_kernel_throughput_batch(ThroughputBatch {
                        neo: &neo,
                        resources: &mut frame_resources,
                        kernel: &reload.active.kernel,
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
                    });
                    match result {
                        Ok(batch) => {
                            throughput.record(batch);
                            throughput.log_if_due(
                                size,
                                frame,
                                presenter.kind(),
                                reload.last_error.as_deref(),
                                options.kernel_cap(),
                            );
                            if options.should_stop_completed(completed_kernels, start.elapsed()) {
                                elwt.exit();
                            }
                        }
                        Err(err) => {
                            eprintln!("throughput error: {err:#}");
                            elwt.exit();
                        }
                    }
                    return;
                }

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
                if let Err(err) = presenter.present(size, resources.host_bgra.as_slice()) {
                    eprintln!("present error: {err:#}");
                    elwt.exit();
                    return;
                }
                timings.render = before_present - frame_start;
                timings.present = before_present.elapsed();
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
            _ => {}
        }
    })?;
    Ok(())
}

#[derive(Debug, Clone)]
struct LiveOptions {
    source_path: PathBuf,
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
}

impl LiveOptions {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            source_path: PathBuf::from("examples/live-window/live.neo"),
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
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--width" => options.width = parse_next(&mut args, "--width")?,
                "--height" => options.height = parse_next(&mut args, "--height")?,
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
                "--help" | "-h" => bail!(
                    "usage: neo-live-window [path.neo] [--width N] [--height N] [--frames N] [--seconds N] [--presenter d3d11|gdi] [--mode live|kernel-throughput] [--sample-every N] [--present-target-fps N] [--kernel-target-fps N] [--max-inflight N]"
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
}

impl std::str::FromStr for RunMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "live" => Ok(Self::Live),
            "kernel-throughput" => Ok(Self::KernelThroughput),
            _ => bail!("unknown mode `{value}`; expected live or kernel-throughput"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresenterKind {
    D3d11,
    Gdi,
}

impl std::str::FromStr for PresenterKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "d3d11" => Ok(Self::D3d11),
            "gdi" => Ok(Self::Gdi),
            _ => bail!("unknown presenter `{value}`; expected d3d11 or gdi"),
        }
    }
}

impl std::fmt::Display for PresenterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D3d11 => f.write_str("d3d11-flip-host"),
            Self::Gdi => f.write_str("win32-gdi"),
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
fn create_window(event_loop: &EventLoop<()>, width: u32, height: u32) -> Result<Window> {
    let attrs = WindowAttributes::default()
        .with_title("Neo Live Window")
        .with_inner_size(PhysicalSize::new(width.max(1), height.max(1)));
    event_loop
        .create_window(attrs)
        .context("failed to create live window")
}

struct LiveKernel {
    kernel: Kernel,
}

impl LiveKernel {
    fn compile(ctx: &NeoContext, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        validate_live_kernel_abi(&source)?;
        let module = neo_runtime::Module::from_neo_source(ctx, &source, &["image"])?;
        Ok(Self {
            kernel: module.kernel("image")?,
        })
    }
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
    for (param, (name, address_space, ty, pointer_depth)) in kernel.params.iter().zip(expected) {
        if param.name != name
            || param.address_space != address_space
            || param.ty.base != ty
            || param.ty.pointer_depth != pointer_depth
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
    resources: &'a mut Option<FrameResources>,
    kernel: &'a Kernel,
    presenter: &'a mut WindowPresenter,
    size: PhysicalSize<u32>,
    start: Instant,
    next_frame: &'a mut u32,
    completed_kernels: &'a mut u64,
    next_sample_at: &'a mut u64,
    next_present_at: &'a mut Option<Instant>,
    sample_every: u32,
    present_interval: Option<Duration>,
    max_inflight: u32,
}

fn run_kernel_throughput_batch(input: ThroughputBatch<'_>) -> Result<ThroughputBatchStats> {
    ensure_frame_resources(input.neo, input.resources, input.size)?;
    let resources = input
        .resources
        .as_mut()
        .expect("frame resources were just created for nonzero size");
    let width = input.size.width;
    let height = input.size.height;
    let dims = LaunchDims::for_2d(width, height, BLOCK);
    let mut stats = ThroughputBatchStats::default();

    for _ in 0..input.max_inflight {
        let launch_start = Instant::now();
        {
            let time = input.start.elapsed().as_secs_f32();
            let frame = *input.next_frame;
            let mut launch = input.kernel.launcher();
            launch
                .arg_buffer_mut(&mut resources.device_pixels)
                .arg(&width)
                .arg(&height)
                .arg(&time)
                .arg(&frame);
            unsafe {
                launch.launch(dims)?;
            }
        }
        stats.launch += launch_start.elapsed();
        stats.completed_kernels += 1;
        *input.next_frame = input.next_frame.wrapping_add(1);
    }

    let wait_start = Instant::now();
    input.neo.synchronize()?;
    stats.completion_wait += wait_start.elapsed();
    *input.completed_kernels += u64::from(stats.completed_kernels);

    let should_sample_by_count =
        input.present_interval.is_none() && *input.completed_kernels >= *input.next_sample_at;
    let should_sample_by_time = input
        .next_present_at
        .is_some_and(|next_present_at| Instant::now() >= next_present_at);

    if should_sample_by_count || should_sample_by_time {
        let sample_start = Instant::now();
        resources
            .device_pixels
            .download_into_readable_pinned(&mut resources.host_bgra)?;
        input.neo.synchronize()?;
        stats.sample_download += sample_start.elapsed();
        stats.sampled_frames += 1;

        let present_start = Instant::now();
        input
            .presenter
            .present(input.size, resources.host_bgra.as_slice())?;
        stats.present += present_start.elapsed();
        stats.presented_frames += 1;

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
    launch: Duration,
    completion_wait: Duration,
    sample_download: Duration,
    present: Duration,
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
        if self.target_fps.is_none() {
            return max_inflight;
        }
        self.refill(now);
        let granted = self.tokens.floor().min(f64::from(max_inflight)) as u32;
        self.tokens -= f64::from(granted);
        granted
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

struct ThroughputCounter {
    last_log: Instant,
    completed_since_log: u64,
    sampled_since_log: u64,
    presented_since_log: u64,
    total_completed: u64,
    total_sampled: u64,
    total_presented: u64,
    launch_accum: Duration,
    completion_wait_accum: Duration,
    sample_download_accum: Duration,
    present_accum: Duration,
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
            launch_accum: Duration::ZERO,
            completion_wait_accum: Duration::ZERO,
            sample_download_accum: Duration::ZERO,
            present_accum: Duration::ZERO,
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
        self.launch_accum += batch.launch;
        self.completion_wait_accum += batch.completion_wait;
        self.sample_download_accum += batch.sample_download;
        self.present_accum += batch.present;
    }

    fn log_if_due(
        &mut self,
        size: PhysicalSize<u32>,
        frame: u32,
        presenter: PresenterKind,
        reload_error: Option<&str>,
        kernel_cap: Option<f32>,
    ) {
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
        let reload_state = if reload_error.is_some() {
            "last-good"
        } else {
            "current"
        };
        let kernel_cap = kernel_cap
            .map(|fps| format!("{fps:.1} fps"))
            .unwrap_or_else(|| "uncapped".to_string());
        println!(
            "kernel_fps {kernel_fps:>9.1} | sample_fps {sample_fps:>6.1} | present_fps {present_fps:>6.1} | completed {:>10} | frame {frame:>8} | {}x{} | launch {launch_us:>5.1} us/k | wait {wait_us:>5.1} us/k | sample_dtoh {sample_us:>6.1} us | present {present_us:>6.1} us | presenter {presenter} | kernel_cap {kernel_cap} | kernel {reload_state}",
            self.total_completed, size.width, size.height
        );
        self.completed_since_log = 0;
        self.sampled_since_log = 0;
        self.presented_since_log = 0;
        self.launch_accum = Duration::ZERO;
        self.completion_wait_accum = Duration::ZERO;
        self.sample_download_accum = Duration::ZERO;
        self.present_accum = Duration::ZERO;
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

#[cfg(windows)]
struct WindowPresenter {
    inner: PresenterImpl,
}

#[cfg(windows)]
enum PresenterImpl {
    D3d11(D3d11Presenter),
    Gdi(GdiPresenter),
}

#[cfg(windows)]
impl WindowPresenter {
    fn new(window: &Window, kind: PresenterKind) -> Result<Self> {
        let inner = match kind {
            PresenterKind::D3d11 => PresenterImpl::D3d11(D3d11Presenter::new(window)?),
            PresenterKind::Gdi => PresenterImpl::Gdi(GdiPresenter::new(window)?),
        };
        Ok(Self { inner })
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<()> {
        match &mut self.inner {
            PresenterImpl::D3d11(presenter) => presenter.present(size, bgra),
            PresenterImpl::Gdi(presenter) => presenter.present(size, bgra),
        }
    }

    fn kind(&self) -> PresenterKind {
        match self.inner {
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

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<()> {
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
        Ok(())
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
struct D3d11Presenter {
    device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    swap_chain: windows::Win32::Graphics::Dxgi::IDXGISwapChain1,
    render_target: Option<windows::Win32::Graphics::Direct3D11::ID3D11RenderTargetView>,
    upload_texture: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    upload_srv: Option<windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView>,
    vertex_shader: windows::Win32::Graphics::Direct3D11::ID3D11VertexShader,
    pixel_shader: windows::Win32::Graphics::Direct3D11::ID3D11PixelShader,
    sampler: windows::Win32::Graphics::Direct3D11::ID3D11SamplerState,
    width: u32,
    height: u32,
    tearing_supported: bool,
}

#[cfg(windows)]
impl D3d11Presenter {
    fn new(window: &Window) -> Result<Self> {
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

        let (vertex_shader, pixel_shader) = create_present_shaders(&device)?;
        let sampler = create_point_sampler(&device)?;
        let mut presenter = Self {
            device,
            context,
            swap_chain,
            render_target: None,
            upload_texture: None,
            upload_srv: None,
            vertex_shader,
            pixel_shader,
            sampler,
            width: size.width.max(1),
            height: size.height.max(1),
            tearing_supported,
        };
        presenter.recreate_backbuffer_views()?;
        presenter.recreate_upload_texture(width, height)?;
        Ok(presenter)
    }

    fn present(&mut self, size: PhysicalSize<u32>, bgra: &[u8]) -> Result<()> {
        let expected = frame_byte_len(size.width, size.height)?;
        if bgra.len() != expected {
            bail!(
                "present buffer size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            );
        }
        self.ensure_size(size)?;
        self.upload_bgra(size.width, size.height, bgra)?;
        self.draw_present_texture()?;
        let flags = if self.tearing_supported {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT_ALLOW_TEARING
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)
        };
        unsafe {
            self.swap_chain.Present(0, flags).ok()?;
        }
        Ok(())
    }

    fn ensure_size(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.render_target = None;
        self.upload_srv = None;
        self.upload_texture = None;
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
        self.recreate_backbuffer_views()?;
        self.recreate_upload_texture(width, height)?;
        Ok(())
    }

    fn recreate_backbuffer_views(&mut self) -> Result<()> {
        use windows::Win32::Graphics::Direct3D11::{ID3D11RenderTargetView, ID3D11Texture2D};

        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0)? };
        let mut render_target: Option<ID3D11RenderTargetView> = None;
        unsafe {
            self.device
                .CreateRenderTargetView(&back_buffer, None, Some(&mut render_target))?;
        }
        self.render_target = Some(render_target.context("D3D11 did not return an RTV")?);
        Ok(())
    }

    fn recreate_upload_texture(&mut self, width: u32, height: u32) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D11::{
                D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_DYNAMIC, ID3D11ShaderResourceView, ID3D11Texture2D,
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
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
        }
        let texture = texture.context("D3D11 did not return an upload texture")?;
        let mut srv: Option<ID3D11ShaderResourceView> = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut srv))?;
        }
        self.upload_texture = Some(texture);
        self.upload_srv = Some(srv.context("D3D11 did not return an upload SRV")?);
        Ok(())
    }

    fn upload_bgra(&self, width: u32, height: u32, bgra: &[u8]) -> Result<()> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE,
        };

        let texture = self
            .upload_texture
            .as_ref()
            .context("D3D11 upload texture is not available")?;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(texture, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))?;
            let src_pitch = width as usize * 4;
            let dst_pitch = mapped.RowPitch as usize;
            let dst_base = mapped.pData.cast::<u8>();
            for y in 0..height as usize {
                let src = bgra.as_ptr().add(y * src_pitch);
                let dst = dst_base.add(y * dst_pitch);
                std::ptr::copy_nonoverlapping(src, dst, src_pitch);
            }
            self.context.Unmap(texture, 0);
        }
        Ok(())
    }

    fn draw_present_texture(&self) -> Result<()> {
        use windows::Win32::Graphics::{
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            Direct3D11::{D3D11_VIEWPORT, ID3D11ShaderResourceView},
        };

        let render_target = self
            .render_target
            .as_ref()
            .context("D3D11 render target is not available")?;
        let upload_srv = self
            .upload_srv
            .as_ref()
            .context("D3D11 upload SRV is not available")?;
        let rtv = [Some(render_target.clone())];
        let srv = [Some(upload_srv.clone())];
        let clear_srv: [Option<ID3D11ShaderResourceView>; 1] = [None];
        let sampler = [Some(self.sampler.clone())];
        let viewport = [D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: self.width as f32,
            Height: self.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }];
        unsafe {
            self.context.RSSetViewports(Some(&viewport));
            self.context.OMSetRenderTargets(
                Some(&rtv),
                None::<&windows::Win32::Graphics::Direct3D11::ID3D11DepthStencilView>,
            );
            self.context
                .IASetInputLayout(None::<&windows::Win32::Graphics::Direct3D11::ID3D11InputLayout>);
            self.context
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.context.VSSetShader(&self.vertex_shader, None);
            self.context.PSSetShader(&self.pixel_shader, None);
            self.context.PSSetSamplers(0, Some(&sampler));
            self.context.PSSetShaderResources(0, Some(&srv));
            self.context.Draw(3, 0);
            self.context.PSSetShaderResources(0, Some(&clear_srv));
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
fn create_present_shaders(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
) -> Result<(
    windows::Win32::Graphics::Direct3D11::ID3D11VertexShader,
    windows::Win32::Graphics::Direct3D11::ID3D11PixelShader,
)> {
    use windows::Win32::Graphics::Direct3D11::{ID3D11PixelShader, ID3D11VertexShader};

    let shader_source = br#"
Texture2D<float4> image_tex : register(t0);
SamplerState image_sampler : register(s0);

struct VsOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VsOut vs_main(uint id : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };
    float2 uvs[3] = {
        float2(0.0,  1.0),
        float2(0.0, -1.0),
        float2(2.0,  1.0),
    };
    VsOut output;
    output.pos = float4(positions[id], 0.0, 1.0);
    output.uv = uvs[id];
    return output;
}

float4 ps_main(VsOut input) : SV_Target {
    return image_tex.Sample(image_sampler, input.uv);
}
"#;

    let vs_blob = compile_hlsl(shader_source, b"vs_main\0", b"vs_5_0\0")?;
    let ps_blob = compile_hlsl(shader_source, b"ps_main\0", b"ps_5_0\0")?;
    let vs_bytes = unsafe {
        std::slice::from_raw_parts(
            vs_blob.GetBufferPointer().cast::<u8>(),
            vs_blob.GetBufferSize(),
        )
    };
    let ps_bytes = unsafe {
        std::slice::from_raw_parts(
            ps_blob.GetBufferPointer().cast::<u8>(),
            ps_blob.GetBufferSize(),
        )
    };
    let mut vertex_shader: Option<ID3D11VertexShader> = None;
    let mut pixel_shader: Option<ID3D11PixelShader> = None;
    unsafe {
        device.CreateVertexShader(
            vs_bytes,
            None::<&windows::Win32::Graphics::Direct3D11::ID3D11ClassLinkage>,
            Some(&mut vertex_shader),
        )?;
        device.CreatePixelShader(
            ps_bytes,
            None::<&windows::Win32::Graphics::Direct3D11::ID3D11ClassLinkage>,
            Some(&mut pixel_shader),
        )?;
    }
    Ok((
        vertex_shader.context("D3D11 did not return a vertex shader")?,
        pixel_shader.context("D3D11 did not return a pixel shader")?,
    ))
}

#[cfg(windows)]
fn compile_hlsl(
    source: &[u8],
    entry: &'static [u8],
    target: &'static [u8],
) -> Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    use windows::{
        Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob, ID3DInclude},
        core::PCSTR,
    };

    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None::<&ID3DInclude>,
            PCSTR::from_raw(entry.as_ptr()),
            PCSTR::from_raw(target.as_ptr()),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(err) = result {
        let message = errors
            .map(|blob| unsafe {
                let bytes = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                String::from_utf8_lossy(bytes).to_string()
            })
            .unwrap_or_else(|| err.to_string());
        bail!("D3D11 present shader compilation failed: {message}");
    }
    code.context("D3DCompile did not return shader bytecode")
}

#[cfg(windows)]
fn create_point_sampler(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
) -> Result<windows::Win32::Graphics::Direct3D11::ID3D11SamplerState> {
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_COMPARISON_NEVER, D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_FLOAT32_MAX,
        D3D11_SAMPLER_DESC, D3D11_TEXTURE_ADDRESS_CLAMP, ID3D11SamplerState,
    };

    let desc = D3D11_SAMPLER_DESC {
        Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D11_COMPARISON_NEVER,
        BorderColor: [0.0; 4],
        MinLOD: 0.0,
        MaxLOD: D3D11_FLOAT32_MAX,
    };
    let mut sampler: Option<ID3D11SamplerState> = None;
    unsafe {
        device.CreateSamplerState(&desc, Some(&mut sampler))?;
    }
    sampler.context("D3D11 did not return a sampler")
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
    fn new(_window: &Window, _kind: PresenterKind) -> Result<Self> {
        bail!("the no-interop live presenter currently targets Windows/Win32")
    }

    fn present(&mut self, _size: PhysicalSize<u32>, _bgra: &[u8]) -> Result<()> {
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
    fn live_kernel_abi_rejects_missing_frame() {
        let source = "kernel fn image(global u8* pixels, u32 width, u32 height, f32 time) {}";
        let err = validate_live_kernel_abi(source).unwrap_err().to_string();
        assert!(err.contains("must have 5 params"));
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
        assert_eq!(
            options.present_interval(),
            Some(Duration::from_secs_f32(1.0 / 60.0))
        );
    }

    #[test]
    fn presenter_kind_accepts_d3d11_and_gdi() {
        assert_eq!(
            "d3d11".parse::<PresenterKind>().unwrap(),
            PresenterKind::D3d11
        );
        assert_eq!("gdi".parse::<PresenterKind>().unwrap(), PresenterKind::Gdi);
        assert_eq!(PresenterKind::D3d11.to_string(), "d3d11-flip-host");
        assert_eq!(PresenterKind::Gdi.to_string(), "win32-gdi");
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
    fn throughput_counter_tracks_completed_separately_from_presented() {
        let mut counter = ThroughputCounter::new();
        counter.record(ThroughputBatchStats {
            completed_kernels: 256,
            sampled_frames: 1,
            presented_frames: 1,
            launch: Duration::from_micros(256),
            completion_wait: Duration::from_micros(512),
            sample_download: Duration::from_micros(200),
            present: Duration::from_micros(100),
        });
        assert_eq!(counter.total_completed, 256);
        assert_eq!(counter.total_sampled, 1);
        assert_eq!(counter.total_presented, 1);
        assert_ne!(counter.total_completed, counter.total_presented);
    }
}
