use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use neo_runtime::{DataLayout, DeviceInfo};

use crate::{DEFAULT_HEIGHT, DEFAULT_INSTANCE_GRID, DEFAULT_WIDTH, HardwareRasterPlan};

#[derive(Debug, Clone)]
pub(crate) struct LiveOptions {
    pub(crate) source_path: PathBuf,
    pub(crate) title: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) max_frames: Option<u32>,
    pub(crate) max_seconds: Option<f32>,
    pub(crate) presenter: PresenterKind,
    pub(crate) mode: RunMode,
    pub(crate) sample_every: u32,
    pub(crate) present_target_fps: Option<f32>,
    pub(crate) kernel_target_fps: Option<f32>,
    pub(crate) max_inflight: u32,
    pub(crate) present_ring: usize,
    pub(crate) instance_grid: InstanceGrid,
    pub(crate) instance_stress_variant: InstanceStressVariant,
    pub(crate) instance_debug_view: InstanceDebugView,
    pub(crate) instance_layout: StressInstanceLayout,
    pub(crate) instance_materials: InstanceMaterials,
    pub(crate) sparse_feedback: SparseFeedbackMode,
    pub(crate) gpu_preset: GpuPreset,
    pub(crate) stress_block: Option<StressBlock>,
    pub(crate) sparse_texture_quality: SparseTextureQuality,
    pub(crate) render_policy: RenderPolicy,
    pub(crate) d3d_upload: D3dUploadMode,
    pub(crate) interop_fallback: InteropFallback,
    pub(crate) hot_reload: bool,
    pub(crate) raster_plan: HardwareRasterPlan,
}

impl LiveOptions {
    pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
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
            instance_debug_view: InstanceDebugView::Off,
            instance_layout: StressInstanceLayout::AoSoA32,
            instance_materials: InstanceMaterials::None,
            sparse_feedback: SparseFeedbackMode::Off,
            gpu_preset: GpuPreset::Auto,
            stress_block: None,
            sparse_texture_quality: SparseTextureQuality::Auto,
            render_policy: RenderPolicy::Auto,
            d3d_upload: D3dUploadMode::MappedCopy,
            interop_fallback: InteropFallback::NoInterop,
            hot_reload: true,
            raster_plan: HardwareRasterPlan::stock(),
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
                "--instance-debug-view" => {
                    options.instance_debug_view = parse_next(&mut args, "--instance-debug-view")?
                }
                "--instance-layout" => {
                    options.instance_layout = parse_next(&mut args, "--instance-layout")?
                }
                "--instance-materials" => {
                    options.instance_materials = parse_next(&mut args, "--instance-materials")?
                }
                "--sparse-feedback" => {
                    options.sparse_feedback = parse_next(&mut args, "--sparse-feedback")?
                }
                "--gpu-preset" => options.gpu_preset = parse_next(&mut args, "--gpu-preset")?,
                "--stress-block" => {
                    options.stress_block = Some(parse_next(&mut args, "--stress-block")?)
                }
                "--sparse-texture-quality" => {
                    options.sparse_texture_quality =
                        parse_next(&mut args, "--sparse-texture-quality")?
                }
                "--draw-policy" | "--raster-draw-policy" => {
                    options.raster_plan.draw_policy = parse_next(&mut args, arg.as_str())?;
                    options.raster_plan.sync_stock_material_to_draw_policy();
                }
                "--draw-depth" | "--raster-depth" => {
                    options.raster_plan.depth = parse_next(&mut args, arg.as_str())?
                }
                "--cull-order" | "--raster-cull-order" => {
                    options.raster_plan.cull_order = parse_next(&mut args, arg.as_str())?
                }
                "--visibility" | "--raster-visibility" => {
                    options.raster_plan.visibility = parse_next(&mut args, arg.as_str())?
                }
                "--min-projected-pixels" | "--raster-min-projected-pixels" => {
                    options.raster_plan.min_projected_millipixels = parse_min_projected_pixels(
                        arg.as_str(),
                        parse_next::<String>(&mut args, arg.as_str())?,
                    )?
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
                    "usage: neo-live-window [path.neo] [--title TEXT] [--width N] [--height N] [--frames N] [--seconds N] [--presenter d3d12-interop|d3d12|d3d11|gdi] [--mode live|kernel-throughput|mesh-demo|instance-stress|draw-stress|raster-stress] [--sample-every N] [--present-target-fps N] [--kernel-target-fps N] [--max-inflight N] [--present-ring N] [--instance-grid XxYxZ] [--instance-stress-variant baseline|fast|culled|tiled|macrocell] [--instance-debug-view off|tile-range|iterations|hit-miss] [--instance-layout aosoa32|aosoa64] [--instance-materials none|sparse-texture] [--sparse-feedback off|sampled|block|missing|atomic] [--gpu-preset auto|pascal|modern] [--stress-block WxH] [--sparse-texture-quality auto|full|pascal-fast] [--draw-policy draw-all|compute-culled] [--draw-depth auto|on|off] [--cull-order atomic-compact|stable-dense] [--visibility frustum|projected-size] [--min-projected-pixels N] [--render-policy auto|force-render|pause-when-empty] [--d3d-upload mapped-copy|update-subresource] [--interop-fallback no-interop|fail] [--hot-reload|--no-hot-reload]"
                ),
                value if value.starts_with('-') => bail!("unknown option `{value}`"),
                value => options.source_path = PathBuf::from(value),
            }
        }
        options.validate()?;
        Ok(options)
    }

    pub(crate) fn should_stop(&self, frame: u32, elapsed: Duration) -> bool {
        self.max_frames.is_some_and(|max| frame >= max)
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    pub(crate) fn should_stop_completed(&self, completed: u64, elapsed: Duration) -> bool {
        self.max_frames
            .is_some_and(|max| completed >= u64::from(max))
            || self
                .max_seconds
                .is_some_and(|max| elapsed.as_secs_f32() >= max)
    }

    pub(crate) fn validate(&self) -> Result<()> {
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
        if self.instance_stress_variant == InstanceStressVariant::Macrocell
            && self.instance_layout != StressInstanceLayout::AoSoA32
        {
            bail!(
                "--instance-stress-variant macrocell currently requires --instance-layout aosoa32"
            );
        }
        if self.instance_materials == InstanceMaterials::SparseTexture
            && self.instance_stress_variant != InstanceStressVariant::Macrocell
        {
            bail!(
                "--instance-materials sparse-texture currently requires --instance-stress-variant macrocell"
            );
        }
        if self.sparse_feedback.records_feedback()
            && self.instance_materials != InstanceMaterials::SparseTexture
        {
            bail!("--sparse-feedback requires --instance-materials sparse-texture");
        }
        Ok(())
    }

    pub(crate) fn present_interval(&self) -> Option<Duration> {
        self.present_target_fps
            .map(|fps| Duration::from_secs_f32(1.0 / fps))
    }

    pub(crate) fn kernel_cap(&self) -> Option<f32> {
        self.kernel_target_fps.filter(|fps| *fps > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunMode {
    Live,
    KernelThroughput,
    MeshDemo,
    InstanceStress,
    DrawStress,
}

impl std::str::FromStr for RunMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "live" => Ok(Self::Live),
            "kernel-throughput" => Ok(Self::KernelThroughput),
            "mesh-demo" => Ok(Self::MeshDemo),
            "instance-stress" => Ok(Self::InstanceStress),
            "draw-stress" | "raster-stress" => Ok(Self::DrawStress),
            _ => bail!(
                "unknown mode `{value}`; expected live, kernel-throughput, mesh-demo, instance-stress, draw-stress, or raster-stress"
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
            Self::DrawStress => f.write_str("draw-stress"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderPolicy {
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
pub(crate) enum RenderVisibility {
    Visible,
    Empty,
    Occluded,
    Minimized,
}

impl RenderVisibility {
    pub(crate) fn limit_max_inflight(self, max_inflight: u32) -> u32 {
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
pub(crate) struct WindowVisibilityState {
    pub(crate) minimized: bool,
    pub(crate) occluded: bool,
    pub(crate) focused: bool,
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
    pub(crate) fn render_visibility(self) -> RenderVisibility {
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
pub struct InstanceGrid {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

impl InstanceGrid {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn x(self) -> u32 {
        self.x
    }

    pub fn y(self) -> u32 {
        self.y
    }

    pub fn z(self) -> u32 {
        self.z
    }

    pub fn validate(self) -> Result<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            bail!("--instance-grid dimensions must be greater than zero");
        }
        self.count()
            .ok_or_else(|| anyhow!("--instance-grid instance count overflow"))?;
        Ok(())
    }

    pub fn count(self) -> Option<u32> {
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
pub(crate) enum InstanceStressVariant {
    Baseline,
    Fast,
    Culled,
    Tiled,
    Macrocell,
}

impl InstanceStressVariant {
    pub(crate) fn source_path(
        self,
        requested: &Path,
        layout: StressInstanceLayout,
        materials: InstanceMaterials,
        debug_view: InstanceDebugView,
        sparse_feedback: SparseFeedbackMode,
        gpu_preset: ResolvedGpuPreset,
    ) -> PathBuf {
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
                Self::Tiled => {
                    return requested.with_file_name(match layout {
                        StressInstanceLayout::AoSoA32 => "three_d_instances_tiled_aosoa32.neo",
                        StressInstanceLayout::AoSoA64 => "three_d_instances_tiled_aosoa64.neo",
                    });
                }
                Self::Macrocell => {
                    if materials == InstanceMaterials::SparseTexture {
                        if gpu_preset == ResolvedGpuPreset::Pascal
                            && layout == StressInstanceLayout::AoSoA32
                            && debug_view == InstanceDebugView::Off
                            && matches!(
                                sparse_feedback,
                                SparseFeedbackMode::Off | SparseFeedbackMode::Sampled
                            )
                        {
                            return requested.with_file_name(
                                "three_d_instances_macrocell_textured_pascal_aosoa32.neo",
                            );
                        }
                        return requested
                            .with_file_name("three_d_instances_macrocell_textured.neo");
                    }
                    return requested.with_file_name("three_d_instances_macrocell_aosoa32.neo");
                }
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
            "macrocell" => Ok(Self::Macrocell),
            _ => bail!(
                "unknown instance stress variant `{value}`; expected baseline, fast, culled, tiled, or macrocell"
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
            Self::Macrocell => f.write_str("macrocell"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstanceDebugView {
    Off,
    TileRange,
    Iterations,
    HitMiss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstanceMaterials {
    None,
    SparseTexture,
}

impl std::str::FromStr for InstanceMaterials {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "sparse-texture" => Ok(Self::SparseTexture),
            _ => bail!("unknown instance materials `{value}`; expected none or sparse-texture"),
        }
    }
}

impl std::fmt::Display for InstanceMaterials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::SparseTexture => f.write_str("sparse-texture"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SparseFeedbackMode {
    Off,
    Sampled,
    Block,
    Missing,
    Atomic,
}

impl SparseFeedbackMode {
    pub(crate) fn records_feedback(self) -> bool {
        self != Self::Off
    }

    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Sampled => 1,
            Self::Block => 2,
            Self::Missing => 3,
            Self::Atomic => 4,
        }
    }

    pub(crate) fn sample_rate(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Sampled => 16,
            Self::Block => 64,
            Self::Missing | Self::Atomic => 1,
        }
    }
}

impl std::str::FromStr for SparseFeedbackMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "on" | "sampled" => Ok(Self::Sampled),
            "block" => Ok(Self::Block),
            "missing" => Ok(Self::Missing),
            "atomic" => Ok(Self::Atomic),
            _ => bail!(
                "unknown sparse feedback mode `{value}`; expected off, sampled, block, missing, or atomic"
            ),
        }
    }
}

impl std::fmt::Display for SparseFeedbackMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::Sampled => f.write_str("sampled"),
            Self::Block => f.write_str("block"),
            Self::Missing => f.write_str("missing"),
            Self::Atomic => f.write_str("atomic"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuPreset {
    Auto,
    Pascal,
    Modern,
}

impl GpuPreset {
    pub(crate) fn resolve(self, device: &DeviceInfo) -> ResolvedGpuPreset {
        match self {
            Self::Pascal => ResolvedGpuPreset::Pascal,
            Self::Modern => ResolvedGpuPreset::Modern,
            Self::Auto if device.is_pascal_sm61() => ResolvedGpuPreset::Pascal,
            Self::Auto => ResolvedGpuPreset::Modern,
        }
    }
}

impl std::str::FromStr for GpuPreset {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "pascal" => Ok(Self::Pascal),
            "modern" => Ok(Self::Modern),
            _ => bail!("unknown GPU preset `{value}`; expected auto, pascal, or modern"),
        }
    }
}

impl std::fmt::Display for GpuPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Pascal => f.write_str("pascal"),
            Self::Modern => f.write_str("modern"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedGpuPreset {
    Pascal,
    Modern,
}

impl ResolvedGpuPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Pascal => "pascal",
            Self::Modern => "modern",
        }
    }

    pub(crate) fn default_stress_block(self) -> StressBlock {
        match self {
            Self::Pascal => StressBlock {
                width: 16,
                height: 8,
            },
            Self::Modern => StressBlock {
                width: 8,
                height: 8,
            },
        }
    }
}

impl std::fmt::Display for ResolvedGpuPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StressBlock {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl StressBlock {
    pub(crate) fn tuple(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl std::str::FromStr for StressBlock {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let parts: Vec<_> = value.split('x').collect();
        if parts.len() != 2 {
            bail!("invalid --stress-block `{value}`; expected WxH");
        }
        let width: u32 = parts[0]
            .parse()
            .with_context(|| format!("invalid width in --stress-block `{value}`"))?;
        let height: u32 = parts[1]
            .parse()
            .with_context(|| format!("invalid height in --stress-block `{value}`"))?;
        if width == 0 || height == 0 {
            bail!("--stress-block dimensions must be greater than zero");
        }
        Ok(Self { width, height })
    }
}

impl std::fmt::Display for StressBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SparseTextureQuality {
    Auto,
    Full,
    PascalFast,
}

impl SparseTextureQuality {
    pub(crate) fn resolve(self, preset: ResolvedGpuPreset) -> ResolvedSparseTextureQuality {
        match self {
            Self::Full => ResolvedSparseTextureQuality::Full,
            Self::PascalFast => ResolvedSparseTextureQuality::PascalFast,
            Self::Auto if preset == ResolvedGpuPreset::Pascal => {
                ResolvedSparseTextureQuality::PascalFast
            }
            Self::Auto => ResolvedSparseTextureQuality::Full,
        }
    }
}

impl std::str::FromStr for SparseTextureQuality {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "full" => Ok(Self::Full),
            "pascal-fast" => Ok(Self::PascalFast),
            _ => bail!(
                "unknown sparse texture quality `{value}`; expected auto, full, or pascal-fast"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedSparseTextureQuality {
    Full,
    PascalFast,
}

impl ResolvedSparseTextureQuality {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::PascalFast => "pascal-fast",
        }
    }
}

impl std::fmt::Display for ResolvedSparseTextureQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl InstanceDebugView {
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::TileRange => 1,
            Self::Iterations => 2,
            Self::HitMiss => 3,
        }
    }
}

impl std::str::FromStr for InstanceDebugView {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "tile-range" => Ok(Self::TileRange),
            "iterations" => Ok(Self::Iterations),
            "hit-miss" => Ok(Self::HitMiss),
            _ => bail!(
                "unknown instance debug view `{value}`; expected off, tile-range, iterations, or hit-miss"
            ),
        }
    }
}

impl std::fmt::Display for InstanceDebugView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::TileRange => f.write_str("tile-range"),
            Self::Iterations => f.write_str("iterations"),
            Self::HitMiss => f.write_str("hit-miss"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressInstanceLayout {
    AoSoA32,
    AoSoA64,
}

impl StressInstanceLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::AoSoA32 => "aosoa32",
            Self::AoSoA64 => "aosoa64",
        }
    }

    pub fn group_size(self) -> u32 {
        match self {
            Self::AoSoA32 => 32,
            Self::AoSoA64 => 64,
        }
    }

    pub fn data_layout(self) -> DataLayout {
        DataLayout::AoSoA {
            group_size: self.group_size(),
        }
    }
}

impl std::str::FromStr for StressInstanceLayout {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "aosoa32" => Ok(Self::AoSoA32),
            "aosoa64" => Ok(Self::AoSoA64),
            _ => bail!("unknown instance layout `{value}`; expected aosoa32 or aosoa64"),
        }
    }
}

impl std::fmt::Display for StressInstanceLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresenterKind {
    D3d12Interop,
    D3d12,
    D3d11,
    Gdi,
}

impl PresenterKind {
    pub(crate) fn uses_present_thread(self) -> bool {
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
pub(crate) enum InteropFallback {
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
pub(crate) enum D3dUploadMode {
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

pub(crate) fn parse_next<T: std::str::FromStr>(
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

pub(crate) fn parse_min_projected_pixels(flag: &str, value: String) -> Result<u32> {
    let pixels = value
        .parse::<f32>()
        .map_err(|err| anyhow!("invalid value for {flag}: {err}"))?;
    if !pixels.is_finite() || pixels < 0.0 {
        bail!("{flag} must be finite and non-negative");
    }
    let millipixels = f64::from(pixels) * 1000.0;
    if millipixels > f64::from(u32::MAX) {
        bail!("{flag} is too large");
    }
    Ok(millipixels.round() as u32)
}
